"""
重排模块 - LLM 驱动的检索结果重排
借鉴 WeKnora 的 PluginRerank 设计，支持 MMR 多样性去重
"""
import asyncio
import logging
from typing import List, Optional

import numpy as np
from openai import AsyncOpenAI

from config.settings import LLMConfig, RetrieverConfig
from core.embedder import Embedder
from models.schemas import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """
    检索结果重排器

    特性：
    - LLM 打分重排：利用 LLM 判断文档与查询的相关性
    - MMR 多样性去重：避免返回内容高度重复的结果
    - 复合评分：综合原始分数 + 重排分数
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        retriever_config: Optional[RetrieverConfig] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.llm_config = llm_config or LLMConfig()
        self.retriever_config = retriever_config or RetrieverConfig()
        self._client = AsyncOpenAI(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url,
            timeout=self.llm_config.timeout,
        )
        self._embedder = embedder

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
        use_llm: bool = True,
        use_mmr: bool = True,
    ) -> List[SearchResult]:
        """
        重排检索结果

        流程：
        1. LLM 打分重排（可选）
        2. MMR 多样性去重（可选）
        3. 取 top_k

        Args:
            query: 查询文本
            results: 原始检索结果
            top_k: 返回数量
            use_llm: 是否使用 LLM 重排
            use_mmr: 是否使用 MMR 去重
        """
        if not results:
            return []

        top_k = top_k or self.retriever_config.rerank_top_k

        # Step 1: LLM 重排
        if use_llm and len(results) > 1:
            results = await self._llm_rerank(query, results)

        # Step 2: MMR 多样性去重
        if use_mmr and len(results) > top_k:
            results = await self._mmr_diversify(query, results, top_k)

        return results[:top_k]

    async def _llm_rerank(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """使用 LLM 对检索结果进行相关性打分"""
        if len(results) <= 3:
            # 结果少时不需要 LLM 重排
            return sorted(results, key=lambda r: r.score, reverse=True)

        # 构建重排 prompt
        doc_list = []
        for i, r in enumerate(results):
            # 截断过长的内容
            content = r.chunk.content[:500]
            doc_list.append(f"[文档{i+1}] (原始分数: {r.score:.3f})\n{content}")

        docs_text = "\n\n".join(doc_list)

        prompt = f"""请根据查询问题，对以下文档片段按相关性从高到低进行排序打分。

查询问题：{query}

{docs_text}

请按以下格式输出，每行一个，不要输出其他内容：
文档编号|相关性分数(0-1)|简要理由

示例：
1|0.95|直接回答了查询问题
3|0.7|部分相关
2|0.3|间接相关"""

        try:
            response = await self._client.chat.completions.create(
                model=self.llm_config.chat_model,
                messages=[
                    {"role": "system", "content": "你是一个精准的文档相关性评估专家。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content.strip()
            return self._parse_rerank_results(result_text, results)

        except Exception as e:
            logger.warning(f"LLM 重排失败，回退到原始排序: {e}")
            return sorted(results, key=lambda r: r.score, reverse=True)

    def _parse_rerank_results(
        self, text: str, original_results: List[SearchResult]
    ) -> List[SearchResult]:
        """解析 LLM 重排结果"""
        reranked: List[tuple[int, float]] = []

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split("|")
                if len(parts) >= 2:
                    doc_idx = int(parts[0].strip()) - 1
                    score = float(parts[1].strip())
                    if 0 <= doc_idx < len(original_results):
                        reranked.append((doc_idx, score))
            except (ValueError, IndexError):
                continue

        if not reranked:
            return sorted(original_results, key=lambda r: r.score, reverse=True)

        # 按重排分数排序
        reranked.sort(key=lambda x: x[1], reverse=True)

        # 构建新结果列表，复合评分
        results = []
        for doc_idx, llm_score in reranked:
            original = original_results[doc_idx]
            # 复合评分：0.6 * LLM分数 + 0.3 * 原始分数 + 0.1 * 来源权重
            composite_score = (
                0.6 * llm_score
                + 0.3 * original.score
                + 0.1 * (1.0 if original.match_type.value == "vector" else 0.8)
            )
            results.append(SearchResult(
                chunk=original.chunk,
                score=composite_score,
                match_type=original.match_type,
            ))

        return results

    async def _mmr_diversify(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Maximal Marginal Relevance (MMR) 多样性选择

        MMR = λ * Sim(q, d) - (1-λ) * max(Sim(d, d_i)) for d_i in selected

        借鉴 WeKnora 的 MMR 实现，λ=0.7
        """
        lam = self.retriever_config.mmr_lambda

        if not self._embedder:
            # 没有 embedder 时退化为简单去重
            return self._simple_diversify(results, top_k)

        # 获取所有结果的嵌入向量
        texts = [r.chunk.content for r in results]
        try:
            embeddings = await self._embedder.embed_documents(texts)
        except Exception as e:
            logger.warning(f"MMR 嵌入失败，回退到简单去重: {e}")
            return self._simple_diversify(results, top_k)

        emb_matrix = np.array(embeddings, dtype=np.float32)
        # L2 归一化
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb_matrix = emb_matrix / norms

        selected_indices: List[int] = []
        remaining = set(range(len(results)))

        # 选择第一个（分数最高的）
        if results:
            best_idx = max(remaining, key=lambda i: results[i].score)
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        while remaining and len(selected_indices) < top_k:
            best_mmr = -float("inf")
            best_idx = -1

            for i in remaining:
                # 相关性分数
                relevance = results[i].score

                # 与已选文档的最大相似度
                if selected_indices:
                    selected_embs = emb_matrix[selected_indices]
                    similarities = np.dot(selected_embs, emb_matrix[i])
                    max_similarity = float(np.max(similarities))
                else:
                    max_similarity = 0.0

                # MMR 分数
                mmr_score = lam * relevance - (1 - lam) * max_similarity

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return [results[i] for i in selected_indices]

    @staticmethod
    def _simple_diversify(results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """简单去重：基于内容相似度去除重复"""
        seen_contents: set[str] = set()
        diverse_results: List[SearchResult] = []

        for result in results:
            # 使用前100字符作为去重键
            content_key = result.chunk.content[:100]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                diverse_results.append(result)

            if len(diverse_results) >= top_k:
                break

        return diverse_results
