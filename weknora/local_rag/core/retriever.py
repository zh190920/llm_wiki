"""
检索器模块 - 统一检索接口
借鉴 WeKnora 的 plugin-based pipeline 设计，实现查询理解 → 混合检索 → 重排的流水线
"""
import asyncio
import logging
from typing import Dict, List, Optional

from config.settings import AppConfig
from core.embedder import Embedder
from core.reranker import Reranker
from core.vector_store import VectorStore
from models.schemas import MatchType, SearchParams, SearchResult

logger = logging.getLogger(__name__)


class Retriever:
    """
    统一检索器 - 编排检索流水线

    流水线（借鉴 WeKnora Chat Pipeline 的插件链设计）：
    1. 查询理解（Query Understanding）：查询改写、意图分类
    2. 混合检索（Hybrid Search）：向量 + BM25 关键词并发检索
    3. 父上下文增强（Parent Context）：子块检索时回溯父块内容
    4. 图增强检索（Graph-enhanced）：基于知识图谱的扩展检索
    5. 重排去重（Rerank & MMR）：LLM 重排 + MMR 多样性选择

    每个阶段都是可选的，支持灵活配置
    """

    def __init__(
        self,
        config: AppConfig,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Optional[Reranker] = None,
        graph_builder=None,
    ):
        self.config = config
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker or Reranker(
            llm_config=config.llm,
            retriever_config=config.retriever,
            embedder=embedder,
        )
        self.graph_builder = graph_builder

    async def retrieve(
        self,
        params: SearchParams,
        use_query_understanding: bool = True,
        use_rerank: bool = True,
        use_graph: bool = False,
    ) -> List[SearchResult]:
        """
        执行完整检索流水线

        Args:
            params: 检索参数
            use_query_understanding: 是否启用查询理解
            use_rerank: 是否启用重排
            use_graph: 是否启用图增强检索

        Returns:
            检索结果列表
        """
        query = params.query

        # Stage 1: 查询理解（可选）
        if use_query_understanding:
            query = await self._query_understanding(params.query)

        # Stage 2: 混合检索
        # 使用 per-query 阈值覆盖（如果提供）
        similarity_threshold = (
            params.similarity_threshold_override
            if params.similarity_threshold_override is not None
            else params.similarity_threshold
        )

        query_embedding = await self.embedder.embed_query(query)

        match_types = params.match_types or [MatchType.VECTOR, MatchType.KEYWORD]

        if len(match_types) > 1 and MatchType.VECTOR in match_types and MatchType.KEYWORD in match_types:
            # 混合检索
            results = await self.vector_store.search_hybrid(
                query=query,
                query_embedding=query_embedding,
                top_k=params.top_k,
                alpha=self.config.retriever.hybrid_alpha,
            )
        elif MatchType.VECTOR in match_types:
            # 仅向量检索
            results = await self.vector_store.search_vector(
                query_embedding=query_embedding,
                top_k=params.top_k,
                similarity_threshold=similarity_threshold,
            )
        else:
            # 仅关键词检索
            results = await self.vector_store.search_keyword(
                query=query,
                top_k=params.top_k,
            )

        # Stage 3: 父上下文增强
        results = self._enrich_parent_context(results)

        # Stage 4: 图增强检索（可选）
        if use_graph and self.graph_builder:
            graph_results = await self._graph_enhanced_retrieval(query, results)
            if graph_results:
                alpha_graph = self.config.retriever.graph_alpha
                results = self._rrf_fuse_three(
                    results, graph_results,
                    alpha_graph=alpha_graph,
                )

        # Stage 5: 重排
        if use_rerank and results:
            results = await self.reranker.rerank(
                query=query,
                results=results,
                top_k=self.config.retriever.rerank_top_k,
            )

        logger.info(f"检索完成: query='{params.query[:50]}', 结果数={len(results)}")
        return results

    def _enrich_parent_context(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        父上下文增强 - 当检索到子块时，将父块内容附加到元数据中

        借鉴 WeKnora 的 Parent-Child 上下文回溯设计：
        - 如果检索结果中的块有 parent_chunk_id，获取父块内容
        - 将父块内容存入 metadata["parent_context"]，供后续生成使用
        """
        for result in results:
            chunk = result.chunk
            if chunk.parent_chunk_id:
                parent_chunk = self.vector_store.get_chunk_by_id(chunk.parent_chunk_id)
                if parent_chunk:
                    chunk.metadata["parent_context"] = parent_chunk.content
                    chunk.metadata["parent_chunk_id"] = chunk.parent_chunk_id
        return results

    async def _graph_enhanced_retrieval(
        self, query: str, existing_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        图增强检索 - 基于知识图谱扩展检索结果

        借鉴 WeKnora 的 GraphRAG 设计：
        1. 从现有检索结果中提取相关实体
        2. 查询知识图谱获取关联块
        3. 返回图增强的检索结果
        """
        if not self.graph_builder or not self.graph_builder._entity_map:
            return []

        # 从检索结果中查找匹配的实体
        related_chunk_ids = set()
        for result in existing_results:
            # 尝试在图谱实体中查找匹配
            content_lower = result.chunk.content.lower()
            for entity_title, entity in self.graph_builder._entity_map.items():
                if entity_title.lower() in content_lower:
                    chunk_ids = self.graph_builder.get_related_chunks(entity_title, depth=1)
                    related_chunk_ids.update(chunk_ids)

        # 获取图增强的块
        graph_results: List[SearchResult] = []
        existing_chunk_ids = {r.chunk.chunk_id for r in existing_results}

        for chunk_id in related_chunk_ids:
            if chunk_id in existing_chunk_ids:
                continue  # 避免重复
            chunk = self.vector_store.get_chunk_by_id(chunk_id)
            if chunk:
                graph_results.append(SearchResult(
                    chunk=chunk,
                    score=0.3,  # 图增强结果给予基础分数
                    match_type=MatchType.GRAPH,
                ))

        logger.info(f"图增强检索: 新增 {len(graph_results)} 个块")
        max_related = self.config.retriever.graph_max_related
        return graph_results[:max_related]

    def _rrf_fuse_three(
        self,
        vector_bm25_results: List[SearchResult],
        graph_results: List[SearchResult],
        rrf_k: int = 60,
        alpha_vb: float = 0.7,
        alpha_graph: float = 0.3,
    ) -> List[SearchResult]:
        """
        三源 RRF 融合 - 向量/BM25 + 图增强

        使用 Reciprocal Rank Fusion 融合三个检索来源
        alpha_vb + alpha_graph 应等于 1.0
        """
        score_map: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}

        # 向量/BM25 结果
        for rank, result in enumerate(vector_bm25_results):
            cid = result.chunk.chunk_id
            rrf_score = alpha_vb / (rrf_k + rank + 1)
            score_map[cid] = score_map.get(cid, 0.0) + rrf_score
            if cid not in chunk_map or result.score > chunk_map[cid].score:
                chunk_map[cid] = result

        # 图增强结果
        for rank, result in enumerate(graph_results):
            cid = result.chunk.chunk_id
            rrf_score = alpha_graph / (rrf_k + rank + 1)
            score_map[cid] = score_map.get(cid, 0.0) + rrf_score
            if cid not in chunk_map:
                chunk_map[cid] = result

        # 按 RRF 分数排序
        sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)

        results: List[SearchResult] = []
        for cid in sorted_ids:
            result = chunk_map[cid]
            results.append(SearchResult(
                chunk=result.chunk,
                score=score_map[cid],
                match_type=result.match_type,
            ))

        return results

    async def _query_understanding(self, query: str) -> str:
        """
        查询理解 - 改写用户查询以提高检索效果

        借鉴 WeKnora 的 PluginQueryUnderstand 设计：
        - 查询改写：将口语化查询转为更精确的检索查询
        - 意图识别：判断是否需要多轮检索
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
            timeout=self.config.llm.timeout,
        )

        # 尝试使用模板管理器获取提示词
        try:
            from agent.prompts import _get_template_manager
            manager = _get_template_manager()
            prompt = manager.get_prompt("query_understanding", query=query)
            if not prompt:
                prompt = self._build_query_understanding_prompt(query)
        except Exception:
            prompt = self._build_query_understanding_prompt(query)

        try:
            response = await client.chat.completions.create(
                model=self.config.llm.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            rewritten = response.choices[0].message.content.strip()
            if rewritten and len(rewritten) > 2:
                logger.info(f"查询改写: '{query}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"查询理解失败，使用原始查询: {e}")

        return query

    @staticmethod
    def _build_query_understanding_prompt(query: str) -> str:
        """构建查询理解提示词（后备）"""
        return f"""请将以下用户查询改写为更适合文档检索的形式。要求：
1. 保留核心意图和关键信息
2. 补充可能缺失的上下文
3. 使用更精确的关键词
4. 如果查询已经很清晰，直接返回原查询

用户查询：{query}

只输出改写后的查询，不要解释："""

    async def quick_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        快速检索 - 跳过查询理解和 LLM 重排，适用于简单查询

        适合 RAG 快速问答场景
        """
        query_embedding = await self.embedder.embed_query(query)
        results = await self.vector_store.search_hybrid(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            alpha=self.config.retriever.hybrid_alpha,
        )

        # 父上下文增强
        results = self._enrich_parent_context(results)

        return results
