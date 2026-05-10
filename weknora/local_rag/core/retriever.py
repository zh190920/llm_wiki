"""
检索器模块 - 统一检索接口
借鉴 WeKnora 的 plugin-based pipeline 设计，实现查询理解 → 混合检索 → 重排的流水线
"""
import asyncio
import logging
from typing import List, Optional

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
    3. 重排去重（Rerank & MMR）：LLM 重排 + MMR 多样性选择

    每个阶段都是可选的，支持灵活配置
    """

    def __init__(
        self,
        config: AppConfig,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Optional[Reranker] = None,
    ):
        self.config = config
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker or Reranker(
            llm_config=config.llm,
            retriever_config=config.retriever,
            embedder=embedder,
        )

    async def retrieve(
        self,
        params: SearchParams,
        use_query_understanding: bool = True,
        use_rerank: bool = True,
        use_graph: bool = False,
        graph_alpha: float = 0.2,
    ) -> List[SearchResult]:
        """
        执行完整检索流水线

        Args:
            params: 检索参数（含 doc_ids 文档范围过滤）
            use_query_understanding: 是否启用查询理解
            use_rerank: 是否启用重排
            use_graph: 是否启用图谱增强检索（三源 RRF）
            graph_alpha: 图谱检索权重

        Returns:
            检索结果列表
        """
        query = params.query

        # Stage 1: 查询理解（可选）
        if use_query_understanding:
            query = await self._query_understanding(params.query)

        # Stage 2: 混合检索
        query_embedding = await self.embedder.embed_query(query)

        match_types = params.match_types or [MatchType.VECTOR, MatchType.KEYWORD]
        doc_ids = params.doc_ids  # 文档路由过滤

        # 图谱增强模式：使用三源 RRF 融合（向量 + 关键词 + 图谱）
        if use_graph and MatchType.GRAPH in match_types:
            results = await self.vector_store.search_hybrid(
                query=query,
                query_embedding=query_embedding,
                top_k=params.top_k,
                alpha=self.config.retriever.hybrid_alpha,
                doc_ids=doc_ids,
                use_graph=True,
                graph_alpha=graph_alpha,
            )
        elif MatchType.GRAPH in match_types and MatchType.VECTOR not in match_types and MatchType.KEYWORD not in match_types:
            # 仅图谱检索
            results = await self.vector_store.search_graph(
                query=query,
                top_k=params.top_k,
                doc_ids=doc_ids,
            )
        elif len(match_types) > 1 and MatchType.VECTOR in match_types and MatchType.KEYWORD in match_types:
            # 混合检索（双源）
            results = await self.vector_store.search_hybrid(
                query=query,
                query_embedding=query_embedding,
                top_k=params.top_k,
                alpha=self.config.retriever.hybrid_alpha,
                doc_ids=doc_ids,
            )
        elif MatchType.VECTOR in match_types:
            # 仅向量检索
            results = await self.vector_store.search_vector(
                query_embedding=query_embedding,
                top_k=params.top_k,
                similarity_threshold=params.similarity_threshold,
                doc_ids=doc_ids,
            )
        else:
            # 仅关键词检索
            results = await self.vector_store.search_keyword(
                query=query,
                top_k=params.top_k,
                doc_ids=doc_ids,
            )

        # Stage 3: 重排
        if use_rerank and results:
            results = await self.reranker.rerank(
                query=query,
                results=results,
                top_k=self.config.retriever.rerank_top_k,
            )

        graph_info = f", 图谱增强=是(权重={graph_alpha})" if use_graph else ""
        if doc_ids:
            logger.info(
                f"检索完成(文档级预筛选): query='{params.query[:50]}', "
                f"限定文档={len(doc_ids)}个, 结果数={len(results)}{graph_info}"
            )
        else:
            logger.info(f"检索完成(全量): query='{params.query[:50]}', 结果数={len(results)}{graph_info}")
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

        prompt = f"""请将以下用户查询改写为更适合文档检索的形式。要求：
1. 保留核心意图和关键信息
2. 补充可能缺失的上下文
3. 使用更精确的关键词
4. 如果查询已经很清晰，直接返回原查询

用户查询：{query}

只输出改写后的查询，不要解释："""

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

    async def quick_search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        use_graph: bool = False,
        graph_alpha: float = 0.2,
    ) -> List[SearchResult]:
        """
        快速检索 - 跳过查询理解和 LLM 重排，适用于简单查询

        适合 RAG 快速问答场景

        Args:
            query: 查询文本
            top_k: 返回结果数
            doc_ids: 限定检索范围的文档 ID 列表
            use_graph: 是否启用图谱增强检索
            graph_alpha: 图谱检索权重
        """
        query_embedding = await self.embedder.embed_query(query)
        results = await self.vector_store.search_hybrid(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            alpha=self.config.retriever.hybrid_alpha,
            doc_ids=doc_ids,
            use_graph=use_graph,
            graph_alpha=graph_alpha,
        )
        graph_info = ", 图谱增强" if use_graph else ""
        if doc_ids:
            logger.info(
                f"快速检索完成(文档级预筛选{graph_info}): query='{query[:50]}', "
                f"限定文档={len(doc_ids)}个, 结果数={len(results)}"
            )
        else:
            logger.info(f"快速检索完成(全量{graph_info}): query='{query[:50]}', 结果数={len(results)}")
        return results
