"""
混合检索引擎 — 向量 + 关键词融合
================================
借鉴 WeKnora 的 HybridSearch 和 RRF Fusion 设计，
将向量检索和 BM25 检索结果通过 Reciprocal Rank Fusion
算法融合，获得更全面准确的检索结果。
"""

import asyncio
from typing import Optional

from loguru import logger

from config import settings
from core.vector_store import VectorStore
from core.keyword_search import BM25SearchEngine
from core.embedding import EmbeddingEngine


class HybridSearchEngine:
    """混合检索引擎：向量检索 + BM25 关键词检索 + RRF 融合"""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_engine: BM25SearchEngine,
        embedding_engine: EmbeddingEngine,
    ):
        self.vector_store = vector_store
        self.bm25_engine = bm25_engine
        self.embedding_engine = embedding_engine

    async def search(
        self,
        query: str,
        top_k: int = 5,
        knowledge_base_ids: Optional[list[str]] = None,
        vector_threshold: float = None,
        keyword_threshold: float = None,
        alpha: float = None,
    ) -> list[dict]:
        """
        混合检索：向量 + 关键词 + RRF 融合

        Args:
            query: 查询文本
            top_k: 返回结果数
            knowledge_base_ids: 限定知识库范围
            vector_threshold: 向量检索阈值
            keyword_threshold: 关键词检索阈值
            alpha: 向量检索权重 (0~1)

        Returns:
            融合后的检索结果列表
        """
        alpha = alpha if alpha is not None else settings.HYBRID_ALPHA
        vector_threshold = vector_threshold or settings.VECTOR_THRESHOLD
        keyword_threshold = keyword_threshold or settings.KEYWORD_THRESHOLD

        # 并发执行向量检索和关键词检索
        vector_task = self._vector_search(query, top_k * 3, knowledge_base_ids, vector_threshold)
        keyword_task = self._keyword_search(query, top_k * 3, knowledge_base_ids, keyword_threshold)

        vector_results, keyword_results = await asyncio.gather(
            vector_task, keyword_task,
            return_exceptions=True,
        )

        if isinstance(vector_results, Exception):
            logger.error(f"向量检索失败: {vector_results}")
            vector_results = []
        if isinstance(keyword_results, Exception):
            logger.error(f"关键词检索失败: {keyword_results}")
            keyword_results = []

        logger.info(f"混合检索: 向量结果 {len(vector_results)}, 关键词结果 {len(keyword_results)}")

        # RRF 融合
        fused = self._rrf_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            alpha=alpha,
        )

        return fused[:top_k]

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        knowledge_base_ids: Optional[list[str]],
        threshold: float,
    ) -> list[dict]:
        """向量检索"""
        query_vector = await self.embedding_engine.embed_query(query)
        return await self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            threshold=threshold,
            knowledge_base_ids=knowledge_base_ids,
        )

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        knowledge_base_ids: Optional[list[str]],
        threshold: float,
    ) -> list[dict]:
        """关键词检索"""
        return self.bm25_engine.search(
            query=query,
            top_k=top_k,
            threshold=threshold,
            knowledge_base_ids=knowledge_base_ids,
        )

    def _rrf_fusion(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        alpha: float = 0.7,
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion 融合算法

        RRF_score = alpha * (1 / (k + rank_vector)) + (1 - alpha) * (1 / (k + rank_keyword))

        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            alpha: 向量检索权重
            k: RRF 常数

        Returns:
            融合排序后的结果列表
        """
        scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        # 向量检索结果排名
        for rank, item in enumerate(vector_results, 1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + alpha / (k + rank)
            chunk_data[cid] = item

        # 关键词检索结果排名
        for rank, item in enumerate(keyword_results, 1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + (1 - alpha) / (k + rank)
            if cid not in chunk_data:
                chunk_data[cid] = item

        # 按融合分数排序
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for cid, score in sorted_items:
            data = chunk_data[cid]
            results.append({
                "chunk_id": cid,
                "score": float(score),
                "vector_score": data.get("score", 0) if data in vector_results else 0,
                "keyword_score": data.get("score", 0) if data in keyword_results else 0,
                "metadata": data.get("metadata", {}),
            })

        return results
