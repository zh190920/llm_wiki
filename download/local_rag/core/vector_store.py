"""
向量存储模块 - 基于 FAISS 的高性能向量检索
借鉴 WeKnora 的 retriever 设计，支持向量检索 + BM25 关键词检索的混合模式
"""
import asyncio
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import RetrieverConfig
from models.schemas import Chunk, MatchType, SearchResult

logger = logging.getLogger(__name__)


class VectorStore:
    """
    向量存储引擎

    特性：
    - FAISS 向量索引（L2 + 内积支持）
    - BM25 关键词检索
    - 混合检索（向量 + 关键词加权融合）
    - 持久化存储（索引 + 元数据）
    - 异步并发安全
    """

    def __init__(self, config: Optional[RetrieverConfig] = None, dim: int = 1536):
        self.config = config or RetrieverConfig()
        self.dim = dim

        # FAISS 索引
        self._index: Optional[faiss.IndexFlatIP] = None  # 内积索引（归一化后即余弦相似度）
        self._chunks: List[Chunk] = []  # chunk_id -> Chunk 映射（有序）
        self._id_map: Dict[str, int] = {}  # chunk_id -> FAISS 内部索引

        # BM25 索引
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: List[List[str]] = []

        # 锁（异步环境下的简单互斥）
        self._lock = asyncio.Lock()

        self._init_index()

    def _init_index(self):
        """初始化 FAISS 索引"""
        self._index = faiss.IndexFlatIP(self.dim)  # 内积索引

    async def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """
        添加文档块和对应的嵌入向量

        Args:
            chunks: 文档块列表
            embeddings: 对应的嵌入向量列表
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks 数量 ({len(chunks)}) 与 embeddings 数量 ({len(embeddings)}) 不匹配")

        async with self._lock:
            vectors = np.array(embeddings, dtype=np.float32)

            # L2 归一化（使内积等于余弦相似度）
            faiss.normalize_L2(vectors)

            start_idx = self._index.ntotal
            self._index.add(vectors)

            for i, chunk in enumerate(chunks):
                faiss_idx = start_idx + i
                self._chunks.append(chunk)
                self._id_map[chunk.chunk_id] = faiss_idx

            # 重建 BM25 索引
            self._rebuild_bm25()

            logger.info(f"添加 {len(chunks)} 个块到向量存储，总计 {self._index.ntotal} 个块")

    async def search_vector(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        similarity_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """向量检索"""
        if self._index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # 搜索 top_k * 2 个结果，然后按阈值过滤
        search_k = min(top_k * 2, self._index.ntotal)
        scores, indices = self._index.search(query_vec, search_k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score < similarity_threshold:
                continue
            if idx < len(self._chunks):
                results.append(SearchResult(
                    chunk=self._chunks[idx],
                    score=float(score),
                    match_type=MatchType.VECTOR,
                ))

        return results[:top_k]

    async def search_keyword(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """BM25 关键词检索"""
        if self._bm25 is None or not self._bm25_corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 获取 top_k 结果
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[SearchResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            if idx < len(self._chunks):
                # BM25 分数归一化到 [0, 1]
                normalized_score = min(float(scores[idx]) / 30.0, 1.0)
                results.append(SearchResult(
                    chunk=self._chunks[idx],
                    score=normalized_score,
                    match_type=MatchType.KEYWORD,
                ))

        return results

    async def search_hybrid(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        alpha: float = 0.7,  # 向量检索权重
    ) -> List[SearchResult]:
        """
        混合检索 - 融合向量检索和关键词检索结果

        借鉴 WeKnora 的 composite retriever 设计：
        - alpha 控制向量和关键词的权重
        - 使用 Reciprocal Rank Fusion (RRF) 融合排序

        Args:
            query: 查询文本
            query_embedding: 查询嵌入向量
            top_k: 返回结果数
            alpha: 向量检索权重 (0~1)
        """
        # 并发执行两种检索
        vector_task = self.search_vector(
            query_embedding, top_k=top_k * 2,
            similarity_threshold=0.0,
        )
        keyword_task = self.search_keyword(query, top_k=top_k * 2)

        vector_results, keyword_results = await asyncio.gather(
            vector_task, keyword_task
        )

        # Reciprocal Rank Fusion (RRF) 融合
        rrf_k = 60  # RRF 常数
        score_map: Dict[str, float] = {}  # chunk_id -> rrf score
        chunk_map: Dict[str, SearchResult] = {}  # chunk_id -> best result

        for rank, result in enumerate(vector_results):
            cid = result.chunk.chunk_id
            rrf_score = alpha / (rrf_k + rank + 1)
            score_map[cid] = score_map.get(cid, 0.0) + rrf_score
            if cid not in chunk_map or result.score > chunk_map[cid].score:
                chunk_map[cid] = result

        for rank, result in enumerate(keyword_results):
            cid = result.chunk.chunk_id
            rrf_score = (1 - alpha) / (rrf_k + rank + 1)
            score_map[cid] = score_map.get(cid, 0.0) + rrf_score
            if cid not in chunk_map:
                chunk_map[cid] = result

        # 按 RRF 分数排序
        sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)

        results: List[SearchResult] = []
        for cid in sorted_ids[:top_k]:
            result = chunk_map[cid]
            # 更新分数为融合分数
            results.append(SearchResult(
                chunk=result.chunk,
                score=score_map[cid],
                match_type=result.match_type,
            ))

        return results

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """删除指定文档的所有块（需要重建索引）"""
        async with self._lock:
            # 找到需要保留的块
            kept_chunks = [c for c in self._chunks if c.doc_id != doc_id]
            removed_count = len(self._chunks) - len(kept_chunks)

            if removed_count > 0:
                self._chunks = kept_chunks
                # 需要重新添加（FAISS 不支持删除）
                # 注意：调用方需要重新嵌入并添加
                self._init_index()
                self._id_map.clear()
                self._rebuild_bm25()
                logger.info(f"删除文档 {doc_id} 的 {removed_count} 个块，索引已清空，需重新添加")

            return removed_count

    def _rebuild_bm25(self):
        """重建 BM25 索引"""
        if not self._chunks:
            self._bm25 = None
            self._bm25_corpus = []
            return

        self._bm25_corpus = [self._tokenize(chunk.content) for chunk in self._chunks]
        self._bm25 = BM25Okapi(self._bm25_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        中文优化的分词器

        中文：尝试 jieba 分词（安装后自动启用），否则按 bigram 切分
        英文：按空格分词
        数字：保留完整数字串
        """
        tokens: List[str] = []

        # --- 中文分词 ---
        # 提取中文片段
        chinese_segments = re.findall(r'[\u4e00-\u9fff]+', text)
        try:
            import jieba
            for seg in chinese_segments:
                tokens.extend(jieba.lcut(seg))
        except ImportError:
            # jieba 未安装，使用 bigram（相邻两字组合）提升召回率
            for seg in chinese_segments:
                # 单字
                tokens.extend(list(seg))
                # bigram
                for i in range(len(seg) - 1):
                    tokens.append(seg[i:i+2])

        # --- 英文分词 ---
        tokens.extend(re.findall(r'[a-zA-Z]+', text.lower()))

        # --- 数字 ---
        tokens.extend(re.findall(r'\d+', text))

        # --- 中英混合术语（如 RAG系统 → rag, 系统）已被上面覆盖 ---

        return tokens

    async def save(self, directory: str):
        """持久化存储向量索引和元数据"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        async with self._lock:
            # 保存 FAISS 索引
            faiss.write_index(self._index, str(dir_path / "faiss.index"))

            # 保存元数据
            metadata = {
                "chunks": [c.model_dump() for c in self._chunks],
                "id_map": self._id_map,
                "dim": self.dim,
            }
            with open(dir_path / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"向量存储已保存到 {directory}，共 {self._index.ntotal} 个块")

    async def load(self, directory: str):
        """从磁盘加载向量索引和元数据"""
        dir_path = Path(directory)

        if not (dir_path / "faiss.index").exists():
            logger.warning(f"向量存储目录不存在: {directory}")
            return

        async with self._lock:
            # 加载 FAISS 索引
            self._index = faiss.read_index(str(dir_path / "faiss.index"))

            # 加载元数据
            with open(dir_path / "metadata.pkl", "rb") as f:
                metadata = pickle.load(f)

            self._chunks = [Chunk(**c) for c in metadata["chunks"]]
            self._id_map = metadata["id_map"]
            self.dim = metadata.get("dim", self.dim)

            # 重建 BM25 索引
            self._rebuild_bm25()

            logger.info(f"向量存储已加载: {directory}，共 {self._index.ntotal} 个块")

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    def get_chunks_by_doc_id(self, doc_id: str) -> List[Chunk]:
        """获取指定文档的所有块"""
        return [c for c in self._chunks if c.doc_id == doc_id]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """根据 chunk_id 获取块"""
        for c in self._chunks:
            if c.chunk_id == chunk_id:
                return c
        return None


# 需要 import re
import re
