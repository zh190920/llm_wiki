"""
向量存储模块 - 基于 FAISS 的高性能向量检索
借鉴 WeKnora 的 retriever 设计，支持向量检索 + BM25 关键词检索的混合模式

核心设计：文档级预筛选检索
- 当指定 doc_ids 时，先构建仅包含匹配文档的子索引
- 在子索引空间中执行检索，而非全量检索后再过滤
- 这确保了"先确定检索手册范围，再在范围内检索"的语义
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
    - 图谱检索（通过 KnowledgeGraphBuilder 图遍历获取相关 chunk）
    - 三源 RRF 融合（向量 + 关键词 + 图谱）
    - 文档级预筛选检索：指定 doc_ids 时先构建子索引再检索
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

        # 文档到块索引的映射：doc_id -> [chunk_index_in_chunks_list, ...]
        # 用于快速定位某文档的所有块，构建子索引
        self._doc_id_to_indices: Dict[str, List[int]] = {}

        # chunk_id 到 chunks 列表索引的反向映射（快速按 chunk_id 查找）
        self._chunk_id_to_index: Dict[str, int] = {}

        # BM25 索引
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: List[List[str]] = []

        # 图谱构建器引用（由外部注入，用于 graph 增强检索）
        self._graph_builder = None

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
                chunk_list_idx = len(self._chunks)
                self._chunks.append(chunk)
                self._id_map[chunk.chunk_id] = faiss_idx

                # 维护 chunk_id -> chunks 列表索引的反向映射
                self._chunk_id_to_index[chunk.chunk_id] = chunk_list_idx

                # 维护 doc_id -> indices 映射
                if chunk.doc_id not in self._doc_id_to_indices:
                    self._doc_id_to_indices[chunk.doc_id] = []
                self._doc_id_to_indices[chunk.doc_id].append(chunk_list_idx)

            # 重建 BM25 索引
            self._rebuild_bm25()

            logger.info(f"添加 {len(chunks)} 个块到向量存储，总计 {self._index.ntotal} 个块")

    def _build_sub_index(self, doc_ids: List[str]) -> Tuple[Optional[faiss.IndexFlatIP], List[int]]:
        """
        构建仅包含指定文档的 FAISS 子索引

        这是实现"先确定检索手册范围，再在范围内检索"的核心方法：
        1. 根据 doc_ids 找到所有属于这些文档的块索引
        2. 从主索引中提取这些块的向量
        3. 构建仅包含这些向量的子索引
        4. 返回子索引和映射表（子索引位置 -> 原始块位置）

        Args:
            doc_ids: 文档 ID 列表

        Returns:
            (sub_index, chunk_map) - 子索引和映射表
            如果指定文档无内容，返回 (None, [])
        """
        doc_id_set = set(doc_ids)

        # 收集匹配文档的所有块索引
        sub_chunk_indices = []
        for doc_id in doc_ids:
            if doc_id in self._doc_id_to_indices:
                sub_chunk_indices.extend(self._doc_id_to_indices[doc_id])

        if not sub_chunk_indices:
            return None, []

        # 从主索引中重建这些块的向量
        # IndexFlatIP 支持 reconstruct 方法
        vectors = []
        for idx in sub_chunk_indices:
            if idx < self._index.ntotal:
                vec = self._index.reconstruct(idx)
                vectors.append(vec)

        if not vectors:
            return None, []

        # 构建子索引
        sub_vectors = np.array(vectors, dtype=np.float32)
        sub_index = faiss.IndexFlatIP(self.dim)
        sub_index.add(sub_vectors)

        return sub_index, sub_chunk_indices

    def _build_sub_bm25(self, doc_ids: List[str]) -> Tuple[Optional[BM25Okapi], List[int]]:
        """
        构建仅包含指定文档的 BM25 子索引

        与 FAISS 子索引类似，BM25 也只在匹配文档的块上构建索引和评分，
        而非全量评分后再过滤。

        Args:
            doc_ids: 文档 ID 列表

        Returns:
            (sub_bm25, chunk_map) - 子 BM25 索引和映射表
            如果指定文档无内容，返回 (None, [])
        """
        doc_id_set = set(doc_ids)

        # 收集匹配文档的所有块索引
        sub_chunk_indices = []
        for doc_id in doc_ids:
            if doc_id in self._doc_id_to_indices:
                sub_chunk_indices.extend(self._doc_id_to_indices[doc_id])

        if not sub_chunk_indices:
            return None, []

        # 构建子语料库
        sub_corpus = [self._bm25_corpus[idx] for idx in sub_chunk_indices if idx < len(self._bm25_corpus)]

        if not sub_corpus:
            return None, []

        sub_bm25 = BM25Okapi(sub_corpus)
        return sub_bm25, sub_chunk_indices

    async def search_vector(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        doc_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        向量检索 - 文档级预筛选

        当指定 doc_ids 时：
        1. 先构建仅包含匹配文档的子 FAISS 索引
        2. 在子索引中检索，确保所有结果都来自匹配文档
        3. 将结果映射回原始块

        Args:
            query_embedding: 查询嵌入向量
            top_k: 返回结果数
            similarity_threshold: 相似度阈值
            doc_ids: 限定检索范围的文档 ID 列表（None 表示全量检索）
        """
        if self._index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        if doc_ids is not None:
            # ========================================
            # 文档级预筛选：先构建子索引，再在子空间中检索
            # ========================================
            sub_index, chunk_map = self._build_sub_index(doc_ids)
            if sub_index is None or sub_index.ntotal == 0:
                logger.info(f"文档级预筛选: 指定文档无内容，doc_ids={doc_ids}")
                return []

            actual_k = min(top_k, sub_index.ntotal)
            scores, indices = sub_index.search(query_vec, actual_k)

            results: List[SearchResult] = []
            for score, sub_idx in zip(scores[0], indices[0]):
                if sub_idx < 0:
                    continue
                if score < similarity_threshold:
                    continue
                # 映射回原始块位置
                original_idx = chunk_map[sub_idx]
                if original_idx < len(self._chunks):
                    results.append(SearchResult(
                        chunk=self._chunks[original_idx],
                        score=float(score),
                        match_type=MatchType.VECTOR,
                    ))

            logger.info(
                f"文档级预筛选向量检索: 搜索空间={sub_index.ntotal}块(来自{len(doc_ids)}个文档), "
                f"命中={len(results)}/{top_k}"
            )
            return results
        else:
            # ========================================
            # 全量检索：不限定文档范围
            # ========================================
            search_k = min(top_k, self._index.ntotal)
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
        doc_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        BM25 关键词检索 - 文档级预筛选

        当指定 doc_ids 时：
        1. 先构建仅包含匹配文档的子 BM25 索引
        2. 在子索引中评分，确保所有结果都来自匹配文档
        3. 将结果映射回原始块

        Args:
            query: 查询文本
            top_k: 返回结果数
            doc_ids: 限定检索范围的文档 ID 列表（None 表示全量检索）
        """
        if self._bm25 is None or not self._bm25_corpus:
            return []

        tokenized_query = self._tokenize(query)

        if doc_ids is not None:
            # ========================================
            # 文档级预筛选：先构建子 BM25 索引，再在子空间中评分
            # ========================================
            sub_bm25, chunk_map = self._build_sub_bm25(doc_ids)
            if sub_bm25 is None:
                logger.info(f"文档级预筛选: 指定文档无 BM25 内容，doc_ids={doc_ids}")
                return []

            scores = sub_bm25.get_scores(tokenized_query)

            # 获取 top_k 结果
            actual_k = min(top_k * 2, len(scores))
            top_sub_indices = np.argsort(scores)[::-1][:actual_k]

            results: List[SearchResult] = []
            for sub_idx in top_sub_indices:
                if scores[sub_idx] <= 0:
                    continue
                # 映射回原始块位置
                original_idx = chunk_map[sub_idx]
                if original_idx < len(self._chunks):
                    # BM25 分数归一化到 [0, 1]
                    normalized_score = min(float(scores[sub_idx]) / 30.0, 1.0)
                    results.append(SearchResult(
                        chunk=self._chunks[original_idx],
                        score=normalized_score,
                        match_type=MatchType.KEYWORD,
                    ))
                if len(results) >= top_k:
                    break

            logger.info(
                f"文档级预筛选BM25检索: 搜索空间={len(chunk_map)}块(来自{len(doc_ids)}个文档), "
                f"命中={len(results)}/{top_k}"
            )
            return results
        else:
            # ========================================
            # 全量检索：不限定文档范围
            # ========================================
            scores = self._bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]

            results: List[SearchResult] = []
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                if idx < len(self._chunks):
                    normalized_score = min(float(scores[idx]) / 30.0, 1.0)
                    results.append(SearchResult(
                        chunk=self._chunks[idx],
                        score=normalized_score,
                        match_type=MatchType.KEYWORD,
                    ))
                if len(results) >= top_k:
                    break

            return results

    async def search_hybrid(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        alpha: float = 0.7,  # 向量检索权重
        doc_ids: Optional[List[str]] = None,
        use_graph: bool = False,
        graph_alpha: float = 0.2,  # 图谱检索权重（仅 use_graph=True 时生效）
    ) -> List[SearchResult]:
        """
        混合检索 - 支持双源（向量+关键词）和三源（向量+关键词+图谱）RRF融合

        三源模式（use_graph=True）：
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  向量检索  │  │ 关键词检索 │  │ 图谱检索  │
        │ (FAISS)  │  │  (BM25)  │  │ (Graph)  │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                    三源 RRF 融合
                            │
                       排序结果
        权重分配：
        - 向量检索: (alpha - graph_alpha/2)  默认 0.6
        - 关键词检索: (1 - alpha - graph_alpha/2) 默认 0.2
        - 图谱检索: graph_alpha  默认 0.2

        双源模式（use_graph=False，兼容原有行为）：
        - 向量检索: alpha  默认 0.7
        - 关键词检索: (1-alpha)  默认 0.3

        借鉴 WeKnora 的 composite retriever 设计：
        - 使用 Reciprocal Rank Fusion (RRF) 融合排序

        Args:
            query: 查询文本
            query_embedding: 查询嵌入向量
            top_k: 返回结果数
            alpha: 向量检索权重 (0~1)
            doc_ids: 限定检索范围的文档 ID 列表（None 表示全量检索）
            use_graph: 是否启用图谱增强检索（三源 RRF）
            graph_alpha: 图谱检索权重（仅 use_graph=True 时生效）
        """
        # 并发执行检索
        vector_task = self.search_vector(
            query_embedding, top_k=top_k * 2,
            similarity_threshold=0.0,
            doc_ids=doc_ids,
        )
        keyword_task = self.search_keyword(query, top_k=top_k * 2, doc_ids=doc_ids)

        if use_graph and self._graph_builder is not None:
            graph_task = self.search_graph(query, top_k=top_k * 2, doc_ids=doc_ids)
            vector_results, keyword_results, graph_results = await asyncio.gather(
                vector_task, keyword_task, graph_task
            )
        else:
            vector_results, keyword_results = await asyncio.gather(
                vector_task, keyword_task
            )
            graph_results = []

        # Reciprocal Rank Fusion (RRF) 融合
        rrf_k = 60  # RRF 常数
        score_map: Dict[str, float] = {}  # chunk_id -> rrf score
        chunk_map: Dict[str, SearchResult] = {}  # chunk_id -> best result

        if use_graph and self._graph_builder is not None and graph_results:
            # ========== 三源 RRF 融合 ==========
            # 权重分配：确保总和为 1
            vector_weight = alpha - graph_alpha / 2
            keyword_weight = 1 - alpha - graph_alpha / 2
            graph_weight = graph_alpha
            # 安全保护
            vector_weight = max(0.1, vector_weight)
            keyword_weight = max(0.1, keyword_weight)

            for rank, result in enumerate(vector_results):
                cid = result.chunk.chunk_id
                rrf_score = vector_weight / (rrf_k + rank + 1)
                score_map[cid] = score_map.get(cid, 0.0) + rrf_score
                if cid not in chunk_map or result.score > chunk_map[cid].score:
                    chunk_map[cid] = result

            for rank, result in enumerate(keyword_results):
                cid = result.chunk.chunk_id
                rrf_score = keyword_weight / (rrf_k + rank + 1)
                score_map[cid] = score_map.get(cid, 0.0) + rrf_score
                if cid not in chunk_map:
                    chunk_map[cid] = result

            for rank, result in enumerate(graph_results):
                cid = result.chunk.chunk_id
                rrf_score = graph_weight / (rrf_k + rank + 1)
                score_map[cid] = score_map.get(cid, 0.0) + rrf_score
                if cid not in chunk_map:
                    chunk_map[cid] = result

            logger.info(
                f"三源RRF融合: vector={len(vector_results)}, keyword={len(keyword_results)}, "
                f"graph={len(graph_results)}, 权重=[v={vector_weight:.2f}, k={keyword_weight:.2f}, g={graph_weight:.2f}]"
            )
        else:
            # ========== 双源 RRF 融合（原有逻辑，完全兼容）==========
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

    async def search_graph(
        self,
        query: str,
        top_k: int = 10,
        doc_ids: Optional[List[str]] = None,
        depth: int = 2,
    ) -> List[SearchResult]:
        """
        图谱检索 - 通过知识图谱遍历获取与查询相关的文档块

        原理：
        1. 从查询中提取可能的实体名称（关键词匹配图谱中的实体标题）
        2. 对匹配到的实体，调用 graph_builder.get_related_chunks() 获取关联 chunk_id
        3. 从向量存储中查找这些 chunk，构建 SearchResult
        4. 如果指定了 doc_ids，只返回属于这些文档的 chunk

        图谱检索的独特价值：
        - 向量检索基于语义相似度，关键词检索基于词频匹配
        - 图谱检索基于实体关系，可以发现「间接关联」的信息
          例如：查询"设备A"时，通过图谱可以找到与"设备A"有关系的"故障码E003"相关文档

        Args:
            query: 查询文本
            top_k: 返回结果数
            doc_ids: 限定检索范围的文档 ID 列表
            depth: 图遍历深度（1=直接关系，2=间接关系）

        Returns:
            检索结果列表
        """
        if self._graph_builder is None:
            logger.info("图谱检索: 未注入 graph_builder，跳过")
            return []

        # 1. 在图谱的实体标题中匹配查询关键词
        matched_chunk_ids = set()
        entity_titles = list(self._graph_builder._entity_map.keys())

        if not entity_titles:
            return []

        # 中文分词提取查询中的关键词
        query_tokens = self._tokenize(query)
        query_lower = query.lower()

        for title in entity_titles:
            title_lower = title.lower()
            # 匹配方式1：实体标题出现在查询中
            if title_lower in query_lower or title in query:
                chunk_ids = self._graph_builder.get_related_chunks(title, depth=depth)
                matched_chunk_ids.update(chunk_ids)
            # 匹配方式2：查询的分词结果出现在实体标题中
            else:
                for token in query_tokens:
                    if len(token) >= 2 and token in title_lower:
                        chunk_ids = self._graph_builder.get_related_chunks(title, depth=depth)
                        matched_chunk_ids.update(chunk_ids)
                        break

        if not matched_chunk_ids:
            logger.info(f"图谱检索: 查询 '{query[:50]}' 未匹配到图谱实体")
            return []

        # 2. 从向量存储中查找这些 chunk
        results: List[SearchResult] = []
        doc_id_set = set(doc_ids) if doc_ids else None

        for chunk_id in matched_chunk_ids:
            if chunk_id in self._chunk_id_to_index:
                idx = self._chunk_id_to_index[chunk_id]
                if idx < len(self._chunks):
                    chunk = self._chunks[idx]
                    # 文档级过滤
                    if doc_id_set and chunk.doc_id not in doc_id_set:
                        continue
                    # 基于图谱关系的分数（实体匹配度 × 关系权重）
                    # 使用中间分数 0.5 作为图谱检索的默认分数
                    results.append(SearchResult(
                        chunk=chunk,
                        score=0.5,  # 图谱检索统一给中间分数，实际排序靠 RRF
                        match_type=MatchType.GRAPH,
                    ))

        # 按分数排序（虽然都是0.5，但保留排序接口）
        results.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            f"图谱检索: query='{query[:50]}', 匹配实体→{len(matched_chunk_ids)}个chunk, "
            f"过滤后→{len(results)}个结果"
        )

        return results[:top_k]

    def set_graph_builder(self, graph_builder):
        """
        注入知识图谱构建器（由 LocalQA 调用）

        Args:
            graph_builder: KnowledgeGraphBuilder 实例
        """
        self._graph_builder = graph_builder
        logger.info(f"图谱构建器已注入 VectorStore，实体数={len(graph_builder._entity_map)}")

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

                # 重建 doc_id -> indices 映射
                self._doc_id_to_indices.clear()
                for idx, chunk in enumerate(self._chunks):
                    if chunk.doc_id not in self._doc_id_to_indices:
                        self._doc_id_to_indices[chunk.doc_id] = []
                    self._doc_id_to_indices[chunk.doc_id].append(idx)

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
                "doc_id_to_indices": self._doc_id_to_indices,
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

            # 加载 doc_id_to_indices（兼容旧数据：如果没有则重建）
            if "doc_id_to_indices" in metadata:
                self._doc_id_to_indices = metadata["doc_id_to_indices"]
            else:
                # 从 chunks 重建映射
                self._doc_id_to_indices = {}
                for idx, chunk in enumerate(self._chunks):
                    if chunk.doc_id not in self._doc_id_to_indices:
                        self._doc_id_to_indices[chunk.doc_id] = []
                    self._doc_id_to_indices[chunk.doc_id].append(idx)

            # 重建 BM25 索引
            self._rebuild_bm25()

            logger.info(f"向量存储已加载: {directory}，共 {self._index.ntotal} 个块，{len(self._doc_id_to_indices)} 个文档")

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

    def get_doc_ids(self) -> List[str]:
        """获取所有文档 ID"""
        return list(self._doc_id_to_indices.keys())

    def get_doc_chunk_count(self, doc_id: str) -> int:
        """获取指定文档的块数量"""
        return len(self._doc_id_to_indices.get(doc_id, []))


# 需要 import re
import re
