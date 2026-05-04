"""
向量存储 — FAISS 本地向量数据库
================================
使用 FAISS 实现高性能本地向量存储和检索，
支持 L2 距离和内积相似度搜索。
"""

import json
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from loguru import logger

from config import settings


class VectorStore:
    """基于 FAISS 的本地向量存储"""

    def __init__(self, dimension: int = None, store_dir: str = None):
        self.dimension = dimension or settings.EMMBEDDING_DIMENSION if hasattr(settings, 'EMMBEDDING_DIMENSION') else settings.EMBEDDING_DIMENSION
        self.store_dir = Path(store_dir) if store_dir else settings.VECTOR_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # FAISS 索引 (使用内积相似度，因为向量已归一化)
        self._index: Optional[faiss.IndexFlatIP] = None
        # ID 映射: faiss 内部 ID → chunk_id
        self._id_map: dict[int, str] = {}
        self._reverse_id_map: dict[str, int] = {}
        # chunk 元数据
        self._metadata: dict[str, dict] = {}
        self._next_id = 0

    def _ensure_index(self):
        """确保索引已初始化"""
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"FAISS 索引已创建, 维度: {self.dimension}")

    async def add_vectors(
        self,
        chunk_ids: list[str],
        vectors: np.ndarray,
        metadata: Optional[list[dict]] = None,
    ) -> int:
        """
        添加向量到索引

        Args:
            chunk_ids: chunk ID 列表
            vectors: 向量矩阵 (N, dimension)
            metadata: 元数据列表

        Returns:
            添加的向量数量
        """
        self._ensure_index()

        if len(chunk_ids) != vectors.shape[0]:
            raise ValueError(f"chunk_ids 数量 ({len(chunk_ids)}) 与向量数量 ({vectors.shape[0]}) 不匹配")

        # 确保向量维度正确
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"向量维度 ({vectors.shape[1]}) 与索引维度 ({self.dimension}) 不匹配")

        # 归一化向量
        faiss.normalize_L2(vectors)

        # 添加到索引
        start_id = self._next_id
        self._index.add(vectors)

        for i, chunk_id in enumerate(chunk_ids):
            faiss_id = start_id + i
            self._id_map[faiss_id] = chunk_id
            self._reverse_id_map[chunk_id] = faiss_id
            if metadata and i < len(metadata):
                self._metadata[chunk_id] = metadata[i]

        self._next_id += len(chunk_ids)
        logger.info(f"已添加 {len(chunk_ids)} 个向量到索引, 总数: {self._index.ntotal}")
        return len(chunk_ids)

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        knowledge_base_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        向量相似度搜索

        Args:
            query_vector: 查询向量 (dimension,)
            top_k: 返回最相似的 K 个结果
            threshold: 相似度阈值
            knowledge_base_ids: 限定搜索的知识库 ID 列表

        Returns:
            [{"chunk_id": str, "score": float, "metadata": dict}, ...]
        """
        self._ensure_index()

        if self._index.ntotal == 0:
            return []

        # 归一化查询向量
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vector)

        # 搜索
        k = min(top_k * 3, self._index.ntotal)  # 过量搜索，后面再过滤
        scores, indices = self._index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            chunk_id = self._id_map.get(int(idx))
            if not chunk_id:
                continue

            meta = self._metadata.get(chunk_id, {})

            # 按知识库 ID 过滤
            if knowledge_base_ids:
                kb_id = meta.get("knowledge_base_id", "")
                if kb_id not in knowledge_base_ids:
                    continue

            if score < threshold:
                continue

            results.append({
                "chunk_id": chunk_id,
                "score": float(score),
                "metadata": meta,
            })

        # 截断到 top_k
        results = results[:top_k]
        return results

    async def delete_by_knowledge_id(self, knowledge_id: str):
        """删除指定文档的所有向量（重建索引）"""
        to_remove = [
            cid for cid, meta in self._metadata.items()
            if meta.get("knowledge_id") == knowledge_id
        ]
        if not to_remove:
            return

        # FAISS 不支持直接删除，需要重建索引
        await self._rebuild_without(to_remove)

    async def delete_by_knowledge_base_id(self, knowledge_base_id: str):
        """删除指定知识库的所有向量"""
        to_remove = [
            cid for cid, meta in self._metadata.items()
            if meta.get("knowledge_base_id") == knowledge_base_id
        ]
        if not to_remove:
            return
        await self._rebuild_without(to_remove)

    async def _rebuild_without(self, chunk_ids_to_remove: set[str]):
        """重建索引，排除指定的 chunk"""
        self._ensure_index()

        # 收集需要保留的向量
        remaining_vectors = []
        remaining_ids = []
        remaining_metadata = {}

        for faiss_id, chunk_id in self._id_map.items():
            if chunk_id in chunk_ids_to_remove:
                continue
            vector = self._index.reconstruct(int(faiss_id))
            remaining_vectors.append(vector)
            remaining_ids.append(chunk_id)
            if chunk_id in self._metadata:
                remaining_metadata[chunk_id] = self._metadata[chunk_id]

        # 重建索引
        self._index = faiss.IndexFlatIP(self.dimension)
        self._id_map = {}
        self._reverse_id_map = {}
        self._metadata = remaining_metadata
        self._next_id = 0

        if remaining_vectors:
            vectors = np.stack(remaining_vectors)
            self._index.add(vectors)
            for i, chunk_id in enumerate(remaining_ids):
                self._id_map[i] = chunk_id
                self._reverse_id_map[chunk_id] = i
            self._next_id = len(remaining_ids)

        logger.info(f"索引重建完成, 剩余向量: {self._index.ntotal}")

    async def save(self, knowledge_base_id: str):
        """持久化索引到磁盘"""
        if self._index is None or self._index.ntotal == 0:
            return

        kb_dir = self.store_dir / knowledge_base_id
        kb_dir.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        faiss.write_index(self._index, str(kb_dir / "index.faiss"))

        # 保存元数据
        meta = {
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "metadata": self._metadata,
            "next_id": self._next_id,
            "dimension": self.dimension,
        }
        with open(kb_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"向量索引已保存: {kb_dir}")

    async def load(self, knowledge_base_id: str) -> bool:
        """从磁盘加载索引"""
        kb_dir = self.store_dir / knowledge_base_id
        index_path = kb_dir / "index.faiss"
        meta_path = kb_dir / "metadata.json"

        if not index_path.exists() or not meta_path.exists():
            return False

        self._index = faiss.read_index(str(index_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._id_map = {int(k): v for k, v in meta["id_map"].items()}
        self._metadata = meta["metadata"]
        self._next_id = meta["next_id"]
        self._reverse_id_map = {v: k for k, v in self._id_map.items()}

        logger.info(f"向量索引已加载: {kb_dir}, 向量数: {self._index.ntotal}")
        return True

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
