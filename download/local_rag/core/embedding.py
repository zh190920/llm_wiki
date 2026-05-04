"""
向量化模块 — 本地 Embedding 模型
================================
使用 sentence-transformers 本地模型生成文本向量，
支持批量编码和异步并发控制。
"""

import asyncio
from typing import Optional

import numpy as np
from loguru import logger

from config import settings


class EmbeddingEngine:
    """本地 Embedding 引擎"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        self.dimension = settings.EMBEDDING_DIMENSION
        self._model = None
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_EMBEDDINGS)
        self._lock = asyncio.Lock()

    async def _ensure_model(self):
        """延迟加载模型（首次调用时加载）"""
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"正在加载 Embedding 模型: {self.model_name}")
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                    )
                    # 更新实际维度
                    self.dimension = self._model.get_sentence_embedding_dimension()
                    logger.info(f"Embedding 模型加载完成, 维度: {self.dimension}")

    async def embed_query(self, text: str) -> np.ndarray:
        """
        对单个查询文本生成向量

        Args:
            text: 查询文本

        Returns:
            float32 向量数组
        """
        async with self._semaphore:
            await self._ensure_model()
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self._model.encode(text, normalize_embeddings=True)
            )
            return embedding.astype(np.float32)

    async def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
        """
        批量对文档文本生成向量

        Args:
            texts: 文档文本列表
            batch_size: 每批处理数量

        Returns:
            向量数组列表
        """
        async with self._semaphore:
            await self._ensure_model()
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=len(texts) > 100,
                )
            )
            return [e.astype(np.float32) for e in embeddings]

    async def embed_documents_numpy(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        批量生成向量，返回 numpy 二维数组

        Returns:
            shape = (len(texts), dimension) 的 float32 数组
        """
        async with self._semaphore:
            await self._ensure_model()
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=len(texts) > 100,
                )
            )
            return embeddings.astype(np.float32)
