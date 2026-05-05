"""
向量嵌入模块 - 使用 OpenAI Embedding API
支持批量嵌入和异步调用
"""
import asyncio
import logging
from typing import List, Optional

import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import LLMConfig

logger = logging.getLogger(__name__)


class Embedder:
    """
    向量嵌入器

    特性：
    - 异步批量嵌入
    - 自动重试机制
    - 嵌入缓存（避免重复计算）
    - 批量大小控制（防止 API 限流）
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        self._cache: dict[str, List[float]] = {}
        self._batch_size = 64  # OpenAI 建议的批量大小

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本"""
        cache_key = f"q:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            response = await self._client.embeddings.create(
                model=self.config.embedding_model,
                input=text,
                dimensions=self.config.embedding_dim,
            )
            embedding = response.data[0].embedding
            self._cache[cache_key] = embedding

            if len(self._cache) > 10000:
                # 清理旧缓存，保留最近的一半
                keys = list(self._cache.keys())
                for k in keys[:len(keys) // 2]:
                    del self._cache[k]

            return embedding
        except Exception as e:
            logger.error(f"嵌入查询失败: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档文本"""
        if not texts:
            return []

        # 检查缓存
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []

        for i, text in enumerate(texts):
            cache_key = f"d:{text[:200]}"  # 用前200字符做缓存key
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)

        # 批量嵌入未缓存的文本
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]

            # 分批处理
            all_embeddings: List[List[float]] = []
            for batch_start in range(0, len(uncached_texts), self._batch_size):
                batch = uncached_texts[batch_start:batch_start + self._batch_size]
                try:
                    response = await self._client.embeddings.create(
                        model=self.config.embedding_model,
                        input=batch,
                        dimensions=self.config.embedding_dim,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"批量嵌入失败 (batch_start={batch_start}): {e}")
                    raise

            # 填充结果并更新缓存
            for idx, embedding in zip(uncached_indices, all_embeddings):
                results[idx] = embedding
                cache_key = f"d:{texts[idx][:200]}"
                self._cache[cache_key] = embedding

        return [r for r in results if r is not None]

    async def embed_chunks(self, chunks) -> List[List[float]]:
        """嵌入 Chunk 对象列表"""
        texts = [chunk.content for chunk in chunks]
        return await self.embed_documents(texts)

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """计算余弦相似度"""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
