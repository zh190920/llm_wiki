"""
关键词搜索 — BM25 稀疏检索
================================
使用 BM25 算法实现关键词检索，
支持中英文分词，与向量检索组合为混合检索。
借鉴 WeKnora 的 KeywordsRetrieverType 设计。
"""

import math
import re
from collections import Counter, defaultdict
from typing import Optional

import jieba
from loguru import logger


class BM25SearchEngine:
    """BM25 关键词搜索引擎"""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # 文档存储: chunk_id → 分词后的词项列表
        self._docs: dict[str, list[str]] = {}
        # 文档长度: chunk_id → 词项数量
        self._doc_lengths: dict[str, int] = {}
        # 平均文档长度
        self._avg_dl: float = 0.0
        # 逆文档频率: term → IDF 值
        self._idf: dict[str, float] = {}
        # 词项频率: chunk_id → Counter
        self._tf: dict[str, Counter] = {}
        # 元数据
        self._metadata: dict[str, dict] = {}

    def tokenize(self, text: str) -> list[str]:
        """
        中英文混合分词

        使用 jieba 对中文分词，正则提取英文单词，
        过滤停用词和短词。
        """
        # 中文分词
        chinese_tokens = jieba.lcut(text)
        # 英文单词提取
        english_tokens = re.findall(r"[a-zA-Z]{2,}", text.lower())

        tokens = []
        for t in chinese_tokens + english_tokens:
            t = t.strip().lower()
            if len(t) >= 2 and not t.isspace():
                tokens.append(t)
        return tokens

    def add_document(
        self,
        chunk_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ):
        """添加文档到索引"""
        tokens = self.tokenize(content)
        self._docs[chunk_id] = tokens
        self._doc_lengths[chunk_id] = len(tokens)
        self._tf[chunk_id] = Counter(tokens)
        if metadata:
            self._metadata[chunk_id] = metadata

    def remove_document(self, chunk_id: str):
        """移除文档"""
        self._docs.pop(chunk_id, None)
        self._doc_lengths.pop(chunk_id, None)
        self._tf.pop(chunk_id, None)
        self._metadata.pop(chunk_id, None)

    def build_index(self):
        """构建/重建 BM25 索引（计算 IDF）"""
        n_docs = len(self._docs)
        if n_docs == 0:
            return

        # 计算平均文档长度
        self._avg_dl = sum(self._doc_lengths.values()) / n_docs

        # 计算每个词项的文档频率
        df = defaultdict(int)
        for tokens in self._docs.values():
            unique_terms = set(tokens)
            for term in unique_terms:
                df[term] += 1

        # 计算 IDF
        self._idf = {}
        for term, freq in df.items():
            idf = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            self._idf[term] = max(idf, self.epsilon)

        logger.info(f"BM25 索引构建完成: {n_docs} 文档, {len(self._idf)} 词项")

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        knowledge_base_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        BM25 关键词搜索

        Args:
            query: 查询文本
            top_k: 返回 top K 结果
            threshold: 分数阈值
            knowledge_base_ids: 限定知识库范围

        Returns:
            [{"chunk_id": str, "score": float, "metadata": dict}, ...]
        """
        if not self._docs:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = {}
        for chunk_id, doc_tokens in self._docs.items():
            # 知识库过滤
            if knowledge_base_ids:
                meta = self._metadata.get(chunk_id, {})
                if meta.get("knowledge_base_id") not in knowledge_base_ids:
                    continue

            score = self._bm25_score(chunk_id, query_tokens)
            if score >= threshold:
                scores[chunk_id] = score

        # 排序
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "chunk_id": cid,
                "score": float(score),
                "metadata": self._metadata.get(cid, {}),
            }
            for cid, score in sorted_results
        ]

    def _bm25_score(self, chunk_id: str, query_tokens: list[str]) -> float:
        """计算 BM25 分数"""
        score = 0.0
        doc_len = self._doc_lengths.get(chunk_id, 0)
        tf_counter = self._tf.get(chunk_id, Counter())

        for term in query_tokens:
            if term not in self._idf:
                continue

            idf = self._idf[term]
            tf = tf_counter.get(term, 0)

            # BM25 公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_dl, 1e-6))
            score += idf * numerator / denominator

        return score
