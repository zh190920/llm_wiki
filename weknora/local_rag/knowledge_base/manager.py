"""
知识库管理器 — 统一管理知识库生命周期
================================
整合文档解析、分块、向量化、BM25 索引、
Wiki 生成等所有流程，提供统一的知识库管理接口。
"""

import asyncio
from typing import Optional

from loguru import logger

from config import settings
from knowledge_base.models import KnowledgeBase, Knowledge, SearchResult
from core.document_parser import DocumentParser
from core.chunker import TextChunker, Chunk
from core.embedding import EmbeddingEngine
from core.vector_store import VectorStore
from core.keyword_search import BM25SearchEngine
from core.hybrid_search import HybridSearchEngine
from core.knowledge_graph import KnowledgeGraphBuilder
from core.llm_client import LLMClient
from wiki.page_manager import WikiPageManager
from wiki.ingest import WikiIngestPipeline


class KnowledgeBaseManager:
    """知识库管理器"""

    def __init__(self):
        # 核心组件
        self.llm_client = LLMClient()
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.bm25_engine = BM25SearchEngine()
        self.hybrid_engine = HybridSearchEngine(
            vector_store=self.vector_store,
            bm25_engine=self.bm25_engine,
            embedding_engine=self.embedding_engine,
        )
        self.chunker = TextChunker()
        self.graph_builder = KnowledgeGraphBuilder(self.llm_client)
        self.page_manager = WikiPageManager()
        self.wiki_pipeline = WikiIngestPipeline(self.llm_client, self.page_manager)

        # 数据存储
        self._knowledge_bases: dict[str, KnowledgeBase] = {}
        self._knowledge_items: dict[str, Knowledge] = {}
        self._chunks: dict[str, list[Chunk]] = {}  # knowledge_id → chunks

        # 锁
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, kb_id: str) -> asyncio.Lock:
        """获取知识库级别的异步锁"""
        if kb_id not in self._locks:
            self._locks[kb_id] = asyncio.Lock()
        return self._locks[kb_id]

    # ── 知识库 CRUD ──

    async def create_knowledge_base(
        self,
        name: str,
        description: str = "",
        kb_type: str = "document",
    ) -> KnowledgeBase:
        """创建知识库"""
        kb = KnowledgeBase(
            name=name,
            description=description,
            type=kb_type,
        )
        self._knowledge_bases[kb.id] = kb

        # 加载已有的 Wiki 页面
        self.page_manager.load_from_disk(kb.id)

        logger.info(f"知识库创建: {name} (ID: {kb.id})")
        return kb

    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        """列出所有知识库"""
        return list(self._knowledge_bases.values())

    def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """获取知识库"""
        return self._knowledge_bases.get(kb_id)

    async def delete_knowledge_base(self, kb_id: str) -> bool:
        """删除知识库"""
        if kb_id not in self._knowledge_bases:
            return False

        # 删除相关向量
        await self.vector_store.delete_by_knowledge_base_id(kb_id)

        # 删除知识条目
        to_remove = [
            kid for kid, k in self._knowledge_items.items()
            if k.knowledge_base_id == kb_id
        ]
        for kid in to_remove:
            self._knowledge_items.pop(kid, None)
            self._chunks.pop(kid, None)

        self._knowledge_bases.pop(kb_id, None)
        logger.info(f"知识库删除: {kb_id}")
        return True

    # ── 文档管理 ──

    async def add_document(
        self,
        kb_id: str,
        file_path: str,
    ) -> Knowledge:
        """
        添加文档到知识库

        完整流程：解析 → 分块 → 向量化 → BM25 索引 → 图谱构建 → Wiki 生成
        """
        async with self._get_lock(kb_id):
            kb = self._knowledge_bases.get(kb_id)
            if not kb:
                raise ValueError(f"知识库不存在: {kb_id}")

            # 1. 解析文档
            logger.info(f"[Pipeline] 解析文档: {file_path}")
            parsed = await DocumentParser.parse(file_path)

            # 2. 创建知识条目
            knowledge = Knowledge(
                knowledge_base_id=kb_id,
                title=parsed["title"],
                file_name=parsed["file_name"],
                file_type=parsed["file_type"],
                file_size=parsed["file_size"],
                content=parsed["content"],
                parse_status="processing",
            )
            self._knowledge_items[knowledge.id] = knowledge

            # 3. 文本分块
            logger.info(f"[Pipeline] 分块处理: {knowledge.id}")
            chunks = self.chunker.chunk_text(
                text=parsed["content"],
                knowledge_id=knowledge.id,
                knowledge_base_id=kb_id,
            )
            self._chunks[knowledge.id] = chunks

            # 4. 向量化并存储
            logger.info(f"[Pipeline] 向量化: {len(chunks)} 个分块")
            chunk_texts = [c.content for c in chunks]
            vectors = await self.embedding_engine.embed_documents_numpy(chunk_texts)

            chunk_ids = [c.id for c in chunks]
            metadata = [
                {
                    "knowledge_id": c.knowledge_id,
                    "knowledge_base_id": c.knowledge_base_id,
                    "chunk_index": c.chunk_index,
                    "content_preview": c.content[:200],
                }
                for c in chunks
            ]
            await self.vector_store.add_vectors(chunk_ids, vectors, metadata)

            # 5. BM25 索引
            logger.info(f"[Pipeline] BM25 索引构建")
            for chunk in chunks:
                self.bm25_engine.add_document(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    metadata={
                        "knowledge_id": chunk.knowledge_id,
                        "knowledge_base_id": chunk.knowledge_base_id,
                    },
                )
            self.bm25_engine.build_index()

            # 6. 知识图谱构建（异步，不阻塞）
            try:
                chunk_data = [{"chunk_id": c.id, "content": c.content} for c in chunks]
                graph_result = await self.graph_builder.build_from_chunks(chunk_data)
                logger.info(f"[Pipeline] 知识图谱: {graph_result['entities']} 实体, {graph_result['relationships']} 关系")
            except Exception as e:
                logger.warning(f"[Pipeline] 知识图谱构建失败: {e}")

            # 7. Wiki 生成（异步，不阻塞）
            try:
                affected = await self.wiki_pipeline.ingest_document(
                    kb_id=kb_id,
                    knowledge_id=knowledge.id,
                    title=parsed["title"],
                    file_name=parsed["file_name"],
                    content=parsed["content"],
                )
                logger.info(f"[Pipeline] Wiki 生成: {len(affected)} 个页面")
            except Exception as e:
                logger.warning(f"[Pipeline] Wiki 生成失败: {e}")

            # 8. 更新知识条目状态
            knowledge.parse_status = "completed"
            knowledge.chunk_count = len(chunks)
            kb.doc_count += 1
            kb.updated_at = knowledge.updated_at

            # 持久化向量索引
            await self.vector_store.save(kb_id)

            logger.info(f"[Pipeline] 文档处理完成: {knowledge.id} ({len(chunks)} 分块)")
            return knowledge

    async def add_text(
        self,
        kb_id: str,
        title: str,
        content: str,
    ) -> Knowledge:
        """添加纯文本到知识库"""
        async with self._get_lock(kb_id):
            kb = self._knowledge_bases.get(kb_id)
            if not kb:
                raise ValueError(f"知识库不存在: {kb_id}")

            knowledge = Knowledge(
                knowledge_base_id=kb_id,
                title=title,
                file_type="text",
                content=content,
                parse_status="processing",
            )
            self._knowledge_items[knowledge.id] = knowledge

            chunks = self.chunker.chunk_text(content, knowledge.id, kb_id)
            self._chunks[knowledge.id] = chunks

            # 向量化
            chunk_texts = [c.content for c in chunks]
            vectors = await self.embedding_engine.embed_documents_numpy(chunk_texts)
            chunk_ids = [c.id for c in chunks]
            metadata = [
                {"knowledge_id": c.knowledge_id, "knowledge_base_id": c.knowledge_base_id}
                for c in chunks
            ]
            await self.vector_store.add_vectors(chunk_ids, vectors, metadata)

            # BM25
            for chunk in chunks:
                self.bm25_engine.add_document(chunk.id, chunk.content, {
                    "knowledge_id": chunk.knowledge_id,
                    "knowledge_base_id": chunk.knowledge_base_id,
                })
            self.bm25_engine.build_index()

            # Wiki 生成
            try:
                await self.wiki_pipeline.ingest_document(
                    kb_id=kb_id,
                    knowledge_id=knowledge.id,
                    title=title,
                    file_name=title,
                    content=content,
                )
            except Exception as e:
                logger.warning(f"Wiki 生成失败: {e}")

            knowledge.parse_status = "completed"
            knowledge.chunk_count = len(chunks)
            kb.doc_count += 1

            await self.vector_store.save(kb_id)
            return knowledge

    def list_documents(self, kb_id: str) -> list[Knowledge]:
        """列出知识库中的文档"""
        return [k for k in self._knowledge_items.values() if k.knowledge_base_id == kb_id]

    # ── 检索 ──

    async def search(
        self,
        query: str,
        knowledge_base_ids: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """混合检索"""
        results = await self.hybrid_engine.search(
            query=query,
            top_k=top_k,
            knowledge_base_ids=knowledge_base_ids,
        )

        search_results = []
        for r in results:
            chunk_id = r["chunk_id"]
            meta = r.get("metadata", {})

            # 获取分块内容
            content = self._get_chunk_content(chunk_id)

            search_results.append(SearchResult(
                chunk_id=chunk_id,
                knowledge_id=meta.get("knowledge_id", ""),
                knowledge_base_id=meta.get("knowledge_base_id", ""),
                content=content,
                score=r.get("score", 0),
                vector_score=r.get("vector_score", 0),
                keyword_score=r.get("keyword_score", 0),
                metadata=meta,
            ))

        return search_results

    def _get_chunk_content(self, chunk_id: str) -> str:
        """获取分块内容"""
        for chunks in self._chunks.values():
            for chunk in chunks:
                if chunk.id == chunk_id:
                    return chunk.content
        # 尝试从元数据中获取预览
        meta = self.vector_store._metadata.get(chunk_id, {})
        return meta.get("content_preview", "")

    # ── RAG 问答 ──

    async def rag_query(
        self,
        query: str,
        knowledge_base_ids: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> dict:
        """
        RAG 快速问答

        流程：检索 → 构建上下文 → LLM 生成回答
        """
        # 1. 检索相关知识
        search_results = await self.search(query, knowledge_base_ids, top_k)

        if not search_results:
            return {
                "answer": "抱歉，在知识库中未找到相关信息。",
                "sources": [],
            }

        # 2. 构建上下文
        context_parts = []
        sources = []
        for i, r in enumerate(search_results, 1):
            context_parts.append(f"[来源 {i}]\n{r.content}\n")
            sources.append({
                "chunk_id": r.chunk_id,
                "knowledge_id": r.knowledge_id,
                "score": r.score,
            })

        context = "\n---\n".join(context_parts)

        # 3. LLM 生成回答
        system_prompt = """你是一个专业的知识问答助手。请根据提供的参考资料回答用户的问题。

要求：
1. 只基于提供的参考资料回答，不要编造信息
2. 引用来源编号（如 [来源 1]）以增强可信度
3. 如果参考资料不足以回答问题，请如实说明
4. 回答要结构清晰、逻辑连贯"""

        user_prompt = f"""参考资料：
{context}

用户问题：{query}

请基于以上参考资料回答问题。"""

        result = await self.llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        return {
            "answer": result["content"],
            "sources": sources,
        }
