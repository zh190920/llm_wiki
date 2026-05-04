"""
FastAPI 服务层 - 异步高并发接口
提供 RAG 问答、Agent 推理、Wiki 生成、文档管理等 REST API
"""
import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from config.settings import AppConfig, load_config
from core.chunker import TextChunker
from core.document_parser import DocumentParser
from core.embedder import Embedder
from core.rag_engine import RAGEngine
from core.reranker import Reranker
from core.retriever import Retriever
from core.vector_store import VectorStore
from agent.engine import AgentEngine
from models.schemas import (
    ChatRequest,
    ChatResponse,
    UploadResponse,
    WikiGenerateRequest,
    WikiGenerateResponse,
)
from wiki.graph_builder import KnowledgeGraphBuilder
from wiki.ingest import WikiIngest
from wiki.page_manager import WikiPageManager

logger = logging.getLogger(__name__)

# ============================================================
# 全局状态管理
# ============================================================


class AppState:
    """应用状态 - 管理所有核心组件"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.parser = DocumentParser()
        self.chunker = TextChunker(config.chunker)
        self.embedder = Embedder(config.llm)
        self.vector_store = VectorStore(config.retriever, dim=config.llm.embedding_dim)
        self.reranker = Reranker(config.llm, config.retriever, self.embedder)
        self.retriever = Retriever(config, self.vector_store, self.embedder, self.reranker)
        self.rag_engine = RAGEngine(config)
        self.agent_engine = AgentEngine(config)
        self.wiki_manager = WikiPageManager(config.wiki.wiki_dir)
        self.wiki_ingest = WikiIngest(config, self.wiki_manager)
        self.graph_builder = KnowledgeGraphBuilder(config)

        # 文档注册表
        self._documents: Dict[str, dict] = {}  # doc_id -> metadata
        self._doc_chunks: Dict[str, list] = {}  # doc_id -> [chunk_ids]

        # 知识库注册表
        self._knowledge_bases: Dict[str, dict] = {}  # kb_id -> info

    async def initialize(self):
        """初始化所有组件"""
        # 创建数据目录
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.vector_store_dir).mkdir(parents=True, exist_ok=True)

        # 初始化 RAG 引擎
        self.rag_engine.initialize(
            embedder=self.embedder,
            vector_store=self.vector_store,
            retriever=self.retriever,
            reranker=self.reranker,
        )

        # 注册 Agent 工具
        self.agent_engine.register_knowledge_tools(self.retriever, self.vector_store)
        self.agent_engine.register_wiki_tools(self.wiki_manager, self.vector_store)

        # 初始化 Wiki 管理器
        await self.wiki_manager.initialize()

        # 尝试加载已有向量存储
        if os.path.exists(self.config.vector_store_dir):
            try:
                await self.vector_store.load(self.config.vector_store_dir)
            except Exception as e:
                logger.warning(f"加载向量存储失败: {e}")

        logger.info("应用状态初始化完成")

    def get_kb_info(self) -> List[dict]:
        """获取知识库信息（供 Agent 提示词使用）"""
        if not self._knowledge_bases:
            doc_count = len(self._documents)
            chunk_count = self.vector_store.total_chunks
            if chunk_count > 0:
                return [{
                    "name": "默认知识库",
                    "description": "所有已上传文档",
                    "doc_count": doc_count,
                    "chunk_count": chunk_count,
                }]
            return []

        return [
            {
                "name": info.get("name", "未命名"),
                "description": info.get("description", ""),
                "doc_count": info.get("doc_count", 0),
                "chunk_count": info.get("chunk_count", 0),
            }
            for info in self._knowledge_bases.values()
        ]


# ============================================================
# FastAPI 应用
# ============================================================


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """创建 FastAPI 应用"""

    if config is None:
        config = load_config()

    state = AppState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        await state.initialize()
        logger.info(f"🚀 Local RAG System 启动完成 - http://{config.host}:{config.port}")
        yield
        # 清理
        try:
            await state.vector_store.save(config.vector_store_dir)
        except Exception as e:
            logger.error(f"保存向量存储失败: {e}")
        logger.info("应用已关闭")

    app = FastAPI(
        title="Local RAG System",
        description="本地 RAG 系统 - 支持 RAG 问答、ReAct Agent 推理、Wiki 知识库生成",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ============================================================
    # 文档管理 API
    # ============================================================

    @app.post("/api/documents/upload", response_model=UploadResponse)
    async def upload_document(file: UploadFile = File(...)):
        """上传文档（支持 PDF、Markdown）"""
        # 验证文件类型
        ext = Path(file.filename).suffix.lower()
        if ext not in state.parser.supported_extensions():
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {ext}，支持: {state.parser.supported_extensions()}",
            )

        # 保存文件
        upload_dir = Path(config.data_dir) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        doc_id = uuid.uuid4().hex[:16]
        file_path = upload_dir / f"{doc_id}_{file.filename}"

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        try:
            # 解析文档
            text, metadata = await state.parser.parse(str(file_path), doc_id)
            metadata.chunk_count = 0

            # 分块
            chunks = state.chunker.chunk_text(text, doc_id, metadata.file_type)
            if not chunks:
                return UploadResponse(
                    doc_id=doc_id,
                    filename=file.filename,
                    chunk_count=0,
                    message="文档解析成功但内容为空",
                )

            # 嵌入
            embeddings = await state.embedder.embed_chunks(chunks)

            # 添加到向量存储
            await state.vector_store.add_chunks(chunks, embeddings)

            # 持久化
            await state.vector_store.save(config.vector_store_dir)

            # 更新元数据
            metadata.chunk_count = len(chunks)
            state._documents[doc_id] = metadata.model_dump()
            state._doc_chunks[doc_id] = [c.chunk_id for c in chunks]

            return UploadResponse(
                doc_id=doc_id,
                filename=file.filename,
                chunk_count=len(chunks),
                message="文档上传并解析成功",
            )

        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    @app.get("/api/documents")
    async def list_documents():
        """列出所有文档"""
        docs = []
        for doc_id, meta in state._documents.items():
            docs.append({
                "doc_id": doc_id,
                "filename": meta.get("filename", ""),
                "file_type": meta.get("file_type", ""),
                "title": meta.get("title", ""),
                "chunk_count": meta.get("chunk_count", 0),
            })
        return {"documents": docs, "total": len(docs)}

    @app.delete("/api/documents/{doc_id}")
    async def delete_document(doc_id: str):
        """删除文档"""
        if doc_id not in state._documents:
            raise HTTPException(status_code=404, detail="文档不存在")

        count = await state.vector_store.delete_by_doc_id(doc_id)
        del state._documents[doc_id]
        if doc_id in state._doc_chunks:
            del state._doc_chunks[doc_id]

        return {"message": f"已删除文档 {doc_id}，移除 {count} 个块"}

    # ============================================================
    # RAG 问答 API
    # ============================================================

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        统一聊天接口

        支持三种模式：
        - rag: RAG 快速问答
        - agent: ReAct Agent 智能推理
        - wiki: Wiki 知识库查询
        """
        try:
            if request.mode == "agent":
                # Agent 模式
                kb_info = state.get_kb_info()
                response = await state.agent_engine.run(
                    query=request.query,
                    knowledge_bases_info=kb_info if kb_info else None,
                )
                return response

            elif request.mode == "wiki":
                # Wiki 模式：在 Wiki 知识库中检索
                wiki_pages = await state.wiki_manager.search_pages(request.query)
                if wiki_pages:
                    context = "\n\n---\n\n".join([
                        f"# {p.title}\n{p.content}" for p in wiki_pages[:3]
                    ])
                    # 使用 RAG 引擎生成回答
                    from models.schemas import SearchResult
                    # 构造伪搜索结果
                    results = []
                    for p in wiki_pages[:3]:
                        from models.schemas import Chunk
                        chunk = Chunk(
                            doc_id="wiki",
                            content=p.content,
                            metadata={"section_title": p.title, "page_type": p.page_type.value},
                        )
                        results.append(SearchResult(chunk=chunk, score=0.9))

                    answer = await state.rag_engine._generate_answer(
                        request.query, context
                    )
                    return ChatResponse(answer=answer, sources=results)
                else:
                    return ChatResponse(answer="Wiki 知识库中暂无相关内容。")

            else:
                # RAG 快速问答（默认）
                if request.query and len(request.query) > 50:
                    # 长查询使用深度模式
                    response = await state.rag_engine.deep_chat(request.query)
                else:
                    response = await state.rag_engine.quick_chat(request.query)
                return response

        except Exception as e:
            logger.error(f"聊天处理失败: {e}")
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest):
        """流式聊天接口（SSE）"""
        async def generate():
            try:
                if request.mode == "agent":
                    async for chunk in state.agent_engine.stream_run(
                        query=request.query,
                        knowledge_bases_info=state.get_kb_info() or None,
                    ):
                        yield f"data: {chunk}\n\n"
                else:
                    async for chunk in state.rag_engine.stream_chat(
                        query=request.query,
                        deep=len(request.query) > 50,
                    ):
                        yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: [ERROR] {str(e)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ============================================================
    # Wiki API
    # ============================================================

    @app.post("/api/wiki/generate", response_model=WikiGenerateResponse)
    async def generate_wiki(request: WikiGenerateRequest):
        """从文档生成 Wiki 知识库"""
        doc_ids = request.document_ids

        if not doc_ids:
            # 使用所有文档
            doc_ids = list(state._documents.keys())

        if not doc_ids:
            raise HTTPException(status_code=400, detail="没有可用的文档")

        try:
            stats = await state.wiki_ingest.ingest_documents(
                doc_ids=doc_ids,
                vector_store=state.vector_store,
                granularity=request.granularity,
            )

            return WikiGenerateResponse(
                status="completed",
                pages_generated=stats.get("pages_created", 0) + stats.get("pages_updated", 0),
                message=f"Wiki 生成完成: 创建 {stats['pages_created']} 页，更新 {stats['pages_updated']} 页",
            )

        except Exception as e:
            logger.error(f"Wiki 生成失败: {e}")
            raise HTTPException(status_code=500, detail=f"Wiki 生成失败: {str(e)}")

    @app.get("/api/wiki/pages")
    async def list_wiki_pages(
        page_type: Optional[str] = None,
        status: Optional[str] = None,
    ):
        """列出 Wiki 页面"""
        from models.schemas import WikiPageType
        pt = WikiPageType(page_type) if page_type else None
        pages = await state.wiki_manager.list_pages(page_type=pt, status=status)
        return {
            "pages": [
                {
                    "slug": p.slug,
                    "title": p.title,
                    "type": p.page_type.value,
                    "status": p.status,
                    "out_links": p.out_links,
                    "updated_at": p.updated_at,
                }
                for p in pages
            ],
            "total": len(pages),
        }

    @app.get("/api/wiki/pages/{slug}")
    async def get_wiki_page(slug: str):
        """获取 Wiki 页面详情"""
        page = await state.wiki_manager.get_page(slug)
        if not page:
            raise HTTPException(status_code=404, detail="页面不存在")
        return page.model_dump()

    @app.get("/api/wiki/export")
    async def export_wiki():
        """导出 Wiki 为 Markdown 文件"""
        md_dir = await state.wiki_manager.export_all_markdown()
        return {"message": f"Wiki 已导出到 {md_dir}", "path": md_dir}

    # ============================================================
    # 知识图谱 API
    # ============================================================

    @app.post("/api/graph/build")
    async def build_knowledge_graph(doc_ids: Optional[List[str]] = None):
        """构建知识图谱"""
        if not doc_ids:
            doc_ids = list(state._documents.keys())

        if not doc_ids:
            raise HTTPException(status_code=400, detail="没有可用的文档")

        # 收集文档块
        chunks = []
        for doc_id in doc_ids:
            doc_chunks = state.vector_store.get_chunks_by_doc_id(doc_id)
            chunks.extend(doc_chunks)

        if not chunks:
            raise HTTPException(status_code=400, detail="文档中没有可用的内容")

        try:
            kg = await state.graph_builder.build_graph(chunks)

            # 保存图谱
            graph_path = Path(config.data_dir) / "knowledge_graph.json"
            with open(graph_path, "w", encoding="utf-8") as f:
                f.write(kg.model_dump_json(indent=2))

            # 生成 Mermaid 可视化
            mermaid = state.graph_builder.to_mermaid()
            mermaid_path = Path(config.data_dir) / "knowledge_graph.mmd"
            with open(mermaid_path, "w", encoding="utf-8") as f:
                f.write(mermaid)

            return {
                "entities": len(kg.entities),
                "relationships": len(kg.relationships),
                "mermaid": mermaid,
                "graph_file": str(graph_path),
                "mermaid_file": str(mermaid_path),
            }

        except Exception as e:
            logger.error(f"知识图谱构建失败: {e}")
            raise HTTPException(status_code=500, detail=f"图谱构建失败: {str(e)}")

    @app.get("/api/graph/mermaid")
    async def get_graph_mermaid():
        """获取知识图谱的 Mermaid 可视化"""
        mermaid = state.graph_builder.to_mermaid()
        return {"mermaid": mermaid}

    # ============================================================
    # 系统 API
    # ============================================================

    @app.get("/api/system/status")
    async def system_status():
        """系统状态"""
        return {
            "status": "running",
            "documents": len(state._documents),
            "total_chunks": state.vector_store.total_chunks,
            "wiki_pages": state.wiki_manager.total_pages,
            "graph_entities": len(state.graph_builder._entity_map),
            "graph_relationships": state.graph_builder._graph.number_of_edges(),
        }

    @app.get("/api/system/tools")
    async def list_agent_tools():
        """列出 Agent 可用工具"""
        tools = state.agent_engine._tool_registry.list_tools()
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                }
                for t in tools
            ]
        }

    return app


# 全局应用实例
app = None


def get_app(config: Optional[AppConfig] = None) -> FastAPI:
    global app
    if app is None:
        app = create_app(config)
    return app
