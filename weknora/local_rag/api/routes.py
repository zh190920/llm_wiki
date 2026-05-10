"""
FastAPI 路由定义
================================
提供 RESTful API 接口，覆盖：
- 知识库 CRUD
- 文档上传和管理
- RAG 快速问答
- Agent 智能推理
- Wiki 浏览和搜索
- 知识图谱查询
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from loguru import logger

from api.schemas import (
    CreateKnowledgeBaseRequest, KnowledgeBaseResponse,
    AddTextRequest, KnowledgeResponse,
    SearchRequest, SearchResponse, SearchResultItem,
    ChatRequest, ChatResponse,
    WikiPageResponse, WikiGraphResponse,
    GraphQueryRequest, GraphDataResponse,
    MessageResponse,
)
from knowledge_base.manager import KnowledgeBaseManager
from agent.engine import AgentEngine
from agent.tools import ToolRegistry
from core.llm_client import LLMClient


def create_router(kb_manager: KnowledgeBaseManager) -> APIRouter:
    """创建 API 路由"""
    router = APIRouter()

    # ──────────── 知识库管理 ────────────

    @router.post("/knowledge-bases", response_model=KnowledgeBaseResponse)
    async def create_knowledge_base(req: CreateKnowledgeBaseRequest):
        """创建知识库"""
        kb = await kb_manager.create_knowledge_base(
            name=req.name,
            description=req.description,
            kb_type=req.type,
        )
        return KnowledgeBaseResponse(**kb.__dict__)

    @router.get("/knowledge-bases", response_model=list[KnowledgeBaseResponse])
    async def list_knowledge_bases():
        """列出所有知识库"""
        kbs = kb_manager.list_knowledge_bases()
        return [KnowledgeBaseResponse(**kb.__dict__) for kb in kbs]

    @router.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseResponse)
    async def get_knowledge_base(kb_id: str):
        """获取知识库详情"""
        kb = kb_manager.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return KnowledgeBaseResponse(**kb.__dict__)

    @router.delete("/knowledge-bases/{kb_id}", response_model=MessageResponse)
    async def delete_knowledge_base(kb_id: str):
        """删除知识库"""
        success = await kb_manager.delete_knowledge_base(kb_id)
        if not success:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return MessageResponse(message="知识库已删除")

    # ──────────── 文档管理 ────────────

    @router.post("/knowledge-bases/{kb_id}/documents/upload", response_model=KnowledgeResponse)
    async def upload_document(kb_id: str, file: UploadFile = File(...)):
        """上传文档到知识库"""
        import tempfile
        import os

        kb = kb_manager.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        # 保存上传文件到临时目录
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            knowledge = await kb_manager.add_document(kb_id, tmp_path)
            return KnowledgeResponse(**knowledge.__dict__)
        finally:
            os.unlink(tmp_path)

    @router.post("/knowledge-bases/{kb_id}/documents/text", response_model=KnowledgeResponse)
    async def add_text_document(kb_id: str, req: AddTextRequest):
        """添加纯文本到知识库"""
        kb = kb_manager.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        knowledge = await kb_manager.add_text(kb_id, req.title, req.content)
        return KnowledgeResponse(**knowledge.__dict__)

    @router.get("/knowledge-bases/{kb_id}/documents", response_model=list[KnowledgeResponse])
    async def list_documents(kb_id: str):
        """列出知识库中的文档"""
        docs = kb_manager.list_documents(kb_id)
        return [KnowledgeResponse(**d.__dict__) for d in docs]

    # ──────────── 检索 ────────────

    @router.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest):
        """混合检索（向量 + 关键词）"""
        results = await kb_manager.search(
            query=req.query,
            knowledge_base_ids=req.knowledge_base_ids,
            top_k=req.top_k,
        )
        items = [
            SearchResultItem(
                chunk_id=r.chunk_id,
                knowledge_id=r.knowledge_id,
                knowledge_base_id=r.knowledge_base_id,
                content=r.content,
                score=r.score,
                vector_score=r.vector_score,
                keyword_score=r.keyword_score,
            )
            for r in results
        ]
        return SearchResponse(results=items, total=len(items))

    # ──────────── 对话（RAG + Agent） ────────────

    @router.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        """
        智能问答对话

        - mode=rag: RAG 快速问答模式
        - mode=agent: ReAct Agent 智能推理模式
        """
        if req.mode == "agent":
            return await _agent_chat(kb_manager, req)
        else:
            return await _rag_chat(kb_manager, req)

    async def _rag_chat(kb_manager: KnowledgeBaseManager, req: ChatRequest) -> ChatResponse:
        """RAG 快速问答"""
        result = await kb_manager.rag_query(
            query=req.query,
            knowledge_base_ids=req.knowledge_base_ids,
            top_k=req.top_k,
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            mode="rag",
        )

    async def _agent_chat(kb_manager: KnowledgeBaseManager, req: ChatRequest) -> ChatResponse:
        """ReAct Agent 智能推理"""
        # 构建工具注册表
        tool_registry = ToolRegistry()

        # 注册知识搜索工具
        async def knowledge_search_func(query: str, top_k: int = 5) -> str:
            results = await kb_manager.search(query, req.knowledge_base_ids, top_k)
            if not results:
                return "未找到相关知识。"
            parts = []
            for i, r in enumerate(results, 1):
                parts.append(f"[{i}] {r.content[:500]}")
            return "\n\n".join(parts)

        tool_registry.register("knowledge_search", knowledge_search_func)

        # 注册知识图谱查询工具
        def query_graph_func(entity: str, depth: int = 1) -> str:
            related = kb_manager.graph_builder.get_entity_relations(entity, top_k=10)
            if not related:
                return f"知识图谱中未找到实体: {entity}"
            return f"与 '{entity}' 相关的实体: {', '.join(related)}"

        tool_registry.register("query_knowledge_graph", query_graph_func)

        # 构建知识库信息
        kb_infos = []
        if req.knowledge_base_ids:
            for kb_id in req.knowledge_base_ids:
                kb = kb_manager.get_knowledge_base(kb_id)
                if kb:
                    kb_infos.append({
                        "id": kb.id,
                        "name": kb.name,
                        "doc_count": kb.doc_count,
                        "description": kb.description,
                        "capabilities": kb.capabilities,
                    })

        has_graph = len(kb_manager.graph_builder.graph.nodes) > 0

        # 创建并执行 Agent
        engine = AgentEngine(
            llm_client=kb_manager.llm_client,
            tool_registry=tool_registry,
            knowledge_bases=kb_infos,
            has_graph=has_graph,
        )

        state = await engine.execute(query=req.query)

        # 转换步骤
        steps = []
        for step in state.round_steps:
            step_data = {
                "iteration": step.iteration,
                "thought": step.thought[:200] if step.thought else "",
                "tool_calls": [
                    {
                        "name": tc.name,
                        "success": tc.success,
                        "duration_ms": tc.duration_ms,
                    }
                    for tc in step.tool_calls
                ],
            }
            steps.append(step_data)

        return ChatResponse(
            answer=state.final_answer,
            mode="agent",
            steps=steps,
        )

    # ──────────── Wiki ────────────

    @router.get("/wiki/{kb_id}/pages", response_model=list[WikiPageResponse])
    async def list_wiki_pages(kb_id: str, page_type: Optional[str] = None):
        """列出 Wiki 页面"""
        pages = kb_manager.page_manager.list_pages(kb_id, page_type)
        return [
            WikiPageResponse(
                slug=p.slug,
                title=p.title,
                page_type=p.page_type,
                content=p.content,
                summary=p.summary,
                aliases=p.aliases,
                out_links=p.out_links,
                in_links=p.in_links,
                updated_at=p.updated_at,
            )
            for p in pages
        ]

    @router.get("/wiki/{kb_id}/pages/{slug}", response_model=WikiPageResponse)
    async def get_wiki_page(kb_id: str, slug: str):
        """获取 Wiki 页面"""
        page = kb_manager.page_manager.get_page(kb_id, slug)
        if not page:
            raise HTTPException(status_code=404, detail="页面不存在")
        return WikiPageResponse(
            slug=page.slug,
            title=page.title,
            page_type=page.page_type,
            content=page.content,
            summary=page.summary,
            aliases=page.aliases,
            out_links=page.out_links,
            in_links=page.in_links,
            updated_at=page.updated_at,
        )

    @router.get("/wiki/{kb_id}/graph", response_model=WikiGraphResponse)
    async def get_wiki_graph(kb_id: str):
        """获取 Wiki 链接图谱数据"""
        data = kb_manager.page_manager.get_graph_data(kb_id)
        return WikiGraphResponse(nodes=data["nodes"], edges=data["edges"])

    @router.get("/wiki/{kb_id}/search", response_model=list[WikiPageResponse])
    async def search_wiki(kb_id: str, q: str, limit: int = 10):
        """搜索 Wiki 页面"""
        pages = kb_manager.page_manager.search_pages(kb_id, q, limit)
        return [
            WikiPageResponse(
                slug=p.slug, title=p.title, page_type=p.page_type,
                content=p.content, summary=p.summary, aliases=p.aliases,
                out_links=p.out_links, in_links=p.in_links, updated_at=p.updated_at,
            )
            for p in pages
        ]

    # ──────────── 知识图谱 ────────────

    @router.get("/graph", response_model=GraphDataResponse)
    async def get_knowledge_graph():
        """获取知识图谱数据"""
        data = kb_manager.graph_builder.get_graph_data()
        mermaid = kb_manager.graph_builder._generate_mermaid()
        return GraphDataResponse(
            nodes=data["nodes"],
            edges=data["edges"],
            mermaid=mermaid,
        )

    @router.post("/graph/query", response_model=GraphDataResponse)
    async def query_knowledge_graph(req: GraphQueryRequest):
        """查询知识图谱"""
        related = kb_manager.graph_builder.get_entity_relations(req.entity, top_k=20)
        data = kb_manager.graph_builder.get_graph_data()
        mermaid = kb_manager.graph_builder._generate_mermaid()

        # 过滤相关节点
        if related:
            related_set = {req.entity} | set(related)
            filtered_nodes = [n for n in data["nodes"] if n["title"] in related_set]
            filtered_edges = [
                e for e in data["edges"]
                if e["source"] in related_set or e["target"] in related_set
            ]
        else:
            filtered_nodes = data["nodes"]
            filtered_edges = data["edges"]

        return GraphDataResponse(
            nodes=filtered_nodes,
            edges=filtered_edges,
            mermaid=mermaid,
        )

    # ──────────── 健康检查 ────────────

    @router.get("/health")
    async def health_check():
        """健康检查"""
        return {"status": "ok", "version": "1.0.0"}

    return router
