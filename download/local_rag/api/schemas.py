"""
API 请求/响应 Schema
================================
定义 FastAPI 接口的数据模型。
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── 知识库 ──

class CreateKnowledgeBaseRequest(BaseModel):
    name: str = Field(..., description="知识库名称", min_length=1, max_length=100)
    description: str = Field(default="", description="知识库描述")
    type: str = Field(default="document", description="类型: document, faq, wiki")


class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: str
    type: str
    doc_count: int
    capabilities: list[str]
    created_at: str
    updated_at: str


# ── 文档 ──

class AddTextRequest(BaseModel):
    title: str = Field(..., description="文本标题")
    content: str = Field(..., description="文本内容", min_length=1)


class KnowledgeResponse(BaseModel):
    id: str
    knowledge_base_id: str
    title: str
    file_name: str
    file_type: str
    file_size: int
    parse_status: str
    chunk_count: int
    created_at: str


# ── 检索 ──

class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询", min_length=1)
    knowledge_base_ids: Optional[list[str]] = Field(default=None, description="限定知识库范围")
    top_k: int = Field(default=5, description="返回结果数", ge=1, le=20)


class SearchResultItem(BaseModel):
    chunk_id: str
    knowledge_id: str
    knowledge_base_id: str
    content: str
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int


# ── 对话 ──

class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题", min_length=1)
    knowledge_base_ids: Optional[list[str]] = Field(default=None, description="知识库范围")
    mode: str = Field(default="rag", description="模式: rag 或 agent")
    top_k: int = Field(default=5, description="检索结果数")


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)
    mode: str = "rag"
    steps: list[dict] = Field(default_factory=list)


# ── Wiki ──

class WikiPageResponse(BaseModel):
    slug: str
    title: str
    page_type: str
    content: str
    summary: str
    aliases: list[str]
    out_links: list[str]
    in_links: list[str]
    updated_at: str


class WikiGraphResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


# ── 知识图谱 ──

class GraphQueryRequest(BaseModel):
    entity: str = Field(..., description="实体名称")
    depth: int = Field(default=1, description="查询深度", ge=1, le=3)


class GraphDataResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]
    mermaid: str = ""


# ── 通用 ──

class MessageResponse(BaseModel):
    message: str
    success: bool = True
