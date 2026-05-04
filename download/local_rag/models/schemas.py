"""
核心数据模型 - 定义系统中所有数据结构
借鉴 WeKnora 的类型设计，使用 Pydantic v2 实现
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# 文档与分块模型
# ============================================================

class DocumentMetadata(BaseModel):
    """文档元数据"""
    doc_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    filename: str = ""
    file_type: str = ""  # pdf, markdown
    title: str = ""
    source: str = ""
    created_at: float = Field(default_factory=time.time)
    chunk_count: int = 0


class Chunk(BaseModel):
    """文档分块 - 最小检索单元"""
    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    doc_id: str = ""
    content: str = ""
    index: int = 0  # 在文档中的顺序
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0


# ============================================================
# 检索模型
# ============================================================

class MatchType(str, Enum):
    """匹配类型"""
    VECTOR = "vector"
    KEYWORD = "keyword"
    GRAPH = "graph"


class SearchResult(BaseModel):
    """检索结果"""
    chunk: Chunk
    score: float = 0.0
    match_type: MatchType = MatchType.VECTOR
    highlighted_content: str = ""


class SearchParams(BaseModel):
    """检索参数"""
    query: str
    top_k: int = 10
    similarity_threshold: float = 0.5
    match_types: List[MatchType] = Field(default_factory=lambda: [MatchType.VECTOR, MatchType.KEYWORD])


# ============================================================
# Agent 模型
# ============================================================

class ToolCall(BaseModel):
    """工具调用请求"""
    call_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """工具执行结果"""
    call_id: str = ""
    name: str = ""
    output: str = ""
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    is_error: bool = False


class AgentStep(BaseModel):
    """Agent 单步执行记录"""
    step_index: int = 0
    thought: str = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
    observation: str = ""


class AgentState(str, Enum):
    """Agent 状态"""
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    ERROR = "error"


# ============================================================
# Wiki 模型
# ============================================================

class WikiPageType(str, Enum):
    """Wiki 页面类型"""
    INDEX = "index"
    SUMMARY = "summary"
    ENTITY = "entity"
    CONCEPT = "concept"
    SYNTHESIS = "synthesis"


class WikiPage(BaseModel):
    """Wiki 页面"""
    slug: str = ""  # URL友好标识
    title: str = ""
    page_type: WikiPageType = WikiPageType.ENTITY
    content: str = ""  # Markdown 内容
    source_doc_ids: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)
    out_links: List[str] = Field(default_factory=list)  # 出链 slug 列表
    status: str = "draft"  # draft / published
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class WikiIssue(BaseModel):
    """Wiki 质量问题"""
    issue_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    page_slug: str = ""
    description: str = ""
    severity: str = "warning"  # info / warning / error
    resolved: bool = False


# ============================================================
# 知识图谱模型
# ============================================================

class Entity(BaseModel):
    """知识图谱实体"""
    entity_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    description: str = ""
    entity_type: str = "generic"  # person / organization / concept / product / ...
    frequency: int = 1
    source_chunk_ids: List[str] = Field(default_factory=list)


class Relationship(BaseModel):
    """知识图谱关系"""
    relation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_entity_id: str = ""
    target_entity_id: str = ""
    relation_type: str = ""
    description: str = ""
    weight: float = 1.0
    source_chunk_ids: List[str] = Field(default_factory=list)


class KnowledgeGraph(BaseModel):
    """知识图谱"""
    entities: Dict[str, Entity] = Field(default_factory=dict)
    relationships: List[Relationship] = Field(default_factory=list)


# ============================================================
# API 请求/响应模型
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求"""
    query: str
    mode: str = "rag"  # rag / agent / wiki
    knowledge_base_ids: List[str] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    conversation_id: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str = ""
    sources: List[SearchResult] = Field(default_factory=list)
    agent_steps: List[AgentStep] = Field(default_factory=list)
    conversation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])


class UploadResponse(BaseModel):
    """文档上传响应"""
    doc_id: str = ""
    filename: str = ""
    chunk_count: int = 0
    message: str = ""


class WikiGenerateRequest(BaseModel):
    """Wiki 生成请求"""
    knowledge_base_id: str = ""
    document_ids: List[str] = Field(default_factory=list)
    granularity: str = "standard"  # focused / standard / exhaustive


class WikiGenerateResponse(BaseModel):
    """Wiki 生成响应"""
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "processing"
    pages_generated: int = 0
    message: str = ""
