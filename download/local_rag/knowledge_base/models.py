"""
知识库数据模型
================================
定义知识库、知识条目、搜索结果等核心数据结构。
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class KnowledgeBase:
    """知识库"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    type: str = "document"  # document, faq, wiki
    doc_count: int = 0
    capabilities: list[str] = field(default_factory=lambda: ["chunks", "wiki"])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Knowledge:
    """知识条目"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    knowledge_base_id: str = ""
    title: str = ""
    file_name: str = ""
    file_type: str = ""
    file_size: int = 0
    content: str = ""
    parse_status: str = "pending"  # pending, processing, completed, failed
    chunk_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SearchResult:
    """搜索结果"""
    chunk_id: str = ""
    knowledge_id: str = ""
    knowledge_base_id: str = ""
    content: str = ""
    score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatMessage:
    """对话消息"""
    role: str = "user"  # user, assistant, system
    content: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Session:
    """对话会话"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    knowledge_base_ids: list[str] = field(default_factory=list)
    mode: str = "rag"  # rag, agent
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
