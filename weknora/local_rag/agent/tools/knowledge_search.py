"""
知识检索工具 - Agent 使用的核心检索工具
"""
import json
import logging
from typing import Any, Dict, List, Optional

from agent.tool_registry import Tool
from models.schemas import SearchResult

logger = logging.getLogger(__name__)


class KnowledgeSearchTool(Tool):
    """
    知识检索工具

    借鉴 WeKnora 的 knowledge_search 工具设计：
    - 语义搜索：基于向量相似度检索
    - 关键词搜索：基于 BM25 的关键词检索
    - 混合搜索：融合两种检索方式
    - 文档路由：只搜索与查询相关的文档
    """

    def __init__(self, retriever, doc_router=None):
        self._retriever = retriever
        self._doc_router = doc_router  # 文档路由器（可选）

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return (
            "在知识库中搜索与查询相关的文档内容。"
            "支持语义搜索和关键词搜索。"
            "当需要查找特定信息、事实、数据或文档内容时使用此工具。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询文本",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["hybrid", "vector", "keyword"],
                    "description": "搜索类型：hybrid=混合搜索（默认），vector=语义搜索，keyword=关键词搜索",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量，默认5",
                },
                "doc_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "限定搜索的文档ID列表（可选，不填则自动路由到相关文档）",
                },
            },
            "required": ["query"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        query = arguments["query"]
        search_type = arguments.get("search_type", "hybrid")
        top_k = arguments.get("top_k", 5)
        doc_ids = arguments.get("doc_ids")

        # 文档路由预筛选：如果没有手动指定 doc_ids，则尝试自动路由
        # 这是"先确定检索手册范围，再在范围内检索"的第一步
        if doc_ids is None and self._doc_router is not None:
            routed = self._doc_router.route(query)
            if routed:
                doc_ids = routed
                logger.info(
                    f"知识检索工具 - 文档路由预筛选: 查询='{query[:50]}' → "
                    f"匹配 {len(doc_ids)} 个文档，将在这些文档的子空间中检索"
                )
            else:
                logger.info(f"知识检索工具 - 文档路由预筛选: 查询='{query[:50]}' → 无匹配，全量检索")
        elif doc_ids:
            logger.info(f"知识检索工具 - 手动指定文档范围: {len(doc_ids)} 个文档")

        from models.schemas import MatchType, SearchParams

        if search_type == "vector":
            match_types = [MatchType.VECTOR]
        elif search_type == "keyword":
            match_types = [MatchType.KEYWORD]
        else:
            match_types = [MatchType.VECTOR, MatchType.KEYWORD]

        params = SearchParams(
            query=query,
            top_k=top_k,
            match_types=match_types,
            doc_ids=doc_ids,
        )

        results = await self._retriever.retrieve(params, use_query_understanding=False, use_rerank=False)

        return self._format_results(results)

    @staticmethod
    def _format_results(results: List[SearchResult]) -> str:
        """格式化检索结果"""
        if not results:
            return "未找到相关内容。"

        parts = []
        for i, r in enumerate(results):
            source = r.chunk.metadata.get("section_title", r.chunk.doc_id)
            parts.append(
                f"[结果{i+1}] (相关度: {r.score:.3f}, 来源: {source})\n"
                f"{r.chunk.content}"
            )

        return "\n\n---\n\n".join(parts)


class GrepChunksTool(Tool):
    """
    关键词搜索工具 - 在文档块中进行精确关键词匹配

    借鉴 WeKnora 的 grep_chunks 工具
    """

    def __init__(self, vector_store):
        self._vector_store = vector_store

    @property
    def name(self) -> str:
        return "grep_chunks"

    @property
    def description(self) -> str:
        return (
            "在文档块中进行精确关键词/短语搜索。"
            "当需要精确查找包含特定关键词或短语的文档内容时使用此工具。"
            "比语义搜索更精确，适合查找专有名词、代码片段等。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "要搜索的关键词或短语",
                },
                "doc_id": {
                    "type": "string",
                    "description": "限定搜索的文档ID（可选）",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量，默认10",
                },
            },
            "required": ["keyword"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        keyword = arguments["keyword"]
        doc_id = arguments.get("doc_id")
        top_k = arguments.get("top_k", 10)

        # 在所有块中搜索关键词
        chunks = self._vector_store._chunks
        if doc_id:
            chunks = [c for c in chunks if c.doc_id == doc_id]

        results = []
        keyword_lower = keyword.lower()
        for chunk in chunks:
            if keyword_lower in chunk.content.lower():
                results.append(chunk)
                if len(results) >= top_k:
                    break

        if not results:
            return f"未找到包含 '{keyword}' 的内容。"

        parts = []
        for i, chunk in enumerate(results):
            # 高亮关键词位置
            content = chunk.content
            parts.append(f"[结果{i+1}] (文档: {chunk.doc_id})\n{content[:500]}")

        return "\n\n---\n\n".join(parts)


class ListKnowledgeChunksTool(Tool):
    """列出文档的所有块"""

    def __init__(self, vector_store):
        self._vector_store = vector_store

    @property
    def name(self) -> str:
        return "list_knowledge_chunks"

    @property
    def description(self) -> str:
        return "列出指定文档的所有文档块。用于了解文档的结构和内容概览。"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "文档ID",
                },
            },
            "required": ["doc_id"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        doc_id = arguments["doc_id"]
        chunks = self._vector_store.get_chunks_by_doc_id(doc_id)

        if not chunks:
            return f"未找到文档 {doc_id} 的内容。"

        parts = [f"文档 {doc_id} 共有 {len(chunks)} 个块：\n"]
        for chunk in chunks:
            preview = chunk.content[:100].replace("\n", " ")
            parts.append(f"  [{chunk.index}] {preview}... (tokens: {chunk.token_count})")

        return "\n".join(parts)
