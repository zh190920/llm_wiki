"""
Wiki 相关工具 - Agent 用于操作 Wiki 知识库的工具集
借鉴 WeKnora 的 wiki_* 系列工具设计
"""
import logging
from typing import Any, Dict, List, Optional

from agent.tool_registry import Tool

logger = logging.getLogger(__name__)


class WikiReadPageTool(Tool):
    """读取 Wiki 页面"""

    def __init__(self, wiki_manager):
        self._wiki_manager = wiki_manager

    @property
    def name(self) -> str:
        return "wiki_read_page"

    @property
    def description(self) -> str:
        return (
            "读取 Wiki 知识库中的页面内容。"
            "可以按页面标题(slug)或关键词搜索并读取页面。"
            "用于获取已有知识的详细信息。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "页面标识符(slug)，如 'rag-overview'",
                },
            },
            "required": ["slug"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        slug = arguments["slug"]
        page = await self._wiki_manager.get_page(slug)

        if page is None:
            available = await self._wiki_manager.list_pages()
            if available:
                slugs = ", ".join([p.slug for p in available[:20]])
                return f"页面 '{slug}' 不存在。可用的页面: {slugs}"
            return f"页面 '{slug}' 不存在。Wiki 知识库当前为空。"

        return (
            f"# {page.title}\n\n"
            f"类型: {page.page_type.value}\n"
            f"状态: {page.status}\n"
            f"出链: {', '.join(page.out_links) if page.out_links else '无'}\n\n"
            f"---\n\n{page.content}"
        )


class WikiWritePageTool(Tool):
    """创建或更新 Wiki 页面"""

    def __init__(self, wiki_manager):
        self._wiki_manager = wiki_manager

    @property
    def name(self) -> str:
        return "wiki_write_page"

    @property
    def description(self) -> str:
        return (
            "创建或更新 Wiki 知识库中的页面。"
            "页面内容使用 Markdown 格式，支持 [[slug|标题]] 语法创建跨页面链接。"
            "用于整理和组织知识。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "页面标识符，使用小写字母和连字符，如 'rag-overview'",
                },
                "title": {
                    "type": "string",
                    "description": "页面标题",
                },
                "content": {
                    "type": "string",
                    "description": "页面内容（Markdown 格式）",
                },
                "page_type": {
                    "type": "string",
                    "enum": ["entity", "concept", "synthesis", "summary"],
                    "description": "页面类型，默认为 concept",
                },
            },
            "required": ["slug", "title", "content"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        from models.schemas import WikiPage, WikiPageType

        slug = arguments["slug"]
        title = arguments["title"]
        content = arguments["content"]
        page_type_str = arguments.get("page_type", "concept")

        try:
            page_type = WikiPageType(page_type_str)
        except ValueError:
            page_type = WikiPageType.CONCEPT

        # 提取出链
        import re
        out_links = re.findall(r'\[\[([^\]|]+)', content)

        page = WikiPage(
            slug=slug,
            title=title,
            page_type=page_type,
            content=content,
            out_links=out_links,
            status="published",
        )

        await self._wiki_manager.save_page(page)
        return f"Wiki 页面 '{title}' (slug: {slug}) 已保存。出链: {len(out_links)} 个。"


class WikiSearchTool(Tool):
    """搜索 Wiki 页面"""

    def __init__(self, wiki_manager):
        self._wiki_manager = wiki_manager

    @property
    def name(self) -> str:
        return "wiki_search"

    @property
    def description(self) -> str:
        return (
            "在 Wiki 知识库中搜索页面。"
            "按标题和内容进行关键词匹配。"
            "用于查找相关的 Wiki 页面。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量，默认5",
                },
            },
            "required": ["query"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)

        results = await self._wiki_manager.search_pages(query, top_k=top_k)

        if not results:
            return f"未找到与 '{query}' 相关的 Wiki 页面。"

        parts = []
        for i, page in enumerate(results):
            preview = page.content[:150].replace("\n", " ")
            parts.append(
                f"[{i+1}] {page.title} (slug: {page.slug}, 类型: {page.page_type.value})\n"
                f"  {preview}..."
            )

        return "\n\n".join(parts)


class WikiReadSourceDocTool(Tool):
    """读取 Wiki 页面关联的原始文档块"""

    def __init__(self, wiki_manager, vector_store):
        self._wiki_manager = wiki_manager
        self._vector_store = vector_store

    @property
    def name(self) -> str:
        return "wiki_read_source_doc"

    @property
    def description(self) -> str:
        return "读取 Wiki 页面引用的原始文档块内容。用于追溯知识来源或获取更详细的信息。"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "Wiki 页面标识符",
                },
            },
            "required": ["slug"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        slug = arguments["slug"]
        page = await self._wiki_manager.get_page(slug)

        if page is None:
            return f"页面 '{slug}' 不存在。"

        if not page.source_chunk_ids:
            return f"页面 '{page.title}' 没有关联的原始文档块。"

        parts = [f"页面 '{page.title}' 的原始文档块：\n"]
        for chunk_id in page.source_chunk_ids:
            chunk = self._vector_store.get_chunk_by_id(chunk_id)
            if chunk:
                parts.append(f"[chunk: {chunk_id}]\n{chunk.content[:300]}...\n")

        return "\n---\n".join(parts)


class WikiFlagIssueTool(Tool):
    """标记 Wiki 页面的质量问题"""

    def __init__(self, wiki_manager):
        self._wiki_manager = wiki_manager

    @property
    def name(self) -> str:
        return "wiki_flag_issue"

    @property
    def description(self) -> str:
        return "标记 Wiki 页面中存在的质量问题，如信息过时、内容矛盾、缺少来源等。"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "page_slug": {
                    "type": "string",
                    "description": "页面标识符",
                },
                "description": {
                    "type": "string",
                    "description": "问题描述",
                },
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "error"],
                    "description": "严重程度，默认 warning",
                },
            },
            "required": ["page_slug", "description"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        from models.schemas import WikiIssue

        issue = WikiIssue(
            page_slug=arguments["page_slug"],
            description=arguments["description"],
            severity=arguments.get("severity", "warning"),
        )

        await self._wiki_manager.add_issue(issue)
        return f"已标记问题: [{issue.severity}] {issue.description} (页面: {issue.page_slug})"
