"""
知识检索工具 - Agent 使用的核心检索工具
"""
import logging
import re
from typing import Any, Dict, List, Optional, Set

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
    """

    def __init__(self, retriever):
        self._retriever = retriever

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
            },
            "required": ["query"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        query = arguments["query"]
        search_type = arguments.get("search_type", "hybrid")
        top_k = arguments.get("top_k", 5)

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
    正则搜索工具 - 在文档块中进行正则表达式匹配

    借鉴 WeKnora 的 grep_chunks 工具，升级支持：
    - 完整正则表达式支持
    - 多模式同时搜索（最多5个）
    - 匹配片段上下文展示（前后各60字符）
    - 每个模式的命中计数
    - already_seen 跨调用跟踪（避免重复结果）
    - MMR 多样性去重（结果>10时使用 Jaccard 相似度）
    """

    def __init__(self, vector_store):
        self._vector_store = vector_store
        self._already_seen: Set[str] = set()  # 跨调用已见 chunk_id 集合

    @property
    def name(self) -> str:
        return "grep_chunks"

    @property
    def description(self) -> str:
        return (
            "在文档块中进行正则表达式搜索。"
            "支持多个搜索模式（最多5个），返回匹配片段及上下文。"
            "比语义搜索更精确，适合查找专有名词、代码片段、特定格式内容等。"
            "支持 Python 正则表达式语法。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要搜索的正则表达式模式列表（最多5个）",
                    "maxItems": 5,
                },
                "doc_id": {
                    "type": "string",
                    "description": "限定搜索的文档ID（可选）",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量，默认10",
                },
                "reset_seen": {
                    "type": "boolean",
                    "description": "是否重置已见记录（默认false），用于开始新的搜索任务时清除之前的跟踪",
                },
            },
            "required": ["patterns"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        patterns = arguments["patterns"]
        doc_id = arguments.get("doc_id")
        top_k = arguments.get("top_k", 10)
        reset_seen = arguments.get("reset_seen", False)

        # 重置已见记录
        if reset_seen:
            self._already_seen.clear()

        # 限制模式数量
        if len(patterns) > 5:
            patterns = patterns[:5]
            logger.warning("搜索模式超过5个，已截断为前5个")

        # 编译正则表达式
        compiled_patterns = []
        for p in patterns:
            try:
                compiled_patterns.append(re.compile(p, re.IGNORECASE))
            except re.error as e:
                return f"正则表达式错误 '{p}': {e}"

        # 在所有块中搜索
        chunks = self._vector_store._chunks
        if doc_id:
            chunks = [c for c in chunks if c.doc_id == doc_id]

        # 搜索匹配
        match_results: List[dict] = []  # {chunk, snippets, hit_counts}
        for chunk in chunks:
            hit_counts: Dict[str, int] = {}
            snippets: List[str] = []

            for i, pattern in enumerate(compiled_patterns):
                matches = list(pattern.finditer(chunk.content))
                if matches:
                    hit_counts[patterns[i]] = len(matches)
                    # 提取匹配片段及上下文
                    for m in matches:
                        start = max(0, m.start() - 60)
                        end = min(len(chunk.content), m.end() + 60)
                        context = chunk.content[start:end]
                        # 标记匹配部分
                        prefix = "..." if start > 0 else ""
                        suffix = "..." if end < len(chunk.content) else ""
                        snippet = f"{prefix}{context}{suffix}"
                        snippets.append(snippet)

            if hit_counts:
                # 检查是否已见过
                is_new = chunk.chunk_id not in self._already_seen
                match_results.append({
                    "chunk": chunk,
                    "snippets": snippets,
                    "hit_counts": hit_counts,
                    "is_new": is_new,
                })
                self._already_seen.add(chunk.chunk_id)

        if not match_results:
            return f"未找到匹配 '{', '.join(patterns)}' 的内容。"

        # MMR 多样性去重（结果 > 10 时）
        if len(match_results) > 10:
            match_results = self._mmr_diversify(match_results, top_k)

        # 格式化输出
        parts = []
        for i, result in enumerate(match_results[:top_k]):
            chunk = result["chunk"]
            new_flag = " [新]" if result["is_new"] else ""
            hit_info = ", ".join(
                f"'{p}': {cnt}次" for p, cnt in result["hit_counts"].items()
            )
            snippet_text = "\n  ".join(result["snippets"][:5])  # 最多显示5个片段

            parts.append(
                f"[结果{i+1}]{new_flag} (文档: {chunk.doc_id}, 命中: {hit_info})\n"
                f"  {snippet_text}"
            )

        # 汇总信息
        total_hits = sum(
            sum(r["hit_counts"].values()) for r in match_results
        )
        summary = (
            f"搜索完成: {len(match_results)} 个块匹配, "
            f"共 {total_hits} 次命中, "
            f"已跟踪 {len(self._already_seen)} 个块"
        )

        return summary + "\n\n" + "\n\n---\n\n".join(parts)

    @staticmethod
    def _mmr_diversify(
        results: List[dict], top_k: int, lambda_param: float = 0.7
    ) -> List[dict]:
        """
        MMR 多样性去重 - 基于 Jaccard 相似度

        在 token 集合上计算 Jaccard 相似度，避免返回内容高度重复的结果
        """
        def tokenize(text: str) -> Set[str]:
            """简单分词：中文按字，英文按词"""
            tokens: Set[str] = set()
            # 英文单词
            tokens.update(re.findall(r'[a-zA-Z]+', text.lower()))
            # 中文字
            tokens.update(re.findall(r'[\u4e00-\u9fff]', text))
            return tokens

        def jaccard(set_a: Set[str], set_b: Set[str]) -> float:
            if not set_a or not set_b:
                return 0.0
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            return intersection / union if union > 0 else 0.0

        # 预计算 token 集合
        token_sets = [tokenize(r["chunk"].content) for r in results]

        # 按命中数排序作为相关性分数
        relevance_scores = [
            sum(r["hit_counts"].values()) for r in results
        ]
        max_rel = max(relevance_scores) if relevance_scores else 1
        normalized_rel = [s / max_rel for s in relevance_scores]

        selected_indices: List[int] = []
        remaining = set(range(len(results)))

        # 选择第一个（最高相关性）
        if results:
            best_idx = max(remaining, key=lambda i: normalized_rel[i])
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        while remaining and len(selected_indices) < top_k:
            best_mmr = -float("inf")
            best_idx = -1

            for i in remaining:
                relevance = normalized_rel[i]
                # 与已选结果的最大相似度
                max_sim = 0.0
                for si in selected_indices:
                    sim = jaccard(token_sets[i], token_sets[si])
                    max_sim = max(max_sim, sim)

                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return [results[i] for i in selected_indices]


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
            parent_info = f" (父块: {chunk.parent_chunk_id})" if chunk.parent_chunk_id else ""
            parts.append(f"  [{chunk.index}] {preview}... (tokens: {chunk.token_count}){parent_info}")

        return "\n".join(parts)
