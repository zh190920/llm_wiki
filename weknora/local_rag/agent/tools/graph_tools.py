"""
知识图谱查询工具 - Agent 用于查询知识图谱和文档信息
借鉴 WeKnora 的 GraphRAG 查询工具设计
"""
import logging
from typing import Any, Dict, List, Optional

from agent.tool_registry import Tool

logger = logging.getLogger(__name__)


class QueryKnowledgeGraphTool(Tool):
    """
    知识图谱查询工具

    借鉴 WeKnora 的 GraphRAG 设计：
    - 根据实体名称查询知识图谱
    - 获取相关实体和关联文档块
    - 支持深度1（直接关系）和深度2（间接关系）
    """

    def __init__(self, graph_builder, vector_store):
        self._graph_builder = graph_builder
        self._vector_store = vector_store

    @property
    def name(self) -> str:
        return "query_knowledge_graph"

    @property
    def description(self) -> str:
        return (
            "查询知识图谱，获取与指定实体相关的信息和文档。"
            "可以通过实体名称查找相关实体、关系和关联的文档内容。"
            "适合探索实体之间的关系和获取结构化知识。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "要查询的实体名称",
                },
                "depth": {
                    "type": "integer",
                    "enum": [1, 2],
                    "description": "关联深度：1=直接关系，2=间接关系（默认1）",
                },
            },
            "required": ["entity_name"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        entity_name = arguments["entity_name"]
        depth = arguments.get("depth", 1)

        # 查找实体
        entity = self._graph_builder._entity_map.get(entity_name)
        if not entity:
            # 尝试模糊匹配
            available = list(self._graph_builder._entity_map.keys())
            close_matches = [
                name for name in available
                if entity_name.lower() in name.lower() or name.lower() in entity_name.lower()
            ]
            if close_matches:
                return (
                    f"未找到实体 '{entity_name}'。"
                    f"相似的实体: {', '.join(close_matches[:5])}"
                )
            return f"未找到实体 '{entity_name}'。知识图谱中暂无此实体。"

        # 获取关联块 ID
        related_chunk_ids = self._graph_builder.get_related_chunks(entity_name, depth=depth)

        # 构建输出
        parts = [f"## 实体: {entity.title}"]
        parts.append(f"- 类型: {entity.entity_type}")
        parts.append(f"- 描述: {entity.description}")
        parts.append(f"- 出现频率: {entity.frequency}")
        parts.append(f"- 关联深度: {depth}")
        parts.append("")

        # 查找直接关联的实体
        related_entities = []
        for neighbor_id in self._graph_builder._graph.neighbors(entity.entity_id):
            neighbor_entity = self._graph_builder._get_entity_by_id(neighbor_id)
            if neighbor_entity:
                # 获取关系类型
                edge_data = self._graph_builder._graph.get_edge_data(
                    entity.entity_id, neighbor_id
                )
                relation_type = edge_data.get("relation_type", "相关") if edge_data else "相关"
                related_entities.append(
                    f"  - {neighbor_entity.title} ({neighbor_entity.entity_type}) "
                    f"[{relation_type}]"
                )

        if related_entities:
            parts.append("### 关联实体:")
            parts.extend(related_entities[:10])
            parts.append("")

        # 获取关联文档块内容
        if related_chunk_ids:
            parts.append(f"### 关联文档 ({len(related_chunk_ids)} 个块):")
            for i, chunk_id in enumerate(related_chunk_ids[:5]):
                chunk = self._vector_store.get_chunk_by_id(chunk_id)
                if chunk:
                    preview = chunk.content[:200].replace("\n", " ")
                    parts.append(f"  [{i+1}] (文档: {chunk.doc_id}) {preview}...")
            if len(related_chunk_ids) > 5:
                parts.append(f"  ...还有 {len(related_chunk_ids) - 5} 个关联块")

        return "\n".join(parts)


class GetDocumentInfoTool(Tool):
    """
    文档信息查询工具 - 返回文档元数据
    """

    def __init__(self, vector_store):
        self._vector_store = vector_store

    @property
    def name(self) -> str:
        return "get_document_info"

    @property
    def description(self) -> str:
        return (
            "获取文档的元数据信息，包括文件名、类型、块数量等。"
            "用于了解知识库中有哪些文档及其基本信息。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "文档ID（可选，不提供则返回所有文档列表）",
                },
            },
            "required": [],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        doc_id = arguments.get("doc_id")

        if doc_id:
            # 获取特定文档信息
            chunks = self._vector_store.get_chunks_by_doc_id(doc_id)
            if not chunks:
                return f"未找到文档 {doc_id}。"

            total_tokens = sum(c.token_count for c in chunks)
            parent_count = sum(1 for c in chunks if c.metadata.get("is_parent"))
            child_count = sum(1 for c in chunks if c.parent_chunk_id)

            parts = [f"## 文档: {doc_id}"]
            parts.append(f"- 总块数: {len(chunks)}")
            parts.append(f"- 总 Token 数: {total_tokens}")
            if parent_count > 0 or child_count > 0:
                parts.append(f"- 父块: {parent_count}, 子块: {child_count}")
            parts.append("")

            # 显示各块概览
            for chunk in chunks[:20]:
                preview = chunk.content[:80].replace("\n", " ")
                parent_info = f" [父块]" if chunk.metadata.get("is_parent") else ""
                child_info = f" [子块→{chunk.parent_chunk_id[:8]}]" if chunk.parent_chunk_id else ""
                parts.append(f"  [{chunk.index}] {preview}... ({chunk.token_count} tokens){parent_info}{child_info}")

            if len(chunks) > 20:
                parts.append(f"  ...还有 {len(chunks) - 20} 个块")

            return "\n".join(parts)
        else:
            # 列出所有文档
            doc_ids = set(c.doc_id for c in self._vector_store._chunks if c.doc_id)
            if not doc_ids:
                return "知识库中暂无文档。"

            parts = [f"知识库中共有 {len(doc_ids)} 个文档：\n"]
            for did in sorted(doc_ids):
                chunks = self._vector_store.get_chunks_by_doc_id(did)
                total_tokens = sum(c.token_count for c in chunks)
                parts.append(f"  - {did}: {len(chunks)} 块, {total_tokens} tokens")

            return "\n".join(parts)
