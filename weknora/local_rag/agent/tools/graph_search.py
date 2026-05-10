"""
图谱检索工具 - Agent 使用的知识图谱检索工具

通过知识图谱的实体关系遍历，发现与查询间接关联的信息。
与 knowledge_search（向量+关键词检索）互补，图谱检索可以发现"隐式关联"。
"""
import logging
from typing import Any, Dict, List, Optional

from agent.tool_registry import Tool

logger = logging.getLogger(__name__)


class GraphSearchTool(Tool):
    """
    图谱检索工具

    借鉴 WeKnora 的 GraphRAG 设计：
    - 通过知识图谱遍历发现与查询实体相关的文档块
    - 支持直接关系和间接关系（2度关联）
    - 发现向量检索和关键词检索无法找到的"隐式关联"

    适用场景：
    - 需要发现实体之间的间接关系
    - 查询涉及多个实体的交叉关联
    - 需要"朋友的朋友"式的扩展检索
    """

    def __init__(self, vector_store, doc_router=None):
        self._vector_store = vector_store
        self._doc_router = doc_router

    @property
    def name(self) -> str:
        return "graph_search"

    @property
    def description(self) -> str:
        return (
            "通过知识图谱检索与查询相关的文档内容。"
            "基于实体关系遍历，可以发现间接关联的信息。"
            "当需要查找实体之间的关联关系、或发现向量搜索无法找到的隐式关联时使用此工具。"
            "例如：查询'设备A'时，可以通过图谱找到与'设备A'有关系（如包含、依赖）的其他实体相关文档。"
            "注意：使用前需要先构建知识图谱（build_knowledge_graph）。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询文本（应包含实体名称，如'设备A'、'故障码E003'）",
                },
                "depth": {
                    "type": "integer",
                    "description": "图遍历深度：1=直接关系，2=间接关系（默认2）",
                    "default": 2,
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量，默认10",
                    "default": 10,
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
        depth = arguments.get("depth", 2)
        top_k = arguments.get("top_k", 10)
        doc_ids = arguments.get("doc_ids")

        # 文档路由预筛选
        if doc_ids is None and self._doc_router is not None:
            routed = self._doc_router.route(query)
            if routed:
                doc_ids = routed
                logger.info(
                    f"图谱检索工具 - 文档路由预筛选: 查询='{query[:50]}' → "
                    f"匹配 {len(doc_ids)} 个文档"
                )

        results = await self._vector_store.search_graph(
            query=query,
            top_k=top_k,
            doc_ids=doc_ids,
            depth=depth,
        )

        return self._format_results(results, depth)

    @staticmethod
    def _format_results(results, depth: int = 2) -> str:
        """格式化图谱检索结果"""
        if not results:
            return "图谱检索未找到相关内容。（可能尚未构建知识图谱，或查询中未包含图谱中的实体名称）"

        parts = [f"图谱检索结果（遍历深度={depth}）：\n"]
        for i, r in enumerate(results):
            source = r.chunk.metadata.get("section_title", r.chunk.doc_id)
            match_type = r.match_type.value
            parts.append(
                f"[结果{i+1}] (匹配类型: {match_type}, 来源: {source})\n"
                f"{r.chunk.content}"
            )

        return "\n\n---\n\n".join(parts)


class GraphEntityInfoTool(Tool):
    """
    图谱实体信息工具 - 查询图谱中的实体及其关系

    用于了解图谱中有哪些实体，以及实体之间的关系。
    """

    def __init__(self, graph_builder):
        self._graph_builder = graph_builder

    @property
    def name(self) -> str:
        return "graph_entity_info"

    @property
    def description(self) -> str:
        return (
            "查询知识图谱中实体的信息和关系。"
            "可以查看某个实体在图谱中的关联实体、关系类型。"
            "当需要了解实体之间的结构化关系时使用此工具。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "实体名称（可选，不填则列出所有实体）",
                },
            },
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        entity_name = arguments.get("entity_name", "")

        if not self._graph_builder or not self._graph_builder._entity_map:
            return "知识图谱为空，请先构建知识图谱（build_knowledge_graph）。"

        if entity_name:
            # 查询特定实体的信息
            entity = self._graph_builder._entity_map.get(entity_name)
            if not entity:
                # 模糊匹配
                candidates = [
                    title for title in self._graph_builder._entity_map.keys()
                    if entity_name.lower() in title.lower()
                ]
                if candidates:
                    return f"未找到实体 '{entity_name}'，但找到相似的实体：{', '.join(candidates[:10])}"
                return f"图谱中未找到实体 '{entity_name}'"

            # 获取关系
            relations = []
            if self._graph_builder._graph.has_node(entity.entity_id):
                for neighbor in self._graph_builder._graph.neighbors(entity.entity_id):
                    neighbor_entity = self._graph_builder._get_entity_by_id(neighbor)
                    edge_data = self._graph_builder._graph.get_edge_data(
                        entity.entity_id, neighbor
                    )
                    rel_type = edge_data.get("relation_type", "相关") if edge_data else "相关"
                    neighbor_title = neighbor_entity.title if neighbor_entity else neighbor
                    relations.append(f"  → {neighbor_title} (关系: {rel_type})")

                for predecessor in self._graph_builder._graph.predecessors(entity.entity_id):
                    if predecessor not in self._graph_builder._graph.neighbors(entity.entity_id):
                        pred_entity = self._graph_builder._get_entity_by_id(predecessor)
                        edge_data = self._graph_builder._graph.get_edge_data(
                            predecessor, entity.entity_id
                        )
                        rel_type = edge_data.get("relation_type", "相关") if edge_data else "相关"
                        pred_title = pred_entity.title if pred_entity else predecessor
                        relations.append(f"  ← {pred_title} (关系: {rel_type})")

            info = (
                f"实体: {entity.title}\n"
                f"类型: {entity.entity_type}\n"
                f"描述: {entity.description}\n"
                f"出现频率: {entity.frequency}\n"
                f"关联数: {len(relations)}\n"
            )
            if relations:
                info += "关系:\n" + "\n".join(relations[:20])
            return info
        else:
            # 列出所有实体
            entities = list(self._graph_builder._entity_map.values())
            entity_info = [f"图谱共有 {len(entities)} 个实体：\n"]
            for e in sorted(entities, key=lambda x: x.frequency, reverse=True)[:30]:
                entity_info.append(
                    f"  - {e.title} (类型: {e.entity_type}, 频率: {e.frequency})"
                )
            return "\n".join(entity_info)
