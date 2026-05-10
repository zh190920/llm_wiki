"""
知识图谱构建器
================================
借鉴 WeKnora 的 graphBuilder 设计，使用 LLM 从文档中
提取实体和关系，构建知识图谱并计算权重，
支持 Mermaid 格式可视化输出。
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
from loguru import logger

from core.llm_client import LLMClient, parse_llm_json
from config import settings


@dataclass
class Entity:
    """实体"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    entity_type: str = ""
    chunk_ids: list[str] = field(default_factory=list)
    frequency: int = 1
    degree: int = 0


@dataclass
class Relationship:
    """关系"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    target: str = ""
    description: str = ""
    relationship_type: str = ""
    strength: int = 5
    weight: float = 0.0
    chunk_ids: list[str] = field(default_factory=list)


# 实体提取 Prompt
ENTITY_EXTRACTION_PROMPT = """你是一个知识提取系统。从以下文本中提取所有重要的实体。

实体类型包括：人物、组织、产品、地点、技术、事件等。

对每个实体，提供：
- title: 实体名称
- description: 简短描述（一句话）
- entity_type: 实体类型

只提取在文本中实质讨论的实体，忽略仅一次提及的名称。

以 JSON 数组格式输出，例如：
[{"title": "RAG", "description": "检索增强生成技术", "entity_type": "技术"}]

文本内容：
{content}"""

# 关系提取 Prompt
RELATIONSHIP_EXTRACTION_PROMPT = """你是一个关系提取系统。分析以下实体列表和文本，提取实体之间的语义关系。

对每个关系，提供：
- source: 源实体名称
- target: 目标实体名称
- description: 关系描述
- relationship_type: 关系类型（如：包含、依赖、使用、影响、属于）
- strength: 关系强度（1-10）

只提取文本中明确体现的关系。

以 JSON 数组格式输出，例如：
[{{"source": "RAG", "target": "向量数据库", "description": "RAG使用向量数据库进行检索", "relationship_type": "使用", "strength": 8}}]

实体列表：{entities}

文本内容：{content}"""


class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.graph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}  # title → Entity
        self._relationships: list[Relationship] = []
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_GRAPH_EXTRACTIONS)

    async def build_from_chunks(self, chunks: list[dict]) -> dict:
        """
        从文档分块构建知识图谱

        Args:
            chunks: [{"chunk_id": str, "content": str, "metadata": dict}, ...]

        Returns:
            {"entities": int, "relationships": int, "mermaid": str}
        """
        logger.info(f"开始构建知识图谱, 共 {len(chunks)} 个分块")

        # 阶段1: 并发提取实体
        all_entities = await self._extract_entities_concurrent(chunks)

        # 阶段2: 批量提取关系
        all_relationships = await self._extract_relationships_concurrent(chunks, all_entities)

        # 阶段3: 构建图并计算权重
        self._build_graph(all_entities, all_relationships)
        self._calculate_weights()
        self._calculate_degrees()

        # 生成可视化
        mermaid = self._generate_mermaid()

        result = {
            "entities": len(self._entities),
            "relationships": len(self._relationships),
            "mermaid": mermaid,
        }

        logger.info(f"知识图谱构建完成: {result['entities']} 实体, {result['relationships']} 关系")
        return result

    async def _extract_entities_concurrent(self, chunks: list[dict]) -> list[Entity]:
        """并发提取实体"""
        tasks = []
        for chunk in chunks:
            tasks.append(self._extract_entities_from_chunk(chunk))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_entities = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"实体提取失败: {result}")
                continue
            all_entities.extend(result)

        # 合并同名实体
        merged = {}
        for entity in all_entities:
            if entity.title in merged:
                merged[entity.title].frequency += 1
                merged[entity.title].chunk_ids.extend(entity.chunk_ids)
            else:
                merged[entity.title] = entity

        self._entities = merged
        return list(merged.values())

    async def _extract_entities_from_chunk(self, chunk: dict) -> list[Entity]:
        """从单个分块提取实体"""
        async with self._semaphore:
            try:
                content = chunk["content"]
                if not content.strip():
                    return []

                prompt = ENTITY_EXTRACTION_PROMPT.format(content=content)
                response = await self.llm_client.generate_with_template(
                    system_prompt="你是知识提取专家。",
                    user_content=prompt,
                    temperature=0.1,
                )

                entities_data = parse_llm_json(response)
                if not isinstance(entities_data, list):
                    return []

                entities = []
                for item in entities_data:
                    if isinstance(item, dict) and "title" in item:
                        entity = Entity(
                            title=item.get("title", ""),
                            description=item.get("description", ""),
                            entity_type=item.get("entity_type", ""),
                            chunk_ids=[chunk.get("chunk_id", "")],
                        )
                        if entity.title:
                            entities.append(entity)

                return entities

            except Exception as e:
                logger.warning(f"实体提取异常: {e}")
                return []

    async def _extract_relationships_concurrent(
        self,
        chunks: list[dict],
        entities: list[Entity],
    ) -> list[Relationship]:
        """批量提取关系"""
        if len(entities) < 2:
            return []

        batch_size = settings.GRAPH_PMI_WEIGHT and 5 or 5  # 默认5个一批
        all_relationships = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_content = "\n\n".join(c["content"] for c in batch)
            entities_json = json.dumps(
                [{"title": e.title, "description": e.description} for e in entities],
                ensure_ascii=False,
            )

            try:
                prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
                    entities=entities_json,
                    content=batch_content[:settings.WIKI_MAX_CONTENT],
                )
                response = await self.llm_client.generate_with_template(
                    system_prompt="你是关系提取专家。",
                    user_content=prompt,
                    temperature=0.1,
                )

                rels_data = parse_llm_json(response)
                if isinstance(rels_data, list):
                    for item in rels_data:
                        if isinstance(item, dict) and "source" in item and "target" in item:
                            rel = Relationship(
                                source=item.get("source", ""),
                                target=item.get("target", ""),
                                description=item.get("description", ""),
                                relationship_type=item.get("relationship_type", ""),
                                strength=item.get("strength", 5),
                            )
                            if rel.source and rel.target:
                                all_relationships.append(rel)

            except Exception as e:
                logger.warning(f"关系提取异常: {e}")

        self._relationships = all_relationships
        return all_relationships

    def _build_graph(self, entities: list[Entity], relationships: list[Relationship]):
        """构建 NetworkX 图"""
        for entity in entities:
            self.graph.add_node(entity.title, **{
                "id": entity.id,
                "description": entity.description,
                "entity_type": entity.entity_type,
                "frequency": entity.frequency,
                "degree": entity.degree,
            })

        for rel in relationships:
            if rel.source in self._entities and rel.target in self._entities:
                self.graph.add_edge(
                    rel.source, rel.target,
                    id=rel.id,
                    description=rel.description,
                    relationship_type=rel.relationship_type,
                    strength=rel.strength,
                    weight=rel.weight,
                )

    def _calculate_weights(self):
        """计算关系权重 (借鉴 WeKnora 的 PMI + Strength 权重计算)"""
        if not self._relationships:
            return

        total_entity_occ = sum(e.frequency for e in self._entities.values())
        total_rel_occ = sum(len(r.chunk_ids) or 1 for r in self._relationships)

        if total_entity_occ == 0 or total_rel_occ == 0:
            return

        pmi_values = {}
        max_pmi = 0.0
        max_strength = 1.0

        for rel in self._relationships:
            source_freq = self._entities.get(rel.source, Entity()).frequency
            target_freq = self._entities.get(rel.target, Entity()).frequency
            rel_freq = len(rel.chunk_ids) or 1

            if source_freq > 0 and target_freq > 0:
                p_source = source_freq / total_entity_occ
                p_target = target_freq / total_entity_occ
                p_rel = rel_freq / total_rel_occ

                import math
                pmi = max(math.log2(p_rel / (p_source * p_target)), 0)
                pmi_values[rel.id] = pmi
                max_pmi = max(max_pmi, pmi)

            max_strength = max(max_strength, float(rel.strength))

        # 归一化并组合
        for rel in self._relationships:
            pmi = pmi_values.get(rel.id, 0)
            norm_pmi = pmi / max_pmi if max_pmi > 0 else 0
            norm_strength = float(rel.strength) / max_strength

            rel.weight = 1.0 + 9.0 * (
                norm_pmi * settings.GRAPH_PMI_WEIGHT +
                norm_strength * settings.GRAPH_STRENGTH_WEIGHT
            )

    def _calculate_degrees(self):
        """计算实体度数"""
        for entity in self._entities.values():
            entity.degree = self.graph.degree(entity.title) if entity.title in self.graph else 0

    def _generate_mermaid(self) -> str:
        """生成 Mermaid 格式知识图谱可视化"""
        if not self.graph.nodes:
            return ""

        lines = ["graph TD"]
        lines.append("  classDef entity fill:#f9f,stroke:#333,stroke-width:1px;")
        lines.append("  classDef highFreq fill:#bbf,stroke:#333,stroke-width:2px;")
        lines.append("")

        # 添加节点
        node_map = {}
        for i, (title, data) in enumerate(self.graph.nodes(data=True)):
            node_id = f"E{i}"
            node_map[title] = node_id
            freq = data.get("frequency", 1)
            style = ":::highFreq" if freq > 2 else ":::entity"
            safe_title = title.replace('"', "'")
            lines.append(f'  {node_id}["{safe_title}"]{style}')

        lines.append("")

        # 添加边
        for source, target, data in self.graph.edges(data=True):
            src_id = node_map.get(source, "")
            tgt_id = node_map.get(target, "")
            if src_id and tgt_id:
                desc = data.get("description", "")
                if desc:
                    desc = desc.replace('"', "'")[:30]
                    lines.append(f'  {src_id} -->|"{desc}"| {tgt_id}')
                else:
                    lines.append(f"  {src_id} --> {tgt_id}")

        return "\n".join(lines)

    def get_entity_relations(self, entity_title: str, top_k: int = 5) -> list[str]:
        """获取与指定实体最相关的实体列表"""
        if entity_title not in self.graph:
            return []

        related = []
        for neighbor in self.graph.neighbors(entity_title):
            edge_data = self.graph.edges[entity_title, neighbor]
            related.append((neighbor, edge_data.get("weight", 0)))

        related.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in related[:top_k]]

    def get_graph_data(self) -> dict:
        """获取图谱数据（用于 API 返回）"""
        nodes = []
        for title, data in self.graph.nodes(data=True):
            nodes.append({
                "id": data.get("id", ""),
                "title": title,
                "description": data.get("description", ""),
                "entity_type": data.get("entity_type", ""),
                "frequency": data.get("frequency", 1),
                "degree": data.get("degree", 0),
            })

        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "description": data.get("description", ""),
                "relationship_type": data.get("relationship_type", ""),
                "weight": data.get("weight", 0),
            })

        return {"nodes": nodes, "edges": edges}
