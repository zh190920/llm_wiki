"""
知识图谱构建器 - 从文档中提取实体和关系，构建可视化知识图谱
借鉴 WeKnora 的 Graph Builder 设计
"""
import asyncio
import json
import logging
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from openai import AsyncOpenAI

from agent.prompts import GRAPH_ENTITY_EXTRACTION_PROMPT
from config.settings import AppConfig
from core.vector_store import VectorStore
from models.schemas import Chunk, Entity, KnowledgeGraph, Relationship

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    知识图谱构建器

    借鉴 WeKnora 的 Graph Builder 设计：
    1. 并发实体提取：从每个文档块中提取实体
    2. 并发关系提取：识别实体间的关系
    3. 权重计算：PMI × 0.6 + Strength × 0.4
    4. 图结构构建：使用 NetworkX 构建有向图
    5. Mermaid 可视化：生成 Mermaid 语法的关系图

    图查询：
    - 直接关系：获取与某实体直接关联的实体和块
    - 间接关系：2度关联（朋友的朋友）
    - 用于图增强检索
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            timeout=config.llm.timeout,
        )
        self._graph = nx.DiGraph()
        self._entity_map: Dict[str, Entity] = {}  # title -> Entity
        self._chunk_entities: Dict[str, List[str]] = defaultdict(list)  # chunk_id -> [entity_titles]

    async def build_graph(
        self,
        chunks: List[Chunk],
        max_concurrent: int = 4,
    ) -> KnowledgeGraph:
        """
        从文档块构建知识图谱

        Args:
            chunks: 文档块列表
            max_concurrent: 最大并发提取数

        Returns:
            知识图谱对象
        """
        if not chunks:
            return KnowledgeGraph()

        logger.info(f"开始构建知识图谱: {len(chunks)} 个块")

        # Step 1: 并发实体提取
        semaphore = asyncio.Semaphore(max_concurrent)
        entity_tasks = [self._extract_entities_from_chunk(semaphore, chunk) for chunk in chunks]
        entity_results = await asyncio.gather(*entity_tasks, return_exceptions=True)

        # 合并实体（去重）
        for result in entity_results:
            if isinstance(result, Exception):
                continue
            for entity_data in result:
                title = entity_data.get("title", "").strip()
                if not title:
                    continue

                if title in self._entity_map:
                    # 更新频率和来源
                    self._entity_map[title].frequency += 1
                    self._entity_map[title].source_chunk_ids.extend(
                        entity_data.get("source_chunk_ids", [])
                    )
                else:
                    self._entity_map[title] = Entity(
                        title=title,
                        description=entity_data.get("description", ""),
                        entity_type=entity_data.get("type", "generic"),
                        source_chunk_ids=entity_data.get("source_chunk_ids", []),
                    )

        logger.info(f"实体提取完成: {len(self._entity_map)} 个唯一实体")

        # Step 2: 并发关系提取
        entity_titles = list(self._entity_map.keys())
        relationship_tasks = []

        # 每批5个块
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_titles_in_batch = set()
            for chunk in batch:
                batch_titles_in_batch.update(self._chunk_entities.get(chunk.chunk_id, []))

            if len(batch_titles_in_batch) >= 2:
                relationship_tasks.append(
                    self._extract_relationships(semaphore, batch, entity_titles)
                )

        rel_results = await asyncio.gather(*relationship_tasks, return_exceptions=True)

        # 合并关系
        all_relationships: List[Relationship] = []
        for result in rel_results:
            if isinstance(result, Exception):
                continue
            all_relationships.extend(result)

        logger.info(f"关系提取完成: {len(all_relationships)} 条关系")

        # Step 3: 构建图结构
        self._build_networkx_graph(all_relationships)

        # Step 4: 计算权重
        self._calculate_weights(all_relationships)

        # 构建 KnowledgeGraph 对象
        kg = KnowledgeGraph(
            entities=self._entity_map,
            relationships=all_relationships,
        )

        return kg

    async def _extract_entities_from_chunk(
        self, semaphore: asyncio.Semaphore, chunk: Chunk
    ) -> List[dict]:
        """从单个块中提取实体"""
        async with semaphore:
            prompt = GRAPH_ENTITY_EXTRACTION_PROMPT.format(text=chunk.content[:2000])

            try:
                response = await self._client.chat.completions.create(
                    model=self.config.llm.chat_model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的实体关系提取专家。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                )

                result_text = response.choices[0].message.content.strip()
                parsed = self._parse_json(result_text)

                entities = []
                if isinstance(parsed, dict):
                    entity_list = parsed.get("entities", [])
                    for e in entity_list:
                        e["source_chunk_ids"] = [chunk.chunk_id]
                        entities.append(e)
                        # 记录块-实体映射
                        title = e.get("title", "")
                        if title:
                            self._chunk_entities[chunk.chunk_id].append(title)

                return entities

            except Exception as e:
                logger.warning(f"实体提取失败 (chunk={chunk.chunk_id}): {e}")
                return []

    async def _extract_relationships(
        self, semaphore: asyncio.Semaphore, chunks: List[Chunk], entity_titles: List[str]
    ) -> List[Relationship]:
        """从一批块中提取关系"""
        async with semaphore:
            content = "\n\n---\n\n".join([c.content for c in chunks])
            if len(content) > 3000:
                content = content[:3000]

            entities_context = ", ".join(entity_titles[:30])

            prompt = f"""请从以下文本中识别实体之间的关系。

已知实体（部分）：{entities_context}

文本内容：
{content}

请按以下 JSON 格式输出：
```json
{{
  "relationships": [
    {{
      "source": "源实体名称",
      "target": "目标实体名称",
      "relation": "关系类型（如：属于、包含、依赖、使用、影响等）",
      "description": "关系描述"
    }}
  ]
}}
```"""

            try:
                response = await self._client.chat.completions.create(
                    model=self.config.llm.chat_model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的关系提取专家。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                )

                result_text = response.choices[0].message.content.strip()
                parsed = self._parse_json(result_text)

                relationships = []
                if isinstance(parsed, dict):
                    rel_list = parsed.get("relationships", [])
                    for r in rel_list:
                        source_title = r.get("source", "")
                        target_title = r.get("target", "")

                        if source_title and target_title:
                            source_entity = self._entity_map.get(source_title)
                            target_entity = self._entity_map.get(target_title)

                            if source_entity and target_entity:
                                relationships.append(Relationship(
                                    source_entity_id=source_entity.entity_id,
                                    target_entity_id=target_entity.entity_id,
                                    relation_type=r.get("relation", "related_to"),
                                    description=r.get("description", ""),
                                    source_chunk_ids=[c.chunk_id for c in chunks],
                                ))

                return relationships

            except Exception as e:
                logger.warning(f"关系提取失败: {e}")
                return []

    def _build_networkx_graph(self, relationships: List[Relationship]):
        """构建 NetworkX 图结构"""
        # 添加实体节点
        for title, entity in self._entity_map.items():
            self._graph.add_node(
                entity.entity_id,
                title=entity.title,
                type=entity.entity_type,
                description=entity.description,
            )

        # 添加关系边
        for rel in relationships:
            self._graph.add_edge(
                rel.source_entity_id,
                rel.target_entity_id,
                relation_type=rel.relation_type,
                description=rel.description,
                weight=rel.weight,
            )

    def _calculate_weights(self, relationships: List[Relationship]):
        """
        计算关系权重

        借鉴 WeKnora 的权重计算：PMI × 0.6 + Strength × 0.4
        """
        # 计算实体频率
        entity_freq = defaultdict(int)
        for entity in self._entity_map.values():
            entity_freq[entity.entity_id] = entity.frequency

        total_chunks = max(1, len(set(
            cid for e in self._entity_map.values() for cid in e.source_chunk_ids
        )))

        for rel in relationships:
            # PMI (Point Mutual Information) - 简化计算
            source_freq = entity_freq.get(rel.source_entity_id, 1)
            target_freq = entity_freq.get(rel.target_entity_id, 1)
            co_freq = len(set(rel.source_chunk_ids))

            p_source = source_freq / total_chunks
            p_target = target_freq / total_chunks
            p_co = co_freq / total_chunks

            if p_source > 0 and p_target > 0 and p_co > 0:
                pmi = min(max(p_co / (p_source * p_target), 0), 10) / 10  # 归一化到 [0, 1]
            else:
                pmi = 0.1

            # Strength - 基于共现频率
            strength = min(co_freq / max(source_freq, target_freq, 1), 1.0)

            # 综合权重
            rel.weight = 0.6 * pmi + 0.4 * strength

            # 归一化到 1-10
            rel.weight = max(1.0, min(10.0, rel.weight * 10))

    def get_related_chunks(self, entity_title: str, depth: int = 1) -> List[str]:
        """
        获取与实体相关的文档块 ID

        Args:
            entity_title: 实体标题
            depth: 关联深度（1=直接关系，2=间接关系）
        """
        entity = self._entity_map.get(entity_title)
        if not entity:
            return []

        related_chunk_ids: Set[str] = set(entity.source_chunk_ids)

        if depth >= 1:
            # 直接关联
            for neighbor in self._graph.neighbors(entity.entity_id):
                neighbor_entity = self._get_entity_by_id(neighbor)
                if neighbor_entity:
                    related_chunk_ids.update(neighbor_entity.source_chunk_ids)

        if depth >= 2:
            # 2度关联
            for neighbor in self._graph.neighbors(entity.entity_id):
                for neighbor2 in self._graph.neighbors(neighbor):
                    neighbor_entity = self._get_entity_by_id(neighbor2)
                    if neighbor_entity:
                        related_chunk_ids.update(neighbor_entity.source_chunk_ids)

        return list(related_chunk_ids)

    def _get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """根据 ID 获取实体"""
        for entity in self._entity_map.values():
            if entity.entity_id == entity_id:
                return entity
        return None

    def to_mermaid(self, max_nodes: int = 30) -> str:
        """
        生成 Mermaid 格式的知识图谱可视化

        Returns:
            Mermaid 语法字符串
        """
        if not self._graph.nodes:
            return "graph LR\n  empty[空图谱]"

        # 选择度数最高的节点
        degrees = dict(self._graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:max_nodes]

        lines = ["graph LR"]

        # 节点
        for node_id in top_nodes:
            node_data = self._graph.nodes[node_id]
            title = node_data.get("title", "unknown")
            node_type = node_data.get("type", "generic")
            # Mermaid 节点 ID 不能包含特殊字符
            safe_id = re.sub(r'[^\w]', '_', node_id)
            lines.append(f"  {safe_id}[\"{title}\"]")

        # 边
        for u, v, data in self._graph.edges(data=True):
            if u in top_nodes and v in top_nodes:
                safe_u = re.sub(r'[^\w]', '_', u)
                safe_v = re.sub(r'[^\w]', '_', v)
                relation = data.get("relation_type", "related")
                lines.append(f"  {safe_u} -->|\"{relation}\"| {safe_v}")

        return "\n".join(lines)

    def to_json(self) -> str:
        """导出为 JSON"""
        kg = KnowledgeGraph(
            entities=self._entity_map,
            relationships=list(self._get_all_relationships()),
        )
        return kg.model_dump_json(indent=2)

    def _get_all_relationships(self) -> List[Relationship]:
        """从图中获取所有关系"""
        relationships = []
        for u, v, data in self._graph.edges(data=True):
            relationships.append(Relationship(
                source_entity_id=u,
                target_entity_id=v,
                relation_type=data.get("relation_type", ""),
                description=data.get("description", ""),
                weight=data.get("weight", 1.0),
            ))
        return relationships

    @staticmethod
    def _parse_json(text: str) -> dict:
        """解析 JSON 文本"""
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return {}

    def save(self, directory: str) -> str:
        """
        保存图谱数据到磁盘

        保存三个文件：
        1. graph_entities.json  — 实体列表（含 entity_id, title, description, type, frequency, source_chunk_ids）
        2. graph_relationships.json — 关系列表（含 source_entity_id, target_entity_id, relation_type, weight 等）
        3. graph_chunk_entities.json — chunk_id → [entity_titles] 映射（用于恢复 _chunk_entities）

        Args:
            directory: 保存目录

        Returns:
            保存路径
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存实体
        entities_data = {}
        for title, entity in self._entity_map.items():
            entities_data[title] = entity.model_dump()

        entities_path = save_dir / "graph_entities.json"
        with open(entities_path, "w", encoding="utf-8") as f:
            json.dump(entities_data, f, ensure_ascii=False, indent=2)

        # 2. 保存关系（从 NetworkX 图中提取，确保权重等已计算）
        relationships_data = []
        for u, v, data in self._graph.edges(data=True):
            relationships_data.append({
                "source_entity_id": u,
                "target_entity_id": v,
                "relation_type": data.get("relation_type", ""),
                "description": data.get("description", ""),
                "weight": data.get("weight", 1.0),
            })

        rels_path = save_dir / "graph_relationships.json"
        with open(rels_path, "w", encoding="utf-8") as f:
            json.dump(relationships_data, f, ensure_ascii=False, indent=2)

        # 3. 保存 chunk_entities 映射
        chunk_entities_data = dict(self._chunk_entities)
        chunk_entities_path = save_dir / "graph_chunk_entities.json"
        with open(chunk_entities_path, "w", encoding="utf-8") as f:
            json.dump(chunk_entities_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"知识图谱已保存到 {directory}: "
            f"{len(self._entity_map)} 个实体, {len(relationships_data)} 条关系"
        )
        return str(save_dir)

    def load(self, directory: str) -> bool:
        """
        从磁盘加载图谱数据并恢复内部状态

        恢复内容：
        1. _entity_map: title → Entity 映射
        2. _chunk_entities: chunk_id → [entity_titles] 映射
        3. _graph: NetworkX DiGraph（从实体和关系重建）

        恢复后可立即用于：
        - get_related_chunks() 图谱遍历检索
        - search_graph() 图谱增强检索
        - 三源 RRF 融合
        - Mermaid 可视化

        Args:
            directory: 图谱数据所在目录

        Returns:
            是否成功加载
        """
        load_dir = Path(directory)

        entities_path = load_dir / "graph_entities.json"
        rels_path = load_dir / "graph_relationships.json"
        chunk_entities_path = load_dir / "graph_chunk_entities.json"

        if not entities_path.exists():
            logger.info(f"图谱数据不存在，跳过恢复: {entities_path}")
            return False

        try:
            # 1. 恢复实体
            with open(entities_path, "r", encoding="utf-8") as f:
                entities_data = json.load(f)

            self._entity_map = {}
            for title, entity_dict in entities_data.items():
                self._entity_map[title] = Entity(**entity_dict)

            # 2. 恢复关系
            relationships: List[Relationship] = []
            if rels_path.exists():
                with open(rels_path, "r", encoding="utf-8") as f:
                    relationships_data = json.load(f)
                for rel_dict in relationships_data:
                    relationships.append(Relationship(**rel_dict))

            # 3. 恢复 chunk_entities 映射
            if chunk_entities_path.exists():
                with open(chunk_entities_path, "r", encoding="utf-8") as f:
                    self._chunk_entities = defaultdict(list, json.load(f))
            else:
                # 从实体数据反向构建 chunk_entities
                self._chunk_entities = defaultdict(list)
                for title, entity in self._entity_map.items():
                    for chunk_id in entity.source_chunk_ids:
                        self._chunk_entities[chunk_id].append(title)

            # 4. 重建 NetworkX 图结构
            self._graph = nx.DiGraph()
            self._build_networkx_graph(relationships)

            logger.info(
                f"知识图谱已从 {directory} 恢复: "
                f"{len(self._entity_map)} 个实体, "
                f"{self._graph.number_of_edges()} 条关系, "
                f"{len(self._chunk_entities)} 个chunk映射"
            )
            return True

        except Exception as e:
            logger.warning(f"加载知识图谱失败（不影响正常使用）: {e}")
            return False

    @property
    def has_data(self) -> bool:
        """图谱是否包含数据"""
        return len(self._entity_map) > 0 and self._graph.number_of_nodes() > 0