"""
Wiki Ingest 管道 - 从原始文档生成 Wiki 知识库
借鉴 WeKnora 的 Map-Reduce Wiki Ingest 设计

流程：
MAP 阶段（每个文档）:
  1. 生成文档摘要页
  2. 提取实体和概念（候选slug）
  3. 对已有页面去重

REDUCE 阶段（每个实体/概念）:
  4. 收集相关文档块
  5. 创建/更新 Wiki 页面

POST 阶段:
  6. 发布所有草稿页面
  7. 重建索引页
  8. 注入跨页面链接
  9. 清理死链接
"""
import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Set

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.prompts import (
    WIKI_DEDUPLICATION_PROMPT,
    WIKI_ENTITY_EXTRACTION_PROMPT,
    WIKI_INDEX_PROMPT,
    WIKI_PAGE_MODIFY_PROMPT,
    WIKI_SUMMARY_PROMPT,
)
from config.settings import AppConfig
from core.vector_store import VectorStore
from models.schemas import Chunk, WikiPage, WikiPageType
from wiki.page_manager import WikiPageManager

logger = logging.getLogger(__name__)


class WikiIngest:
    """
    Wiki 生成管道

    借鉴 WeKnora 的 Map-Reduce 架构：
    - MAP 阶段：每个文档独立提取实体和概念
    - REDUCE 阶段：每个实体/概念创建或更新页面
    - POST 阶段：链接注入和索引构建

    粒度控制（granularity）：
    - focused: 少量核心实体/概念
    - standard: 适度提取
    - exhaustive: 尽可能提取所有实体/概念
    """

    def __init__(self, config: AppConfig, wiki_manager: WikiPageManager):
        self.config = config
        self.wiki_manager = wiki_manager
        self._client = AsyncOpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            timeout=config.llm.timeout,
        )

        # 粒度映射到提取指令
        self._granularity_instructions = {
            "focused": "只提取最核心的、最重要的实体和概念（5-10个）。",
            "standard": "提取所有重要的实体和概念（10-30个）。",
            "exhaustive": "尽可能提取所有提到的实体和概念，包括次要的（不限制数量）。",
        }

    async def ingest_documents(
        self,
        doc_ids: List[str],
        vector_store: VectorStore,
        granularity: str = "standard",
    ) -> Dict[str, int]:
        """
        对指定文档执行 Wiki 生成管道

        Args:
            doc_ids: 文档 ID 列表
            vector_store: 向量存储（用于获取文档块）
            granularity: 提取粒度

        Returns:
            统计信息 {"pages_created": int, "pages_updated": int, "links_injected": int}
        """
        stats = {"pages_created": 0, "pages_updated": 0, "links_injected": 0}

        # 收集所有相关文档块，按文档分组
        doc_chunks: Dict[str, List[Chunk]] = {}
        for doc_id in doc_ids:
            chunks = vector_store.get_chunks_by_doc_id(doc_id)
            if chunks:
                doc_chunks[doc_id] = chunks

        if not doc_chunks:
            logger.warning("没有找到指定文档的内容")
            return stats

        # ========== MAP 阶段 ==========

        # 1. 为每个文档生成摘要页
        summary_tasks = []
        for doc_id, chunks in doc_chunks.items():
            summary_tasks.append(self._generate_summary_page(doc_id, chunks, granularity))

        summary_pages = await asyncio.gather(*summary_tasks, return_exceptions=True)

        for result in summary_pages:
            if isinstance(result, Exception):
                logger.error(f"生成摘要页失败: {result}")
                continue
            if result:
                await self.wiki_manager.save_page(result)
                stats["pages_created"] += 1

        # 2. 从每个文档提取实体和概念（2-pass）
        extraction_tasks = []
        for doc_id, chunks in doc_chunks.items():
            extraction_tasks.append(self._extract_entities(doc_id, chunks, granularity))

        extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # 收集所有候选实体
        all_candidates: List[dict] = []
        for result in extraction_results:
            if isinstance(result, Exception):
                logger.error(f"提取实体失败: {result}")
                continue
            all_candidates.extend(result)

        # 3. 去重
        existing_pages = await self.wiki_manager.list_pages()
        deduplicated = await self._deduplicate_candidates(all_candidates, existing_pages)
        logger.info(f"实体去重: {len(all_candidates)} → {len(deduplicated)} 个候选")

        # ========== REDUCE 阶段 ==========

        # 4. 为每个实体/概念创建 Wiki 页面
        page_tasks = []
        for candidate in deduplicated:
            # 收集与此实体相关的文档块
            related_chunks = self._find_related_chunks(
                candidate, doc_chunks
            )
            if related_chunks:
                page_tasks.append(
                    self._create_entity_page(candidate, related_chunks)
                )

        # 限制并发数
        semaphore = asyncio.Semaphore(self.config.wiki.max_concurrent_extractions)

        async def _limited_create(task):
            async with semaphore:
                return await task

        page_results = await asyncio.gather(
            *[_limited_create(t) for t in page_tasks],
            return_exceptions=True,
        )

        for result in page_results:
            if isinstance(result, Exception):
                logger.error(f"创建页面失败: {result}")
                continue
            if result:
                existing = await self.wiki_manager.get_page(result.slug)
                if existing:
                    await self.wiki_manager.save_page(result)
                    stats["pages_updated"] += 1
                else:
                    await self.wiki_manager.save_page(result)
                    stats["pages_created"] += 1

        # ========== POST 阶段 ==========

        # 6. 发布草稿页面
        for page in await self.wiki_manager.list_pages():
            if page.status == "draft":
                page.status = "published"
                await self.wiki_manager.save_page(page)

        # 7. 重建索引页
        index_page = await self._rebuild_index()
        if index_page:
            await self.wiki_manager.save_page(index_page)

        # 8. 注入跨页面链接
        await self.wiki_manager.inject_cross_links()
        stats["links_injected"] = self.wiki_manager.total_pages

        logger.info(f"Wiki 生成完成: {stats}")
        return stats

    async def _generate_summary_page(
        self, doc_id: str, chunks: List[Chunk], granularity: str
    ) -> Optional[WikiPage]:
        """MAP: 为文档生成摘要页"""
        # 合并文档内容
        content = "\n\n".join([c.content for c in chunks[:10]])  # 限制长度
        if len(content) > 4000:
            content = content[:4000]

        prompt = WIKI_SUMMARY_PROMPT.format(content=content, title=doc_id)

        try:
            response = await self._client.chat.completions.create(
                model=self.config.llm.chat_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识整理专家。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            page_content = response.choices[0].message.content.strip()

            slug = self._title_to_slug(f"summary-{doc_id}")
            return WikiPage(
                slug=slug,
                title=f"文档摘要: {doc_id}",
                page_type=WikiPageType.SUMMARY,
                content=page_content,
                source_doc_ids=[doc_id],
                source_chunk_ids=[c.chunk_id for c in chunks[:10]],
                status="draft",
            )

        except Exception as e:
            logger.error(f"生成摘要页失败 (doc={doc_id}): {e}")
            return None

    async def _extract_entities(
        self, doc_id: str, chunks: List[Chunk], granularity: str
    ) -> List[dict]:
        """
        MAP: 从文档中提取实体和概念

        借鉴 WeKnora 的 2-pass 提取设计：
        - Pass 0: 提取候选实体（轻量级）
        - Pass 1-N: 为每批块分配引用（哪些块提到哪些实体）
        """
        granularity_instruction = self._granularity_instructions.get(
            granularity, self._granularity_instructions["standard"]
        )

        # 分批处理（避免上下文过长）
        batch_size = self.config.wiki.chunk_batch_size
        all_entities: List[dict] = []
        seen_titles: Set[str] = set()

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_content = "\n\n---\n\n".join([c.content for c in batch])

            prompt = (
                f"{WIKI_ENTITY_EXTRACTION_PROMPT.format(content=batch_content)}\n\n"
                f"提取要求：{granularity_instruction}"
            )

            try:
                response = await self._client.chat.completions.create(
                    model=self.config.llm.chat_model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的实体和概念提取专家。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=2000,
                )

                result_text = response.choices[0].message.content.strip()
                entities = self._parse_json_list(result_text)

                for entity in entities:
                    title = entity.get("title", "").strip()
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        entity["source_doc_id"] = doc_id
                        entity["source_chunk_ids"] = [c.chunk_id for c in batch]
                        all_entities.append(entity)

            except Exception as e:
                logger.error(f"实体提取失败 (doc={doc_id}, batch={i}): {e}")
                continue

        return all_entities

    async def _deduplicate_candidates(
        self,
        candidates: List[dict],
        existing_pages: List[WikiPage],
    ) -> List[dict]:
        """对候选实体与已有页面进行去重"""
        if not existing_pages or not candidates:
            return candidates

        # 简单去重：标题完全匹配
        existing_titles = {p.title.lower() for p in existing_pages}
        existing_slugs = {p.slug for p in existing_pages}

        unique_candidates = []
        for candidate in candidates:
            title = candidate.get("title", "")
            slug = candidate.get("slug", self._title_to_slug(title))

            if title.lower() not in existing_titles and slug not in existing_slugs:
                unique_candidates.append(candidate)

        # 如果候选数较多，使用 LLM 进行语义去重
        if len(unique_candidates) > 5 and existing_pages:
            try:
                new_entities_text = json.dumps(
                    [{"title": e.get("title"), "slug": e.get("slug")}
                     for e in unique_candidates],
                    ensure_ascii=False,
                )
                existing_text = "\n".join(
                    [f"- {p.title} (slug: {p.slug})" for p in existing_pages[:20]]
                )

                prompt = WIKI_DEDUPLICATION_PROMPT.format(
                    new_entities=new_entities_text,
                    existing_pages=existing_text,
                )

                response = await self._client.chat.completions.create(
                    model=self.config.llm.chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1000,
                )

                dedup_result = self._parse_json_list(response.choices[0].message.content)

                # 根据去重结果过滤
                merge_slugs = set()
                for item in dedup_result:
                    if item.get("action") == "merge":
                        merge_slug = item.get("merge_to", "")
                        if merge_slug:
                            merge_slugs.add(merge_slug)

                # 保留 action=create 的候选
                filtered = []
                for candidate, dedup_item in zip(unique_candidates, dedup_result):
                    if dedup_item.get("action") != "merge":
                        filtered.append(candidate)

                return filtered if filtered else unique_candidates

            except Exception as e:
                logger.warning(f"LLM 去重失败，使用简单去重: {e}")

        return unique_candidates

    def _find_related_chunks(
        self, entity: dict, doc_chunks: Dict[str, List[Chunk]]
    ) -> List[Chunk]:
        """查找与实体相关的文档块"""
        title = entity.get("title", "")
        description = entity.get("description", "")
        source_chunk_ids = entity.get("source_chunk_ids", [])

        related = []
        seen_ids = set()

        # 优先使用源文档块
        for doc_chunks_list in doc_chunks.values():
            for chunk in doc_chunks_list:
                if chunk.chunk_id in source_chunk_ids and chunk.chunk_id not in seen_ids:
                    related.append(chunk)
                    seen_ids.add(chunk.chunk_id)

        # 补充：标题出现在内容中的块
        if title:
            for doc_chunks_list in doc_chunks.values():
                for chunk in doc_chunks_list:
                    if chunk.chunk_id not in seen_ids and title.lower() in chunk.content.lower():
                        related.append(chunk)
                        seen_ids.add(chunk.chunk_id)
                        if len(related) >= 10:
                            break

        return related[:10]  # 限制每个实体最多10个块

    async def _create_entity_page(
        self, entity: dict, chunks: List[Chunk]
    ) -> Optional[WikiPage]:
        """REDUCE: 为实体创建 Wiki 页面"""
        title = entity.get("title", "")
        slug = entity.get("slug", self._title_to_slug(title))
        description = entity.get("description", "")
        entity_type = entity.get("type", "concept")

        chunks_text = "\n\n---\n\n".join([f"[chunk: {c.chunk_id}]\n{c.content[:500]}" for c in chunks])

        prompt = WIKI_PAGE_MODIFY_PROMPT.format(
            entity_title=title,
            entity_description=description,
            entity_type=entity_type,
            chunks=chunks_text,
        )

        try:
            response = await self._client.chat.completions.create(
                model=self.config.llm.chat_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识库编辑。请根据提供的文档内容创建准确、详细的 Wiki 页面。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=3000,
            )

            page_content = response.choices[0].message.content.strip()

            # 提取出链
            out_links = re.findall(r'\[\[([^\]|]+)', page_content)

            page_type = WikiPageType.ENTITY if entity_type == "entity" else WikiPageType.CONCEPT

            return WikiPage(
                slug=slug,
                title=title,
                page_type=page_type,
                content=page_content,
                source_doc_ids=list(set(entity.get("source_doc_id", "") for e in [entity] if entity.get("source_doc_id"))),
                source_chunk_ids=[c.chunk_id for c in chunks],
                out_links=out_links,
                status="draft",
            )

        except Exception as e:
            logger.error(f"创建实体页面失败 (entity={title}): {e}")
            return None

    async def _rebuild_index(self) -> Optional[WikiPage]:
        """重建 Wiki 索引页"""
        pages = await self.wiki_manager.list_pages()

        # 按类型分组
        pages_text = ""
        for page_type in WikiPageType:
            type_pages = [p for p in pages if p.page_type == page_type]
            if type_pages:
                pages_text += f"\n### {page_type.value.upper()}\n"
                for p in type_pages:
                    pages_text += f"- [[{p.slug}|{p.title}]]\n"

        prompt = WIKI_INDEX_PROMPT.format(pages=pages_text)

        try:
            response = await self._client.chat.completions.create(
                model=self.config.llm.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()

            return WikiPage(
                slug="index",
                title="知识库索引",
                page_type=WikiPageType.INDEX,
                content=content,
                status="published",
            )

        except Exception as e:
            logger.error(f"重建索引页失败: {e}")
            return None

    @staticmethod
    def _title_to_slug(title: str) -> str:
        """将标题转换为 URL-friendly slug"""
        # 转小写
        slug = title.lower()
        # 替换空格和特殊字符为连字符
        slug = re.sub(r'[\s_]+', '-', slug)
        # 只保留字母、数字、连字符和中文
        slug = re.sub(r'[^\w\-\u4e00-\u9fff]', '', slug)
        # 截断
        slug = slug[:64]
        return slug or "untitled"

    @staticmethod
    def _parse_json_list(text: str) -> List[dict]:
        """解析 LLM 输出中的 JSON 列表"""
        # 尝试提取 JSON 代码块
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        try:
            result = json.loads(text.strip())
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return [result]
        except json.JSONDecodeError:
            pass

        # 尝试更宽松的解析
        try:
            # 找到 [ 和 ] 的范围
            start = text.find('[')
            end = text.rfind(']')
            if start >= 0 and end > start:
                result = json.loads(text[start:end + 1])
                if isinstance(result, list):
                    return result
        except json.JSONDecodeError:
            pass

        logger.warning(f"无法解析 JSON 列表: {text[:200]}")
        return []
