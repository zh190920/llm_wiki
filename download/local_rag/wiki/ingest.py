"""
Wiki Ingest Pipeline — 文档自动生成 Wiki 知识库
================================
借鉴 WeKnora 的 wikiIngestService 设计，
实现 Map-Reduce 流水线:
  Map:   从文档中提取实体/概念/摘要
  Reduce: 合并/去重/更新 Wiki 页面

核心特点:
1. LLM 驱动的实体和概念提取
2. 基于分块引用的两阶段提取（候选提取 + 分块引用）
3. 实体去重和合并
4. Wiki 页面的增量更新
5. 交叉链接自动注入
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Optional

from loguru import logger

from core.llm_client import LLMClient, parse_llm_json, repair_json
from wiki.page_manager import WikiPage, WikiPageManager
from config import settings


# ── Prompt 模板 (借鉴 WeKnora 的 prompts_wiki.go) ──

WIKI_SUMMARY_PROMPT = """你是一个 Wiki 编辑。根据以下文档内容，创建一个结构化的 Wiki 摘要页面。

<document>
<title>{title}</title>
<file_name>{file_name}</file_name>
<content>
{content}
</content>
</document>

<available_wiki_pages>
{available_slugs}
</available_wiki_pages>

<instructions>
1. 输出的第一行必须是: SUMMARY: {一句话描述文档内容，15-40字}
2. 在 SUMMARY 行之后，用 Markdown 格式撰写文档的综合摘要
3. 包含关键事实、论点和结论
4. 使用正确的标题层级 (## 用于章节，### 用于子节)
5. **Wiki 链接规则**: 当你提到 available_wiki_pages 列表中的名称时，必须写成 [[slug|显示名称]] 的格式
6. 最后包含一个 "## 要点总结" 章节
7. 用{language}撰写
8. 保持摘要简洁但全面（500-1500字）
</instructions>

先输出 SUMMARY 行，然后是 Markdown 内容。"""

WIKI_KNOWLEDGE_EXTRACT_PROMPT = """你是一个知识提取系统。分析以下文档并提取所有重要实体和关键概念。

<document>
<title>{title}</title>
<content>
{content}
</content>
</document>

<previous_slugs>
{previous_slugs}
</previous_slugs>

<instructions>
返回一个 JSON 对象，包含两个数组："entities" 和 "concepts"。
**重要：所有名称、描述和详情用{language}撰写。**

### 实体 (人物、组织、产品、地点、技术、事件等)
每个实体应包含:
- "name": 实体名称
- "slug": URL 友好的标识符，格式 "entity/<小写连字符名>"
- "aliases": 别名数组（同义名称）
- "description": 一句话索引描述（15-40字）
- "details": 2-5句关键事实摘要

### 概念 (主题、方法论、理论等)
每个概念应包含:
- "name": 概念名称
- "slug": URL 友好的标识符，格式 "concept/<小写连字符名>"
- "aliases": 别名数组
- "description": 一句话定义（15-40字）
- "details": 2-5句解释

### 去重规则
- 具体命名事物放入 "entities"
- 抽象想法/方法论放入 "concepts"
- 两个数组之间不重复

只输出有效的 JSON。"""

WIKI_PAGE_MODIFY_PROMPT = """你是一个 Wiki 编辑，负责更新现有 Wiki 页面。

<page_metadata>
  <slug>{page_slug}</slug>
  <title>{page_title}</title>
  <type>{page_type}</type>
</page_metadata>

此 Wiki 页面专门关于 **{page_title}**。

<existing_page_content>
{existing_content}
</existing_page_content>

<new_information>
{new_content}
</new_information>

<valid_wiki_links>
{available_slugs}
</valid_wiki_links>

<instructions>
1. 输出的第一行必须是: SUMMARY: {一句话描述此页面更新后的内容，15-40字}
2. 将新信息合并到页面中。贴近原文措辞，不要过度改写。
3. 如果新信息与旧内容矛盾，优先采用新信息。
4. 保留仍有效的现有信息。
5. 只保留 <valid_wiki_links> 中存在的 [[slug|name]] 链接。
6. 用{language}撰写
</instructions>

先输出 SUMMARY 行，然后是更新后的 Markdown 内容。"""

WIKI_DEDUPLICATION_PROMPT = """你是一个严格去重系统。给定一组新提取的项目和现有 Wiki 页面列表，判断哪些新项目与现有页面指的是同一个事物。

<new_items>
{new_items}
</new_items>

<existing_pages>
{existing_pages}
</existing_pages>

<instructions>
### 合并标准 — 必须全部满足:
1. 新项目和现有页面指的是**同一个真实事物**
2. 匹配是名称变体：缩写 ↔ 全称、翻译、或轻微拼写差异
3. 类型兼容：实体与实体合并，概念与概念合并

### 关键原则: **相关 ≠ 相同**。有疑问时不要合并。

返回 JSON 对象，键是新项目的 slug，值是应合并到的现有页面 slug。
如果没有匹配，返回: {{"merges": {{}}}}

只输出有效的 JSON。"""


class WikiIngestPipeline:
    """Wiki 知识库自动生成流水线"""

    def __init__(
        self,
        llm_client: LLMClient,
        page_manager: WikiPageManager,
    ):
        self.llm_client = llm_client
        self.page_manager = page_manager

    async def ingest_document(
        self,
        kb_id: str,
        knowledge_id: str,
        title: str,
        file_name: str,
        content: str,
        language: str = None,
    ) -> list[str]:
        """
        入库单个文档，生成 Wiki 页面

        流程 (借鉴 WeKnora 的 Map-Reduce Pipeline):
        1. Map: 生成文档摘要页 + 提取实体/概念
        2. Dedup: 与已有页面去重
        3. Reduce: 创建/更新 Wiki 页面
        4. Link: 注入交叉链接

        Args:
            kb_id: 知识库 ID
            knowledge_id: 文档 ID
            title: 文档标题
            file_name: 文件名
            content: 文档内容
            language: 语言

        Returns:
            受影响的页面 slug 列表
        """
        language = language or settings.WIKI_LANGUAGE
        content = content[:settings.WIKI_MAX_CONTENT]

        logger.info(f"[Wiki] 开始入库: {title} (KB: {kb_id})")

        affected_slugs = []

        # ── Phase 1: 生成摘要页 ──
        summary_slug = f"summary/{self._slugify(title)}"
        available_slugs = self._get_available_slugs(kb_id)

        summary_content = await self._generate_summary(
            title=title,
            file_name=file_name,
            content=content,
            available_slugs=available_slugs,
            language=language,
        )

        summary, summary_body = self._split_summary_line(summary_content)

        summary_page = WikiPage(
            knowledge_base_id=kb_id,
            slug=summary_slug,
            title=f"摘要: {title}",
            page_type="summary",
            content=summary_body,
            summary=summary or title,
            source_refs=[knowledge_id],
        )
        self.page_manager.create_page(summary_page)
        affected_slugs.append(summary_slug)

        # ── Phase 2: 提取实体和概念 ──
        extraction = await self._extract_knowledge(
            title=title,
            content=content,
            previous_slugs=available_slugs,
            language=language,
        )

        entities = extraction.get("entities", [])
        concepts = extraction.get("concepts", [])

        logger.info(f"[Wiki] 提取结果: {len(entities)} 实体, {len(concepts)} 概念")

        # ── Phase 3: 去重 ──
        entities, concepts = await self._deduplicate(
            kb_id=kb_id,
            entities=entities,
            concepts=concepts,
        )

        # ── Phase 4: 创建/更新页面 ──
        for item in entities + concepts:
            item_type = "entity" if item in entities else "concept"
            slug = item.get("slug", f"{item_type}/{self._slugify(item.get('name', ''))}")

            existing = self.page_manager.get_page(kb_id, slug)

            if existing:
                # 更新已有页面
                updated_content = await self._modify_page(
                    kb_id=kb_id,
                    page_slug=slug,
                    page_title=existing.title,
                    page_type=existing.page_type,
                    existing_content=existing.content,
                    new_content=item.get("details", ""),
                    available_slugs=self._get_available_slugs(kb_id),
                    language=language,
                )
                _, body = self._split_summary_line(updated_content)
                existing.content = body
                existing.source_refs = list(set(existing.source_refs + [knowledge_id]))
                self.page_manager.update_page(existing)
            else:
                # 创建新页面
                new_summary, new_body = self._split_summary_line(item.get("details", ""))
                page = WikiPage(
                    knowledge_base_id=kb_id,
                    slug=slug,
                    title=item.get("name", ""),
                    page_type=item_type,
                    content=f"# {item.get('name', '')}\n\n{new_body}",
                    summary=item.get("description", new_summary or ""),
                    aliases=item.get("aliases", []),
                    source_refs=[knowledge_id],
                )
                self.page_manager.create_page(page)

            affected_slugs.append(slug)

        # ── Phase 5: 更新索引页 ──
        await self._rebuild_index(kb_id, language)

        # ── Phase 6: 注入交叉链接 ──
        self._inject_cross_links(kb_id, affected_slugs)

        # 持久化
        self.page_manager.save_to_disk(kb_id)

        logger.info(f"[Wiki] 入库完成: {title}, 影响 {len(affected_slugs)} 个页面")
        return affected_slugs

    async def _generate_summary(
        self,
        title: str,
        file_name: str,
        content: str,
        available_slugs: str,
        language: str,
    ) -> str:
        """生成文档摘要页"""
        prompt = WIKI_SUMMARY_PROMPT.format(
            title=title,
            file_name=file_name,
            content=content,
            available_slugs=available_slugs,
            language=language,
        )
        return await self.llm_client.generate_with_template(
            system_prompt="你是专业的 Wiki 编辑。",
            user_content=prompt,
            temperature=0.1,
        )

    async def _extract_knowledge(
        self,
        title: str,
        content: str,
        previous_slugs: str,
        language: str,
    ) -> dict:
        """提取实体和概念"""
        prompt = WIKI_KNOWLEDGE_EXTRACT_PROMPT.format(
            title=title,
            content=content,
            previous_slugs=previous_slugs,
            language=language,
        )
        response = await self.llm_client.generate_with_template(
            system_prompt="你是知识提取专家。",
            user_content=prompt,
            temperature=0.1,
        )
        return parse_llm_json(response)

    async def _deduplicate(
        self,
        kb_id: str,
        entities: list[dict],
        concepts: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """实体/概念去重"""
        existing_pages = self.page_manager.list_pages(kb_id)
        entity_concept_pages = [
            p for p in existing_pages if p.page_type in ("entity", "concept")
        ]

        if not entity_concept_pages or (not entities and not concepts):
            return entities, concepts

        # 构建去重 Prompt 输入
        new_items_str = self._format_items_for_dedup(entities, concepts)
        existing_pages_str = self._format_existing_pages_for_dedup(entity_concept_pages)

        prompt = WIKI_DEDUPLICATION_PROMPT.format(
            new_items=new_items_str,
            existing_pages=existing_pages_str,
        )

        try:
            response = await self.llm_client.generate_with_template(
                system_prompt="你是去重专家。",
                user_content=prompt,
                temperature=0.0,
            )
            result = parse_llm_json(response)
            merges = result.get("merges", {})

            if merges:
                # 处理合并
                for new_slug, existing_slug in merges.items():
                    # 从新提取中移除被合并的项
                    entities = [e for e in entities if e.get("slug") != new_slug]
                    concepts = [c for c in concepts if c.get("slug") != new_slug]

                logger.info(f"[Wiki] 去重合并: {len(merges)} 项")

        except Exception as e:
            logger.warning(f"[Wiki] 去重失败: {e}")

        return entities, concepts

    async def _modify_page(
        self,
        kb_id: str,
        page_slug: str,
        page_title: str,
        page_type: str,
        existing_content: str,
        new_content: str,
        available_slugs: str,
        language: str,
    ) -> str:
        """更新已有 Wiki 页面"""
        prompt = WIKI_PAGE_MODIFY_PROMPT.format(
            page_slug=page_slug,
            page_title=page_title,
            page_type=page_type,
            existing_content=existing_content[:4000],
            new_content=new_content[:2000],
            available_slugs=available_slugs,
            language=language,
        )
        return await self.llm_client.generate_with_template(
            system_prompt="你是专业的 Wiki 编辑。",
            user_content=prompt,
            temperature=0.1,
        )

    async def _rebuild_index(self, kb_id: str, language: str):
        """重建索引页"""
        all_pages = self.page_manager.list_pages(kb_id)
        non_system = [p for p in all_pages if p.page_type not in ("index", "log")]

        # 按类型分组
        grouped = {}
        for p in non_system:
            grouped.setdefault(p.page_type, []).append(p)

        type_labels = {
            "summary": "文档摘要",
            "entity": "实体",
            "concept": "概念",
            "synthesis": "综合",
        }

        # 构建目录
        dir_lines = []
        for ptype, labels in type_labels.items():
            pages = grouped.get(ptype, [])
            if not pages:
                continue
            dir_lines.append(f"\n## {labels} ({len(pages)})\n")
            for p in pages:
                dir_lines.append(f"[[{p.slug}]] — {p.summary}")

        intro = "# Wiki 索引\n\n本 Wiki 包含从上传文档中自动提取的知识。\n"
        index_content = intro + "\n".join(dir_lines)

        index_page = self.page_manager.get_index(kb_id)
        if index_page:
            index_page.content = index_content
            self.page_manager.update_page(index_page)
        else:
            self.page_manager.create_page(WikiPage(
                knowledge_base_id=kb_id,
                slug="index",
                title="Wiki 索引",
                page_type="index",
                content=index_content,
                summary="Wiki 知识库索引",
            ))

    def _inject_cross_links(self, kb_id: str, affected_slugs: list[str]):
        """
        注入交叉链接 (借鉴 WeKnora 的 injectCrossLinks)

        扫描受影响页面，将内容中出现的其他 Wiki 页面标题
        自动转换为 [[slug|显示名称]] 链接格式。
        """
        all_pages = self.page_manager.list_pages(kb_id)
        if len(all_pages) < 2:
            return

        # 构建链接引用表
        refs = []
        for p in all_pages:
            if p.page_type in ("index", "log"):
                continue
            if p.title:
                refs.append((p.slug, p.title))
            for alias in p.aliases:
                if alias:
                    refs.append((p.slug, alias))

        # 按长度降序排列（优先匹配长名称）
        refs.sort(key=lambda x: len(x[1]), reverse=True)

        affected_set = set(affected_slugs)
        updated = 0

        for page in all_pages:
            if page.slug not in affected_set:
                continue
            if page.page_type in ("index", "log"):
                continue

            new_content = page.content
            changed = False

            for slug, match_text in refs:
                if slug == page.slug:
                    continue  # 不自我链接

                # 跳过已存在的链接
                if f"[[{slug}" in new_content:
                    continue

                # 简单文本替换（不在代码块或已有链接中）
                if match_text in new_content:
                    # 排除已在 [[...]] 中的
                    pattern = re.compile(
                        r"(?<!\[\[)" + re.escape(match_text) + r"(?![\]\|])"
                    )
                    replacement = f"[[{slug}|{match_text}]]"
                    new_content, count = pattern.subn(replacement, new_content, count=1)
                    if count > 0:
                        changed = True

            if changed:
                page.content = new_content
                self.page_manager.update_page(page)
                updated += 1

        if updated > 0:
            logger.info(f"[Wiki] 交叉链接注入: 更新了 {updated} 个页面")

    def _get_available_slugs(self, kb_id: str) -> str:
        """获取当前可用的 Wiki 页面 slug 列表"""
        pages = self.page_manager.list_pages(kb_id)
        if not pages:
            return "(暂无 Wiki 页面)"

        lines = []
        for p in pages:
            alias_str = f" (别名: {', '.join(p.aliases)})" if p.aliases else ""
            lines.append(f"[[{p.slug}]] = {p.title}{alias_str}")
        return "\n".join(lines)

    @staticmethod
    def _split_summary_line(raw: str) -> tuple[str, str]:
        """分离 SUMMARY 行和正文"""
        raw = raw.strip()
        if raw.startswith("SUMMARY:") or raw.startswith("SUMMARY："):
            idx = raw.find("\n")
            if idx < 0:
                summary = raw.split(":", 1)[1].strip().split("：", 1)[-1].strip()
                return summary, ""
            summary_line = raw[:idx]
            summary = summary_line.split(":", 1)[1].strip().split("：", 1)[-1].strip()
            return summary, raw[idx + 1:].strip()
        return "", raw

    @staticmethod
    def _slugify(text: str) -> str:
        """将文本转换为 URL 友好的 slug"""
        import unicodedata
        import re

        # 对中文进行简单拼音化处理（这里用连字符替代）
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_]+", "-", text)
        text = text.strip("-")
        return text or "untitled"

    @staticmethod
    def _format_items_for_dedup(entities: list[dict], concepts: list[dict]) -> str:
        """格式化新提取项用于去重"""
        lines = []
        for item in entities:
            lines.append(f'  <item slug="{item.get("slug", "")}" type="entity">')
            lines.append(f'    <name>{item.get("name", "")}</name>')
            for alias in item.get("aliases", []):
                lines.append(f"    <alias>{alias}</alias>")
            lines.append("  </item>")
        for item in concepts:
            lines.append(f'  <item slug="{item.get("slug", "")}" type="concept">')
            lines.append(f'    <name>{item.get("name", "")}</name>')
            for alias in item.get("aliases", []):
                lines.append(f"    <alias>{alias}</alias>")
            lines.append("  </item>")
        return "\n".join(lines)

    @staticmethod
    def _format_existing_pages_for_dedup(pages: list[WikiPage]) -> str:
        """格式化现有页面用于去重"""
        lines = []
        for p in pages:
            lines.append(f'  <item slug="{p.slug}" type="{p.page_type}">')
            lines.append(f"    <name>{p.title}</name>")
            for alias in p.aliases:
                lines.append(f"    <alias>{alias}</alias>")
            lines.append("  </item>")
        return "\n".join(lines)
