"""
Wiki 页面管理器
================================
借鉴 WeKnora 的 wikiPageService 设计，
管理 Wiki 页面的 CRUD、双向链接解析和图谱数据生成。
"""

import re
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings


@dataclass
class WikiPage:
    """Wiki 页面数据模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    knowledge_base_id: str = ""
    slug: str = ""
    title: str = ""
    page_type: str = "entity"  # entity, concept, summary, index, log, synthesis
    content: str = ""
    summary: str = ""
    status: str = "published"  # published, draft, archived
    version: int = 1
    aliases: list[str] = field(default_factory=list)
    out_links: list[str] = field(default_factory=list)
    in_links: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


# Wiki 链接正则: [[slug]] 或 [[slug|显示名称]]
WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


class WikiPageManager:
    """Wiki 页面管理器"""

    def __init__(self):
        self._pages: dict[str, dict[str, WikiPage]] = {}  # kb_id → {slug → WikiPage}

    def create_page(self, page: WikiPage) -> WikiPage:
        """创建 Wiki 页面"""
        kb_id = page.knowledge_base_id
        if kb_id not in self._pages:
            self._pages[kb_id] = {}

        # 解析出站链接
        page.out_links = self._parse_out_links(page.content)

        self._pages[kb_id][page.slug] = page

        # 更新入站链接
        self._update_in_links(kb_id, page.slug, page.out_links)

        logger.info(f"Wiki 页面创建: {page.slug} (KB: {kb_id})")
        return page

    def update_page(self, page: WikiPage) -> WikiPage:
        """更新 Wiki 页面"""
        kb_id = page.knowledge_base_id
        existing = self._get_page(kb_id, page.slug)
        if not existing:
            return self.create_page(page)

        old_out_links = existing.out_links

        # 更新字段
        existing.title = page.title
        existing.content = page.content
        existing.summary = page.summary
        existing.page_type = page.page_type
        existing.source_refs = page.source_refs
        existing.aliases = page.aliases
        existing.status = page.status
        existing.version += 1
        existing.updated_at = datetime.now().isoformat()

        # 重新解析出站链接
        existing.out_links = self._parse_out_links(existing.content)

        # 更新入站链接
        self._remove_in_links(kb_id, existing.slug, old_out_links)
        self._update_in_links(kb_id, existing.slug, existing.out_links)

        return existing

    def delete_page(self, kb_id: str, slug: str) -> bool:
        """删除 Wiki 页面"""
        page = self._get_page(kb_id, slug)
        if not page:
            return False

        # 移除入站链接引用
        self._remove_in_links(kb_id, slug, page.out_links)

        # 删除页面
        self._pages[kb_id].pop(slug, None)
        return True

    def get_page(self, kb_id: str, slug: str) -> Optional[WikiPage]:
        """获取 Wiki 页面"""
        return self._get_page(kb_id, slug)

    def list_pages(self, kb_id: str, page_type: Optional[str] = None) -> list[WikiPage]:
        """列出 Wiki 页面"""
        pages = self._pages.get(kb_id, {}).values()
        if page_type:
            pages = [p for p in pages if p.page_type == page_type]
        return sorted(pages, key=lambda p: p.updated_at, reverse=True)

    def get_index(self, kb_id: str) -> Optional[WikiPage]:
        """获取索引页"""
        return self._get_page(kb_id, "index")

    def get_graph_data(self, kb_id: str) -> dict:
        """
        获取 Wiki 链接图谱数据（借鉴 WeKnora 的 GetGraph 设计）

        Returns:
            {"nodes": [...], "edges": [...]}
        """
        pages = self._pages.get(kb_id, {})
        nodes = []
        edges = []

        for slug, page in pages.items():
            link_count = len(page.in_links) + len(page.out_links)
            nodes.append({
                "slug": slug,
                "title": page.title,
                "page_type": page.page_type,
                "link_count": link_count,
            })

            for target in page.out_links:
                if target in pages:
                    edges.append({
                        "source": slug,
                        "target": target,
                    })

        return {"nodes": nodes, "edges": edges}

    def search_pages(self, kb_id: str, query: str, limit: int = 10) -> list[WikiPage]:
        """搜索 Wiki 页面"""
        all_pages = self._pages.get(kb_id, {}).values()
        query_lower = query.lower()
        results = []

        for page in all_pages:
            score = 0
            if query_lower in page.title.lower():
                score += 10
            if query_lower in page.content.lower():
                score += 5
            if query_lower in page.summary.lower():
                score += 3
            for alias in page.aliases:
                if query_lower in alias.lower():
                    score += 7

            if score > 0:
                results.append((page, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:limit]]

    def rebuild_links(self, kb_id: str):
        """重建所有页面的双向链接"""
        pages = self._pages.get(kb_id, {})

        # 清除所有入站链接
        for page in pages.values():
            page.in_links = []

        # 重新解析并建立链接
        for page in pages.values():
            page.out_links = self._parse_out_links(page.content)
            for target in page.out_links:
                if target in pages:
                    if page.slug not in pages[target].in_links:
                        pages[target].in_links.append(page.slug)

    def _get_page(self, kb_id: str, slug: str) -> Optional[WikiPage]:
        """获取页面内部方法"""
        return self._pages.get(kb_id, {}).get(slug)

    def _parse_out_links(self, content: str) -> list[str]:
        """解析 Markdown 中的 [[wiki-link]] 链接"""
        links = []
        seen = set()

        for match in WIKI_LINK_PATTERN.finditer(content):
            link_text = match.group(1).strip()
            # 处理 [[slug|display name]] 格式
            if "|" in link_text:
                slug = link_text.split("|")[0].strip()
            else:
                slug = link_text
            slug = slug.lower().replace(" ", "-")
            if slug and slug not in seen:
                seen.add(slug)
                links.append(slug)

        return links

    def _update_in_links(self, kb_id: str, source_slug: str, targets: list[str]):
        """更新入站链接"""
        pages = self._pages.get(kb_id, {})
        for target in targets:
            if target in pages:
                target_page = pages[target]
                if source_slug not in target_page.in_links:
                    target_page.in_links.append(source_slug)

    def _remove_in_links(self, kb_id: str, source_slug: str, targets: list[str]):
        """移除入站链接"""
        pages = self._pages.get(kb_id, {})
        for target in targets:
            if target in pages:
                target_page = pages[target]
                target_page.in_links = [
                    s for s in target_page.in_links if s != source_slug
                ]

    def save_to_disk(self, kb_id: str):
        """持久化 Wiki 页面到磁盘"""
        wiki_dir = settings.WIKI_DIR / kb_id
        wiki_dir.mkdir(parents=True, exist_ok=True)

        pages = self._pages.get(kb_id, {})
        data = {}
        for slug, page in pages.items():
            data[slug] = {
                "id": page.id,
                "slug": page.slug,
                "title": page.title,
                "page_type": page.page_type,
                "content": page.content,
                "summary": page.summary,
                "status": page.status,
                "version": page.version,
                "aliases": page.aliases,
                "out_links": page.out_links,
                "in_links": page.in_links,
                "source_refs": page.source_refs,
                "created_at": page.created_at,
                "updated_at": page.updated_at,
            }

        with open(wiki_dir / "pages.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Wiki 页面已保存: {wiki_dir} ({len(pages)} 页)")

    def load_from_disk(self, kb_id: str) -> bool:
        """从磁盘加载 Wiki 页面"""
        wiki_dir = settings.WIKI_DIR / kb_id
        pages_file = wiki_dir / "pages.json"

        if not pages_file.exists():
            return False

        with open(pages_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._pages[kb_id] = {}
        for slug, page_data in data.items():
            page = WikiPage(
                id=page_data.get("id", str(uuid.uuid4())),
                knowledge_base_id=kb_id,
                slug=page_data.get("slug", slug),
                title=page_data.get("title", ""),
                page_type=page_data.get("page_type", "entity"),
                content=page_data.get("content", ""),
                summary=page_data.get("summary", ""),
                status=page_data.get("status", "published"),
                version=page_data.get("version", 1),
                aliases=page_data.get("aliases", []),
                out_links=page_data.get("out_links", []),
                in_links=page_data.get("in_links", []),
                source_refs=page_data.get("source_refs", []),
                created_at=page_data.get("created_at", ""),
                updated_at=page_data.get("updated_at", ""),
            )
            self._pages[kb_id][slug] = page

        logger.info(f"Wiki 页面已加载: {kb_id} ({len(data)} 页)")
        return True
