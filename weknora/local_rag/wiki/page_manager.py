"""
Wiki 页面管理器 - Wiki 页面的 CRUD 操作
借鉴 WeKnora 的 WikiPageService 设计
"""
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from models.schemas import WikiIssue, WikiPage, WikiPageType

logger = logging.getLogger(__name__)


class WikiPageManager:
    """
    Wiki 页面管理器

    功能：
    - 页面 CRUD
    - 页面搜索（标题 + 内容）
    - 跨链接管理
    - 持久化存储（JSON 文件）
    - Markdown 文件导出
    """

    def __init__(self, wiki_dir: str = "./wiki_output"):
        self._wiki_dir = Path(wiki_dir)
        self._pages: Dict[str, WikiPage] = {}  # slug -> WikiPage
        self._issues: List[WikiIssue] = []
        self._lock = asyncio.Lock()

    async def initialize(self):
        """初始化：创建目录并加载已有页面"""
        self._wiki_dir.mkdir(parents=True, exist_ok=True)
        await self._load_pages()
        logger.info(f"Wiki 管理器初始化完成: {len(self._pages)} 个页面")

    async def _load_pages(self):
        """从磁盘加载页面"""
        pages_file = self._wiki_dir / "pages.json"
        if pages_file.exists():
            try:
                with open(pages_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for slug, page_data in data.items():
                    self._pages[slug] = WikiPage(**page_data)
            except Exception as e:
                logger.error(f"加载 Wiki 页面失败: {e}")

        issues_file = self._wiki_dir / "issues.json"
        if issues_file.exists():
            try:
                with open(issues_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._issues = [WikiIssue(**i) for i in data]
            except Exception as e:
                logger.error(f"加载 Wiki 问题失败: {e}")

    async def _save_pages(self):
        """持久化页面到磁盘"""
        pages_file = self._wiki_dir / "pages.json"
        data = {slug: page.model_dump() for slug, page in self._pages.items()}
        with open(pages_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        issues_file = self._wiki_dir / "issues.json"
        with open(issues_file, "w", encoding="utf-8") as f:
            json.dump([i.model_dump() for i in self._issues], f, ensure_ascii=False, indent=2)

    async def get_page(self, slug: str) -> Optional[WikiPage]:
        """获取页面"""
        return self._pages.get(slug)

    async def save_page(self, page: WikiPage):
        """保存页面（创建或更新）"""
        async with self._lock:
            page.updated_at = __import__("time").time()
            self._pages[page.slug] = page
            await self._save_pages()
            await self._export_markdown(page)

    async def delete_page(self, slug: str) -> bool:
        """删除页面"""
        async with self._lock:
            if slug in self._pages:
                del self._pages[slug]
                # 清理其他页面中指向此页面的链接
                for page in self._pages.values():
                    if slug in page.out_links:
                        page.out_links.remove(slug)
                        # 替换内容中的死链接
                        page.content = re.sub(
                            rf'\[\[{re.escape(slug)}\|[^\]]+\]\]',
                            '', page.content
                        )
                        page.content = re.sub(
                            rf'\[\[{re.escape(slug)}\]\]',
                            '', page.content
                        )
                await self._save_pages()
                return True
            return False

    async def list_pages(
        self,
        page_type: Optional[WikiPageType] = None,
        status: Optional[str] = None,
    ) -> List[WikiPage]:
        """列出页面"""
        pages = list(self._pages.values())
        if page_type:
            pages = [p for p in pages if p.page_type == page_type]
        if status:
            pages = [p for p in pages if p.status == status]
        return sorted(pages, key=lambda p: p.updated_at, reverse=True)

    async def search_pages(self, query: str, top_k: int = 5) -> List[WikiPage]:
        """搜索页面（标题 + 内容关键词匹配）"""
        query_lower = query.lower()
        scored_pages: List[tuple[float, WikiPage]] = []

        for page in self._pages.values():
            score = 0.0
            # 标题匹配（权重高）
            if query_lower in page.title.lower():
                score += 2.0
            # 内容匹配
            content_lower = page.content.lower()
            score += content_lower.count(query_lower) * 0.1
            # slug 匹配
            if query_lower in page.slug.lower():
                score += 1.0

            if score > 0:
                scored_pages.append((score, page))

        scored_pages.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored_pages[:top_k]]

    async def add_issue(self, issue: WikiIssue):
        """添加质量问题"""
        self._issues.append(issue)
        async with self._lock:
            await self._save_pages()

    async def get_issues(self, page_slug: Optional[str] = None) -> List[WikiIssue]:
        """获取质量问题"""
        if page_slug:
            return [i for i in self._issues if i.page_slug == page_slug]
        return self._issues

    async def resolve_issue(self, issue_id: str) -> bool:
        """解决质量问题"""
        for issue in self._issues:
            if issue.issue_id == issue_id:
                issue.resolved = True
                await self._save_pages()
                return True
        return False

    async def _export_markdown(self, page: WikiPage):
        """将页面导出为 Markdown 文件"""
        md_dir = self._wiki_dir / "markdown"
        md_dir.mkdir(exist_ok=True)

        # 清理 slug 作为文件名
        safe_slug = re.sub(r'[^\w\-.]', '_', page.slug)
        md_path = md_dir / f"{safe_slug}.md"

        # 构建页面头部信息
        header = (
            f"---\n"
            f"title: \"{page.title}\"\n"
            f"slug: \"{page.slug}\"\n"
            f"type: \"{page.page_type.value}\"\n"
            f"status: \"{page.status}\"\n"
            f"out_links: {json.dumps(page.out_links)}\n"
            f"---\n\n"
        )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(header + page.content)

    async def export_all_markdown(self) -> str:
        """导出所有页面为 Markdown 文件"""
        md_dir = self._wiki_dir / "markdown"
        md_dir.mkdir(exist_ok=True)

        for page in self._pages.values():
            await self._export_markdown(page)

        return str(md_dir)

    async def inject_cross_links(self):
        """
        注入跨页面链接

        借鉴 WeKnora 的纯文本跨链接注入设计：
        - 遍历所有页面
        - 检测页面内容中出现的其他页面标题
        - 将标题替换为 [[slug|标题]] 格式
        """
        async with self._lock:
            title_to_slug: Dict[str, str] = {}
            for page in self._pages.values():
                title_to_slug[page.title] = page.slug

            for page in self._pages.values():
                modified = False
                content = page.content

                for title, slug in title_to_slug.items():
                    if slug == page.slug:
                        continue  # 不自引用

                    # 检查是否已存在链接
                    link_pattern = f"[[{slug}|{title}]]"
                    if link_pattern in content:
                        continue

                    # 替换标题文本为链接（不替换标题行中的文本）
                    # 只替换正文中的出现
                    lines = content.split("\n")
                    new_lines = []
                    for line in lines:
                        if line.startswith("#"):
                            new_lines.append(line)
                        else:
                            new_line = line.replace(title, f"[[{slug}|{title}]]")
                            if new_line != line:
                                modified = True
                            new_lines.append(new_line)

                    content = "\n".join(new_lines)

                if modified:
                    page.content = content
                    # 更新出链
                    page.out_links = re.findall(r'\[\[([^\]|]+)', content)

            await self._save_pages()
            logger.info("跨页面链接注入完成")

    @property
    def total_pages(self) -> int:
        return len(self._pages)
