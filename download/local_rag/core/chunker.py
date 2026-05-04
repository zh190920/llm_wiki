"""
文本分块器
================================
借鉴 WeKnora 的 chunker 设计，支持多种分块策略：
- 固定大小分块（带重叠）
- 语义分块（按段落/标题分割）
- 父子分块策略
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from config import settings


@dataclass
class Chunk:
    """文档分块数据模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    knowledge_id: str = ""
    knowledge_base_id: str = ""
    content: str = ""
    start_at: int = 0
    end_at: int = 0
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)
    # 父子分块
    parent_chunk_id: Optional[str] = None


class TextChunker:
    """文本分块器"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separator: str = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separator = separator or settings.CHUNK_SEPARATOR

    def chunk_text(
        self,
        text: str,
        knowledge_id: str = "",
        knowledge_base_id: str = "",
    ) -> list[Chunk]:
        """
        对文本进行分块

        策略: 优先按语义边界（标题/段落）分割，
        超长段落再按固定大小切割，保证重叠。
        """
        if not text.strip():
            return []

        # 第一层：按语义边界分割
        sections = self._split_by_semantic_boundary(text)

        chunks = []
        char_offset = 0

        for section in sections:
            # 第二层：对超长段落进行固定大小分块
            if len(section) > self.chunk_size:
                sub_chunks = self._fixed_size_chunk(section)
                for i, sub in enumerate(sub_chunks):
                    chunk = Chunk(
                        knowledge_id=knowledge_id,
                        knowledge_base_id=knowledge_base_id,
                        content=sub,
                        start_at=char_offset,
                        end_at=char_offset + len(sub),
                        chunk_index=len(chunks),
                        metadata={"sub_chunk_index": i},
                    )
                    chunks.append(chunk)
                    char_offset += len(sub)
            else:
                chunk = Chunk(
                    knowledge_id=knowledge_id,
                    knowledge_base_id=knowledge_base_id,
                    content=section,
                    start_at=char_offset,
                    end_at=char_offset + len(section),
                    chunk_index=len(chunks),
                )
                chunks.append(chunk)
                char_offset += len(section)

        logger.info(f"文本分块完成: {len(text)} 字符 → {len(chunks)} 个分块")
        return chunks

    def chunk_with_parent(
        self,
        text: str,
        knowledge_id: str = "",
        knowledge_base_id: str = "",
        parent_size: int = 2048,
    ) -> list[Chunk]:
        """
        父子分块策略 (借鉴 WeKnora Parent-Child Chunking)

        大块（parent）用于提供上下文，小块（child）用于精确检索。
        检索时先命中 child，再展开其 parent 获得完整上下文。
        """
        # 先生成大块
        parent_chunks = self.chunk_text(text, knowledge_id, knowledge_base_id)
        # 调整 chunk_size 生成小块
        original_size = self.chunk_size
        original_overlap = self.chunk_overlap

        self.chunk_size = original_size // 4
        self.chunk_overlap = original_overlap // 2

        all_chunks = []
        for parent in parent_chunks:
            # 每个大块内部再切分小块
            child_chunks = self._fixed_size_chunk(parent.content)
            parent_id = parent.id
            all_chunks.append(parent)

            for j, child_content in enumerate(child_chunks):
                child = Chunk(
                    knowledge_id=knowledge_id,
                    knowledge_base_id=knowledge_base_id,
                    content=child_content,
                    chunk_index=len(all_chunks),
                    parent_chunk_id=parent_id,
                    metadata={"is_child": True, "child_index": j},
                )
                all_chunks.append(child)

        # 恢复原始设置
        self.chunk_size = original_size
        self.chunk_overlap = original_overlap

        logger.info(f"父子分块完成: {len(parent_chunks)} 个父块 + 子块 = {len(all_chunks)} 总块数")
        return all_chunks

    def _split_by_semantic_boundary(self, text: str) -> list[str]:
        """按语义边界分割：标题、空行、段落"""
        # 按 Markdown 标题分割
        heading_pattern = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)
        sections = []
        last_end = 0

        for match in heading_pattern.finditer(text):
            if match.start() > last_end:
                pre_text = text[last_end:match.start()].strip()
                if pre_text:
                    sections.append(pre_text)
            last_end = match.start()

        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sections.append(remaining)

        # 如果没有标题，按双换行分段落
        if not sections:
            sections = [s.strip() for s in text.split(self.separator) if s.strip()]

        # 如果段落仍然太长，按单换行再分
        final_sections = []
        for section in sections:
            if len(section) > self.chunk_size * 2:
                sub_paragraphs = [p.strip() for p in section.split("\n") if p.strip()]
                current = ""
                for para in sub_paragraphs:
                    if len(current) + len(para) + 1 <= self.chunk_size * 2:
                        current = f"{current}\n{para}" if current else para
                    else:
                        if current:
                            final_sections.append(current)
                        current = para
                if current:
                    final_sections.append(current)
            else:
                final_sections.append(section)

        return final_sections if final_sections else [text]

    def _fixed_size_chunk(self, text: str) -> list[str]:
        """固定大小分块，带重叠"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # 在边界处尽量按句子分割
            if end < len(text):
                # 查找最后一个句号/感叹号/问号
                last_sentence = max(
                    chunk_text.rfind("。"),
                    chunk_text.rfind("！"),
                    chunk_text.rfind("？"),
                    chunk_text.rfind("."),
                    chunk_text.rfind("\n"),
                )
                if last_sentence > self.chunk_size // 2:
                    chunk_text = chunk_text[: last_sentence + 1]
                    end = start + last_sentence + 1

            chunks.append(chunk_text.strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c]
