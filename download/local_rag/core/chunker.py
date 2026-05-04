"""
文本分块模块 - 支持语义感知的文本切分
借鉴 WeKnora 的 chunker 设计，支持按标题层级和固定大小分块
"""
import logging
import re
from typing import List, Optional

import tiktoken

from config.settings import ChunkerConfig
from models.schemas import Chunk

logger = logging.getLogger(__name__)


class TextChunker:
    """
    文本分块器

    策略：
    1. 对于 Markdown 文本，优先按标题层级分块（语义分块）
    2. 对于普通文本（如 PDF 提取的），使用固定大小 + 重叠窗口分块
    3. 两者结合：先按标题分段，长段再切分
    """

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None
            logger.warning("tiktoken 加载失败，将使用字符数估算 token 数")

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数"""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # 粗略估算：中文约 1.5 字符/token，英文约 4 字符/token
        return int(len(text) / 2.5)

    def chunk_text(
        self,
        text: str,
        doc_id: str = "",
        file_type: str = "",
    ) -> List[Chunk]:
        """
        将文本分块

        Args:
            text: 原始文本
            doc_id: 文档 ID
            file_type: 文件类型 (pdf/markdown)

        Returns:
            分块列表
        """
        if not text.strip():
            return []

        # 根据文件类型选择分块策略
        if file_type == "markdown":
            chunks = self._chunk_markdown(text, doc_id)
        else:
            chunks = self._chunk_fixed_size(text, doc_id)

        # 设置 token 计数和索引
        for i, chunk in enumerate(chunks):
            chunk.index = i
            chunk.token_count = self.count_tokens(chunk.content)

        logger.info(f"文本分块完成: doc_id={doc_id}, 总块数={len(chunks)}")
        return chunks

    def _chunk_markdown(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Markdown 语义分块 - 按标题层级分段，长段再切分
        借鉴 WeKnora 的 header-based splitting 思想
        """
        # 按标题分段
        sections = self._split_by_headers(text)

        chunks = []
        for section_title, section_content in sections:
            section_text = section_content.strip()
            if not section_text:
                continue

            section_tokens = self.count_tokens(section_text)

            if section_tokens <= self.config.chunk_size:
                # 短段直接作为一个块
                chunks.append(Chunk(
                    doc_id=doc_id,
                    content=section_text,
                    metadata={"section_title": section_title},
                ))
            else:
                # 长段使用固定大小分块
                sub_chunks = self._split_fixed(
                    section_text,
                    self.config.chunk_size,
                    self.config.chunk_overlap,
                )
                for j, sub_text in enumerate(sub_chunks):
                    chunks.append(Chunk(
                        doc_id=doc_id,
                        content=sub_text,
                        metadata={
                            "section_title": section_title,
                            "sub_index": j,
                        },
                    ))

        return chunks

    def _split_by_headers(self, text: str) -> List[tuple[str, str]]:
        """按 Markdown 标题分段"""
        lines = text.split("\n")
        sections: List[tuple[str, str]] = []
        current_title = ""
        current_content: List[str] = []

        for line in lines:
            if re.match(r'^#{1,6}\s+', line):
                # 遇到新标题，保存当前段
                if current_content:
                    sections.append((current_title, "\n".join(current_content)))
                current_title = line.strip()
                current_content = [line]
            else:
                current_content.append(line)

        # 最后一段
        if current_content:
            sections.append((current_title, "\n".join(current_content)))

        # 如果没有标题，整体作为一段
        if not sections:
            sections.append(("", text))

        return sections

    def _chunk_fixed_size(self, text: str, doc_id: str) -> List[Chunk]:
        """固定大小 + 重叠窗口分块"""
        sub_texts = self._split_fixed(
            text,
            self.config.chunk_size,
            self.config.chunk_overlap,
        )
        return [
            Chunk(doc_id=doc_id, content=t)
            for t in sub_texts
        ]

    def _split_fixed(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> List[str]:
        """
        固定大小分块（以 token 为单位）

        策略：先按段落分割，然后合并段落直到达到 token 上限
        """
        # 先按段落分割
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # 单个段落超长，需要强制切分
            if para_tokens > max_tokens:
                # 先保存当前块
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # 按句子切分长段落
                sentences = re.split(r'(?<=[。！？.!?])\s*', para)
                for sent in sentences:
                    sent_tokens = self.count_tokens(sent)
                    if current_tokens + sent_tokens > max_tokens and current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        # 重叠处理：保留最后一部分
                        overlap_text = "\n\n".join(current_chunk)
                        overlap_count = min(overlap_tokens, self.count_tokens(overlap_text))
                        if overlap_count > 0:
                            # 简化处理：从当前块尾部取重叠内容
                            overlap_para = self._get_overlap_paragraphs(
                                current_chunk, overlap_count
                            )
                            current_chunk = overlap_para
                            current_tokens = self.count_tokens("\n\n".join(current_chunk))
                        else:
                            current_chunk = []
                            current_tokens = 0

                    current_chunk.append(sent)
                    current_tokens += sent_tokens
            else:
                if current_tokens + para_tokens > max_tokens and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    # 重叠处理
                    overlap_text = "\n\n".join(current_chunk)
                    overlap_count = min(overlap_tokens, self.count_tokens(overlap_text))
                    if overlap_count > 0:
                        overlap_para = self._get_overlap_paragraphs(
                            current_chunk, overlap_count
                        )
                        current_chunk = overlap_para
                        current_tokens = self.count_tokens("\n\n".join(current_chunk))
                    else:
                        current_chunk = []
                        current_tokens = 0

                current_chunk.append(para)
                current_tokens += para_tokens

        # 最后一个块
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _get_overlap_paragraphs(
        self, paragraphs: List[str], overlap_token_count: int
    ) -> List[str]:
        """从段落列表尾部获取不超过 overlap_token_count 的重叠段落"""
        result: List[str] = []
        token_count = 0
        for para in reversed(paragraphs):
            para_tokens = self.count_tokens(para)
            if token_count + para_tokens > overlap_token_count:
                break
            result.insert(0, para)
            token_count += para_tokens
        return result
