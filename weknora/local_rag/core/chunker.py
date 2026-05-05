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
        """计算文本的 token 数（中文优化）"""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # 中文约 1.5 字符/token，英文约 4 字符/token，混合取 2
        chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        non_chinese_len = len(text) - chinese_count
        return int(chinese_count / 1.5 + non_chinese_len / 4)

    def chunk_text(
        self,
        text: str,
        doc_id: str = "",
        file_type: str = "",
        hierarchical: bool = False,
    ) -> List[Chunk]:
        """
        将文本分块

        Args:
            text: 原始文本
            doc_id: 文档 ID
            file_type: 文件类型 (pdf/markdown)
            hierarchical: 是否启用层级分块（父子块）

        Returns:
            分块列表
        """
        if not text.strip():
            return []

        # 层级分块模式
        if hierarchical or self.config.hierarchical:
            return self.chunk_text_hierarchical(text, doc_id, file_type)

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

    def chunk_text_hierarchical(
        self,
        text: str,
        doc_id: str = "",
        file_type: str = "",
    ) -> List[Chunk]:
        """
        层级分块 - 先创建父块，再将父块拆分为子块

        借鉴 WeKnora 的 Parent-Child Chunking 设计：
        - 父块较大（如 2048 tokens），提供完整上下文
        - 子块较小（如 512 tokens），作为精确检索单元
        - 子块通过 parent_chunk_id 关联到父块
        - 检索时命中子块，可回溯获取父块的完整上下文
        """
        parent_size = self.config.chunk_size_parent
        child_size = self.config.chunk_size
        child_overlap = self.config.chunk_overlap

        # Step 1: 创建父块
        if file_type == "markdown":
            parent_texts = self._chunk_markdown(text, doc_id)
        else:
            parent_texts = self._chunk_fixed_size(text, doc_id)

        # 如果父块太大，进一步切分
        oversized_parents = []
        for p in parent_texts:
            if self.count_tokens(p.content) > parent_size:
                sub_texts = self._split_fixed(p.content, parent_size, child_overlap)
                for j, sub_text in enumerate(sub_texts):
                    oversized_parents.append(Chunk(
                        doc_id=doc_id,
                        content=sub_text,
                        metadata={**p.metadata, "sub_parent_index": j},
                    ))
            else:
                oversized_parents.append(p)

        parent_texts = oversized_parents

        # Step 2: 将每个父块拆分为子块
        all_chunks: List[Chunk] = []
        chunk_index = 0

        for parent_idx, parent_chunk in enumerate(parent_texts):
            # 设置父块属性
            parent_chunk.index = chunk_index
            parent_chunk.token_count = self.count_tokens(parent_chunk.content)
            parent_chunk.metadata["is_parent"] = True
            parent_chunk.metadata["child_count"] = 0

            parent_token_count = self.count_tokens(parent_chunk.content)

            # 如果父块够小，不需要拆分，直接作为一个块（既是父也是子）
            if parent_token_count <= child_size:
                parent_chunk.metadata["is_parent"] = False
                all_chunks.append(parent_chunk)
                chunk_index += 1
                continue

            # 添加父块
            all_chunks.append(parent_chunk)
            parent_index = chunk_index
            chunk_index += 1

            # 拆分子块
            child_texts = self._split_fixed(
                parent_chunk.content, child_size, child_overlap
            )

            child_count = 0
            for child_text in child_texts:
                child_chunk = Chunk(
                    doc_id=doc_id,
                    content=child_text,
                    parent_chunk_id=parent_chunk.chunk_id,
                    metadata={
                        **parent_chunk.metadata,
                        "is_parent": False,
                        "parent_index": parent_idx,
                    },
                )
                child_chunk.index = chunk_index
                child_chunk.token_count = self.count_tokens(child_text)
                all_chunks.append(child_chunk)
                chunk_index += 1
                child_count += 1

            parent_chunk.metadata["child_count"] = child_count

        logger.info(
            f"层级分块完成: doc_id={doc_id}, 总块数={len(all_chunks)}, "
            f"父块数={sum(1 for c in all_chunks if c.metadata.get('is_parent'))}"
        )
        return all_chunks

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

                # 按句子切分长段落（支持中英文标点）
                sentences = re.split(r'(?<=[。！？；.!?;])\s*', para)
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
