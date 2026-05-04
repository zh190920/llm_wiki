"""
文档解析模块 - 支持 PDF 和 Markdown 格式
借鉴 WeKnora docreader 的设计思想，实现统一的文档解析接口
"""
import asyncio
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from models.schemas import Chunk, DocumentMetadata

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """文档解析器基类 - 定义统一接口"""

    @abstractmethod
    async def parse(self, file_path: str, doc_id: str) -> tuple[str, DocumentMetadata]:
        """解析文档，返回 (纯文本内容, 文档元数据)"""
        ...

    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名"""
        ...


class PDFParser(BaseParser):
    """
    PDF 文档解析器
    使用 PyMuPDF (fitz) 解析 PDF，提取文本并保留结构信息
    """

    async def parse(self, file_path: str, doc_id: str) -> tuple[str, DocumentMetadata]:
        """解析 PDF 文件，提取文本内容"""
        try:
            text_content = await asyncio.to_thread(self._parse_sync, file_path)
        except Exception as e:
            logger.error(f"PDF 解析失败: {file_path}, 错误: {e}")
            raise

        filename = Path(file_path).name
        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type="pdf",
            title=Path(file_path).stem,
            source=file_path,
        )
        return text_content, metadata

    def _parse_sync(self, file_path: str) -> str:
        """同步解析 PDF（在线程池中执行）"""
        import fitz  # pymupdf

        doc = fitz.open(file_path)
        pages_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # 清理和格式化
            text = self._clean_text(text)

            if text.strip():
                # 添加页码标记，便于后续追溯
                pages_text.append(f"[第 {page_num + 1} 页]\n{text}")

        doc.close()
        return "\n\n".join(pages_text)

    def _clean_text(self, text: str) -> str:
        """清理 PDF 提取的文本"""
        # 移除多余的空白行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除行内多余空格
        text = re.sub(r' +', ' ', text)
        # 修复断行问题（中文不需要空格连接）
        text = re.sub(r'(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])', '', text)
        return text.strip()

    def supported_extensions(self) -> List[str]:
        return [".pdf"]


class MarkdownParser(BaseParser):
    """
    Markdown 文档解析器
    保留 Markdown 结构信息（标题层级），便于语义分块
    """

    async def parse(self, file_path: str, doc_id: str) -> tuple[str, DocumentMetadata]:
        """解析 Markdown 文件"""
        try:
            content = await asyncio.to_thread(self._parse_sync, file_path)
        except Exception as e:
            logger.error(f"Markdown 解析失败: {file_path}, 错误: {e}")
            raise

        filename = Path(file_path).name
        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type="markdown",
            title=self._extract_title(content) or Path(file_path).stem,
            source=file_path,
        )
        return content, metadata

    def _parse_sync(self, file_path: str) -> str:
        """同步解析 Markdown"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content

    def _extract_title(self, content: str) -> Optional[str]:
        """从 Markdown 中提取第一个标题"""
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return None

    def supported_extensions(self) -> List[str]:
        return [".md", ".markdown"]


class DocumentParser:
    """
    文档解析引擎 - 统一管理所有解析器
    借鉴 WeKnora 的 engine_registry 设计，支持解析器注册和自动选择
    """

    def __init__(self):
        self._parsers: dict[str, BaseParser] = {}
        self._register_default_parsers()

    def _register_default_parsers(self):
        """注册默认解析器"""
        for parser_cls in [PDFParser, MarkdownParser]:
            parser = parser_cls()
            for ext in parser.supported_extensions():
                self._parsers[ext.lower()] = parser

    def register_parser(self, extension: str, parser: BaseParser):
        """注册自定义解析器（先注册优先）"""
        ext = extension.lower()
        if ext not in self._parsers:
            self._parsers[ext] = parser
        else:
            logger.warning(f"解析器已存在: {ext}，跳过注册")

    def get_parser(self, file_path: str) -> Optional[BaseParser]:
        """根据文件扩展名获取解析器"""
        ext = Path(file_path).suffix.lower()
        return self._parsers.get(ext)

    async def parse(self, file_path: str, doc_id: Optional[str] = None) -> tuple[str, DocumentMetadata]:
        """
        解析文档 - 自动选择合适的解析器

        Args:
            file_path: 文件路径
            doc_id: 文档ID，如不提供则自动生成

        Returns:
            (文本内容, 文档元数据)
        """
        if doc_id is None:
            doc_id = DocumentMetadata().doc_id

        parser = self.get_parser(file_path)
        if parser is None:
            supported = ", ".join(self._parsers.keys())
            raise ValueError(
                f"不支持的文件类型: {Path(file_path).suffix}，"
                f"当前支持: {supported}"
            )

        text, metadata = await parser.parse(file_path, doc_id)
        logger.info(
            f"文档解析完成: {metadata.filename}, "
            f"文本长度={len(text)}, doc_id={doc_id}"
        )
        return text, metadata

    def supported_extensions(self) -> List[str]:
        """返回所有支持的文件扩展名"""
        return list(self._parsers.keys())
