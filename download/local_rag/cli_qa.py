"""
无端口本地文档问答 - 直接读取 PDF/MD 文件进行问答

使用方式一（Python 函数调用）：
    from cli_qa import LocalQA

    qa = LocalQA(
        api_key="sk-xxxxx",
        base_url="https://api.openai.com/v1",   # 可选
        chat_model="gpt-4o-mini",                # 可选
        embedding_model="text-embedding-3-small", # 可选
    )

    # 加载文档（支持 pdf 和 md）
    qa.load_file("操作手册.pdf")
    qa.load_file("技术文档.md")

    # 问答
    answer = qa.ask("设备故障码 E003 怎么处理？")
    print(answer)

    # 带来源的问答
    result = qa.ask_with_sources("安全操作规程有哪些？")
    print(result["answer"])
    print(result["sources"])

    # 交互式问答
    qa.interactive()

使用方式二（命令行）：
    python cli_qa.py --api-key sk-xxxxx 操作手册.pdf
    python cli_qa.py --api-key sk-xxxxx --base-url https://your-api.com/v1 技术文档.md
    python cli_qa.py --api-key sk-xxxxx --chat-model gpt-4o 操作手册.pdf 技术文档.md

使用方式三（环境变量）：
    set OPENAI_API_KEY=sk-xxxxx
    set OPENAI_BASE_URL=https://api.openai.com/v1
    python cli_qa.py 操作手册.pdf
"""
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import AppConfig, LLMConfig, RetrieverConfig, ChunkerConfig
from core.chunker import TextChunker
from core.document_parser import DocumentParser
from core.embedder import Embedder
from core.rag_engine import RAGEngine
from core.reranker import Reranker
from core.retriever import Retriever
from core.vector_store import VectorStore
from models.schemas import ChatResponse, DocumentMetadata

logger = logging.getLogger(__name__)


class LocalQA:
    """
    本地文档问答 - 无需启动服务端口

    核心用法：
        qa = LocalQA(api_key="sk-xxxxx")
        qa.load_file("手册.pdf")
        answer = qa.ask("xxx怎么操作？")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        chat_model: str = "gpt-4o-mini",
        embedding_model: str = "BAAI/bge-m3",
        embedding_dim: int = 1024,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        temperature: float = 0.3,
    ):
        """
        初始化本地问答引擎

        Args:
            api_key: OpenAI API Key（也可通过环境变量 OPENAI_API_KEY 设置）
            base_url: OpenAI API 地址
            chat_model: 聊天模型名称
            embedding_model: 嵌入模型名称
            embedding_dim: 嵌入维度
            chunk_size: 文档分块大小（token 数）
            chunk_overlap: 分块重叠大小
            top_k: 检索返回的文档块数量
            temperature: 生成温度
        """
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not api_key:
            raise ValueError(
                "请提供 api_key 参数或设置 OPENAI_API_KEY 环境变量\n"
                "  示例: qa = LocalQA(api_key='sk-xxxxx')\n"
                "  或:  set OPENAI_API_KEY=sk-xxxxx"
            )

        # 构建配置
        self._config = AppConfig(
            llm=LLMConfig(
                api_key=api_key,
                base_url=base_url,
                chat_model=chat_model,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                temperature=temperature,
            ),
            chunker=ChunkerConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ),
            retriever=RetrieverConfig(
                rerank_top_k=top_k,
                vector_top_k=top_k * 2,
                keyword_top_k=top_k * 2,
            ),
        )

        # 初始化组件
        self._parser = DocumentParser()
        self._chunker = TextChunker(self._config.chunker)
        self._embedder = Embedder(self._config.llm)
        self._vector_store = VectorStore(self._config.retriever, dim=embedding_dim)
        self._reranker = Reranker(self._config.llm, self._config.retriever, self._embedder)
        self._retriever = Retriever(self._config, self._vector_store, self._embedder, self._reranker)
        self._rag_engine = RAGEngine(self._config)
        self._rag_engine.initialize(
            embedder=self._embedder,
            vector_store=self._vector_store,
            retriever=self._retriever,
            reranker=self._reranker,
        )

        # 文档记录
        self._documents: Dict[str, DocumentMetadata] = {}
        self._conversation_history: List[dict] = []

        logger.info("LocalQA 初始化完成")

    @property
    def loaded_documents(self) -> List[dict]:
        """已加载的文档列表"""
        return [
            {
                "doc_id": meta.doc_id,
                "filename": meta.filename,
                "file_type": meta.file_type,
                "title": meta.title,
                "chunk_count": meta.chunk_count,
            }
            for meta in self._documents.values()
        ]

    @property
    def total_chunks(self) -> int:
        """已加载的文档块总数"""
        return self._vector_store.total_chunks

    def load_file(self, file_path: str) -> dict:
        """
        加载单个文件（同步接口）

        支持 PDF 和 Markdown 格式

        Args:
            file_path: 文件路径

        Returns:
            加载结果 {"doc_id": ..., "filename": ..., "chunk_count": ...}
        """
        return asyncio.get_event_loop().run_until_complete(
            self._load_file_async(file_path)
        )

    def load_directory(self, dir_path: str, extensions: Optional[List[str]] = None) -> List[dict]:
        """
        加载目录下的所有文档（同步接口）

        Args:
            dir_path: 目录路径
            extensions: 文件扩展名过滤，默认 [".pdf", ".md", ".markdown"]

        Returns:
            加载结果列表
        """
        return asyncio.get_event_loop().run_until_complete(
            self._load_directory_async(dir_path, extensions)
        )

    def ask(self, question: str, deep: bool = False) -> str:
        """
        提问并获取回答（同步接口）

        Args:
            question: 问题
            deep: 是否使用深度模式（查询理解+重排，更准但更慢）

        Returns:
            回答文本
        """
        result = asyncio.get_event_loop().run_until_complete(
            self._ask_async(question, deep=deep)
        )

        # 记录对话历史
        self._conversation_history.append({"role": "user", "content": question})
        self._conversation_history.append({"role": "assistant", "content": result.answer})

        # 保留最近 10 轮
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        return result.answer

    def ask_with_sources(self, question: str, deep: bool = False) -> Dict:
        """
        提问并获取带来源引用的回答（同步接口）

        Args:
            question: 问题
            deep: 是否使用深度模式

        Returns:
            {"answer": str, "sources": [{"content": ..., "score": ..., "doc_id": ...}]}
        """
        result = asyncio.get_event_loop().run_until_complete(
            self._ask_async(question, deep=deep)
        )

        # 记录对话历史
        self._conversation_history.append({"role": "user", "content": question})
        self._conversation_history.append({"role": "assistant", "content": result.answer})

        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        return {
            "answer": result.answer,
            "sources": [
                {
                    "content": s.chunk.content[:200],
                    "score": round(s.score, 4),
                    "doc_id": s.chunk.doc_id,
                    "section": s.chunk.metadata.get("section_title", ""),
                    "match_type": s.match_type.value,
                }
                for s in result.sources
            ],
        }

    def clear_history(self):
        """清空对话历史"""
        self._conversation_history = []

    def interactive(self, prompt_text: str = "❓ 请输入问题（输入 q 退出，c 清空历史）: "):
        """
        交互式问答模式

        Args:
            prompt_text: 提示文本
        """
        print("\n" + "=" * 60)
        print("  📚 本地文档问答系统")
        print(f"  已加载文档: {len(self._documents)} 个 | 文档块: {self.total_chunks} 个")
        for meta in self._documents.values():
            print(f"    - {meta.filename} ({meta.chunk_count} 块)")
        print("  命令: q=退出, c=清空历史, d=深度模式切换")
        print("=" * 60 + "\n")

        deep_mode = False
        while True:
            try:
                question = input(prompt_text).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not question:
                continue
            if question.lower() == "q":
                print("再见！")
                break
            if question.lower() == "c":
                self.clear_history()
                print("✅ 对话历史已清空\n")
                continue
            if question.lower() == "d":
                deep_mode = not deep_mode
                print(f"✅ 深度模式: {'开启' if deep_mode else '关闭'}\n")
                continue

            try:
                result = self.ask_with_sources(question, deep=deep_mode)
                print(f"\n💬 回答:\n{result['answer']}")

                if result["sources"]:
                    print(f"\n📎 参考来源:")
                    for i, src in enumerate(result["sources"][:3]):
                        section = src["section"] or src["doc_id"]
                        print(f"  [{i+1}] {section} (相关度: {src['score']})")
                print()

            except Exception as e:
                print(f"\n❌ 出错: {e}\n")

    # ============================================================
    # 异步内部方法
    # ============================================================

    async def _load_file_async(self, file_path: str) -> dict:
        """异步加载文件"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        if ext not in self._parser.supported_extensions():
            raise ValueError(
                f"不支持的文件类型: {ext}，"
                f"当前支持: {self._parser.supported_extensions()}"
            )

        doc_id = DocumentMetadata().doc_id

        # 1. 解析
        text, metadata = await self._parser.parse(file_path, doc_id)

        # 2. 分块
        chunks = self._chunker.chunk_text(text, doc_id, metadata.file_type)
        if not chunks:
            return {"doc_id": doc_id, "filename": metadata.filename, "chunk_count": 0, "message": "文档内容为空"}

        # 3. 嵌入
        embeddings = await self._embedder.embed_chunks(chunks)

        # 4. 添加到向量存储
        await self._vector_store.add_chunks(chunks, embeddings)

        # 5. 更新元数据
        metadata.chunk_count = len(chunks)
        self._documents[doc_id] = metadata

        result = {
            "doc_id": doc_id,
            "filename": metadata.filename,
            "chunk_count": len(chunks),
            "message": f"成功加载 {metadata.filename}，共 {len(chunks)} 个文档块",
        }

        logger.info(f"文件加载完成: {metadata.filename}, {len(chunks)} 块")
        return result

    async def _load_directory_async(
        self, dir_path: str, extensions: Optional[List[str]] = None
    ) -> List[dict]:
        """异步加载目录"""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"不是有效目录: {dir_path}")

        extensions = extensions or [".pdf", ".md", ".markdown"]
        results = []

        for ext in extensions:
            for file_path in dir_path.glob(f"*{ext}"):
                try:
                    result = await self._load_file_async(str(file_path))
                    results.append(result)
                except Exception as e:
                    logger.error(f"加载文件失败 {file_path}: {e}")
                    results.append({
                        "doc_id": "",
                        "filename": file_path.name,
                        "chunk_count": 0,
                        "message": f"加载失败: {e}",
                    })

        return results

    async def _ask_async(self, question: str, deep: bool = False) -> ChatResponse:
        """异步问答"""
        if self.total_chunks == 0:
            return ChatResponse(answer="尚未加载任何文档，请先使用 load_file() 加载文档。")

        if deep:
            return await self._rag_engine.deep_chat(
                question,
                conversation_history=self._conversation_history or None,
            )
        else:
            return await self._rag_engine.quick_chat(
                question,
                conversation_history=self._conversation_history or None,
            )


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="本地文档问答 - 无需启动服务，直接读取 PDF/MD 文件进行问答",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python cli_qa.py --api-key sk-xxxxx 操作手册.pdf

  # 指定 API 地址和模型
  python cli_qa.py --api-key sk-xxxxx --base-url https://your-api.com/v1 --chat-model gpt-4o 操作手册.pdf

  # 加载多个文件
  python cli_qa.py --api-key sk-xxxxx 手册.pdf 技术文档.md

  # 加载整个目录
  python cli_qa.py --api-key sk-xxxxx ./docs/

  # 使用环境变量
  set OPENAI_API_KEY=sk-xxxxx
  set OPENAI_BASE_URL=https://api.openai.com/v1
  python cli_qa.py 操作手册.pdf
        """,
    )
    parser.add_argument("files", nargs="+", help="文件或目录路径（支持 PDF/MD）")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI API 地址")
    parser.add_argument("--chat-model", type=str, default="gpt-4o-mini", help="聊天模型")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3", help="嵌入模型")
    parser.add_argument("--chunk-size", type=int, default=512, help="分块大小")
    parser.add_argument("--top-k", type=int, default=5, help="检索结果数")
    parser.add_argument("--question", "-q", type=str, default=None, help="直接提问（不进入交互模式）")
    parser.add_argument("--deep", action="store_true", help="使用深度模式")

    args = parser.parse_args()

    # 构建 LocalQA
    try:
        qa = LocalQA(
            api_key=args.api_key,
            base_url=args.base_url,
            chat_model=args.chat_model,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            top_k=args.top_k,
        )
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)

    # 加载文件
    print("\n📂 正在加载文档...")
    for file_path in args.files:
        path = Path(file_path)
        if path.is_dir():
            results = qa.load_directory(file_path)
            for r in results:
                print(f"  {'✅' if r['chunk_count'] > 0 else '⚠️'} {r['filename']}: {r['message']}")
        else:
            try:
                result = qa.load_file(file_path)
                print(f"  ✅ {result['filename']}: {result['message']}")
            except Exception as e:
                print(f"  ❌ {file_path}: {e}")

    if qa.total_chunks == 0:
        print("\n❌ 没有成功加载任何文档")
        sys.exit(1)

    # 问答
    if args.question:
        # 单次问答模式
        answer = qa.ask(args.question, deep=args.deep)
        print(f"\n💬 回答:\n{answer}")
    else:
        # 交互模式
        qa.interactive()


if __name__ == "__main__":
    main()
