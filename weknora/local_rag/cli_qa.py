"""
无端口本地文档问答 - 直接读取 PDF/MD 文件进行问答

核心特性：
  ✅ 数据全部持久化 — 切块、向量索引、对话历史、文档元数据均自动存盘
  ✅ 增量加载 — 相同文件不重复嵌入，文件变更自动重新索引
  ✅ 对话历史 — 自动保存/恢复，支持跨会话连续对话
  ✅ 中文优先 — 分词、断句、提示词全面中文优化

使用方式一（Python 函数调用）：
    from cli_qa import LocalQA

    qa = LocalQA(
        api_key="sk-xxxxx",
        base_url="https://api.openai.com/v1",   # 可选
        chat_model="gpt-4o-mini",                # 可选
        embedding_model="text-embedding-3-small", # 可选
        workspace="./my_workspace",               # 可选，工作目录（默认 ./rag_workspace）
    )

    # 加载文档（支持 pdf 和 md，已加载的文件自动跳过）
    qa.load_file("操作手册.pdf")
    qa.load_file("技术文档.md")
    qa.load_directory("./docs/")  # 加载整个目录

    # 问答
    answer = qa.ask("设备故障码 E003 怎么处理？")
    print(answer)

    # 带来源的问答
    result = qa.ask_with_sources("安全操作规程有哪些？")
    print(result["answer"])
    print(result["sources"])

    # 交互式问答
    qa.interactive()

    # 退出时数据自动保存，下次启动自动恢复

使用方式二（命令行）：
    python cli_qa.py --api-key sk-xxxxx 操作手册.pdf
    python cli_qa.py --api-key sk-xxxxx --base-url https://your-api.com/v1 技术文档.md
    python cli_qa.py --api-key sk-xxxxx --chat-model gpt-4o 操作手册.pdf 技术文档.md

使用方式三（环境变量）：
    set OPENAI_API_KEY=sk-xxxxx
    set OPENAI_BASE_URL=https://api.openai.com/v1
    python cli_qa.py 操作手册.pdf

数据持久化结构（workspace 目录）：
    rag_workspace/
    ├── vector_store/           # FAISS 向量索引 + BM25 元数据
    │   ├── faiss.index
    │   └── metadata.pkl
    ├── chunks/                 # 文档切块缓存（JSON）
    │   └── {doc_id}.json
    ├── documents.json          # 文档注册表（元数据 + 文件哈希）
    ├── conversation.json       # 对话历史
    └── config.json             # 运行时配置快照
"""
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
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
from models.schemas import ChatResponse, Chunk, DocumentMetadata

logger = logging.getLogger(__name__)


class LocalQA:
    """
    本地文档问答 - 无需启动服务端口，数据全部持久化

    核心用法：
        qa = LocalQA(api_key="sk-xxxxx")
        qa.load_file("手册.pdf")
        answer = qa.ask("xxx怎么操作？")

    持久化策略：
        - 文档元数据 + 文件哈希 → documents.json（检测文件变更，避免重复嵌入）
        - 文档切块 → chunks/{doc_id}.json（可独立读取）
        - 向量索引 → vector_store/（FAISS + BM25 元数据）
        - 对话历史 → conversation.json（跨会话保持上下文）
        - 配置快照 → config.json（确保加载时配置一致）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        chat_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        embedding_model: str = "BAAI/bge-m3",
        embedding_dim: int = 1024,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        temperature: float = 0.3,
        workspace: str = "./rag_workspace",
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
            workspace: 工作目录，所有持久化数据存放于此
        """
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not api_key:
            raise ValueError(
                "请提供 api_key 参数或设置 OPENAI_API_KEY 环境变量\n"
                "  示例: qa = LocalQA(api_key='sk-xxxxx')\n"
                "  或:  set OPENAI_API_KEY=sk-xxxxx"
            )

        # 工作目录
        self._workspace = Path(workspace)
        self._workspace.mkdir(parents=True, exist_ok=True)
        (self._workspace / "chunks").mkdir(exist_ok=True)
        (self._workspace / "vector_store").mkdir(exist_ok=True)

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
        self._graph_builder = None  # 延迟初始化，在调用 build_graph() 时创建
        self._reranker = Reranker(self._config.llm, self._config.retriever, self._embedder)
        self._retriever = Retriever(
            self._config, self._vector_store, self._embedder, self._reranker,
            graph_builder=None,  # 初始为空，build_graph() 后会更新
        )
        self._rag_engine = RAGEngine(self._config)
        self._rag_engine.initialize(
            embedder=self._embedder,
            vector_store=self._vector_store,
            retriever=self._retriever,
            reranker=self._reranker,
        )

        # 运行时状态
        self._documents: Dict[str, dict] = {}       # doc_id -> {metadata, file_hash, abs_path, chunk_ids}
        self._conversation_history: List[dict] = []
        self._conversation_meta: dict = {"created_at": time.time(), "turn_count": 0}

        # 从磁盘恢复
        self._restore_state()

        logger.info(
            f"LocalQA 初始化完成 | 工作目录: {self._workspace} | "
            f"文档: {len(self._documents)} | 块: {self._vector_store.total_chunks} | "
            f"对话轮次: {self._conversation_meta.get('turn_count', 0)}"
        )

    # ================================================================
    # 公共属性
    # ================================================================

    @property
    def loaded_documents(self) -> List[dict]:
        """已加载的文档列表"""
        return [
            {
                "doc_id": info.get("doc_id", ""),
                "filename": info.get("filename", ""),
                "file_type": info.get("file_type", ""),
                "title": info.get("title", ""),
                "chunk_count": info.get("chunk_count", 0),
                "file_path": info.get("abs_path", ""),
                "loaded_at": info.get("loaded_at", 0),
            }
            for info in self._documents.values()
        ]

    @property
    def total_chunks(self) -> int:
        """已加载的文档块总数"""
        return self._vector_store.total_chunks

    @property
    def conversation_history(self) -> List[dict]:
        """当前对话历史"""
        return list(self._conversation_history)

    @property
    def workspace_path(self) -> str:
        """工作目录路径"""
        return str(self._workspace.resolve())

    # ================================================================
    # 公共方法 - 文档加载
    # ================================================================

    def load_file(self, file_path: str, force_reindex: bool = False) -> dict:
        """
        加载单个文件（同步接口）

        持久化行为：
        - 文件未变更 → 直接从缓存恢复切块和向量，跳过嵌入
        - 文件已变更 → 重新解析、分块、嵌入，更新索引
        - force_reindex=True → 强制重新索引

        Args:
            file_path: 文件路径
            force_reindex: 是否强制重新索引

        Returns:
            加载结果 {"doc_id": ..., "filename": ..., "chunk_count": ..., "from_cache": ...}
        """
        return self._run_async(self._load_file_async(file_path, force_reindex))

    def load_directory(self, dir_path: str, extensions: Optional[List[str]] = None,
                       force_reindex: bool = False) -> List[dict]:
        """
        加载目录下的所有文档（同步接口）

        Args:
            dir_path: 目录路径
            extensions: 文件扩展名过滤，默认 [".pdf", ".md", ".markdown"]
            force_reindex: 是否强制重新索引

        Returns:
            加载结果列表
        """
        return self._run_async(self._load_directory_async(dir_path, extensions, force_reindex))

    def remove_document(self, doc_id: str) -> bool:
        """
        移除已加载的文档（同步接口）

        从向量索引、切块缓存、文档注册表中移除

        Args:
            doc_id: 文档 ID

        Returns:
            是否移除成功
        """
        return self._run_async(self._remove_document_async(doc_id))

    # ================================================================
    # 公共方法 - 知识图谱与图增强检索
    # ================================================================

    def build_graph(self, doc_ids: Optional[List[str]] = None) -> dict:
        """
        构建知识图谱（同步接口）

        构建后自动开启图增强检索。图谱会持久化到工作目录。

        Args:
            doc_ids: 文档 ID 列表（None=使用所有已加载文档）

        Returns:
            {"entities": int, "relationships": int, "graph_enabled": True}
        """
        from wiki.graph_builder import KnowledgeGraphBuilder

        if self._graph_builder is None:
            self._graph_builder = KnowledgeGraphBuilder(self._config)

        # 收集文档块
        if doc_ids is None:
            doc_ids = list(self._documents.keys())

        chunks = []
        for doc_id in doc_ids:
            chunks.extend(self._vector_store.get_chunks_by_doc_id(doc_id))

        if not chunks:
            return {"entities": 0, "relationships": 0, "graph_enabled": False,
                    "message": "没有可用的文档块，请先加载文档"}

        # 构建图谱
        kg = self._run_async(self._graph_builder.build_graph(chunks))

        # 将 graph_builder 注入 retriever
        self._retriever.graph_builder = self._graph_builder

        # 自动开启图增强检索
        self._config.retriever.graph_enabled = True

        # 持久化图谱
        graph_path = self._workspace / "knowledge_graph.json"
        with open(graph_path, "w", encoding="utf-8") as f:
            f.write(kg.model_dump_json(indent=2))

        mermaid = self._graph_builder.to_mermaid()
        mermaid_path = self._workspace / "knowledge_graph.mmd"
        with open(mermaid_path, "w", encoding="utf-8") as f:
            f.write(mermaid)

        result = {
            "entities": len(kg.entities),
            "relationships": len(kg.relationships),
            "graph_enabled": True,
            "message": f"图谱构建完成: {len(kg.entities)} 个实体, {len(kg.relationships)} 条关系，图增强检索已自动开启",
        }
        logger.info(result["message"])
        return result

    def enable_graph_search(self, enabled: bool = True):
        """
        开启/关闭图增强检索

        前提：需先调用 build_graph() 构建知识图谱，或已有持久化的图谱（启动时自动恢复）

        Args:
            enabled: True=开启, False=关闭
        """
        if enabled and (self._graph_builder is None or not self._graph_builder._entity_map):
            logger.warning("知识图谱尚未构建，请先调用 build_graph()")
            return

        self._config.retriever.graph_enabled = enabled
        if enabled and self._graph_builder:
            self._retriever.graph_builder = self._graph_builder

        logger.info(f"图增强检索已{'开启' if enabled else '关闭'}")

    def load_graph(self, graph_path: Optional[str] = None) -> dict:
        """
        从磁盘加载已保存的知识图谱（同步接口）

        无需重新调用 LLM 构建图谱，直接从 knowledge_graph.json 恢复。
        加载后自动开启图增强检索。

        Args:
            graph_path: 图谱文件路径（None=使用工作目录下的 knowledge_graph.json）

        Returns:
            {"entities": int, "relationships": int, "graph_enabled": True}
        """
        path = Path(graph_path) if graph_path else self._workspace / "knowledge_graph.json"

        if not path.exists():
            return {
                "entities": 0, "relationships": 0, "graph_enabled": False,
                "message": f"图谱文件不存在: {path}",
            }

        try:
            self._restore_graph()

            if self._graph_builder and self._graph_builder._entity_map:
                entity_count = len(self._graph_builder._entity_map)
                rel_count = self._graph_builder._graph.number_of_edges()
                return {
                    "entities": entity_count,
                    "relationships": rel_count,
                    "graph_enabled": True,
                    "message": f"图谱加载完成: {entity_count} 个实体, {rel_count} 条关系，图增强检索已开启",
                }
            else:
                return {
                    "entities": 0, "relationships": 0, "graph_enabled": False,
                    "message": "图谱文件为空或格式无效",
                }
        except Exception as e:
            return {
                "entities": 0, "relationships": 0, "graph_enabled": False,
                "message": f"图谱加载失败: {e}",
            }

    # ================================================================
    # 公共方法 - 问答
    # ================================================================

    def ask(self, question: str, deep: bool = False) -> str:
        """
        提问并获取回答（同步接口）

        每次问答后自动保存对话历史。

        Args:
            question: 问题
            deep: 是否使用深度模式（查询理解+重排，更准但更慢）

        Returns:
            回答文本
        """
        result = self._run_async(self._ask_async(question, deep=deep))

        # 记录对话历史
        self._conversation_history.append({"role": "user", "content": question, "timestamp": time.time()})
        self._conversation_history.append({"role": "assistant", "content": result.answer, "timestamp": time.time()})
        self._conversation_meta["turn_count"] = self._conversation_meta.get("turn_count", 0) + 1

        # 保留最近 20 轮
        if len(self._conversation_history) > 40:
            self._conversation_history = self._conversation_history[-40:]

        # 持久化对话历史
        self._save_conversation()

        return result.answer

    def ask_with_sources(self, question: str, deep: bool = False) -> Dict:
        """
        提问并获取带来源引用的回答（同步接口）

        Args:
            question: 问题
            deep: 是否使用深度模式

        Returns:
            {"answer": str, "sources": [...], "turn_count": int}
        """
        result = self._run_async(self._ask_async(question, deep=deep))

        # 记录对话历史
        self._conversation_history.append({"role": "user", "content": question, "timestamp": time.time()})
        self._conversation_history.append({"role": "assistant", "content": result.answer, "timestamp": time.time()})
        self._conversation_meta["turn_count"] = self._conversation_meta.get("turn_count", 0) + 1

        if len(self._conversation_history) > 40:
            self._conversation_history = self._conversation_history[-40:]

        self._save_conversation()

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
            "turn_count": self._conversation_meta.get("turn_count", 0),
        }

    # ================================================================
    # 公共方法 - 对话管理
    # ================================================================

    def clear_history(self):
        """清空对话历史（同时删除磁盘文件）"""
        self._conversation_history = []
        self._conversation_meta["turn_count"] = 0
        conv_file = self._workspace / "conversation.json"
        if conv_file.exists():
            conv_file.unlink()
        logger.info("对话历史已清空")

    def get_history(self, last_n: int = 10) -> List[dict]:
        """
        获取最近 N 轮对话历史

        Args:
            last_n: 轮数（1轮 = 1问+1答）

        Returns:
            对话历史列表
        """
        return self._conversation_history[-(last_n * 2):]

    # ================================================================
    # 公共方法 - 数据管理
    # ================================================================

    def save(self):
        """手动保存所有状态到磁盘"""
        self._run_async(self._save_all())

    def reset(self):
        """
        重置工作空间（删除所有持久化数据）

        ⚠️ 这将删除所有文档索引、切块缓存和对话历史！
        """
        import shutil
        if self._workspace.exists():
            shutil.rmtree(self._workspace)
        self._workspace.mkdir(parents=True, exist_ok=True)
        (self._workspace / "chunks").mkdir(exist_ok=True)
        (self._workspace / "vector_store").mkdir(exist_ok=True)

        self._documents = {}
        self._conversation_history = []
        self._conversation_meta = {"created_at": time.time(), "turn_count": 0}
        self._vector_store = VectorStore(self._config.retriever, dim=self._config.llm.embedding_dim)
        logger.info("工作空间已重置")

    def status(self) -> dict:
        """获取当前系统状态"""
        graph_info = {}
        if self._graph_builder and self._graph_builder._entity_map:
            graph_info = {
                "entities": len(self._graph_builder._entity_map),
                "relationships": self._graph_builder._graph.number_of_edges(),
                "graph_enabled": self._config.retriever.graph_enabled,
            }
        return {
            "workspace": str(self._workspace.resolve()),
            "documents": len(self._documents),
            "total_chunks": self.total_chunks,
            "conversation_turns": self._conversation_meta.get("turn_count", 0),
            "conversation_messages": len(self._conversation_history),
            "vector_store_size": self._vector_store._index.ntotal if self._vector_store._index else 0,
            "graph": graph_info if graph_info else {"graph_enabled": self._config.retriever.graph_enabled, "built": False},
            "config": {
                "chat_model": self._config.llm.chat_model,
                "embedding_model": self._config.llm.embedding_model,
                "chunk_size": self._config.chunker.chunk_size,
                "top_k": self._config.retriever.rerank_top_k,
                "graph_enabled": self._config.retriever.graph_enabled,
            },
        }

    def interactive(self, prompt_text: str = "❓ 请输入问题（q=退出, c=清空历史, d=深度模式, s=状态）: "):
        """
        交互式问答模式

        退出时自动保存所有状态。
        """
        status = self.status()
        print("\n" + "=" * 60)
        print("  📚 本地文档问答系统（数据自动持久化）")
        print(f"  工作目录: {status['workspace']}")
        print(f"  已加载文档: {status['documents']} 个 | 文档块: {status['total_chunks']} 个")
        print(f"  对话轮次: {status['conversation_turns']}")
        for doc in self.loaded_documents:
            print(f"    - {doc['filename']} ({doc['chunk_count']} 块)")
        print("  命令: q=退出, c=清空历史, d=深度模式, s=状态, r=重置")
        print("=" * 60 + "\n")

        deep_mode = False
        while True:
            try:
                question = input(prompt_text).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n💾 正在保存数据...")
                self.save()
                print("再见！")
                break

            if not question:
                continue
            if question.lower() == "q":
                print("💾 正在保存数据...")
                self.save()
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
            if question.lower() == "s":
                st = self.status()
                print(f"\n📊 系统状态:")
                print(f"  文档数: {st['documents']}")
                print(f"  文档块: {st['total_chunks']}")
                print(f"  对话轮次: {st['conversation_turns']}")
                print(f"  向量索引大小: {st['vector_store_size']}")
                print(f"  工作目录: {st['workspace']}\n")
                continue
            if question.lower() == "r":
                confirm = input("⚠️  确定要重置工作空间吗？(y/N): ").strip().lower()
                if confirm == "y":
                    self.reset()
                    print("✅ 工作空间已重置\n")
                else:
                    print("已取消\n")
                continue

            try:
                result = self.ask_with_sources(question, deep=deep_mode)
                print(f"\n💬 回答:\n{result['answer']}")

                if result["sources"]:
                    print(f"\n📎 参考来源:")
                    for i, src in enumerate(result["sources"][:3]):
                        section = src["section"] or src["doc_id"]
                        print(f"  [{i+1}] {section} (相关度: {src['score']})")
                print(f"  [对话轮次: {result['turn_count']}]")
                print()

            except Exception as e:
                print(f"\n❌ 出错: {e}\n")

    # ================================================================
    # 内部方法 - 异步调度
    # ================================================================

    def _run_async(self, coro):
        """在事件循环中运行异步协程"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已在异步环境中，创建新的线程运行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # ================================================================
    # 内部方法 - 文件加载
    # ================================================================

    async def _load_file_async(self, file_path: str, force_reindex: bool = False) -> dict:
        """异步加载文件（带缓存检测）"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        if ext not in self._parser.supported_extensions():
            raise ValueError(
                f"不支持的文件类型: {ext}，"
                f"当前支持: {self._parser.supported_extensions()}"
            )

        abs_path = str(path.resolve())
        file_hash = self._compute_file_hash(abs_path)

        # ---- 检查缓存 ----
        if not force_reindex:
            cached_doc = self._find_doc_by_path(abs_path)
            if cached_doc and cached_doc.get("file_hash") == file_hash:
                # 文件未变更，检查向量索引中是否已有
                doc_id = cached_doc["doc_id"]
                existing_chunks = self._vector_store.get_chunks_by_doc_id(doc_id)
                if existing_chunks:
                    logger.info(f"文件未变更，跳过重新索引: {path.name}")
                    return {
                        "doc_id": doc_id,
                        "filename": cached_doc["filename"],
                        "chunk_count": len(existing_chunks),
                        "from_cache": True,
                        "message": f"文件未变更，从缓存加载 {path.name}，共 {len(existing_chunks)} 个文档块",
                    }
                else:
                    # 有注册信息但向量索引中没有，需要重新索引
                    logger.info(f"缓存不完整，重新索引: {path.name}")

            # 尝试从切块缓存恢复
            cached_doc = self._find_doc_by_path(abs_path)
            if cached_doc and cached_doc.get("file_hash") == file_hash:
                doc_id = cached_doc["doc_id"]
                cached_chunks = self._load_chunks_from_disk(doc_id)
                if cached_chunks:
                    # 从缓存的切块恢复，但仍需重新嵌入
                    logger.info(f"从切块缓存恢复: {path.name}, {len(cached_chunks)} 块")
                    embeddings = await self._embedder.embed_chunks(cached_chunks)
                    await self._vector_store.add_chunks(cached_chunks, embeddings)
                    await self._save_vector_store()

                    return {
                        "doc_id": doc_id,
                        "filename": cached_doc["filename"],
                        "chunk_count": len(cached_chunks),
                        "from_cache": True,
                        "message": f"从切块缓存恢复 {path.name}，共 {len(cached_chunks)} 个文档块",
                    }

        # ---- 文件变更或首次加载，需要重新索引 ----
        # 如果旧版本存在，先移除
        old_doc = self._find_doc_by_path(abs_path)
        if old_doc:
            await self._remove_document_async(old_doc["doc_id"])

        doc_id = DocumentMetadata().doc_id

        # 1. 解析
        text, metadata = await self._parser.parse(abs_path, doc_id)

        # 2. 分块
        chunks = self._chunker.chunk_text(text, doc_id, metadata.file_type)
        if not chunks:
            return {"doc_id": doc_id, "filename": metadata.filename, "chunk_count": 0,
                    "from_cache": False, "message": "文档内容为空"}

        # 3. 嵌入
        embeddings = await self._embedder.embed_chunks(chunks)

        # 4. 添加到向量存储
        await self._vector_store.add_chunks(chunks, embeddings)

        # 5. 保存切块到磁盘
        self._save_chunks_to_disk(doc_id, chunks)

        # 6. 保存文档注册信息
        self._documents[doc_id] = {
            "doc_id": doc_id,
            "filename": metadata.filename,
            "file_type": metadata.file_type,
            "title": metadata.title,
            "chunk_count": len(chunks),
            "abs_path": abs_path,
            "file_hash": file_hash,
            "loaded_at": time.time(),
            "chunk_ids": [c.chunk_id for c in chunks],
        }
        self._save_documents()

        # 7. 保存向量索引
        await self._save_vector_store()

        # 8. 保存配置快照
        self._save_config_snapshot()

        result = {
            "doc_id": doc_id,
            "filename": metadata.filename,
            "chunk_count": len(chunks),
            "from_cache": False,
            "message": f"成功加载 {metadata.filename}，共 {len(chunks)} 个文档块",
        }

        logger.info(f"文件加载完成: {metadata.filename}, {len(chunks)} 块")
        return result

    async def _load_directory_async(
        self, dir_path: str, extensions: Optional[List[str]] = None,
        force_reindex: bool = False,
    ) -> List[dict]:
        """异步加载目录"""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"不是有效目录: {dir_path}")

        extensions = extensions or [".pdf", ".md", ".markdown"]
        results = []

        for ext in extensions:
            for file_path in sorted(dir_path.glob(f"*{ext}")):
                try:
                    result = await self._load_file_async(str(file_path), force_reindex)
                    results.append(result)
                except Exception as e:
                    logger.error(f"加载文件失败 {file_path}: {e}")
                    results.append({
                        "doc_id": "", "filename": file_path.name,
                        "chunk_count": 0, "from_cache": False,
                        "message": f"加载失败: {e}",
                    })

        return results

    async def _remove_document_async(self, doc_id: str) -> bool:
        """异步移除文档"""
        if doc_id not in self._documents:
            return False

        # 从向量存储移除
        await self._vector_store.delete_by_doc_id(doc_id)

        # 删除切块缓存
        chunk_file = self._workspace / "chunks" / f"{doc_id}.json"
        if chunk_file.exists():
            chunk_file.unlink()

        # 从文档注册表移除
        del self._documents[doc_id]
        self._save_documents()

        # 保存向量索引
        await self._save_vector_store()

        logger.info(f"文档已移除: {doc_id}")
        return True

    async def _ask_async(self, question: str, deep: bool = False) -> ChatResponse:
        """异步问答"""
        if self.total_chunks == 0:
            return ChatResponse(answer="尚未加载任何文档，请先使用 load_file() 加载文档。")

        # 深度模式下根据配置决定是否启用图增强检索
        use_graph = self._config.retriever.graph_enabled and self._graph_builder is not None

        if deep:
            return await self._rag_engine.deep_chat(
                question,
                conversation_history=self._conversation_history[-20:] or None,
                use_graph=use_graph,
            )
        else:
            return await self._rag_engine.quick_chat(
                question,
                conversation_history=self._conversation_history[-20:] or None,
            )

    # ================================================================
    # 内部方法 - 持久化
    # ================================================================

    def _restore_state(self):
        """从磁盘恢复所有状态"""
        # 1. 恢复文档注册表
        self._load_documents()

        # 2. 恢复向量索引
        vs_dir = self._workspace / "vector_store"
        if (vs_dir / "faiss.index").exists():
            try:
                self._run_async(self._vector_store.load(str(vs_dir)))
                logger.info(f"向量索引已恢复: {self._vector_store.total_chunks} 个块")
            except Exception as e:
                logger.warning(f"向量索引恢复失败: {e}")

        # 3. 恢复对话历史
        self._load_conversation()

        # 4. 恢复知识图谱（如果之前已构建）
        self._restore_graph()

        # 5. 校验文档注册表与向量索引一致性
        self._validate_consistency()

    def _restore_graph(self):
        """从磁盘恢复知识图谱（如果之前已构建并保存）"""
        graph_path = self._workspace / "knowledge_graph.json"
        if not graph_path.exists():
            return

        try:
            from wiki.graph_builder import KnowledgeGraphBuilder
            from models.schemas import KnowledgeGraph

            # 读取图谱数据
            with open(graph_path, "r", encoding="utf-8") as f:
                kg_json = f.read()

            kg = KnowledgeGraph.model_validate_json(kg_json)
            if not kg.entities:
                return

            # 重建 graph_builder
            self._graph_builder = KnowledgeGraphBuilder(self._config)

            # 恢复实体映射
            self._graph_builder._entity_map = kg.entities

            # 恢复 chunk_entities 映射
            from collections import defaultdict
            self._graph_builder._chunk_entities = defaultdict(list)
            for title, entity in kg.entities.items():
                for cid in entity.source_chunk_ids:
                    self._graph_builder._chunk_entities[cid].append(title)

            # 恢复 NetworkX 图结构
            self._graph_builder._build_networkx_graph(kg.relationships)

            # 将 graph_builder 注入 retriever
            self._retriever.graph_builder = self._graph_builder

            # 自动开启图增强检索
            self._config.retriever.graph_enabled = True

            logger.info(
                f"知识图谱已恢复: {len(kg.entities)} 个实体, "
                f"{len(kg.relationships)} 条关系, 图增强检索已自动开启"
            )
        except Exception as e:
            logger.warning(f"知识图谱恢复失败: {e}")

    def _validate_consistency(self):
        """校验文档注册表与向量索引的一致性"""
        registered_docs = set(self._documents.keys())
        indexed_docs = set()

        for chunk in self._vector_store._chunks:
            indexed_docs.add(chunk.doc_id)

        # 清理注册表中没有向量数据的文档
        stale_docs = registered_docs - indexed_docs
        for doc_id in stale_docs:
            logger.warning(f"文档 {doc_id} 注册信息存在但向量索引中无数据，移除注册")
            del self._documents[doc_id]

        if stale_docs:
            self._save_documents()

    def _save_documents(self):
        """保存文档注册表"""
        doc_file = self._workspace / "documents.json"
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False, indent=2)

    def _load_documents(self):
        """加载文档注册表"""
        doc_file = self._workspace / "documents.json"
        if doc_file.exists():
            try:
                with open(doc_file, "r", encoding="utf-8") as f:
                    self._documents = json.load(f)
                logger.info(f"文档注册表已恢复: {len(self._documents)} 个文档")
            except Exception as e:
                logger.warning(f"文档注册表加载失败: {e}")
                self._documents = {}

    def _save_chunks_to_disk(self, doc_id: str, chunks: List[Chunk]):
        """保存文档切块到磁盘"""
        chunk_file = self._workspace / "chunks" / f"{doc_id}.json"
        data = [c.model_dump() for c in chunks]
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"切块已保存: {doc_id}, {len(chunks)} 块")

    def _load_chunks_from_disk(self, doc_id: str) -> Optional[List[Chunk]]:
        """从磁盘加载文档切块"""
        chunk_file = self._workspace / "chunks" / f"{doc_id}.json"
        if not chunk_file.exists():
            return None
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [Chunk(**c) for c in data]
        except Exception as e:
            logger.warning(f"切块缓存加载失败 ({doc_id}): {e}")
            return None

    def _save_conversation(self):
        """保存对话历史"""
        conv_file = self._workspace / "conversation.json"
        data = {
            "meta": self._conversation_meta,
            "messages": self._conversation_history,
        }
        with open(conv_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_conversation(self):
        """加载对话历史"""
        conv_file = self._workspace / "conversation.json"
        if conv_file.exists():
            try:
                with open(conv_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._conversation_meta = data.get("meta", {"created_at": time.time(), "turn_count": 0})
                self._conversation_history = data.get("messages", [])
                logger.info(f"对话历史已恢复: {self._conversation_meta.get('turn_count', 0)} 轮")
            except Exception as e:
                logger.warning(f"对话历史加载失败: {e}")

    async def _save_vector_store(self):
        """保存向量索引"""
        vs_dir = self._workspace / "vector_store"
        await self._vector_store.save(str(vs_dir))

    def _save_config_snapshot(self):
        """保存配置快照（用于检测配置变更）"""
        config_file = self._workspace / "config.json"
        data = {
            "chat_model": self._config.llm.chat_model,
            "embedding_model": self._config.llm.embedding_model,
            "embedding_dim": self._config.llm.embedding_dim,
            "chunk_size": self._config.chunker.chunk_size,
            "chunk_overlap": self._config.chunker.chunk_overlap,
            "saved_at": time.time(),
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def _save_all(self):
        """保存所有状态"""
        self._save_documents()
        self._save_conversation()
        await self._save_vector_store()
        self._save_config_snapshot()
        logger.info("所有数据已保存到磁盘")

    # ================================================================
    # 内部方法 - 工具函数
    # ================================================================

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """计算文件内容哈希（SHA256）"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()[:32]

    def _find_doc_by_path(self, abs_path: str) -> Optional[dict]:
        """根据绝对路径查找已注册的文档"""
        for doc_info in self._documents.values():
            if doc_info.get("abs_path") == abs_path:
                return doc_info
        return None


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

  # 指定工作目录（持久化数据存放位置）
  python cli_qa.py --api-key sk-xxxxx --workspace ./my_data 操作手册.pdf

  # 强制重新索引
  python cli_qa.py --api-key sk-xxxxx --force-reindex 操作手册.pdf
        """,
    )
    parser.add_argument("files", nargs="+", help="文件或目录路径（支持 PDF/MD）")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI API 地址")
    parser.add_argument("--chat-model", type=str, default="gpt-4o-mini", help="聊天模型")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small", help="嵌入模型")
    parser.add_argument("--chunk-size", type=int, default=512, help="分块大小")
    parser.add_argument("--top-k", type=int, default=5, help="检索结果数")
    parser.add_argument("--workspace", type=str, default="./rag_workspace", help="工作目录")
    parser.add_argument("--force-reindex", action="store_true", help="强制重新索引所有文件")
    parser.add_argument("--question", "-q", type=str, default=None, help="直接提问（不进入交互模式）")
    parser.add_argument("--deep", action="store_true", help="使用深度模式")
    parser.add_argument("--reset", action="store_true", help="重置工作空间")

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
            workspace=args.workspace,
        )
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)

    # 重置
    if args.reset:
        qa.reset()
        print("✅ 工作空间已重置")

    # 加载文件
    print(f"\n📂 正在加载文档（工作目录: {qa.workspace_path}）...")
    for file_path in args.files:
        path = Path(file_path)
        if path.is_dir():
            results = qa.load_directory(file_path, force_reindex=args.force_reindex)
            for r in results:
                icon = "✅" if r["chunk_count"] > 0 else "⚠️"
                cache_tag = " [缓存]" if r.get("from_cache") else " [新索引]"
                print(f"  {icon}{cache_tag} {r['filename']}: {r['message']}")
        else:
            try:
                result = qa.load_file(file_path, force_reindex=args.force_reindex)
                cache_tag = " [缓存]" if result.get("from_cache") else " [新索引]"
                print(f"  ✅{cache_tag} {result['filename']}: {result['message']}")
            except Exception as e:
                print(f"  ❌ {file_path}: {e}")

    if qa.total_chunks == 0:
        print("\n❌ 没有成功加载任何文档")
        sys.exit(1)

    # 问答
    if args.question:
        answer = qa.ask(args.question, deep=args.deep)
        print(f"\n💬 回答:\n{answer}")
        qa.save()
    else:
        qa.interactive()


if __name__ == "__main__":
    main()
