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

    # 快速问答
    answer = qa.ask("设备故障码 E003 怎么处理？")
    print(answer)

    # 带来源的问答
    result = qa.ask_with_sources("安全操作规程有哪些？")
    print(result["answer"])
    print(result["sources"])

    # ReAct Agent 模式（多步推理，适合复杂问题）
    result = qa.ask_agent("设备A的故障码E003和安全操作规程有什么关联？")
    print(result["answer"])
    for step in result["steps"]:
        print(f"  思考: {step['thought'][:100]}")

    # 构建知识图谱
    graph = qa.build_knowledge_graph()
    print(f"实体数: {graph['entities']}, 关系数: {graph['relationships']}")
    print(graph['mermaid'])  # Mermaid 可视化语法

    # 生成 Wiki 知识库
    wiki = qa.generate_wiki(granularity="standard")
    print(f"创建了 {wiki['pages_created']} 个页面")

    # Wiki 模式问答
    result = qa.ask_wiki("安全操作有哪些注意事项？")
    print(result["answer"])

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
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import AppConfig, LLMConfig, RetrieverConfig, ChunkerConfig
from core.chunker import TextChunker
from core.doc_router import DocRouter
from core.document_parser import DocumentParser
from core.embedder import Embedder
from core.rag_engine import RAGEngine
from core.reranker import Reranker
from core.retriever import Retriever
from core.vector_store import VectorStore
from agent.engine import AgentEngine
from wiki.graph_builder import KnowledgeGraphBuilder
from wiki.ingest import WikiIngest
from wiki.page_manager import WikiPageManager
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
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        temperature: float = 0.3,
        data_dir: str = "./local_qa_data",
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
            data_dir: 数据持久化目录（保存向量索引、文档元数据、路由注册表）
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
        self._agent_engine = AgentEngine(self._config)
        self._agent_engine.register_knowledge_tools(
            self._retriever, self._vector_store, doc_router=None  # 初始不传，后面 _update_agent_router 会更新
        )

        # Wiki 管理器
        wiki_dir = os.path.join(data_dir, "wiki_output")
        self._wiki_manager = WikiPageManager(wiki_dir)
        self._wiki_ingest = WikiIngest(self._config, self._wiki_manager)

        # 知识图谱构建器
        self._graph_builder = KnowledgeGraphBuilder(self._config)

        # 数据持久化目录
        self._data_dir = data_dir
        self._vector_store_dir = os.path.join(data_dir, "vector_store")

        # 文档路由器
        self._doc_router = DocRouter()

        # 文档记录
        self._documents: Dict[str, DocumentMetadata] = {}
        self._conversation_history: List[dict] = []

        # 从磁盘恢复状态（向量索引、文档元数据、路由注册表）
        self._restore_state()

        logger.info(f"LocalQA 初始化完成（已加载 {len(self._documents)} 个文档，{self.total_chunks} 个文档块）")

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

    def ask(self, question: str, deep: bool = False, use_graph: bool = False) -> str:
        """
        提问并获取回答（同步接口）

        Args:
            question: 问题
            deep: 是否使用深度模式（查询理解+重排，更准但更慢）
            use_graph: 是否启用图谱增强检索（需先调用 build_knowledge_graph()）

        Returns:
            回答文本
        """
        result = asyncio.get_event_loop().run_until_complete(
            self._ask_async(question, deep=deep, use_graph=use_graph)
        )

        # 记录对话历史
        self._conversation_history.append({"role": "user", "content": question})
        self._conversation_history.append({"role": "assistant", "content": result.answer})

        # 保留最近 10 轮
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        return result.answer

    def ask_with_sources(self, question: str, deep: bool = False, use_graph: bool = False) -> Dict:
        """
        提问并获取带来源引用的回答（同步接口）

        Args:
            question: 问题
            deep: 是否使用深度模式
            use_graph: 是否启用图谱增强检索（需先调用 build_knowledge_graph()）

        Returns:
            {"answer": str, "sources": [{"content": ..., "score": ..., "doc_id": ...}]}
        """
        result = asyncio.get_event_loop().run_until_complete(
            self._ask_async(question, deep=deep, use_graph=use_graph)
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

    def set_aliases(self, aliases: Dict[str, str]):
        """
        设置文档别名/同义词映射（用于文档路由）

        当用户查询中使用了与文档名不同的称呼时，通过别名映射可以正确路由。

        Args:
            aliases: {别名: 标准名} 映射

        示例:
            qa.set_aliases({
                "设备A": "XX型设备操作手册",
                "安全规程": "安全操作规程",
            })
        """
        self._doc_router.set_aliases(aliases)

    def ask_agent(self, question: str) -> Dict:
        """
        ReAct Agent 模式问答（同步接口）

        Agent 模式会进行多步推理：思考 → 检索 → 分析 → 再检索 → 综合回答。
        适合需要多步推理、交叉验证的复杂问题。

        流程：
        1. 文档路由预筛选：根据问题关键词匹配文档
        2. Agent 多步推理：Think → Act → Observe 循环
        3. LLM 综合整理：基于检索内容生成最终答案

        Args:
            question: 问题

        Returns:
            {
                "answer": str,              # 最终答案
                "steps": [{"thought": ..., "tools": [...]}],  # 推理步骤
                "routed_docs": [...],        # 预筛选匹配的文档
            }

        示例:
            result = qa.ask_agent("设备A的故障码E003和安全操作规程有什么关联？")
            print(result["answer"])
        """
        result = asyncio.get_event_loop().run_until_complete(
            self._ask_agent_async(question)
        )

        # 记录对话历史
        self._conversation_history.append({"role": "user", "content": question})
        self._conversation_history.append({"role": "assistant", "content": result.answer})

        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        # 提取路由信息
        doc_ids = self._route_query(question)
        routed_docs = []
        if doc_ids:
            routed_docs = [
                self._documents[did].filename
                for did in doc_ids if did in self._documents
            ]

        # 格式化推理步骤
        steps = []
        for step in result.agent_steps:
            step_info = {
                "thought": step.thought[:200] if step.thought else "",
                "tools": [
                    {
                        "name": tc.name,
                        "arguments": str(tc.arguments)[:100],
                    }
                    for tc in step.tool_calls
                ],
            }
            steps.append(step_info)

        return {
            "answer": result.answer,
            "steps": steps,
            "routed_docs": routed_docs,
        }

    def build_knowledge_graph(self, doc_ids: Optional[List[str]] = None) -> Dict:
        """
        构建知识图谱（同步接口）

        从已加载的文档中提取实体和关系，构建可视化知识图谱。
        图谱可用于图增强检索（GraphRAG），也支持导出 Mermaid 可视化。

        流程：
        1. 收集指定文档的文档块
        2. 并发提取实体（人名、组织、概念、产品等）
        3. 并发提取关系（实体间的关联）
        4. 计算权重（PMI × 0.6 + Strength × 0.4）
        5. 构建 NetworkX 图结构

        Args:
            doc_ids: 文档 ID 列表（None 表示所有文档）

        Returns:
            {
                "entities": int,        # 实体数量
                "relationships": int,   # 关系数量
                "mermaid": str,         # Mermaid 可视化语法
                "graph_json": str,      # 图谱 JSON 文件路径
            }

        示例:
            result = qa.build_knowledge_graph()
            print(f"实体数: {result['entities']}, 关系数: {result['relationships']}")
            print(result['mermaid'])  # Mermaid 语法
        """
        return asyncio.get_event_loop().run_until_complete(
            self._build_knowledge_graph_async(doc_ids)
        )

    def generate_wiki(
        self,
        doc_ids: Optional[List[str]] = None,
        granularity: str = "standard",
    ) -> Dict:
        """
        生成 Wiki 知识库（同步接口）

        从已加载的文档生成结构化的 Wiki 知识库，包含摘要页、实体页、概念页、索引页。

        流程（Map-Reduce-Post）：
        1. MAP 阶段：每个文档生成摘要页 + 提取实体和概念
        2. REDUCE 阶段：每个实体/概念创建或更新 Wiki 页面
        3. POST 阶段：发布草稿 → 重建索引 → 注入跨页面链接

        粒度控制（granularity）：
        - focused: 少量核心实体/概念（5-10个）
        - standard: 适度提取（10-30个）
        - exhaustive: 尽可能提取所有实体/概念

        Args:
            doc_ids: 文档 ID 列表（None 表示所有文档）
            granularity: 提取粒度，focused/standard/exhaustive

        Returns:
            {
                "pages_created": int,    # 新创建的页面数
                "pages_updated": int,    # 更新的页面数
                "links_injected": int,   # 注入的跨页面链接数
                "total_pages": int,      # Wiki 总页面数
                "message": str,          # 完成信息
            }

        示例:
            result = qa.generate_wiki(granularity="standard")
            print(f"创建了 {result['pages_created']} 个页面")
        """
        return asyncio.get_event_loop().run_until_complete(
            self._generate_wiki_async(doc_ids, granularity)
        )

    def ask_wiki(self, question: str) -> Dict:
        """
        Wiki 模式问答（同步接口）

        在 Wiki 知识库中搜索并回答问题。先确定文档范围，再在匹配文档的 Wiki 页面中检索。

        流程：
        1. 文档路由预筛选：根据问题关键词匹配文档名
        2. 在匹配文档的 Wiki 页面中搜索
        3. 基于搜索结果生成结构化回答

        Args:
            question: 问题

        Returns:
            {
                "answer": str,           # 回答
                "sources": [...],        # 参考的 Wiki 页面
                "routed_docs": [...],    # 预筛选匹配的文档
                "wiki_pages_searched": int,  # 搜索的页面数
            }

        示例:
            result = qa.ask_wiki("安全操作有哪些注意事项？")
            print(result["answer"])
        """
        return asyncio.get_event_loop().run_until_complete(
            self._ask_wiki_async(question)
        )

    def get_wiki_pages(self) -> List[Dict]:
        """
        获取所有 Wiki 页面列表（同步接口）

        Returns:
            Wiki 页面信息列表
        """
        pages = asyncio.get_event_loop().run_until_complete(
            self._wiki_manager.list_pages()
        )
        return [
            {
                "slug": p.slug,
                "title": p.title,
                "type": p.page_type.value,
                "status": p.status,
                "out_links": p.out_links,
                "source_doc_ids": p.source_doc_ids,
            }
            for p in pages
        ]

    def export_wiki(self) -> str:
        """
        导出 Wiki 为 Markdown 文件（同步接口）

        Returns:
            Markdown 文件目录路径
        """
        md_dir = asyncio.get_event_loop().run_until_complete(
            self._wiki_manager.export_all_markdown()
        )
        return md_dir

    def get_graph_mermaid(self) -> str:
        """
        获取知识图谱的 Mermaid 可视化语法

        Returns:
            Mermaid 语法字符串
        """
        return self._graph_builder.to_mermaid()

    def route_query(self, question: str) -> List[str]:
        """
        文档路由：根据查询关键词匹配相关文档（同步接口）

        返回匹配的 doc_id 列表。如果为空，表示无匹配，应回退到全量检索。

        Args:
            question: 用户查询

        Returns:
            匹配的 doc_id 列表
        """
        return self._doc_router.route(question)

    def clear_history(self):
        """清空对话历史"""
        self._conversation_history = []

    # ============================================================
    # 数据持久化
    # ============================================================

    def save(self, directory: Optional[str] = None):
        """
        手动保存所有状态到磁盘（同步接口）

        保存内容包括：
        - FAISS 向量索引 + BM25 索引
        - 文档元数据（文件名、标题、块数量等）
        - 文档路由注册表
        - 对话历史

        Args:
            directory: 保存目录（默认使用初始化时的 data_dir）
        """
        save_dir = directory or self._data_dir
        asyncio.get_event_loop().run_until_complete(self._save_graph_data(save_dir))

    def load(self, directory: Optional[str] = None):
        """
        手动从磁盘恢复状态（同步接口）

        Args:
            directory: 加载目录（默认使用初始化时的 data_dir）
        """
        load_dir = directory or self._data_dir
        asyncio.get_event_loop().run_until_complete(self._restore_state_async(load_dir))

    async def _save_graph_data(self, directory: Optional[str] = None):
        """
        异步保存知识图谱数据到磁盘

        Args:
            directory: 保存目录（默认使用 data_dir/knowledge_graph）
        """

        if self._graph_builder and self._graph_builder.has_data:
            save_dir = Path(directory or self._data_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            graph_dir = save_dir / "knowledge_graph"
            self._graph_builder.save(str(graph_dir))
            logger.info(f"知识图谱已保存到 {graph_dir}")

    async def _save_state_async(self, directory: Optional[str] = None):
        """
        异步保存状态到磁盘

        保存内容：
        1. 向量索引（FAISS + 元数据 + BM25）→ vector_store 子目录
        2. 文档元数据 → documents.json
        3. 文档路由注册表 → doc_router.json
        4. 对话历史 → conversation_history.json
        """
        save_dir = Path(directory or self._data_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. 保存向量索引
            vs_dir = save_dir / "vector_store"
            vs_dir.mkdir(parents=True, exist_ok=True)
            await self._vector_store.save(str(vs_dir))

            # 2. 保存文档元数据
            docs_data = {}
            for doc_id, meta in self._documents.items():
                docs_data[doc_id] = meta.model_dump()
            docs_path = save_dir / "documents.json"
            with open(docs_path, "w", encoding="utf-8") as f:
                json.dump(docs_data, f, ensure_ascii=False, indent=2)

            # 3. 保存文档路由注册表
            router_data = {}
            for doc_id, info in self._doc_router._docs.items():
                router_data[doc_id] = {
                    "doc_id": info.doc_id,
                    "filename": info.filename,
                    "title": info.title,
                    "keywords": info.keywords,
                    "metadata": info.metadata,
                }
            router_path = save_dir / "doc_router.json"
            with open(router_path, "w", encoding="utf-8") as f:
                json.dump(router_data, f, ensure_ascii=False, indent=2)

            # 4. 保存对话历史
            history_path = save_dir / "conversation_history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self._conversation_history, f, ensure_ascii=False, indent=2)

            # 5. 保存知识图谱数据（如果已构建）
            if self._graph_builder and self._graph_builder.has_data:
                graph_dir = save_dir / "knowledge_graph"
                self._graph_builder.save(str(graph_dir))
                logger.info(f"知识图谱已保存到 {graph_dir}")

            logger.info(f"状态已保存到 {save_dir}（{len(self._documents)} 个文档，{self.total_chunks} 个块）")
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
            raise

    async def _restore_state_async(self, directory: Optional[str] = None):
        """
        异步从磁盘恢复状态

        恢复内容：
        1. 向量索引（FAISS + 元数据 + BM25）
        2. 文档元数据
        3. 文档路由注册表
        4. 对话历史
        5. 知识图谱数据
        """
        load_dir = Path(directory or self._data_dir)

        if not load_dir.exists():
            logger.info(f"数据目录不存在，跳过恢复: {load_dir}")
            return

        try:
            # 1. 恢复向量索引
            vs_dir = load_dir / "vector_store"
            if vs_dir.exists() and (vs_dir / "faiss.index").exists():
                await self._vector_store.load(str(vs_dir))
                logger.info(f"向量索引已恢复: {self.total_chunks} 个块")

            # 2. 恢复文档元数据
            docs_path = load_dir / "documents.json"
            if docs_path.exists():
                with open(docs_path, "r", encoding="utf-8") as f:
                    docs_data = json.load(f)
                for doc_id, meta_dict in docs_data.items():
                    if doc_id in self._documents:
                        continue
                    self._documents[doc_id] = DocumentMetadata(**meta_dict)
                logger.info(f"文档元数据已恢复: {len(self._documents)} 个文档")

            # 3. 恢复文档路由注册表
            router_path = load_dir / "doc_router.json"
            filter_docs = []
            if router_path.exists():
                with open(router_path, "r", encoding="utf-8") as f:
                    router_data = json.load(f)
                for doc_id, info in router_data.items():
                    if doc_id in filter_docs:
                        continue

                    self._doc_router.register_document(
                        doc_id=info["doc_id"],
                        filename=info.get("filename", ""),
                        title=info.get("title", ""),
                        keywords=info.get("keywords", []),
                        metadata=info.get("metadata", {}),
                    )
                    filter_docs.append(doc_id)
                logger.info(f"文档路由注册表已恢复: {len(router_data)} 个文档")

            # 4. 恢复对话历史
            # history_path = load_dir / "conversation_history.json"
            # if history_path.exists():
            #     with open(history_path, "r", encoding="utf-8") as f:
            #         self._conversation_history = json.load(f)
            #     logger.info(f"对话历史已恢复: {len(self._conversation_history)} 条消息")


             # 5. 恢复知识图谱数据（如果存在）
            graph_dir = load_dir / "knowledge_graph"
            if graph_dir.exists() and (graph_dir / "graph_entities.json").exists():
                graph_loaded = self._graph_builder.load(str(graph_dir))
                if graph_loaded:
                    # 注入图谱到 VectorStore，启用三源 RRF 图谱增强检索
                    self._vector_store.set_graph_builder(self._graph_builder)
                    # 更新 Agent 的图谱工具（使 Agent 也能使用图谱检索）
                    if self._documents:
                        self._update_agent_router()
                    logger.info(
                        f"知识图谱已恢复并注入 VectorStore: "
                        f"{len(self._graph_builder._entity_map)} 个实体, "
                        f"{self._graph_builder._graph.number_of_edges()} 条关系"
                    )

            # 更新 Agent 的文档路由器
            if self._documents:
                self._update_agent_router()


        except Exception as e:
            logger.warning(f"恢复状态失败（不影响正常使用）: {e}")


    def _restore_state(self):
        """从磁盘恢复状态（同步接口，初始化时调用）"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已在事件循环中，创建任务
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._restore_state_async()
                    )
                    future.result()
            else:
                loop.run_until_complete(self._restore_state_async())
        except Exception as e:
            logger.warning(f"恢复状态失败（不影响正常使用）: {e}")

    def _save_state(self):
        """保存状态到磁盘（同步接口，加载文件后调用）"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._save_state_async()
                    )
                    future.result()
            else:
                loop.run_until_complete(self._save_state_async())
        except Exception as e:
            logger.warning(f"保存状态失败: {e}")

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
        print("  命令: q=退出, c=清空历史, d=深度模式, a=Agent模式, w=Wiki模式")
        print("        g=构建知识图谱, wg=生成Wiki")
        print("=" * 60 + "\n")

        deep_mode = False
        agent_mode = False
        wiki_mode = False
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
                wiki_mode = False
                agent_mode = False
                print(f"✅ 深度模式: {'开启' if deep_mode else '关闭'}\n")
                continue
            if question.lower() == "a":
                agent_mode = not agent_mode
                wiki_mode = False
                deep_mode = False
                print(f"✅ Agent模式: {'开启' if agent_mode else '关闭'}\n")
                continue
            if question.lower() == "w":
                wiki_mode = not wiki_mode
                agent_mode = False
                deep_mode = False
                print(f"✅ Wiki模式: {'开启' if wiki_mode else '关闭'}\n")
                continue
            if question.lower() == "g":
                # 构建知识图谱
                print("\n🔧 正在构建知识图谱...")
                try:
                    result = self.build_knowledge_graph()
                    print(f"✅ 知识图谱构建完成: {result['entities']} 个实体, {result['relationships']} 条关系")
                    if result.get('mermaid'):
                        print(f"\n📊 Mermaid 可视化（前10行）:")
                        for line in result['mermaid'].split('\n')[:10]:
                            print(f"  {line}")
                except Exception as e:
                    print(f"❌ 构建知识图谱失败: {e}")
                print()
                continue
            if question.lower() == "wg":
                # 生成 Wiki
                print("\n🔧 正在生成 Wiki 知识库...")
                try:
                    result = self.generate_wiki()
                    print(f"✅ Wiki 生成完成: 创建 {result['pages_created']} 页, 更新 {result['pages_updated']} 页")
                    print(f"   注入跨页面链接: {result['links_injected']} 个")
                    print(f"   Wiki 总页面数: {result['total_pages']}")
                except Exception as e:
                    print(f"❌ Wiki 生成失败: {e}")
                print()
                continue

            try:
                if agent_mode:
                    # Agent 模式
                    result = self.ask_agent(question)
                    print(f"\n💬 回答 (Agent模式):\n{result['answer']}")
                    if result["routed_docs"]:
                        print(f"\n📎 预筛选文档: {', '.join(result['routed_docs'])}")
                    if result["steps"]:
                        print(f"\n🧠 推理步骤 ({len(result['steps'])}步):")
                        for i, step in enumerate(result["steps"]):
                            if step["thought"]:
                                print(f"  [{i+1}] 思考: {step['thought'][:100]}")
                            for tool in step["tools"]:
                                print(f"      工具: {tool['name']}({tool['arguments'][:50]})")
                elif wiki_mode:
                    # Wiki 模式
                    result = self.ask_wiki(question)
                    print(f"\n💬 回答 (Wiki模式):\n{result['answer']}")
                    if result.get("routed_docs"):
                        print(f"\n📎 预筛选文档: {', '.join(result['routed_docs'])}")
                    if result.get("sources"):
                        print(f"\n📎 Wiki 参考页面:")
                        for i, src in enumerate(result["sources"][:3]):
                            print(f"  [{i+1}] {src['title']} (类型: {src['type']})")
                else:
                    # RAG 快速/深度问答
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

        # 6. 注册到文档路由器
        self._doc_router.register_document(
            doc_id=doc_id,
            filename=metadata.filename,
            title=metadata.title,
            keywords=[],  # 可以后续通过 set_aliases 补充
        )

        result = {
            "doc_id": doc_id,
            "filename": metadata.filename,
            "chunk_count": len(chunks),
            "message": f"成功加载 {metadata.filename}，共 {len(chunks)} 个文档块",
        }

        logger.info(f"文件加载完成: {metadata.filename}, {len(chunks)} 块")

        # 7. 更新 Agent 的文档路由器
        self._update_agent_router()

        # 8. 持久化到磁盘（向量索引、文档元数据、路由注册表）
        try:
            await self._save_state_async()
        except Exception as e:
            logger.warning(f"文件加载后自动保存失败（不影响当前使用）: {e}")

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

    def _route_query(self, question: str) -> Optional[List[str]]:
        """
        文档路由：根据查询关键词匹配相关文档

        这是"先确定检索手册范围，再在范围内检索"的第一步：
        1. 从用户问题中提取关键词
        2. 与文档名/标题/别名进行匹配
        3. 返回匹配的 doc_id 列表

        返回 None 表示全量检索，返回 List[str] 表示只在匹配文档中检索
        """
        if len(self._documents) <= 1:
            return None  # 只有一个文档，无需路由

        routed = self._doc_router.route(question)
        if routed:
            matched_names = [
                self._documents[did].filename
                for did in routed if did in self._documents
            ]
            logger.info(
                f"文档路由预筛选: 查询='{question[:50]}' → "
                f"匹配 {len(routed)}/{len(self._documents)} 个文档: {matched_names}，"
                f"将在这些文档的子空间中检索"
            )
            return routed
        else:
            logger.info(
                f"文档路由预筛选: 查询='{question[:50]}' → 无匹配，使用全量检索"
            )
            return None

    async def _ask_async(self, question: str, deep: bool = False, use_graph: bool = False) -> ChatResponse:
        """
        异步问答（含文档路由预筛选，可选图谱增强）

        流程：
        1. 文档路由：根据查询关键词匹配相关文档 → 确定 doc_ids
        2. 在匹配文档的子索引空间中检索（不是全量检索后过滤）
        3. 生成回答

        Args:
            question: 用户问题
            deep: 是否使用深度模式
            use_graph: 是否启用图谱增强检索（需先构建图谱）
        """
        if self.total_chunks == 0:
            return ChatResponse(answer="尚未加载任何文档，请先使用 load_file() 加载文档。")

        # Step 1: 文档路由 - 先确定检索手册范围
        doc_ids = self._route_query(question)

        # Step 2: 在确定的手册范围内检索
        if deep:
            return await self._rag_engine.deep_chat(
                question,
                conversation_history=self._conversation_history or None,
                doc_ids=doc_ids,
                use_graph=use_graph,
            )
        else:
            return await self._rag_engine.quick_chat(
                question,
                conversation_history=self._conversation_history or None,
                doc_ids=doc_ids,
                use_graph=use_graph,
            )

    def _update_agent_router(self):
        """更新 Agent 引擎的文档路由器（在加载新文档后调用）"""
        # 重新注册知识检索工具，传入最新的 doc_router
        self._agent_engine._tool_registry.unregister("knowledge_search")
        self._agent_engine.register_knowledge_tools(
            self._retriever, self._vector_store, doc_router=self._doc_router
        )
        # 如果已构建图谱，注册图谱检索工具
        if self._graph_builder and self._graph_builder._entity_map:
            # 先取消旧的注册（避免重复）
            self._agent_engine._tool_registry.unregister("graph_search")
            self._agent_engine._tool_registry.unregister("graph_entity_info")
            self._agent_engine.register_graph_tools(
                self._vector_store, self._graph_builder, doc_router=self._doc_router
            )

    async def _ask_agent_async(self, question: str) -> ChatResponse:
        """
        Agent 模式异步问答

        流程：
        1. 文档路由预筛选
        2. Agent ReAct 循环（Think → Act → Observe）
        3. LLM 综合整理最终答案
        """
        if self.total_chunks == 0:
            return ChatResponse(answer="尚未加载任何文档，请先使用 load_file() 加载文档。")

        # 确保文档路由器已更新
        self._update_agent_router()

        # 构建知识库信息（含路由结果）
        kb_info = [{
            "name": "本地文档知识库",
            "description": f"已加载 {len(self._documents)} 个文档，共 {self.total_chunks} 个文档块",
            "doc_count": len(self._documents),
            "chunk_count": self.total_chunks,
            "doc_names": [meta.filename for meta in self._documents.values()],
        }]

        # 文档路由预筛选
        doc_ids = self._route_query(question)
        if doc_ids:
            kb_info[0]["routed_doc_ids"] = doc_ids
            kb_info[0]["routed_doc_names"] = [
                self._documents[did].filename
                for did in doc_ids if did in self._documents
            ]

        response = await self._agent_engine.run(
            query=question,
            knowledge_bases_info=kb_info,
            conversation_history=self._conversation_history or None,
        )
        return response

    async def _build_knowledge_graph_async(
        self, doc_ids: Optional[List[str]] = None
    ) -> Dict:
        """
        异步构建知识图谱

        流程：
        1. 收集指定文档的文档块
        2. 并发提取实体和关系
        3. 计算权重（PMI × 0.6 + Strength × 0.4）
        4. 构建 NetworkX 图结构
        5. 保存图谱到磁盘
        """
        if self.total_chunks == 0:
            return {"entities": 0, "relationships": 0, "mermaid": "", "graph_json": "",
                    "error": "尚未加载任何文档"}

        # 确定文档范围
        if doc_ids is None:
            doc_ids = list(self._documents.keys())

        if not doc_ids:
            return {"entities": 0, "relationships": 0, "mermaid": "", "graph_json": "",
                    "error": "没有可用的文档"}

        # 收集文档块
        chunks = []
        for doc_id in doc_ids:
            doc_chunks = self._vector_store.get_chunks_by_doc_id(doc_id)
            chunks.extend(doc_chunks)

        if not chunks:
            return {"entities": 0, "relationships": 0, "mermaid": "", "graph_json": "",
                    "error": "文档中没有可用的内容"}

        chunks = chunks[500:505]  # 限制最大块数，避免过大导致提取失败
        # 构建知识图谱
        kg = await self._graph_builder.build_graph(chunks)

        # 将图谱构建器注入到 VectorStore，启用三源 RRF 图谱增强检索
        self._vector_store.set_graph_builder(self._graph_builder)

        # 保存图谱
        graph_path = os.path.join(self._data_dir, "knowledge_graph.json")
        try:
            with open(graph_path, "w", encoding="utf-8") as f:
                f.write(kg.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"保存知识图谱失败: {e}")
            graph_path = ""

        # 生成 Mermaid 可视化
        mermaid = self._graph_builder.to_mermaid()
        mermaid_path = os.path.join(self._data_dir, "knowledge_graph.mmd")
        try:
            with open(mermaid_path, "w", encoding="utf-8") as f:
                f.write(mermaid)
        except Exception as e:
            logger.warning(f"保存 Mermaid 文件失败: {e}")

        logger.info(f"知识图谱构建完成: {len(kg.entities)} 个实体, {len(kg.relationships)} 条关系")

        return {
            "entities": len(kg.entities),
            "relationships": len(kg.relationships),
            "mermaid": mermaid,
            "graph_json": graph_path,
        }

    async def _generate_wiki_async(
        self,
        doc_ids: Optional[List[str]] = None,
        granularity: str = "standard",
    ) -> Dict:
        """
        异步生成 Wiki 知识库

        流程（Map-Reduce-Post）：
        1. MAP: 每个文档生成摘要页 + 提取实体和概念
        2. REDUCE: 每个实体/概念创建或更新 Wiki 页面
        3. POST: 发布草稿 → 重建索引 → 注入跨页面链接
        """
        if self.total_chunks == 0:
            return {"pages_created": 0, "pages_updated": 0, "links_injected": 0,
                    "total_pages": 0, "message": "尚未加载任何文档"}

        # 确定文档范围
        if doc_ids is None:
            doc_ids = list(self._documents.keys())

        if not doc_ids:
            return {"pages_created": 0, "pages_updated": 0, "links_injected": 0,
                    "total_pages": 0, "message": "没有可用的文档"}

        # 初始化 Wiki 管理器（确保目录存在）
        await self._wiki_manager.initialize()

        # 执行 Wiki 生成管道
        stats = await self._wiki_ingest.ingest_documents(
            doc_ids=doc_ids,
            vector_store=self._vector_store,
            granularity=granularity,
        )

        total_pages = self._wiki_manager.total_pages

        message = (
            f"Wiki 生成完成: 创建 {stats['pages_created']} 页, "
            f"更新 {stats['pages_updated']} 页, "
            f"注入 {stats['links_injected']} 个跨页面链接"
        )

        logger.info(message)

        return {
            "pages_created": stats["pages_created"],
            "pages_updated": stats["pages_updated"],
            "links_injected": stats["links_injected"],
            "total_pages": total_pages,
            "message": message,
        }

    async def _ask_wiki_async(self, question: str) -> Dict:
        """
        Wiki 模式异步问答

        流程：
        1. 文档路由预筛选：根据问题关键词匹配文档名
        2. 在匹配文档的 Wiki 页面中搜索
        3. 基于搜索结果生成结构化回答
        """
        if self._wiki_manager.total_pages == 0:
            return {
                "answer": "Wiki 知识库为空，请先使用 generate_wiki() 生成 Wiki 页面。",
                "sources": [],
                "routed_docs": [],
                "wiki_pages_searched": 0,
            }

        # Step 1: 文档路由预筛选
        doc_ids = self._route_query(question)

        # Step 2: 在匹配文档的 Wiki 页面中搜索
        wiki_pages = await self._wiki_manager.search_pages(
            question, doc_ids=doc_ids
        )

        # Step 3: 生成回答
        if not wiki_pages:
            return {
                "answer": "Wiki 知识库中暂无与您问题相关的内容。",
                "sources": [],
                "routed_docs": [
                    self._documents[did].filename
                    for did in (doc_ids or []) if did in self._documents
                ],
                "wiki_pages_searched": 0,
            }

        # 构建上下文
        context = "\n\n---\n\n".join([
            f"# {p.title}\n{p.content}" for p in wiki_pages[:3]
        ])

        # 使用 RAG 引擎生成回答
        answer = await self._rag_engine._generate_answer(question, context)

        # 路由信息
        routed_docs = []
        if doc_ids:
            routed_docs = [
                self._documents[did].filename
                for did in doc_ids if did in self._documents
            ]

        # 来源信息
        sources = [
            {
                "slug": p.slug,
                "title": p.title,
                "type": p.page_type.value,
                "status": p.status,
            }
            for p in wiki_pages[:3]
        ]

        return {
            "answer": answer,
            "sources": sources,
            "routed_docs": routed_docs,
            "wiki_pages_searched": len(wiki_pages),
        }


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
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small", help="嵌入模型")
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
