"""
RAG 问答引擎 - 快速问答模式
借鉴 WeKnora 的 Chat Pipeline 设计，实现 检索→重排→生成的完整流程
"""
import asyncio
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import AppConfig
from core.embedder import Embedder
from core.reranker import Reranker
from core.retriever import Retriever
from core.vector_store import VectorStore
from models.schemas import ChatResponse, SearchResult

logger = logging.getLogger(__name__)

# RAG 系统提示词模板（中文优化）
RAG_SYSTEM_PROMPT = """你是一个专业的中文知识问答助手。请根据提供的参考文档内容来回答用户的问题。

## 回答原则：
1. **基于事实**：只根据参考文档中的信息回答，不要编造内容
2. **引用来源**：回答时标注信息来源（如"根据文档第X页..."、"根据参考内容..."）
3. **承认不确定**：如果参考文档中没有相关信息，请诚实说明
4. **结构化回答**：使用清晰的段落和列表组织回答
5. **中文回答**：请使用中文回答，除非用户明确要求其他语言

## 参考文档：
{context}
"""

NO_CONTEXT_PROMPT = """你是一个专业的中文知识问答助手。当前没有可用的参考文档来回答用户的问题。
请根据你的通用知识提供回答，但需要明确告知用户此回答未经过知识库验证。请使用中文回答。"""


class RAGEngine:
    """
    RAG 问答引擎

    两种模式：
    - 快速问答（quick_chat）：跳过查询理解，直接检索+生成，适合日常知识查询
    - 深度问答（deep_chat）：查询理解+混合检索+重排+生成，适合复杂查询

    设计思想借鉴 WeKnora 的 Chat Pipeline：
    - 插件化流水线：每个阶段可独立开关
    - 流式输出：支持 SSE 流式生成
    - 来源追踪：返回引用的文档块信息
    - 多轮上下文：支持会话级别的对话历史管理
    - 推荐问题：生成后续推荐问题
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            timeout=config.llm.timeout,
        )

        # 初始化子模块（由外部注入或自行创建）
        self._embedder: Optional[Embedder] = None
        self._vector_store: Optional[VectorStore] = None
        self._retriever: Optional[Retriever] = None
        self._reranker: Optional[Reranker] = None

        # 多轮上下文管理
        self._max_context_turns: int = 3
        self._conversation_contexts: Dict[str, List[dict]] = {}  # conversation_id -> messages
        self._context_token_counts: Dict[str, int] = {}  # conversation_id -> estimated tokens

    def initialize(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        retriever: Optional[Retriever] = None,
        reranker: Optional[Reranker] = None,
    ):
        """注入依赖模块"""
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker or Reranker(
            llm_config=self.config.llm,
            retriever_config=self.config.retriever,
            embedder=embedder,
        )
        self._retriever = retriever or Retriever(
            config=self.config,
            vector_store=vector_store,
            embedder=embedder,
            reranker=self._reranker,
        )

    @property
    def is_initialized(self) -> bool:
        return all([self._embedder, self._vector_store])

    def _get_conversation_history(
        self, conversation_id: Optional[str], max_turns: Optional[int] = None
    ) -> List[dict]:
        """获取对话历史，限制上下文轮次"""
        if not conversation_id or conversation_id not in self._conversation_contexts:
            return []
        history = self._conversation_contexts[conversation_id]
        max_t = max_turns or self._max_context_turns
        # 每轮对话包含 user + assistant 两条消息
        max_messages = max_t * 2
        return history[-max_messages:]

    def _append_to_conversation(
        self, conversation_id: Optional[str], user_msg: str, assistant_msg: str
    ):
        """追加消息到对话历史，管理 token 计数"""
        if not conversation_id:
            return

        if conversation_id not in self._conversation_contexts:
            self._conversation_contexts[conversation_id] = []
            self._context_token_counts[conversation_id] = 0

        self._conversation_contexts[conversation_id].append(
            {"role": "user", "content": user_msg}
        )
        self._conversation_contexts[conversation_id].append(
            {"role": "assistant", "content": assistant_msg}
        )

        # 估算 token 数
        self._context_token_counts[conversation_id] = (
            sum(len(m.get("content", "")) for m in self._conversation_contexts[conversation_id]) // 2
        )

        # 如果超过限制，清理旧的对话
        max_context_tokens = self.config.agent.max_context_tokens // 2
        if self._context_token_counts[conversation_id] > max_context_tokens:
            # 保留最近的对话轮次
            max_messages = self._max_context_turns * 2
            self._conversation_contexts[conversation_id] = (
                self._conversation_contexts[conversation_id][-max_messages:]
            )
            self._context_token_counts[conversation_id] = (
                sum(len(m.get("content", "")) for m in self._conversation_contexts[conversation_id]) // 2
            )

    async def quick_chat(
        self,
        query: str,
        top_k: int = 5,
        conversation_history: Optional[List[dict]] = None,
        conversation_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        快速问答 - 适合日常知识查询

        流程：检索 → 构建上下文 → 生成回答
        不含查询理解和 LLM 重排，延迟更低
        """
        if not self.is_initialized:
            return ChatResponse(answer="系统尚未初始化，请先上传文档。")

        start_time = time.time()

        # 1. 快速检索
        search_results = await self._retriever.quick_search(query, top_k=top_k)

        # 2. 简单去重
        search_results = Reranker._simple_diversify(
            search_results, top_k
        )

        # 3. 构建上下文
        context = self._build_context(search_results)

        # 4. 合并对话历史
        history = conversation_history or []
        if conversation_id:
            session_history = self._get_conversation_history(conversation_id)
            if session_history:
                history = session_history

        # 5. 生成回答
        answer = await self._generate_answer(query, context, history)

        # 更新对话历史
        if conversation_id:
            self._append_to_conversation(conversation_id, query, answer)

        elapsed = time.time() - start_time
        logger.info(f"快速问答完成: query='{query[:30]}', 耗时={elapsed:.2f}s, 结果数={len(search_results)}")

        # 生成推荐问题
        recommended = await self._generate_recommended_questions(query, search_results)

        return ChatResponse(
            answer=answer,
            sources=search_results,
            conversation_id=conversation_id or "",
            recommended_questions=recommended,
        )

    async def deep_chat(
        self,
        query: str,
        top_k: int = 10,
        conversation_history: Optional[List[dict]] = None,
        conversation_id: Optional[str] = None,
        use_graph: Optional[bool] = None,
    ) -> ChatResponse:
        """
        深度问答 - 适合复杂查询

        流程：查询理解 → 混合检索 → 图增强(可选) → LLM 重排 + MMR → 构建上下文 → 生成回答

        Args:
            query: 用户查询
            top_k: 检索结果数
            conversation_history: 对话历史
            conversation_id: 会话 ID
            use_graph: 是否启用图增强检索（None=跟随配置, True/False=强制开关）
        """
        if not self.is_initialized:
            return ChatResponse(answer="系统尚未初始化，请先上传文档。")

        start_time = time.time()

        # 判断是否启用图增强检索
        enable_graph = (
            use_graph if use_graph is not None
            else self.config.retriever.graph_enabled
        )

        # 1. 完整检索流水线
        from models.schemas import SearchParams, MatchType
        params = SearchParams(
            query=query,
            top_k=top_k,
            similarity_threshold=self.config.retriever.similarity_threshold,
            match_types=[MatchType.VECTOR, MatchType.KEYWORD],
        )
        search_results = await self._retriever.retrieve(
            params,
            use_graph=enable_graph,
        )

        # 2. 构建上下文
        context = self._build_context(search_results)

        # 3. 合并对话历史
        history = conversation_history or []
        if conversation_id:
            session_history = self._get_conversation_history(conversation_id)
            if session_history:
                history = session_history

        # 4. 生成回答
        answer = await self._generate_answer(query, context, history)

        # 更新对话历史
        if conversation_id:
            self._append_to_conversation(conversation_id, query, answer)

        elapsed = time.time() - start_time
        logger.info(f"深度问答完成: query='{query[:30]}', 耗时={elapsed:.2f}s, 结果数={len(search_results)}")

        # 生成推荐问题
        recommended = await self._generate_recommended_questions(query, search_results)

        return ChatResponse(
            answer=answer,
            sources=search_results,
            conversation_id=conversation_id or "",
            recommended_questions=recommended,
        )

    async def stream_chat(
        self,
        query: str,
        top_k: int = 5,
        conversation_history: Optional[List[dict]] = None,
        deep: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        流式问答 - 逐步输出回答内容

        Args:
            query: 用户查询
            top_k: 检索结果数
            conversation_history: 对话历史
            deep: 是否使用深度问答模式
        """
        if not self.is_initialized:
            yield "系统尚未初始化，请先上传文档。"
            return

        # 1. 检索
        if deep:
            from models.schemas import SearchParams, MatchType
            params = SearchParams(query=query, top_k=top_k)
            search_results = await self._retriever.retrieve(params)
        else:
            search_results = await self._retriever.quick_search(query, top_k=top_k)

        # 2. 构建上下文
        context = self._build_context(search_results)

        # 3. 流式生成回答
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context) if context else NO_CONTEXT_PROMPT
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})

        try:
            stream = await self._client.chat.completions.create(
                model=self.config.llm.chat_model,
                messages=messages,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"\n\n[生成回答时出错: {str(e)}]"

    async def _generate_recommended_questions(
        self, query: str, search_results: List[SearchResult]
    ) -> List[str]:
        """生成推荐后续问题"""
        try:
            from core.question_generator import QuestionGenerator
            generator = QuestionGenerator(self.config)
            return await generator.generate(query, search_results)
        except Exception as e:
            logger.warning(f"生成推荐问题失败: {e}")
            return []

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _generate_answer(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> str:
        """调用 LLM 生成回答"""
        # 尝试使用模板管理器
        try:
            from agent.prompts import _get_template_manager
            manager = _get_template_manager()
            if context:
                system_prompt = manager.get_prompt("rag_system", context=context)
            else:
                system_prompt = manager.get_prompt("rag_system", context="")
            if not system_prompt:
                system_prompt = RAG_SYSTEM_PROMPT.format(context=context) if context else NO_CONTEXT_PROMPT
        except Exception:
            system_prompt = RAG_SYSTEM_PROMPT.format(context=context) if context else NO_CONTEXT_PROMPT

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-6:])  # 保留最近3轮对话
        messages.append({"role": "user", "content": query})

        response = await self._client.chat.completions.create(
            model=self.config.llm.chat_model,
            messages=messages,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )

        return response.choices[0].message.content

    @staticmethod
    def _build_context(search_results: List[SearchResult], max_tokens: int = 4000) -> str:
        """
        构建上下文 - 将检索结果格式化为 LLM 可用的上下文

        借鉴 WeKnora 的 merge 插件设计：
        - 去重（相同内容只保留分数最高的）
        - 截断（防止超过上下文长度）
        - 结构化标注来源
        - 父上下文增强（如果有 parent_context）
        """
        if not search_results:
            return ""

        seen_contents: set[str] = set()
        context_parts: List[str] = []
        current_tokens = 0

        for i, result in enumerate(search_results):
            # 去重
            content_key = result.chunk.content[:100]
            if content_key in seen_contents:
                continue
            seen_contents.add(content_key)

            # 如果有父上下文，使用父上下文（更完整的上下文）
            display_content = result.chunk.content
            if result.chunk.metadata.get("parent_context"):
                parent_content = result.chunk.metadata["parent_context"]
                # 如果父上下文不太长，直接使用父上下文
                if len(parent_content) <= 3000:
                    display_content = parent_content
                else:
                    # 否则附加父上下文摘要
                    display_content = f"{result.chunk.content}\n\n[父块上下文]: {parent_content[:500]}..."

            # 粗略估算 token 数
            est_tokens = len(display_content) // 2
            if current_tokens + est_tokens > max_tokens:
                break

            # 格式化来源信息
            source_info = f"[文档: {result.chunk.metadata.get('section_title', result.chunk.doc_id)}"
            if result.chunk.metadata.get("sub_index") is not None:
                source_info += f", 段落: {result.chunk.metadata['sub_index'] + 1}"
            if result.chunk.parent_chunk_id:
                source_info += f", 含父块上下文"
            source_info += f", 相关度: {result.score:.2f}]"

            context_parts.append(f"{source_info}\n{display_content}")
            current_tokens += est_tokens

        return "\n\n---\n\n".join(context_parts)
