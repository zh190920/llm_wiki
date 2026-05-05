"""
ReAct Agent 引擎 - 核心推理循环
借鉴 WeKnora 的 AgentEngine 设计，实现 Think → Analyze → Act → Observe 循环
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from openai import AsyncOpenAI

from agent.prompts import build_agent_system_prompt
from agent.tool_registry import ToolRegistry
from agent.tools.knowledge_search import (
    GrepChunksTool,
    KnowledgeSearchTool,
    ListKnowledgeChunksTool,
)
from agent.tools.thinking_and_answer import (
    FinalAnswerTool,
    ThinkingTool,
    TodoWriteTool,
    DatabaseQueryTool,
)
from agent.tools.wiki_tools import (
    WikiFlagIssueTool,
    WikiReadPageTool,
    WikiReadSourceDocTool,
    WikiSearchTool,
    WikiWritePageTool,
    WikiRenamePageTool,
    WikiDeletePageTool,
    WikiReplaceTextTool,
)
from agent.tools.graph_tools import (
    QueryKnowledgeGraphTool,
    GetDocumentInfoTool,
)
from config.settings import AppConfig
from models.schemas import AgentStep, AgentState, ChatResponse, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class AgentEngine:
    """
    ReAct Agent 引擎

    核心循环（借鉴 WeKnora 的 executeLoop 设计）：
    1. THINK:  调用 LLM，获取响应（含可能的工具调用）
    2. ANALYZE: 分析响应，判断终止条件
    3. ACT:    执行工具调用
    4. OBSERVE: 将工具结果添加到上下文

    安全特性：
    - 卡死检测：连续相同内容检测
    - 上下文窗口管理：Token 估算与压缩
    - 最大轮次限制：防止无限循环
    - 优雅降级：LLM 失败时从工具结果合成答案
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            timeout=config.llm.timeout,
        )
        self._tool_registry = ToolRegistry(
            max_output_size=config.agent.max_tool_output_size
        )
        self._thinking_tool = ThinkingTool()
        self._todo_tool = TodoWriteTool()
        self._final_answer_tool = FinalAnswerTool()

        # 注册内置工具
        self._register_builtin_tools()

        # 状态追踪
        self._last_response_hash: Optional[str] = None
        self._stuck_count = 0
        self._max_stuck = 3

    def _register_builtin_tools(self):
        """注册内置工具"""
        self._tool_registry.register(self._thinking_tool)
        self._tool_registry.register(self._todo_tool)
        self._tool_registry.register(self._final_answer_tool)
        self._tool_registry.register(DatabaseQueryTool())

    def register_knowledge_tools(self, retriever, vector_store):
        """注册知识检索工具"""
        self._tool_registry.register(KnowledgeSearchTool(retriever))
        self._tool_registry.register(GrepChunksTool(vector_store))
        self._tool_registry.register(ListKnowledgeChunksTool(vector_store))
        logger.info("知识检索工具已注册")

    def register_wiki_tools(self, wiki_manager, vector_store):
        """注册 Wiki 工具"""
        self._tool_registry.register(WikiReadPageTool(wiki_manager))
        self._tool_registry.register(WikiWritePageTool(wiki_manager))
        self._tool_registry.register(WikiSearchTool(wiki_manager))
        self._tool_registry.register(WikiReadSourceDocTool(wiki_manager, vector_store))
        self._tool_registry.register(WikiFlagIssueTool(wiki_manager))
        self._tool_registry.register(WikiRenamePageTool(wiki_manager))
        self._tool_registry.register(WikiDeletePageTool(wiki_manager))
        self._tool_registry.register(WikiReplaceTextTool(wiki_manager))
        logger.info("Wiki 工具已注册")

    def register_graph_tools(self, graph_builder, vector_store):
        """注册知识图谱查询工具"""
        self._tool_registry.register(QueryKnowledgeGraphTool(graph_builder, vector_store))
        self._tool_registry.register(GetDocumentInfoTool(vector_store))
        logger.info("知识图谱工具已注册")

    async def run(
        self,
        query: str,
        knowledge_bases_info: Optional[List[dict]] = None,
        conversation_history: Optional[List[dict]] = None,
        on_step: Optional[Callable[[AgentStep], None]] = None,
    ) -> ChatResponse:
        """
        运行 ReAct 循环

        Args:
            query: 用户查询
            knowledge_bases_info: 知识库信息列表
            conversation_history: 对话历史
            on_step: 每步回调

        Returns:
            聊天响应（含 Agent 步骤记录）
        """
        has_kb = knowledge_bases_info is not None and len(knowledge_bases_info) > 0
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 构建系统提示词
        system_prompt = build_agent_system_prompt(
            has_knowledge_base=has_kb,
            knowledge_bases_info=knowledge_bases_info,
            current_time=current_time,
        )

        # 初始化消息列表
        messages: List[dict] = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-8:])
        messages.append({"role": "user", "content": query})

        # 工具定义
        tools_schema = self._tool_registry.get_openai_tools_schema()

        # ReAct 循环
        steps: List[AgentStep] = []
        final_answer = ""
        state = AgentState.THINKING

        for iteration in range(self.config.agent.max_iterations):
            logger.info(f"ReAct 循环 - 第 {iteration + 1} 轮")

            try:
                # THINK: 调用 LLM
                state = AgentState.THINKING
                response = await self._call_llm(messages, tools_schema)

                assistant_message = response.choices[0].message
                assistant_content = assistant_message.content or ""
                tool_calls_data = assistant_message.tool_calls

                # ANALYZE: 分析响应
                state = AgentState.ACTING

                # 卡死检测
                if self._is_stuck(assistant_content):
                    logger.warning(f"检测到卡死（第 {self._stuck_count} 次），尝试打破循环")
                    if self._stuck_count >= self._max_stuck:
                        final_answer = self._synthesize_answer(steps, query)
                        break

                # 检查是否为最终回答
                if not tool_calls_data:
                    # LLM 自然结束（finish_reason=stop）
                    final_answer = assistant_content
                    state = AgentState.COMPLETED
                    break

                # 处理工具调用
                step = AgentStep(
                    step_index=iteration,
                    thought=assistant_content,
                )

                # 构建助手消息（含工具调用）
                assistant_msg = {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls_data
                    ],
                }
                messages.append(assistant_msg)

                # ACT: 解析并执行工具调用
                tool_calls: List[ToolCall] = []
                for tc in tool_calls_data:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {"raw_input": tc.function.arguments}

                    tool_calls.append(ToolCall(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    ))

                step.tool_calls = tool_calls

                # 执行工具
                if self.config.agent.parallel_tool_calls and len(tool_calls) > 1:
                    # 并行执行
                    tool_results = await self._tool_registry.execute_tools_parallel(tool_calls)
                else:
                    # 串行执行
                    tool_results = []
                    for tc in tool_calls:
                        result = await self._tool_registry.execute_tool(tc)
                        tool_results.append(result)

                step.tool_results = tool_results

                # 检查是否为最终回答
                for result in tool_results:
                    if result.name == "final_answer" and not result.is_error:
                        final_answer = self._extract_final_answer(result.output)
                        state = AgentState.COMPLETED
                        break

                if state == AgentState.COMPLETED:
                    steps.append(step)
                    if on_step:
                        on_step(step)
                    break

                # OBSERVE: 将工具结果添加到消息
                for result in tool_results:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.output,
                    }
                    messages.append(tool_msg)

                # 上下文窗口管理
                messages = await self._manage_context(messages, system_prompt)

                steps.append(step)
                if on_step:
                    on_step(step)

            except Exception as e:
                logger.error(f"ReAct 循环异常: {e}")
                if steps:
                    final_answer = self._synthesize_answer(steps, query)
                else:
                    final_answer = f"处理过程中发生错误: {str(e)}"
                state = AgentState.ERROR
                break

        # 超过最大轮次
        if state != AgentState.COMPLETED and state != AgentState.ERROR:
            if steps:
                final_answer = self._synthesize_answer(steps, query)
            else:
                final_answer = "抱歉，我无法在规定轮次内完成此任务。请尝试简化您的问题。"

        logger.info(f"ReAct 循环结束: 状态={state.value}, 轮次={len(steps)}")

        return ChatResponse(
            answer=final_answer,
            agent_steps=steps,
        )

    async def stream_run(
        self,
        query: str,
        knowledge_bases_info: Optional[List[dict]] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式 ReAct 循环 - 逐步输出推理过程

        每一步产出格式：
        - [思考] ... 
        - [工具] tool_name(args) 
        - [结果] ... 
        - [回答] ...
        """
        has_kb = knowledge_bases_info is not None and len(knowledge_bases_info) > 0
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = build_agent_system_prompt(
            has_knowledge_base=has_kb,
            knowledge_bases_info=knowledge_bases_info,
            current_time=current_time,
        )

        messages: List[dict] = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-8:])
        messages.append({"role": "user", "content": query})

        tools_schema = self._tool_registry.get_openai_tools_schema()

        for iteration in range(self.config.agent.max_iterations):
            try:
                # THINK
                response = await self._call_llm(messages, tools_schema)
                assistant_message = response.choices[0].message
                assistant_content = assistant_message.content or ""
                tool_calls_data = assistant_message.tool_calls

                if assistant_content:
                    yield f"\n[思考] {assistant_content}\n"

                if not tool_calls_data:
                    yield f"\n[回答] {assistant_content}\n"
                    return

                # ACT: 执行工具调用（只执行一次，结果同时用于显示和消息构建）
                tool_results_map = {}  # call_id -> ToolResult
                for tc in tool_calls_data:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    yield f"\n[工具] {tc.function.name}({json.dumps(args, ensure_ascii=False)[:100]})\n"

                    tool_call = ToolCall(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                    result = await self._tool_registry.execute_tool(tool_call)
                    tool_results_map[tc.id] = result

                    # 检查最终回答
                    if result.name == "final_answer" and not result.is_error:
                        answer = self._extract_final_answer(result.output)
                        yield f"\n[回答] {answer}\n"
                        return

                    # 截断输出
                    output_preview = result.output[:300]
                    yield f"[结果] {output_preview}{'...' if len(result.output) > 300 else ''}\n"

                # 构建消息（复用已执行的工具结果）
                assistant_msg = {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls_data
                    ],
                }
                messages.append(assistant_msg)

                # 使用已执行的结果构建工具消息，不再重复执行
                for tc in tool_calls_data:
                    result = tool_results_map[tc.id]
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.output,
                    })

                messages = await self._manage_context(messages, system_prompt)

            except Exception as e:
                yield f"\n[错误] {str(e)}\n"
                return

        yield "\n[回答] 抱歉，我无法在规定轮次内完成此任务。\n"

    async def _call_llm(self, messages: List[dict], tools_schema: List[dict]) -> Any:
        """调用 LLM"""
        return await self._client.chat.completions.create(
            model=self.config.llm.chat_model,
            messages=messages,
            tools=tools_schema if tools_schema else None,
            tool_choice="auto" if tools_schema else None,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )

    def _is_stuck(self, content: str) -> bool:
        """卡死检测：连续相同内容"""
        import hashlib
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if content_hash == self._last_response_hash:
            self._stuck_count += 1
            return True
        else:
            self._stuck_count = 0
            self._last_response_hash = content_hash
            return False

    @staticmethod
    def _extract_final_answer(output: str) -> str:
        """从 final_answer 工具输出中提取答案"""
        if output.startswith("[FINAL_ANSWER]"):
            return output[len("[FINAL_ANSWER]"):].strip()
        return output

    def _synthesize_answer(self, steps: List[AgentStep], query: str) -> str:
        """从已有步骤合成最终答案（优雅降级）"""
        all_observations = []
        for step in steps:
            for result in step.tool_results:
                if not result.is_error:
                    all_observations.append(result.output[:500])

        if all_observations:
            return (
                f"基于已收集的信息，以下是对 '{query}' 的回答：\n\n"
                + "\n\n".join(all_observations[:3])
            )
        return "抱歉，由于处理过程中出现问题，无法生成完整答案。"

    async def _manage_context(self, messages: List[dict], system_prompt: str) -> List[dict]:
        """
        上下文窗口管理

        借鉴 WeKnora 的上下文管理策略：
        - 估算当前 token 数
        - 超限时压缩早期对话和工具结果
        - 保留系统提示词和最近对话
        """
        # 粗略估算 token 数（中英文混合约 2 字符/token）
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 2

        max_tokens = self.config.agent.max_context_tokens
        if estimated_tokens <= max_tokens * 0.8:
            return messages

        logger.info(f"上下文接近上限 ({estimated_tokens} tokens)，开始压缩")

        # 压缩策略：保留系统提示 + 最近的对话轮次
        # 1. 保留系统提示
        system_msg = messages[0] if messages[0]["role"] == "system" else None
        recent_messages = messages[-6:]  # 保留最近3轮

        # 2. 压缩中间消息（截断工具结果）
        compressed_messages = []
        if system_msg:
            compressed_messages.append(system_msg)

        # 添加压缩提示
        compressed_messages.append({
            "role": "system",
            "content": "[之前的对话和检索结果已被压缩。以下是关键信息摘要：]",
        })

        # 截断中间的工具结果
        for msg in messages[1:-6]:
            content = msg.get("content", "")
            if len(content) > 500:
                msg = {**msg, "content": content[:500] + "\n[...已截断...]"}
            compressed_messages.append(msg)

        compressed_messages.extend(recent_messages)
        return compressed_messages
