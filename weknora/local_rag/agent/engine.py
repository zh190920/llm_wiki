"""
ReAct Agent 引擎 - 核心推理循环
借鉴 WeKnora 的 AgentEngine 设计，实现 Think → Analyze → Act → Observe 循环
"""
import asyncio
import hashlib
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
)
from agent.tools.graph_search import GraphSearchTool, GraphEntityInfoTool
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
    - 卡死检测：基于完整响应指纹（含工具调用），避免误判
    - 上下文窗口管理：Token 估算与压缩
    - 最大轮次限制：防止无限循环
    - 优雅降级：LLM 失败时调用 LLM 基于检索内容重新生成答案
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

    def register_knowledge_tools(self, retriever, vector_store, doc_router=None):
        """注册知识检索工具"""
        self._tool_registry.register(KnowledgeSearchTool(retriever, doc_router=doc_router))
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
        logger.info("Wiki 工具已注册")

    def register_graph_tools(self, vector_store, graph_builder, doc_router=None):
        """
        注册图谱检索工具

        Args:
            vector_store: 向量存储（用于执行图谱检索）
            graph_builder: 知识图谱构建器（用于查询实体信息）
            doc_router: 文档路由器（可选，用于文档级预筛选）
        """
        self._tool_registry.register(GraphSearchTool(vector_store, doc_router=doc_router))
        self._tool_registry.register(GraphEntityInfoTool(graph_builder))
        logger.info("图谱检索工具已注册")

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

        # 重置卡死检测状态
        self._last_response_hash = None
        self._stuck_count = 0

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

                # 卡死检测（基于完整响应指纹，包含工具调用信息）
                if self._is_stuck(assistant_content, tool_calls_data):
                    logger.warning(f"检测到卡死（第 {self._stuck_count} 次），尝试打破循环")
                    if self._stuck_count >= self._max_stuck:
                        # 卡死时用 LLM 重新整理答案，而不是简单拼接
                        final_answer = await self._llm_synthesize_answer(steps, query)
                        break
                    # 未达最大卡死次数，注入提示让 LLM 改变策略
                    messages.append({
                        "role": "system",
                        "content": (
                            "检测到你可能陷入了重复循环。"
                            "请尝试不同的检索策略或直接用 final_answer 提交当前已收集的信息。"
                            "如果你已经有足够的信息来回答问题，请立即使用 final_answer 工具。"
                        ),
                    })
                    continue

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
                # 注意：不再直接使用 final_answer 工具的输出作为最终答案，
                # 而是标记完成，循环结束后统一由 LLM 综合整理
                for result in tool_results:
                    if result.name == "final_answer" and not result.is_error:
                        # 提取 agent 草稿答案，用于后续 LLM 综合整理
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
                    final_answer = await self._llm_synthesize_answer(steps, query)
                else:
                    final_answer = f"处理过程中发生错误: {str(e)}"
                state = AgentState.ERROR
                break

        # 超过最大轮次
        if state != AgentState.COMPLETED and state != AgentState.ERROR:
            if steps:
                final_answer = await self._llm_synthesize_answer(steps, query)
            else:
                final_answer = "抱歉，我无法在规定轮次内完成此任务。请尝试简化您的问题。"

        # ============================================================
        # 最终答案综合整理：无论 Agent 循环如何结束，
        # 始终调用 LLM 对所有检索内容进行综合整理，
        # 输出根据检索内容和用户问题的结构化最终答案
        # ============================================================
        if steps and state != AgentState.ERROR:
            final_answer = await self._llm_synthesize_answer(
                steps, query, draft_answer=final_answer
            )

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
        - [回答] ...（最终综合整理的答案）
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

        # 收集所有步骤（用于最终综合整理）
        all_steps: List[AgentStep] = []
        final_answer = ""

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
                    # LLM 自然结束
                    final_answer = assistant_content
                    break

                # ACT
                step = AgentStep(step_index=iteration, thought=assistant_content)
                tool_calls: List[ToolCall] = []
                tool_results: List[ToolResult] = []

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
                    tool_calls.append(tool_call)
                    result = await self._tool_registry.execute_tool(tool_call)
                    tool_results.append(result)

                    # 检查最终回答
                    if result.name == "final_answer" and not result.is_error:
                        final_answer = self._extract_final_answer(result.output)

                    # 截断输出
                    output_preview = result.output[:300]
                    yield f"[结果] {output_preview}{'...' if len(result.output) > 300 else ''}\n"

                step.tool_calls = tool_calls
                step.tool_results = tool_results
                all_steps.append(step)

                if final_answer:
                    break

                # 构建消息（继续循环）
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

                for result in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.output,
                    })

                messages = await self._manage_context(messages, system_prompt)

            except Exception as e:
                yield f"\n[错误] {str(e)}\n"
                return

        # 最终答案综合整理
        if all_steps:
            synthesized = await self._llm_synthesize_answer(
                all_steps, query, draft_answer=final_answer
            )
            yield f"\n[回答] {synthesized}\n"
        elif final_answer:
            yield f"\n[回答] {final_answer}\n"
        else:
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

    # 推理辅助工具，不参与卡死检测
    _REASONING_TOOLS = {"thinking", "todo_write"}

    def _is_stuck(self, content: str, tool_calls_data=None) -> bool:
        """
        卡死检测：基于「实际动作」指纹（排除推理辅助工具）

        修复策略（V4 优化）：
        - 排除 thinking / todo_write 等推理辅助工具——
          这些工具的输出只是"思考已记录"，LLM 很容易连续调用产生相同指纹
        - 只基于「实际动作」工具（knowledge_search, grep_chunks, final_answer 等）构建指纹
        - 如果连续两轮只有推理工具没有实际动作，不算卡死（可能是在深度思考）
        - 只有当「实际动作」指纹连续相同，才判定卡死
        - 至少连续 2 次相同实际动作才算卡死（首次重复只是计数+1，不立即判定）
        """
        # 构建实际动作指纹（排除推理辅助工具）
        action_parts = []

        if content and content.strip():
            # 如果有文本内容，取前200字符作为指纹的一部分
            action_parts.append(f"text:{content[:200]}")

        if tool_calls_data:
            for tc in tool_calls_data:
                tool_name = tc.function.name
                if tool_name in self._REASONING_TOOLS:
                    continue  # 跳过推理辅助工具
                action_parts.append(f"{tool_name}:{tc.function.arguments}")

        # 如果没有实际动作（只有推理工具或纯空响应），不算卡死
        if not action_parts:
            self._stuck_count = 0
            return False

        action_fingerprint = "|||".join(action_parts)
        action_hash = hashlib.md5(action_fingerprint.encode()).hexdigest()

        # 特殊情况：空内容且无工具调用，不算卡死
        if not content and not tool_calls_data:
            self._last_response_hash = action_hash
            self._stuck_count = 0
            return False

        if action_hash == self._last_response_hash:
            self._stuck_count += 1
            # 至少连续 2 次相同实际动作才判定卡死
            return self._stuck_count >= 2
        else:
            self._stuck_count = 0
            self._last_response_hash = action_hash
            return False

    @staticmethod
    def _extract_final_answer(output: str) -> str:
        """从 final_answer 工具输出中提取答案"""
        if output.startswith("[FINAL_ANSWER]"):
            return output[len("[FINAL_ANSWER]"):].strip()
        return output

    async def _llm_synthesize_answer(
        self, steps: List[AgentStep], query: str, draft_answer: str = ""
    ) -> str:
        """
        用 LLM 基于检索内容重新整理最终答案

        这是 Agent 循环结束后的核心方法：
        1. 收集所有步骤中的检索结果（非错误、非 thinking/todo_write 的工具输出）
        2. 构建提示词，让 LLM 根据检索内容和用户问题生成结构化的最终答案
        3. 如果 Agent 已有草稿答案（final_answer 工具输出），也一并传入参考
        4. 答案包含来源引用

        无论 Agent 是正常结束、卡死还是超时，都通过此方法统一输出。
        """
        # 收集所有有价值的检索结果
        all_observations = []
        for step in steps:
            for result in step.tool_results:
                if result.is_error:
                    continue
                # 跳过 thinking 和 todo_write 的输出（只是推理记录，不是检索内容）
                if result.name in ("thinking", "todo_write"):
                    continue
                all_observations.append({
                    "tool": result.name,
                    "content": result.output[:1500],  # 限制单条长度
                })

        if not all_observations:
            # 没有检索结果，返回草稿答案或默认提示
            if draft_answer:
                return draft_answer
            return "抱歉，未能检索到相关信息来回答您的问题。"

        # 构建上下文
        context_parts = []
        for i, obs in enumerate(all_observations[:8]):  # 最多8条检索结果
            context_parts.append(
                f"[检索结果{i+1}] (来源工具: {obs['tool']})\n{obs['content']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # 构建提示词，让 LLM 整理最终答案
        draft_section = ""
        if draft_answer:
            draft_section = f"""
## Agent 草稿答案
{draft_answer}

（请参考以上草稿答案，结合检索信息进行补充和修正，输出更完整准确的结构化答案。）
"""

        prompt = f"""请根据以下检索到的信息，回答用户的问题。

## 用户问题
{query}

## 检索到的信息
{context}
{draft_section}
## 回答要求
1. 基于检索到的信息回答，不要编造内容
2. 回答要结构化、清晰，使用标题和列表组织
3. 引用信息来源（标注来自哪个检索结果）
4. 如果信息不足，诚实说明哪些方面信息不够
5. 使用中文回答
6. 直接给出回答，不需要重复问题
7. 如果有多个方面的信息，分别列出并标注来源
8. 对关键数据和结论加粗标注"""

        try:
            response = await self._client.chat.completions.create(
                model=self.config.llm.chat_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的中文知识问答助手，擅长根据检索结果整理出清晰、准确、结构化的答案。请基于检索到的信息，综合整理出对用户问题的完整回答。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=self.config.llm.max_tokens,
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM 综合整理答案完成，长度={len(answer)}")
            return answer
        except Exception as e:
            logger.error(f"LLM 综合整理答案失败: {e}，回退到草稿答案或简单拼接")
            # 回退逻辑
            if draft_answer:
                return draft_answer
            simple_parts = [obs["content"][:500] for obs in all_observations[:3]]
            return f"基于已收集的信息，以下是对 '{query}' 的回答：\n\n" + "\n\n".join(simple_parts)

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
