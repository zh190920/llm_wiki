"""
ReAct Agent 引擎
================================
借鉴 WeKnora 的 AgentEngine 设计，实现 ReAct 推理循环：
Think → Analyze → Act → Observe

核心设计:
1. 渐进式多步推理，自主编排知识检索和工具调用
2. 支持并行工具调用
3. 上下文窗口管理
4. 循环检测和优雅退出
"""

import asyncio
import json
import time
from typing import AsyncIterator, Callable, Optional

from loguru import logger

from agent.state import AgentState, AgentStep, ToolCall
from agent.tools import ToolRegistry, TOOL_DEFINITIONS
from agent.prompts import build_rag_system_prompt, build_pure_agent_prompt
from core.llm_client import LLMClient
from config import settings


# 循环检测阈值
MAX_REPEATED_RESPONSE_ROUNDS = 3
MAX_EMPTY_RESPONSE_RETRIES = 2


class AgentEngine:
    """ReAct Agent 引擎"""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        knowledge_bases: Optional[list[dict]] = None,
        has_graph: bool = False,
        max_iterations: Optional[int] = None,
        parallel_tool_calls: bool = True,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.knowledge_bases = knowledge_bases or []
        self.has_graph = has_graph
        self.max_iterations = max_iterations or settings.AGENT_MAX_ITERATIONS
        self.parallel_tool_calls = parallel_tool_calls

    async def execute(
        self,
        query: str,
        session_id: str = "",
        history: Optional[list[dict]] = None,
        stream_callback: Optional[Callable] = None,
    ) -> AgentState:
        """
        执行 Agent ReAct 循环

        Args:
            query: 用户查询
            session_id: 会话 ID
            history: 对话历史
            stream_callback: 流式回调函数

        Returns:
            Agent 最终状态
        """
        logger.info(f"[Agent] 开始执行: session={session_id}, query={query[:100]}")

        # 初始化状态
        state = AgentState()

        # 构建系统提示词
        system_prompt = self._build_system_prompt()

        # 构建消息列表
        messages = self._build_messages(system_prompt, query, history or [])

        # 工具定义
        tools = self.tool_registry.get_definitions()

        # ReAct 循环
        empty_retries = 0
        last_content = ""
        consecutive_same = 0

        while state.current_round < self.max_iterations:
            state.current_round += 1
            round_start = time.time()
            round_num = state.current_round

            logger.info(f"[Agent][Round-{round_num}/{self.max_iterations}] 开始推理")

            try:
                # 1. Think: 调用 LLM
                response = await self.llm_client.chat(
                    messages=messages,
                    tools=tools if state.current_round > 0 else None,
                )

                content = response.get("content", "")
                tool_calls = response.get("tool_calls", [])
                finish_reason = response.get("finish_reason", "")

                # 流式回调
                if stream_callback and content:
                    await stream_callback("thinking", content)

                # 循环检测
                if not tool_calls and content:
                    if content == last_content:
                        consecutive_same += 1
                    else:
                        consecutive_same = 0
                    last_content = content

                    if consecutive_same >= MAX_REPEATED_RESPONSE_ROUNDS:
                        logger.warning(f"[Agent][Round-{round_num}] 检测到循环，终止")
                        state.final_answer = content
                        state.is_complete = True
                        break

                # 2. Analyze: 检查是否完成
                # 检查 final_answer 工具调用
                for tc in tool_calls:
                    if tc["name"] == "final_answer":
                        try:
                            args = json.loads(tc["arguments"])
                            state.final_answer = args.get("answer", content)
                        except json.JSONDecodeError:
                            state.final_answer = content
                        state.is_complete = True

                        step = AgentStep(
                            iteration=state.current_round - 1,
                            thought=content,
                            tool_calls=[ToolCall(
                                id=tc["id"],
                                name="final_answer",
                                arguments=tc["arguments"],
                                output=state.final_answer,
                                success=True,
                            )],
                        )
                        state.round_steps.append(step)

                        if stream_callback:
                            await stream_callback("final_answer", state.final_answer)

                        logger.info(f"[Agent] 完成: {round_num} 轮, 最终答案长度 {len(state.final_answer)}")
                        return state

                # 自然停止（无工具调用）
                if not tool_calls and finish_reason == "stop":
                    if not content.strip():
                        empty_retries += 1
                        if empty_retries <= MAX_EMPTY_RESPONSE_RETRIES:
                            messages.append({"role": "user", "content": "请使用 final_answer 工具给出你的回答。"})
                            continue
                        state.final_answer = "抱歉，我无法生成有效的回答。"
                    else:
                        state.final_answer = content
                    state.is_complete = True
                    break

                # 3. Act: 执行工具调用
                step = AgentStep(
                    iteration=state.current_round - 1,
                    thought=content,
                )

                if tool_calls:
                    # 并行或串行执行
                    if self.parallel_tool_calls and len(tool_calls) > 1:
                        tool_results = await self._execute_tools_parallel(tool_calls, session_id)
                    else:
                        tool_results = await self._execute_tools_sequential(tool_calls, session_id)

                    # 添加到 step
                    step.tool_calls = tool_results

                    # 4. Observe: 将工具结果添加到消息
                    if content:
                        messages.append({"role": "assistant", "content": content,
                                        "tool_calls": [{"id": tc["id"], "type": "function",
                                                       "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                                      for tc in tool_calls]})
                    for tr in tool_results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tr.id,
                            "content": tr.output if tr.success else f"错误: {tr.error}",
                        })

                        if stream_callback:
                            await stream_callback("tool_result", {
                                "name": tr.name,
                                "output": tr.output[:200] if tr.output else tr.error,
                                "success": tr.success,
                            })

                state.round_steps.append(step)

                duration_ms = int((time.time() - round_start) * 1000)
                logger.info(f"[Agent][Round-{round_num}] 完成: {len(step.tool_calls)} 工具调用, {duration_ms}ms")

            except Exception as e:
                logger.error(f"[Agent][Round-{round_num}] 执行异常: {e}")
                state.final_answer = f"Agent 执行出错: {str(e)}"
                state.is_complete = True
                break

        # 超过最大轮次
        if not state.is_complete:
            state.final_answer = await self._generate_fallback_answer(query, messages)
            state.is_complete = True

        return state

    async def _execute_tools_parallel(
        self,
        tool_calls: list[dict],
        session_id: str,
    ) -> list[ToolCall]:
        """并行执行工具调用"""
        tasks = [self._execute_single_tool(tc, session_id) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tool_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tc = tool_calls[i]
                tool_results.append(ToolCall(
                    id=tc.get("id", str(i)),
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", ""),
                    error=str(result),
                    success=False,
                ))
            else:
                tool_results.append(result)

        return tool_results

    async def _execute_tools_sequential(
        self,
        tool_calls: list[dict],
        session_id: str,
    ) -> list[ToolCall]:
        """串行执行工具调用"""
        results = []
        for tc in tool_calls:
            result = await self._execute_single_tool(tc, session_id)
            results.append(result)
        return results

    async def _execute_single_tool(self, tc: dict, session_id: str) -> ToolCall:
        """执行单个工具调用"""
        tc_id = tc.get("id", str(hash(tc.get("name", ""))))
        name = tc.get("name", "")
        arguments = tc.get("arguments", "{}")

        start_time = time.time()
        result = await self.tool_registry.execute(name, arguments)
        duration_ms = int((time.time() - start_time) * 1000)

        return ToolCall(
            id=tc_id,
            name=name,
            arguments=arguments,
            output=result.get("output", ""),
            error=result.get("error", ""),
            success=result.get("success", False),
            duration_ms=duration_ms,
        )

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        if self.knowledge_bases:
            return build_rag_system_prompt(self.knowledge_bases, self.has_graph)
        return build_pure_agent_prompt()

    def _build_messages(
        self,
        system_prompt: str,
        query: str,
        history: list[dict],
    ) -> list[dict]:
        """构建消息列表"""
        messages = [{"role": "system", "content": system_prompt}]

        # 添加历史对话
        for msg in history[-10:]:  # 保留最近10轮
            messages.append(msg)

        # 添加当前查询
        messages.append({"role": "user", "content": query})

        return messages

    async def _generate_fallback_answer(self, query: str, messages: list[dict]) -> str:
        """超过最大轮次时生成兜底回答"""
        try:
            summary_prompt = "基于以上推理过程，请总结你的最终回答。"
            messages.append({"role": "user", "content": summary_prompt})
            result = await self.llm_client.chat(messages, temperature=0.3)
            return result.get("content", "抱歉，经过多轮推理仍未能得出完整结论。")
        except Exception:
            return "抱歉，经过多轮推理仍未能得出完整结论。"
