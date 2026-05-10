"""
LLM 客户端 — 异步高并发兼容
================================
提供与 OpenAI 兼容的异步 LLM 调用接口，
支持流式/非流式输出、重试机制和并发控制。
借鉴 WeKnora 的 chat.Chat 抽象设计。
"""

import asyncio
import json
import re
from typing import AsyncIterator, Optional

from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings


class LLMClient:
    """异步 LLM 客户端，兼容 OpenAI API 格式"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = base_url or settings.LLM_BASE_URL
        self.api_key = api_key or settings.LLM_API_KEY
        self.model = model or settings.LLM_MODEL
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_LLM_CALLS)
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
    ) -> dict:
        """
        非流式聊天接口

        Args:
            messages: OpenAI 格式消息列表
            temperature: 采样温度
            max_tokens: 最大输出 token 数
            tools: Function calling 工具定义列表
            tool_choice: 工具选择策略

        Returns:
            包含 content, tool_calls, usage 的字典
        """
        async with self._semaphore:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or settings.LLM_TEMPERATURE,
                "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            result = {
                "content": choice.message.content or "",
                "tool_calls": [],
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            }

            # 解析 tool_calls
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    result["tool_calls"].append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    })

            return result

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        流式聊天接口

        Yields:
            逐 token 的文本片段
        """
        async with self._semaphore:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or settings.LLM_TEMPERATURE,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    async def generate_with_template(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.1,
    ) -> str:
        """
        使用模板生成文本（用于 Wiki 页面生成等场景）

        Args:
            system_prompt: 系统提示词
            user_content: 用户内容
            temperature: 采样温度

        Returns:
            生成的文本内容
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = await self.chat(messages, temperature=temperature)
        return result["content"]


def repair_json(json_str: str) -> str:
    """修复 LLM 输出中常见的 JSON 格式问题"""
    # 移除 markdown 代码块标记
    json_str = re.sub(r"```(?:json)?\s*", "", json_str)
    json_str = re.sub(r"```\s*$", "", json_str)
    json_str = json_str.strip()

    # 尝试提取 JSON 对象/数组
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, json_str, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    return json_str


def parse_llm_json(json_str: str) -> dict | list:
    """解析 LLM 输出的 JSON，自动修复常见问题"""
    cleaned = repair_json(json_str)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"JSON 解析失败，原始内容: {json_str[:200]}")
        return {}
