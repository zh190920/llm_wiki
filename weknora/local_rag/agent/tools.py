"""
Agent 工具定义
================================
借鉴 WeKnora 的 ToolRegistry 设计，定义 Agent 可用的工具。
包含知识搜索、深度思考、最终回答等工具，
支持工具注册和执行。
"""

import json
import time
import uuid
from typing import Callable, Optional

from loguru import logger


# ── 工具定义 (OpenAI Function Calling 格式) ──

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": "在知识库中搜索相关信息。支持语义检索和关键词检索的混合搜索模式，返回与查询最相关的文档片段。当需要查找特定知识、概念、事实时应使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询文本",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，默认5",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sequential_thinking",
            "description": "使用结构化的思维链进行深度推理。当你需要分析复杂问题、制定计划、或进行多步推理时使用此工具。有助于将复杂问题分解为可管理的步骤。",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "当前的思考内容",
                    },
                    "next_step_needed": {
                        "type": "boolean",
                        "description": "是否还需要继续推理",
                        "default": True,
                    },
                },
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_graph",
            "description": "查询知识图谱，获取实体之间的关系。当需要了解概念间的关联、查找相关实体、或获取结构化知识时应使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "要查询的实体名称",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "查询深度，默认1（直接关联），2表示二度关联",
                        "default": 1,
                    },
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "当已经收集到足够的信息并可以给出最终回答时，调用此工具输出最终答案。必须在你确信已经完整回答了用户问题时才调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "对用户问题的最终回答",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._definitions: list[dict] = []

    def register(self, name: str, func: Callable, definition: Optional[dict] = None):
        """注册工具"""
        self._tools[name] = func
        if definition:
            self._definitions.append(definition)

    def get_definitions(self) -> list[dict]:
        """获取所有工具的 OpenAI 格式定义"""
        return self._definitions if self._definitions else TOOL_DEFINITIONS

    async def execute(self, name: str, arguments: str) -> dict:
        """
        执行工具调用

        Args:
            name: 工具名称
            arguments: JSON 格式的参数字符串

        Returns:
            {"output": str, "error": str, "success": bool}
        """
        start_time = time.time()

        try:
            func = self._tools.get(name)
            if not func:
                return {
                    "output": "",
                    "error": f"未知工具: {name}",
                    "success": False,
                }

            # 解析参数
            try:
                args = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                return {
                    "output": "",
                    "error": f"参数 JSON 解析失败: {arguments[:100]}",
                    "success": False,
                }

            # 执行
            if name == "final_answer":
                result = args.get("answer", "")
            elif name == "sequential_thinking":
                result = args.get("thought", "")
            else:
                result = await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "output": str(result) if not isinstance(result, dict) else json.dumps(result, ensure_ascii=False),
                "error": "",
                "success": True,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"工具执行失败 [{name}]: {e}")
            return {
                "output": "",
                "error": str(e),
                "success": False,
                "duration_ms": duration_ms,
            }


import asyncio
