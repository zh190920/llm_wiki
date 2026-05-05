"""
工具注册表 - 借鉴 WeKnora 的 ToolRegistry 设计
支持工具注册、验证、执行和安全控制
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from models.schemas import ToolCall, ToolResult

logger = logging.getLogger(__name__)


class Tool(ABC):
    """
    工具基类 - 所有 Agent 工具必须实现此接口

    借鉴 WeKnora 的 types.Tool 设计：
    - 名称、描述、参数 Schema
    - 异步执行
    - 输出截断保护
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（供 LLM 理解用途）"""
        ...

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema 格式的参数定义"""
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> str:
        """
        执行工具

        Args:
            arguments: 工具参数

        Returns:
            工具执行结果（字符串形式）
        """
        ...

    async def cleanup(self):
        """清理资源（可选）"""
        pass


class ToolRegistry:
    """
    工具注册表

    借鉴 WeKnora 的安全设计：
    - 先注册优先（防止工具劫持）
    - 参数 JSON Schema 验证
    - 输出截断保护（防止上下文窗口中毒）
    - 工具生命周期管理
    """

    def __init__(self, max_output_size: int = 16384):
        self._tools: Dict[str, Tool] = {}
        self.max_output_size = max_output_size

    def register(self, tool: Tool) -> bool:
        """
        注册工具（先注册优先，防止工具名冲突）

        Args:
            tool: 工具实例

        Returns:
            是否注册成功
        """
        if tool.name in self._tools:
            logger.warning(f"工具已存在: {tool.name}，跳过注册")
            return False

        self._tools[tool.name] = tool
        logger.debug(f"注册工具: {tool.name}")
        return True

    def unregister(self, name: str) -> bool:
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """列出所有已注册工具"""
        return list(self._tools.values())

    def get_openai_tools_schema(self) -> List[Dict[str, Any]]:
        """
        生成 OpenAI function calling 格式的工具定义

        Returns:
            OpenAI tools 参数格式的列表
        """
        tools_schema = []
        for tool in self._tools.values():
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                }
            }
            tools_schema.append(schema)
        return tools_schema

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        执行工具调用

        包含安全控制：
        1. 参数验证
        2. 执行超时
        3. 输出截断
        4. 错误捕获
        """
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                output=f"错误：未知工具 '{tool_call.name}'",
                is_error=True,
                error=f"Tool not found: {tool_call.name}",
            )

        try:
            # 参数预处理：修复 LLM 常见的参数格式问题
            arguments = self._fix_arguments(tool_call.arguments, tool.parameters_schema)

            # 执行工具
            output = await tool.execute(arguments)

            # 输出截断保护
            if len(output) > self.max_output_size:
                original_len = len(output)
                output = output[:self.max_output_size]
                output += f"\n\n[输出已截断：原始长度 {original_len} 字符，截断至 {self.max_output_size} 字符]"
                logger.warning(f"工具 {tool.name} 输出截断: {original_len} → {self.max_output_size}")

            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                output=output,
            )

        except Exception as e:
            logger.error(f"工具执行失败: {tool.name}, 错误: {e}")
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                output=f"工具执行错误: {str(e)}",
                is_error=True,
                error=str(e),
            )

    async def execute_tools_parallel(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """并行执行多个工具调用"""
        import asyncio
        tasks = [self.execute_tool(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)

    async def cleanup_all(self):
        """清理所有工具资源"""
        for tool in self._tools.values():
            try:
                await tool.cleanup()
            except Exception as e:
                logger.warning(f"工具清理失败: {tool.name}, 错误: {e}")

    @staticmethod
    def _fix_arguments(arguments: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        修复 LLM 生成的参数格式问题

        常见问题：
        - 字符串 "true" → 布尔值 true
        - 字符串数字 → 数字
        - JSON 字符串 → 解析后的对象
        """
        if not isinstance(arguments, dict):
            # 尝试解析为 JSON
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"input": arguments}
            else:
                arguments = {"input": str(arguments)}

        properties = schema.get("properties", {})

        for key, value in arguments.items():
            if key in properties:
                prop_type = properties[key].get("type", "string")

                if prop_type == "boolean" and isinstance(value, str):
                    arguments[key] = value.lower() in ("true", "1", "yes")
                elif prop_type == "number" and isinstance(value, str):
                    try:
                        arguments[key] = float(value)
                    except ValueError:
                        pass
                elif prop_type == "integer" and isinstance(value, str):
                    try:
                        arguments[key] = int(value)
                    except ValueError:
                        pass
                elif prop_type == "object" and isinstance(value, str):
                    try:
                        arguments[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass

        return arguments
