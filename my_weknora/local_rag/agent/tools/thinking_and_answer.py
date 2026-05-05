"""
思考工具和最终回答工具 - ReAct Agent 的核心控制工具
借鉴 WeKnora 的 thinking + final_answer + todo_write 工具设计
"""
import logging
from typing import Any, Dict

from agent.tool_registry import Tool

logger = logging.getLogger(__name__)


class ThinkingTool(Tool):
    """
    思考工具 - Agent 用于展示推理过程

    借鉴 WeKnora 的 thinking 工具：
    - 帮助 Agent 在执行前进行结构化推理
    - 提高复杂任务的推理质量
    - 使 Agent 的决策过程可追溯
    """

    @property
    def name(self) -> str:
        return "thinking"

    @property
    def description(self) -> str:
        return (
            "在采取行动之前进行深度思考。"
            "使用此工具来分析问题、制定计划、评估选项或回顾已有信息。"
            "这对于复杂的多步骤问题特别有用。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "你的思考内容：分析、推理、计划等",
                },
            },
            "required": ["thought"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        # 思考工具不执行实际操作，只是记录推理过程
        thought = arguments["thought"]
        logger.debug(f"[Agent 思考] {thought[:200]}")
        return f"思考已记录。继续推理或执行下一步操作。"


class TodoWriteTool(Tool):
    """
    待办事项工具 - Agent 用于创建和跟踪研究计划

    借鉴 WeKnora 的 todo_write 工具
    """

    def __init__(self):
        self._todos: list[dict] = []

    @property
    def name(self) -> str:
        return "todo_write"

    @property
    def description(self) -> str:
        return (
            "创建或更新研究计划/待办事项列表。"
            "在处理复杂多步骤任务时，先用此工具制定计划，再逐步执行。"
            "有助于保持任务聚焦和可追溯性。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "待办内容"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "状态",
                            },
                        },
                        "required": ["content", "status"],
                    },
                    "description": "待办事项列表",
                },
            },
            "required": ["todos"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        self._todos = arguments["todos"]

        status_icons = {
            "pending": "⬜",
            "in_progress": "🔄",
            "completed": "✅",
        }

        lines = ["计划已更新："]
        for i, todo in enumerate(self._todos):
            icon = status_icons.get(todo["status"], "⬜")
            lines.append(f"  {icon} {i+1}. {todo['content']} [{todo['status']}]")

        return "\n".join(lines)

    @property
    def current_todos(self) -> list[dict]:
        return self._todos


class FinalAnswerTool(Tool):
    """
    最终回答工具 - Agent 用于提交最终答案并终止循环

    借鉴 WeKnora 的 final_answer 工具：
    - Agent 明确提交最终答案
    - 终止 ReAct 循环
    - 包含引用来源
    """

    @property
    def name(self) -> str:
        return "final_answer"

    @property
    def description(self) -> str:
        return (
            "提交最终答案并完成任务。"
            "当你已经收集到足够的信息并可以回答用户问题时，使用此工具。"
            "这将终止推理循环。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "最终答案内容",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "对答案的信心程度",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "答案的来源引用列表",
                },
            },
            "required": ["answer"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        # 此工具的执行结果由 Agent Engine 特殊处理
        # 实际的答案提取在 engine.py 中完成
        answer = arguments["answer"]
        confidence = arguments.get("confidence", "medium")
        sources = arguments.get("sources", [])

        result = f"[FINAL_ANSWER] {answer}"
        if confidence != "high":
            result += f"\n[信心程度: {confidence}]"
        if sources:
            result += f"\n[来源: {', '.join(sources)}]"

        return result


class DatabaseQueryTool(Tool):
    """
    数据查询工具 - 对结构化数据执行 SQL 查询

    借鉴 WeKnora 的 database_query 工具（简化版）
    """

    @property
    def name(self) -> str:
        return "database_query"

    @property
    def description(self) -> str:
        return (
            "对已加载的结构化数据执行 SQL 查询。"
            "当需要从表格数据中提取特定信息时使用。"
            "仅支持 SELECT 查询，不允许修改数据。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL 查询语句（仅支持 SELECT）",
                },
            },
            "required": ["query"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        # 简化实现：返回提示信息
        query = arguments["query"].strip()

        # 安全检查
        upper_query = query.upper()
        forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]
        for word in forbidden:
            if word in upper_query:
                return f"安全限制：不允许执行 {word} 操作。仅支持 SELECT 查询。"

        return f"SQL 查询已接收：{query}\n注意：数据查询功能需要配置数据源后使用。"
