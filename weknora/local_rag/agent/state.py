"""
Agent 状态管理
================================
借鉴 WeKnora 的 AgentState 和 AgentStep 设计，
管理 ReAct 推理循环的状态。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolCall:
    """工具调用记录"""
    id: str = ""
    name: str = ""
    arguments: str = ""
    output: str = ""
    error: str = ""
    success: bool = True
    duration_ms: int = 0


@dataclass
class AgentStep:
    """Agent 单步执行记录"""
    iteration: int = 0
    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class AgentState:
    """Agent 执行状态"""
    current_round: int = 0
    round_steps: list[AgentStep] = field(default_factory=list)
    knowledge_refs: list[dict] = field(default_factory=list)
    final_answer: str = ""
    is_complete: bool = False
