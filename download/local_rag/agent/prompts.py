"""
Agent 系统提示词
================================
借鉴 WeKnora 的 BuildSystemPrompt 设计，
为 Agent 构建渐进式 RAG 系统提示词，
包含知识库信息、工具使用指南和推理策略。
"""


def build_rag_system_prompt(
    knowledge_bases: list[dict],
    has_graph: bool = False,
) -> str:
    """
    构建 RAG Agent 系统提示词

    Args:
        knowledge_bases: 知识库信息列表
        has_graph: 是否有知识图谱可用

    Returns:
        系统提示词文本
    """
    kb_section = _format_knowledge_bases(knowledge_bases)
    graph_section = "\n### 知识图谱\n\n你可以使用 `query_knowledge_graph` 工具查询实体之间的关系和关联知识。" if has_graph else ""

    return f"""你是一个智能知识助手，擅长通过知识检索和推理来回答用户的问题。

## 可用知识库

{kb_section}
{graph_section}

## 工具使用策略

你需要根据问题复杂度选择合适的工具组合：

### 简单查询（日常知识查询）
- 直接使用 `knowledge_search` 搜索相关知识
- 获取到相关信息后，立即调用 `final_answer` 给出回答

### 复杂推理（多步分析任务）
1. 先使用 `sequential_thinking` 分析问题，制定推理计划
2. 按计划逐步使用 `knowledge_search` 收集信息
3. 如果有知识图谱，使用 `query_knowledge_graph` 获取实体关系
4. 综合所有信息，使用 `final_answer` 输出完整回答

### 重要规则
- **必须调用 final_answer 工具**来给出最终回答，不要直接在文本中输出答案
- 每次搜索使用**具体、明确**的查询词，避免模糊搜索
- 如果搜索结果不够充分，尝试**换一组关键词**重新搜索
- 回答时必须**基于检索到的知识**，不要编造信息
- 如果知识库中没有相关信息，如实告知用户

## 回答质量要求
- 结构清晰，逻辑连贯
- 引用知识来源，增强可信度
- 中文问题用中文回答，英文问题用英文回答
"""


def build_pure_agent_prompt() -> str:
    """构建纯 Agent（无知识库）提示词"""
    return """你是一个智能推理助手。虽然当前没有绑定知识库，你可以使用深度思考工具来分析和回答问题。

## 工具使用策略

1. 使用 `sequential_thinking` 进行结构化推理
2. 将复杂问题分解为多个小步骤
3. 逐步推理，最终使用 `final_answer` 给出回答

## 重要规则
- 必须调用 final_answer 工具给出最终回答
- 基于已知信息回答，如不确定请明确说明
"""


def _format_knowledge_bases(kbs: list[dict]) -> str:
    """格式化知识库信息"""
    if not kbs:
        return "(当前未绑定知识库)"

    lines = []
    for kb in kbs:
        kb_id = kb.get("id", "")
        name = kb.get("name", "")
        doc_count = kb.get("doc_count", 0)
        desc = kb.get("description", "")
        caps = kb.get("capabilities", [])

        cap_str = ", ".join(caps) if caps else "向量检索"
        lines.append(f"- **{name}** (ID: {kb_id})")
        if desc:
            lines.append(f"  描述: {desc}")
        lines.append(f"  文档数: {doc_count}, 能力: {cap_str}")

    return "\n".join(lines)
