"""
Agent 提示词管理 - 动态构建系统提示词
借鉴 WeKnora 的动态提示词构建设计
"""
from typing import List, Optional


# Agent 系统提示词模板（RAG 模式）
AGENT_RAG_SYSTEM_PROMPT = """你是一个智能知识助手，拥有强大的推理和知识检索能力。

## 你的能力
你可以使用工具来检索知识库、搜索文档、管理 Wiki 知识库，以及进行深度推理。

## 工具使用原则
1. **先思考后行动**：在调用工具前，先用 thinking 工具分析问题
2. **按需检索**：根据问题复杂度决定检索策略
3. **多步推理**：对于复杂问题，分步检索、分析和综合
4. **来源追溯**：回答时引用知识来源
5. **适时终止**：收集到足够信息后，用 final_answer 提交答案

## 可用知识库
{knowledge_bases}

## 工作流程
1. 分析用户问题，确定需要哪些信息
2. 使用 knowledge_search 搜索相关知识
3. 如需更详细的信息，使用 grep_chunks 或 list_knowledge_chunks
4. 如需查阅 Wiki 知识库，使用 wiki_read_page 或 wiki_search
5. 综合分析所有信息，用 final_answer 提交答案

## 重要提醒
- 只基于检索到的信息回答，不要编造内容
- 如果信息不足，诚实地说明
- 回答要结构化、清晰、有引用
- 当前时间: {current_time}
"""

# Agent 系统提示词模板（纯推理模式，无知识库）
AGENT_PURE_SYSTEM_PROMPT = """你是一个智能推理助手，具备强大的分析和推理能力。

## 你的能力
你可以使用思考工具进行深度推理，使用待办事项工具管理任务计划。

## 工作流程
1. 使用 thinking 工具分析问题
2. 对于复杂任务，使用 todo_write 制定计划
3. 逐步推理和分析
4. 用 final_answer 提交答案

当前时间: {current_time}
"""


def build_agent_system_prompt(
    has_knowledge_base: bool = True,
    knowledge_bases_info: Optional[List[dict]] = None,
    current_time: str = "",
) -> str:
    """
    动态构建 Agent 系统提示词

    借鉴 WeKnora 的提示词构建逻辑：
    - 根据是否绑定知识库选择模板
    - 填充运行时信息
    """
    if has_knowledge_base and knowledge_bases_info:
        # 构建知识库信息
        kb_parts = []
        for kb in knowledge_bases_info:
            name = kb.get("name", "未命名")
            doc_count = kb.get("doc_count", 0)
            chunk_count = kb.get("chunk_count", 0)
            description = kb.get("description", "")
            kb_parts.append(
                f"- **{name}**: {description} (文档数: {doc_count}, 块数: {chunk_count})"
            )
        kb_text = "\n".join(kb_parts) if kb_parts else "暂无可用知识库"

        return AGENT_RAG_SYSTEM_PROMPT.format(
            knowledge_bases=kb_text,
            current_time=current_time,
        )
    else:
        return AGENT_PURE_SYSTEM_PROMPT.format(current_time=current_time)


# Wiki 相关提示词
WIKI_SUMMARY_PROMPT = """请根据以下文档内容，生成一份结构化的摘要页面。

文档内容：
{content}

请按以下格式输出：
# {title} 摘要

## 概述
[文档的整体概述，2-3句话]

## 关键要点
- [要点1]
- [要点2]
- ...

## 核心概念
- **概念1**: 简要说明
- **概念2**: 简要说明
"""

WIKI_ENTITY_EXTRACTION_PROMPT = """请从以下文档内容中提取所有值得记录的实体和概念。

文档内容：
{content}

提取规则：
1. 实体：人名、组织名、产品名、技术名称等专有名词
2. 概念：重要的理论、方法、模型等抽象概念
3. 每个实体/概念应有一个简短描述

请按以下 JSON 格式输出：
```json
[
  {{
    "title": "实体/概念名称",
    "slug": "url-friendly-identifier",
    "description": "简短描述",
    "type": "entity|concept"
  }}
]
```
"""

WIKI_PAGE_MODIFY_PROMPT = """请根据以下原始文档块内容，为 "{entity_title}" 创建/更新一个详细的 Wiki 页面。

实体描述: {entity_description}
实体类型: {entity_type}

相关文档块：
{chunks}

请生成 Markdown 格式的页面内容，要求：
1. 综合所有相关文档块的信息
2. 使用 [[slug|标题]] 语法链接到相关的其他 Wiki 页面
3. 内容要详细、准确、有结构
4. 如果信息有矛盾，注明不同来源的说法

直接输出 Markdown 内容，不要包裹在代码块中：
"""

WIKI_DEDUPLICATION_PROMPT = """请判断以下新的实体/概念是否与已有页面重复或高度相似。

新实体：
{new_entities}

已有页面：
{existing_pages}

对于每个新实体，判断是否与已有页面重复。如果重复，建议合并到哪个页面。
请按以下 JSON 格式输出：
```json
[
  {{
    "new_title": "新实体名称",
    "action": "create|merge",
    "merge_to": "合并到的页面slug（仅merge时需要）",
    "reason": "原因"
  }}
]
```
"""

WIKI_INDEX_PROMPT = """请根据以下 Wiki 页面列表，生成一个索引页面。

页面列表：
{pages}

请生成 Markdown 格式的索引页面，要求：
1. 按类别分组（实体、概念、摘要等）
2. 使用 [[slug|标题]] 语法创建链接
3. 在开头添加一段简介，概述知识库的整体内容
4. 按重要性和关联性排列

直接输出 Markdown 内容：
"""

# 知识图谱相关提示词
GRAPH_ENTITY_EXTRACTION_PROMPT = """请从以下文本中提取实体和它们之间的关系。

文本内容：
{text}

请提取：
1. 实体（人物、组织、概念、产品等）
2. 实体之间的关系

按以下 JSON 格式输出：
```json
{{
  "entities": [
    {{
      "title": "实体名称",
      "description": "实体描述",
      "type": "person|organization|concept|product|other"
    }}
  ],
  "relationships": [
    {{
      "source": "源实体名称",
      "target": "目标实体名称",
      "relation": "关系类型",
      "description": "关系描述"
    }}
  ]
}}
```
"""
