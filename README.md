# Local RAG System 模块详细解析

> 基于 Tencent/WeKnora 核心设计思想的本地 RAG 系统
> 全部使用 Python 实现，兼容异步高并发，中文优先

---

## 一、项目总览

```
local_rag/
├── config/                     # 配置层
│   ├── __init__.py
│   └── settings.py             # 统一配置管理
├── models/                     # 数据模型层
│   ├── __init__.py
│   └── schemas.py              # Pydantic 数据模型
├── core/                       # 核心引擎层
│   ├── __init__.py
│   ├── document_parser.py      # 文档解析
│   ├── chunker.py              # 文本分块
│   ├── embedder.py             # 向量嵌入
│   ├── vector_store.py         # 向量存储+BM25
│   ├── reranker.py             # 重排+MMR去重
│   ├── retriever.py            # 统一检索器
│   └── rag_engine.py           # RAG问答引擎
├── agent/                      # Agent推理层
│   ├── __init__.py
│   ├── engine.py               # ReAct引擎
│   ├── tool_registry.py        # 工具注册表
│   ├── prompts.py              # 提示词管理
│   └── tools/                  # 工具实现
│       ├── __init__.py
│       ├── knowledge_search.py # 知识检索工具
│       ├── thinking_and_answer.py # 思考/回答工具
│       └── wiki_tools.py       # Wiki操作工具
├── wiki/                       # Wiki知识库层
│   ├── __init__.py
│   ├── page_manager.py         # Wiki页面管理
│   ├── ingest.py               # Wiki生成管道
│   └── graph_builder.py        # 知识图谱构建
├── api/                        # API接口层
│   ├── __init__.py
│   └── server.py               # FastAPI服务
├── cli_qa.py                   # ⭐ 无端口问答（核心入口）
├── main.py                     # 服务启动入口
├── config.yaml                 # YAML配置文件
└── requirements.txt            # Python依赖
```

---

## 二、数据流全景

### 2.1 RAG 快速问答流程

```
用户提问 "E003故障怎么处理？"
     │
     ▼
┌─────────────────────────────────┐
│  cli_qa.LocalQA.ask()           │
│  或 RAGEngine.quick_chat()      │
└─────────────┬───────────────────┘
              │
     ┌────────▼────────┐
     │  Retriever       │  统一检索器
     │  .quick_search() │  跳过查询理解，直接检索
     └────────┬─────────┘
              │
     ┌────────▼────────────────────────────┐
     │  VectorStore.search_hybrid()         │  混合检索
     │  ┌──────────────┐ ┌──────────────┐  │
     │  │ search_vector │ │search_keyword│  │  并发执行
     │  │  (FAISS)      │ │  (BM25)      │  │
     │  └──────┬───────┘ └──────┬───────┘  │
     │         └───────┬────────┘           │
     │                 ▼                    │
     │         RRF 分数融合                  │
     │     (Reciprocal Rank Fusion)         │
     └────────────────┬─────────────────────┘
                      │
     ┌────────────────▼─────────────────────┐
     │  Reranker._simple_diversify()         │  简单去重
     └────────────────┬─────────────────────┘
                      │
     ┌────────────────▼─────────────────────┐
     │  RAGEngine._build_context()           │  构建上下文
     │  格式：[文档: xxx, 相关度: 0.85]       │
     │        原始文档内容...                  │
     └────────────────┬─────────────────────┘
                      │
     ┌────────────────▼─────────────────────┐
     │  LLM (OpenAI API)                     │  生成回答
     │  system: RAG_SYSTEM_PROMPT + 上下文    │
     │  user:   用户问题                      │
     └────────────────┬─────────────────────┘
                      │
                      ▼
              "E003是传感器异常..."
              + 对话历史自动保存到 conversation.json
```

### 2.2 ReAct Agent 推理流程

```
用户提问（复杂多步问题）
     │
     ▼
┌──────────────────────────────────┐
│  AgentEngine.run()               │
│  构建 system_prompt + tools_schema│
└─────────────┬────────────────────┘
              │
     ┌────────▼────────┐  循环开始
     │   THINK          │  调用 LLM
     │   流式输出响应     │  含 tool_calls 或纯文本
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │   ANALYZE        │  检查终止条件
     │   - final_answer?│  → 提取答案，结束
     │   - 无工具调用？  │  → 自然结束
     │   - 卡死检测？    │  → 强制合成答案
     └────────┬────────┘
              │ 继续循环
     ┌────────▼────────┐
     │   ACT            │  执行工具调用
     │   - 参数验证      │
     │   - 并行/串行执行 │
     │   - 输出截断保护  │
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │   OBSERVE        │  工具结果加入上下文
     │   - 上下文窗口管理│  token 估算+压缩
     └────────┬────────┘
              │
              └──→ 回到 THINK（下一轮循环）
```

### 2.3 Wiki 生成流程

```
文档上传
     │
     ▼
┌──────────────────────────────────────────┐
│          MAP 阶段（每文档并发）            │
│                                          │
│  1. 生成摘要页                            │
│     WIKI_SUMMARY_PROMPT → LLM → WikiPage │
│                                          │
│  2. 提取实体+概念（2-pass）               │
│     Pass 0: 候选提取（轻量）              │
│     Pass 1-N: 分配引用块                  │
│     WIKI_ENTITY_EXTRACTION_PROMPT        │
│                                          │
│  3. 去重                                  │
│     WIKI_DEDUPLICATION_PROMPT            │
│     与已有页面比对                         │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│         REDUCE 阶段（每实体）              │
│                                          │
│  4. 收集相关文档块                         │
│     - 源文档块优先                         │
│     - 标题关键词匹配补充                   │
│                                          │
│  5. 创建/更新 Wiki 页面                    │
│     WIKI_PAGE_MODIFY_PROMPT → LLM        │
│     输出 Markdown + [[slug|标题]] 链接    │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│         POST 阶段                         │
│                                          │
│  6. 发布草稿页面                          │
│  7. 重建索引页                            │
│  8. 注入跨页面链接（纯文本替换，不调LLM） │
│  9. 清理死链接                            │
└──────────────────────────────────────────┘
```

---

## 三、核心模块详解

### 3.1 `config/settings.py` — 统一配置管理

**职责**：集中管理所有配置项，支持 YAML 文件 + 环境变量 + 代码传参三种方式。

```python
# 配置优先级：代码传参 > 环境变量 > YAML 文件 > 默认值
config = load_config("config.yaml")       # 从 YAML 加载
qa = LocalQA(api_key="sk-xxx", ...)       # 代码传参覆盖
```

**核心配置类**：

| 类 | 管理内容 | 关键参数 |
|---|---------|---------|
| `LLMConfig` | OpenAI 接口配置 | `api_key`, `base_url`, `chat_model`, `embedding_model`, `embedding_dim` |
| `RetrieverConfig` | 检索参数 | `vector_top_k`, `keyword_top_k`, `rerank_top_k`, `mmr_lambda`, `hybrid_alpha` |
| `ChunkerConfig` | 分块参数 | `chunk_size`(512), `chunk_overlap`(64) |
| `AgentConfig` | Agent参数 | `max_iterations`(10), `max_context_tokens`(128K), `parallel_tool_calls` |
| `WikiConfig` | Wiki参数 | `granularity`(standard/focused/exhaustive), `max_concurrent_extractions` |
| `AppConfig` | 全局聚合 | 聚合以上所有 + `data_dir`, `host`, `port` |

---

### 3.2 `models/schemas.py` — 数据模型层

**职责**：使用 Pydantic v2 定义所有数据结构，提供类型校验和序列化。

**核心模型**：

| 模型 | 用途 | 关键字段 |
|------|------|---------|
| `DocumentMetadata` | 文档元数据 | `doc_id`, `filename`, `file_type`, `title`, `chunk_count` |
| `Chunk` | 文档分块（最小检索单元） | `chunk_id`, `doc_id`, `content`, `index`, `metadata`, `token_count` |
| `SearchResult` | 检索结果 | `chunk`, `score`, `match_type`(vector/keyword/graph) |
| `ToolCall` | 工具调用请求 | `call_id`, `name`, `arguments` |
| `ToolResult` | 工具执行结果 | `call_id`, `name`, `output`, `error`, `is_error` |
| `AgentStep` | Agent单步记录 | `step_index`, `thought`, `tool_calls`, `tool_results` |
| `WikiPage` | Wiki页面 | `slug`, `title`, `page_type`, `content`, `out_links`, `source_chunk_ids` |
| `Entity` | 知识图谱实体 | `entity_id`, `title`, `description`, `entity_type`, `frequency` |
| `Relationship` | 知识图谱关系 | `source_entity_id`, `target_entity_id`, `relation_type`, `weight` |

---

### 3.3 `core/document_parser.py` — 文档解析

**职责**：将 PDF / Markdown 文件解析为纯文本，保留结构信息。

**设计模式**：策略模式 + 注册表模式

```
DocumentParser (注册表)
   ├── PDFParser      (策略：PyMuPDF 提取文本)
   └── MarkdownParser (策略：直接读取，保留标题层级)
```

**中文优化**：
- PDF：修复中文断行（`中文\n中文` → `中文中文`）、清除中文与标点间的异常空格
- 自动按文件扩展名选择解析器
- 解析在线程池中执行（不阻塞事件循环）

**关键方法**：
```python
parser = DocumentParser()
text, metadata = await parser.parse("手册.pdf", doc_id="xxx")
# text: 纯文本（含页码标记 "[第 3 页]"）
# metadata: DocumentMetadata(filename, file_type, title, ...)
```

---

### 3.4 `core/chunker.py` — 文本分块

**职责**：将长文本切分为适合检索的小块。

**两种策略**：

| 策略 | 适用场景 | 实现 |
|------|---------|------|
| **语义分块** | Markdown | 按 `#` 标题分段，长段再切分 |
| **固定分块** | PDF等纯文本 | 按段落合并 → 句子切分 → 重叠窗口 |

**中文优化**：
- Token 计数：中文 1.5 字符/token，英文 4 字符/token
- 断句支持：`。！？；` 等中文标点
- 重叠窗口：相邻块共享内容，避免语义断裂

```python
chunker = TextChunker(ChunkerConfig(chunk_size=512, chunk_overlap=64))
chunks = chunker.chunk_text(text, doc_id="xxx", file_type="markdown")
# chunks: [Chunk(content=..., index=0, token_count=150, metadata={"section_title": "## 第一章"}), ...]
```

---

### 3.5 `core/embedder.py` — 向量嵌入

**职责**：调用 OpenAI Embedding API 将文本转为向量。

**特性**：
- **批量嵌入**：自动分批（每批100条），避免 API 限流
- **嵌入缓存**：相同文本不重复调用 API
- **自动重试**：指数退避，最多3次
- **异步**：全 async/await

```python
embedder = Embedder(llm_config)
query_vec = await embedder.embed_query("传感器异常怎么处理？")      # 单条
doc_vecs = await embedder.embed_chunks(chunks)                     # 批量
```

---

### 3.6 `core/vector_store.py` — 向量存储 + BM25 混合检索

**职责**：存储文档块的向量索引和 BM25 索引，支持三种检索模式。

**存储结构**：
```
VectorStore
   ├── FAISS IndexFlatIP    # 内积索引（归一化后=余弦相似度）
   ├── chunks: List[Chunk]  # 文档块列表（有序）
   ├── id_map               # chunk_id → FAISS索引 映射
   └── BM25Okapi            # 关键词检索索引
```

**三种检索模式**：

| 模式 | 方法 | 适用场景 |
|------|------|---------|
| **向量检索** | `search_vector()` | 语义相似，"怎么处理故障" ≈ "故障排除方法" |
| **关键词检索** | `search_keyword()` | 精确匹配，"E003" "Modbus" |
| **混合检索** | `search_hybrid()` | 两者融合，RRF排序，默认推荐 |

**中文优化**：BM25 分词优先使用 jieba，未安装时自动降级为 bigram（相邻两字组合）。

**混合检索算法（RRF）**：
```python
# Reciprocal Rank Fusion
score(chunk) = α × 1/(k+rank_vector) + (1-α) × 1/(k+rank_keyword)
# α=0.7, k=60 (默认)
# 向量检索权重更高，关键词检索补充精确匹配
```

**持久化**：
```python
await vector_store.save("./vector_store/")   # 保存 FAISS + metadata.pkl
await vector_store.load("./vector_store/")   # 恢复
```

---

### 3.7 `core/reranker.py` — LLM 重排 + MMR 去重

**职责**：对检索结果重排序，提高相关性，去除重复。

**两级重排**：

1. **LLM 重排**：让 LLM 对每条结果打分（0-1），复合评分 = 0.6×LLM + 0.3×原始 + 0.1×来源权重
2. **MMR 去重**：Maximal Marginal Relevance，平衡相关性和多样性

```python
# MMR 公式
mmr(d) = λ × Sim(q,d) - (1-λ) × max(Sim(d, d_selected))
# λ=0.7 → 偏向相关性，同时保证多样性
```

---

### 3.8 `core/retriever.py` — 统一检索器

**职责**：编排完整检索流水线，是 RAG 引擎和 Agent 的检索入口。

**流水线**（借鉴 WeKnora Chat Pipeline）：
```
查询理解(可选) → 混合检索 → LLM重排(可选) → MMR去重(可选)
```

```python
retriever = Retriever(config, vector_store, embedder, reranker)

# 快速检索（RAG快速问答用）
results = await retriever.quick_search("E003故障", top_k=5)

# 完整检索（深度问答/Agent用）
results = await retriever.retrieve(SearchParams(query="...", top_k=10))
```

---

### 3.9 `core/rag_engine.py` — RAG 问答引擎

**职责**：检索 + 上下文构建 + LLM 生成，提供三种输出模式。

| 模式 | 方法 | 特点 |
|------|------|------|
| **快速问答** | `quick_chat()` | 跳过查询理解和重排，延迟低 |
| **深度问答** | `deep_chat()` | 完整流水线，更准但更慢 |
| **流式输出** | `stream_chat()` | SSE 逐步输出 |

**上下文构建**：
```python
# _build_context() 将检索结果格式化
"""
[文档: ## 第一章 设备概述, 相关度: 0.85]
本设备为工业级自动化控制系统，型号为 AC-2024...

---

[文档: ### E003 传感器异常, 相关度: 0.72]
传感器信号超出正常范围...
"""
```

---

### 3.10 `agent/engine.py` — ReAct Agent 引擎

**职责**：实现 Think → Analyze → Act → Observe 的推理循环。

**安全机制**：
- **卡死检测**：连续3次相同输出 → 强制合成答案
- **上下文管理**：Token 估算 + 自动压缩早期对话
- **输出截断**：工具输出最大 16KB，防止上下文中毒
- **优雅降级**：LLM 失败时从已有工具结果合成答案
- **最大轮次**：默认10轮，防止无限循环

```python
engine = AgentEngine(config)
engine.register_knowledge_tools(retriever, vector_store)
engine.register_wiki_tools(wiki_manager, vector_store)

response = await engine.run(
    query="分析所有故障代码的处理方法",
    knowledge_bases_info=[{"name": "设备手册", "chunk_count": 50}],
)
# response.answer: 最终答案
# response.agent_steps: [AgentStep(thought=..., tool_calls=..., tool_results=...), ...]
```

---

### 3.11 `agent/tool_registry.py` — 工具注册表

**职责**：管理所有 Agent 工具的生命周期，提供安全防护。

**安全设计**（借鉴 WeKnora）：
- **先注册优先**：防止同名工具劫持
- **参数修复**：LLM 输出 `"true"` → `true`，JSON 字符串 → 对象
- **输出截断**：默认 16KB 上限
- **错误捕获**：工具异常不中断循环

```python
registry = ToolRegistry(max_output_size=16384)
registry.register(KnowledgeSearchTool(retriever))
schema = registry.get_openai_tools_schema()  # 生成 OpenAI function calling 格式
result = await registry.execute_tool(tool_call)
```

---

### 3.12 `agent/tools/` — 内置工具集

| 工具 | 文件 | 用途 |
|------|------|------|
| `thinking` | thinking_and_answer.py | 深度思考/推理（不执行操作） |
| `todo_write` | thinking_and_answer.py | 创建研究计划/待办事项 |
| `final_answer` | thinking_and_answer.py | 提交最终答案（终止循环） |
| `database_query` | thinking_and_answer.py | SQL查询（安全限制） |
| `knowledge_search` | knowledge_search.py | 语义/关键词/混合搜索 |
| `grep_chunks` | knowledge_search.py | 精确关键词匹配 |
| `list_knowledge_chunks` | knowledge_search.py | 列出文档所有块 |
| `wiki_read_page` | wiki_tools.py | 读取Wiki页面 |
| `wiki_write_page` | wiki_tools.py | 创建/更新Wiki页面 |
| `wiki_search` | wiki_tools.py | 搜索Wiki页面 |
| `wiki_read_source_doc` | wiki_tools.py | 读取页面关联的原始块 |
| `wiki_flag_issue` | wiki_tools.py | 标记质量问题 |

---

### 3.13 `wiki/page_manager.py` — Wiki 页面管理

**职责**：Wiki 页面的 CRUD、搜索、跨链接管理、Markdown 导出。

**页面类型**：index / summary / entity / concept / synthesis

**关键方法**：
```python
manager = WikiPageManager(wiki_dir="./wiki_output")
await manager.initialize()
page = await manager.get_page("rag-overview")
await manager.save_page(WikiPage(slug="xxx", title="xxx", content="..."))
await manager.inject_cross_links()     # 自动注入 [[slug|标题]] 链接
await manager.export_all_markdown()    # 导出为 .md 文件
```

---

### 3.14 `wiki/ingest.py` — Wiki 生成管道

**职责**：从原始文档自动生成结构化 Wiki 知识库（Map-Reduce 架构）。

**粒度控制**：
- `focused`：5-10 个核心实体
- `standard`：10-30 个重要实体
- `exhaustive`：尽可能提取所有实体

```python
ingest = WikiIngest(config, wiki_manager)
stats = await ingest.ingest_documents(
    doc_ids=["doc1", "doc2"],
    vector_store=vector_store,
    granularity="standard",
)
# stats: {"pages_created": 15, "pages_updated": 3, "links_injected": 20}
```

---

### 3.15 `wiki/graph_builder.py` — 知识图谱构建

**职责**：从文档中提取实体和关系，构建可视化知识图谱。

**权重计算**（借鉴 WeKnora）：
```python
weight = 0.6 × PMI + 0.4 × Strength
# PMI: 点互信息（共现频率的统计显著性）
# Strength: 共现频率占最大频率的比例
# 归一化到 1-10
```

**输出**：
```python
builder = KnowledgeGraphBuilder(config)
kg = await builder.build_graph(chunks)
mermaid = builder.to_mermaid()   # 生成 Mermaid 可视化语法
```

---

### 3.16 `cli_qa.py` — ⭐ 无端口问答（核心入口）

**职责**：提供最简洁的 Python 函数调用接口，数据全部自动持久化。

**持久化结构**：
```
rag_workspace/
├── vector_store/           # FAISS 向量索引 + BM25 元数据
│   ├── faiss.index         #   FAISS 二进制索引
│   └── metadata.pkl        #   文档块元数据 + id映射
├── chunks/                 # 文档切块缓存（JSON格式）
│   └── {doc_id}.json       #   每个文档一个文件
├── documents.json          # 文档注册表（元数据 + 文件哈希 + 路径）
├── conversation.json       # 对话历史（跨会话保持）
└── config.json             # 运行时配置快照
```

**智能缓存**：
- **文件哈希检测**：相同文件不重复嵌入（SHA256 哈希比对）
- **切块缓存恢复**：向量索引丢失时可从切块缓存快速恢复（仅需重新嵌入）
- **一致性校验**：启动时自动校验注册表与向量索引的一致性

**API 一览**：

| 方法 | 说明 |
|------|------|
| `load_file(path)` | 加载文件（自动缓存检测） |
| `load_directory(dir)` | 加载目录 |
| `remove_document(doc_id)` | 移除文档 |
| `ask(question)` | 问答 → 返回字符串 |
| `ask_with_sources(question)` | 问答 → 返回答案+来源引用 |
| `get_history(last_n)` | 获取对话历史 |
| `clear_history()` | 清空对话历史 |
| `save()` | 手动保存所有状态 |
| `reset()` | 重置工作空间 |
| `status()` | 查看系统状态 |
| `interactive()` | 交互式问答模式 |

---

## 四、使用示例

### 4.1 最简问答

```python
from cli_qa import LocalQA

qa = LocalQA(api_key="sk-xxxxx")
qa.load_file("设备操作手册.pdf")
answer = qa.ask("E003故障怎么处理？")
print(answer)
```

### 4.2 带来源引用

```python
result = qa.ask_with_sources("安全操作规程有哪些？")
print(result["answer"])
for src in result["sources"]:
    print(f"  来源: {src['section']}, 相关度: {src['score']}")
```

### 4.3 加载目录 + 持久化

```python
# 第一次运行：加载并索引
qa = LocalQA(api_key="sk-xxxxx", workspace="./my_data")
qa.load_directory("./docs/")

# 第二次运行：自动从缓存恢复，无需重新嵌入
qa2 = LocalQA(api_key="sk-xxxxx", workspace="./my_data")
qa2.load_file("./docs/手册.pdf")  # → from_cache=True，瞬间加载

# 对话历史也自动恢复
print(qa2.get_history())  # 之前的对话记录
```

### 4.4 使用 Agent 模式

```python
from agent.engine import AgentEngine

engine = AgentEngine(config)
engine.register_knowledge_tools(retriever, vector_store)
response = await engine.run("分析手册中所有故障代码，并整理成表格")
print(response.answer)
```

### 4.5 生成 Wiki 知识库

```python
from wiki.ingest import WikiIngest
from wiki.page_manager import WikiPageManager

manager = WikiPageManager("./wiki_output")
await manager.initialize()
ingest = WikiIngest(config, manager)
stats = await ingest.ingest_documents(
    doc_ids=["doc1"], vector_store=vector_store, granularity="standard"
)
print(f"生成 {stats['pages_created']} 个Wiki页面")
```

### 4.6 构建知识图谱

```python
from wiki.graph_builder import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder(config)
kg = await builder.build_graph(chunks)
mermaid = builder.to_mermaid()  # Mermaid 可视化语法
print(mermaid)
```
