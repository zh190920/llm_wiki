# Local RAG System

> 借鉴 [WeKnora](https://github.com/Tencent/WeKnora) 核心思想的本地 RAG 系统

## 核心理念

本项目从腾讯 WeKnora 项目中提取三大核心思想，用纯 Python 实现了一个**完全本地运行**的知识管理框架：

| 能力 | 说明 |
|------|------|
| **RAG 快速问答** | 向量检索 + BM25 关键词检索混合搜索，通过 RRF 融合算法获得更全面的检索结果，适合日常知识查询 |
| **ReAct Agent 智能推理** | 渐进式多步推理引擎，自主编排知识检索、图谱查询、深度思考等工具完成复杂多步任务 |
| **Wiki 模式** | 从原始文档中自动提取实体和概念，生成相互链接的 Markdown Wiki 知识库，并构建可视化知识图谱 |

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI RESTful API                    │
│  /chat (RAG/Agent)  /search  /wiki  /graph  /documents  │
└──────────────────────┬──────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
┌───▼───┐      ┌──────▼──────┐    ┌──────▼──────┐
│  RAG  │      │ Agent Engine │    │   Wiki      │
│ Query │      │  (ReAct)     │    │  Pipeline   │
│ Mode  │      │  Think→Act   │    │  Map-Reduce │
└───┬───┘      │  →Observe    │    └──────┬──────┘
    │          └──────┬──────┘           │
    │                 │                  │
    ▼                 ▼                  ▼
┌──────────────────────────────────────────────────┐
│              Hybrid Search Engine                 │
│     Vector Search (FAISS) + BM25 + RRF Fusion    │
└────────────────────┬─────────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼────┐    ┌──────▼──────┐  ┌─────▼─────┐
│ FAISS  │    │   BM25      │  │  Knowledge │
│ Vector │    │  Keywords   │  │   Graph    │
│ Store  │    │   Search    │  │ (NetworkX) │
└───┬────┘    └─────────────┘  └───────────┘
    │
┌───▼────────┐    ┌──────────────────┐
│ Embedding  │    │  Document Parser │
│ (Sentence  │    │  PDF/DOCX/TXT/   │
│ Transformers│   │  MD/HTML/PPTX    │
└────────────┘    └──────────────────┘
```

## 关键设计借鉴

### 1. 从 WeKnora AgentEngine 借鉴

- **ReAct 循环**: Think → Analyze → Act → Observe 四阶段迭代
- **并行工具调用**: 多工具并发执行，提升效率
- **循环检测**: 识别重复响应，避免无限循环
- **上下文窗口管理**: 自动裁剪历史消息，控制 token 消耗
- **final_answer 工具**: 强制 Agent 使用工具输出最终答案

### 2. 从 WeKnora HybridSearch 借鉴

- **RRF 融合**: 使用 Reciprocal Rank Fusion 合并向量检索和关键词检索结果
- **过量搜索**: 搜索 5x 候选再截断，保证召回率
- **可配置权重**: alpha 参数控制向量/关键词权重比例

### 3. 从 WeKnora WikiIngestService 借鉴

- **Map-Reduce 流水线**: Map 阶段提取实体/概念，Reduce 阶段合并更新
- **两阶段提取**: 候选 Slug 提取 + 分块引用关联
- **实体去重**: LLM 驱动的智能去重，防止同名实体重复创建
- **交叉链接注入**: 自动检测内容中的 Wiki 页面标题并转换为链接
- **索引页自动重建**: 每次入库后自动更新 Wiki 索引

### 4. 从 WeKnora graphBuilder 借鉴

- **PMI + Strength 权重**: 使用点互信息和关系强度综合计算边权重
- **并发实体/关系提取**: 限制并发度，避免 LLM 过载
- **Mermaid 可视化**: 自动生成知识图谱的 Mermaid 格式图表

## 快速开始

### 环境要求

- Python 3.10+
- Ollama (或其他 OpenAI-compatible LLM 服务)
- 至少 4GB 内存

### 安装

```bash
cd local_rag
pip install -r requirements.txt
```

### 配置

创建 `.env` 文件（可选，所有配置有默认值）：

```env
# LLM 配置 (默认使用 Ollama)
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b

# Embedding 模型
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu  # 或 cuda

# 检索参数
VECTOR_TOP_K=5
KEYWORD_TOP_K=5
HYBRID_ALPHA=0.7  # 向量检索权重
```

### 启动

```bash
python main.py
```

访问 http://localhost:8000/docs 查看 API 文档。

### 使用示例

```python
import httpx
import asyncio

BASE = "http://localhost:8000/api/v1"

async def main():
    async with httpx.AsyncClient(timeout=120) as client:
        # 1. 创建知识库
        resp = await client.post(f"{BASE}/knowledge-bases", json={
            "name": "技术文档库",
            "description": "存放技术文档和知识",
        })
        kb = resp.json()
        kb_id = kb["id"]

        # 2. 上传文档
        with open("doc.pdf", "rb") as f:
            resp = await client.post(
                f"{BASE}/knowledge-bases/{kb_id}/documents/upload",
                files={"file": ("doc.pdf", f, "application/pdf")},
            )

        # 3. RAG 快速问答
        resp = await client.post(f"{BASE}/chat", json={
            "query": "什么是 RAG？",
            "knowledge_base_ids": [kb_id],
            "mode": "rag",
        })
        print(resp.json()["answer"])

        # 4. Agent 智能推理
        resp = await client.post(f"{BASE}/chat", json={
            "query": "对比 RAG 和微调的优缺点",
            "knowledge_base_ids": [kb_id],
            "mode": "agent",
        })
        print(resp.json()["answer"])

        # 5. 浏览 Wiki
        resp = await client.get(f"{BASE}/wiki/{kb_id}/pages")
        pages = resp.json()
        for p in pages:
            print(f"  [{p['page_type']}] {p['title']}")

        # 6. 知识图谱
        resp = await client.get(f"{BASE}/wiki/{kb_id}/graph")
        graph = resp.json()
        print(f"节点: {len(graph['nodes'])}, 边: {len(graph['edges'])}")

asyncio.run(main())
```

## 项目结构

```
local_rag/
├── main.py                      # FastAPI 应用入口
├── config.py                    # 全局配置管理
├── requirements.txt             # 依赖
│
├── core/                        # 核心模块
│   ├── document_parser.py       # 多格式文档解析
│   ├── chunker.py              # 文本分块（语义+固定大小+父子分块）
│   ├── embedding.py            # 本地 Embedding (sentence-transformers)
│   ├── vector_store.py         # FAISS 向量存储
│   ├── keyword_search.py       # BM25 关键词搜索
│   ├── hybrid_search.py        # 混合检索 + RRF 融合
│   ├── knowledge_graph.py      # 知识图谱构建
│   └── llm_client.py           # 异步 LLM 客户端
│
├── agent/                       # ReAct Agent 模块
│   ├── engine.py               # Agent 引擎 (Think→Act→Observe)
│   ├── tools.py                # 工具注册与执行
│   ├── prompts.py              # 系统提示词构建
│   └── state.py                # Agent 状态管理
│
├── wiki/                        # Wiki 模式模块
│   ├── page_manager.py         # Wiki 页面 CRUD + 双向链接
│   └── ingest.py               # Map-Reduce 入库流水线
│
├── knowledge_base/              # 知识库管理
│   ├── manager.py              # 统一管理器（整合所有流程）
│   └── models.py               # 数据模型
│
└── api/                         # API 层
    ├── routes.py               # FastAPI 路由
    └── schemas.py              # 请求/响应模型
```

## 异步高并发设计

- **异步 LLM 客户端**: 使用 `AsyncOpenAI` + `asyncio.Semaphore` 控制并发
- **并发 Embedding**: 批量编码 + 信号量限流
- **并行工具调用**: Agent 支持同时执行多个工具
- **异步文档处理**: 文档解析、分块、向量化全链路 async
- **知识图谱并发提取**: 限制并发度的实体/关系提取

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| LLM | OpenAI-compatible API (Ollama 等) |
| Embedding | sentence-transformers (BAAI/bge-small-zh-v1.5) |
| 向量存储 | FAISS (IndexFlatIP) |
| 关键词搜索 | BM25 + jieba 中文分词 |
| 知识图谱 | NetworkX |
| 文档解析 | pypdf, python-docx, python-pptx, openpyxl, BeautifulSoup |

## 许可证

MIT License
