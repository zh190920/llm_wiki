"""
Local RAG System — 主入口
================================
借鉴 WeKnora 核心思想的本地 RAG 系统

三大核心能力:
1. RAG 快速问答 — 向量+关键词混合检索，适合日常知识查询
2. ReAct Agent 智能推理 — 自主编排知识检索和工具调用，完成复杂多步任务
3. Wiki 模式 — Agent 从原始文档中自治生成相互链接的 Markdown 知识库与可视化知识图谱

特性:
- 纯 Python 实现，兼容异步高并发
- 无需外源知识、网络搜索
- 本地 Embedding + FAISS + BM25 + LLM
- 完整的 FastAPI RESTful API
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from knowledge_base.manager import KnowledgeBaseManager
from api.routes import create_router


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="Local RAG System",
        description="""
## 本地 RAG 系统 API

借鉴 WeKnora 核心思想的本地知识管理框架。

### 三大核心能力

| 能力 | 说明 |
|------|------|
| **RAG 快速问答** | 向量+关键词混合检索，快速准确回答日常知识查询 |
| **ReAct Agent 智能推理** | 自主编排知识检索、图谱查询等工具，完成复杂多步推理任务 |
| **Wiki 模式** | 从原始文档自动生成相互链接的 Markdown 知识库与可视化知识图谱 |

### 快速开始

1. `POST /knowledge-bases` — 创建知识库
2. `POST /knowledge-bases/{kb_id}/documents/upload` — 上传文档
3. `POST /chat` — 开始问答（mode=rag 或 mode=agent）
4. `GET /wiki/{kb_id}/pages` — 浏览自动生成的 Wiki 知识库
5. `GET /wiki/{kb_id}/graph` — 查看知识图谱
""",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 初始化知识库管理器
    kb_manager = KnowledgeBaseManager()

    # 注册路由
    router = create_router(kb_manager)
    app.include_router(router, prefix="/api/v1")

    @app.on_event("startup")
    async def startup():
        print(f"""
╔══════════════════════════════════════════════════════╗
║                                                      ║
║   🚀 Local RAG System 已启动                         ║
║                                                      ║
║   借鉴 WeKnora 核心思想的本地知识管理框架              ║
║                                                      ║
║   📚 RAG 快速问答  —  向量+关键词混合检索             ║
║   🤖 Agent 智能推理  —  ReAct 多步推理               ║
║   📖 Wiki 模式  —  自动生成链接知识库+知识图谱         ║
║                                                      ║
║   API 文档: http://{settings.HOST}:{settings.PORT}/docs    ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        workers=1,
        loop="uvloop",
        log_level="info",
    )
