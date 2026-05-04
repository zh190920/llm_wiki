"""
Local RAG System Configuration
================================
借鉴 WeKnora 核心思想的本地 RAG 系统配置管理。
支持环境变量覆盖，所有组件可本地运行，无需外源依赖。
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """全局配置，支持 .env 文件和环境变量覆盖"""

    # ── 基础路径 ──
    BASE_DIR: Path = Field(default=Path(__file__).parent)
    DATA_DIR: Path = Field(default=Path(__file__).parent / "data")
    UPLOAD_DIR: Path = Field(default=Path(__file__).parent / "data" / "uploads")
    VECTOR_DIR: Path = Field(default=Path(__file__).parent / "data" / "vectors")
    WIKI_DIR: Path = Field(default=Path(__file__).parent / "data" / "wiki")
    GRAPH_DIR: Path = Field(default=Path(__file__).parent / "data" / "graphs")

    # ── 服务端口 ──
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── LLM 配置 (OpenAI-compatible API) ──
    LLM_BASE_URL: str = "http://localhost:11434/v1"  # 默认 Ollama
    LLM_API_KEY: str = "ollama"
    LLM_MODEL: str = "qwen2.5:7b"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4096

    # ── Embedding 配置 ──
    # 本地模型: sentence-transformers
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DIMENSION: int = 512
    EMBEDDING_DEVICE: str = "cpu"  # "cuda" 或 "cpu"

    # ── 文档分块策略 ──
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    CHUNK_SEPARATOR: str = "\n\n"

    # ── 检索配置 ──
    VECTOR_TOP_K: int = 5
    KEYWORD_TOP_K: int = 5
    VECTOR_THRESHOLD: float = 0.3
    KEYWORD_THRESHOLD: float = 0.1
    RRF_K: int = 60  # Reciprocal Rank Fusion 参数
    HYBRID_ALPHA: float = 0.7  # 向量检索权重 (1-alpha 为关键词权重)

    # ── Agent 配置 ──
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_PARALLEL_TOOL_CALLS: bool = True
    AGENT_MAX_CONTEXT_TOKENS: int = 8192

    # ── Wiki 配置 ──
    WIKI_GRANULARITY: str = "standard"  # focused / standard / exhaustive
    WIKI_LANGUAGE: str = "中文"
    WIKI_MAX_CONTENT: int = 32768
    WIKI_MAX_DOCS_PER_BATCH: int = 5

    # ── 并发配置 ──
    MAX_CONCURRENT_EMBEDDINGS: int = 8
    MAX_CONCURRENT_LLM_CALLS: int = 4
    MAX_CONCURRENT_GRAPH_EXTRACTIONS: int = 4

    # ── 知识图谱配置 ──
    GRAPH_PMI_WEIGHT: float = 0.6
    GRAPH_STRENGTH_WEIGHT: float = 0.4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        """确保所有数据目录存在"""
        for d in [self.DATA_DIR, self.UPLOAD_DIR, self.VECTOR_DIR,
                  self.WIKI_DIR, self.GRAPH_DIR]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
