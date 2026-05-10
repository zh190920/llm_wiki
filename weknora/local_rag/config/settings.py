"""
配置管理模块 - 基于 pydantic-settings 的统一配置管理
支持 YAML 配置文件 + 环境变量覆盖
"""
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """LLM 相关配置"""
    api_key: str = Field(default="", alias="OPENAI_API_KEY")
    base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    chat_model: str = Field(default="gpt-4o-mini", alias="CHAT_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.3)
    timeout: float = Field(default=120.0)

    model_config = {"populate_by_name": True}


class RetrieverConfig(BaseSettings):
    """检索相关配置"""
    vector_top_k: int = Field(default=10, description="向量检索返回数量")
    keyword_top_k: int = Field(default=10, description="关键词检索返回数量")
    rerank_top_k: int = Field(default=5, description="重排后保留数量")
    similarity_threshold: float = Field(default=0.5, description="相似度阈值")
    mmr_lambda: float = Field(default=0.7, description="MMR 多样性参数 (0=最大多样性, 1=最大相关性)")
    hybrid_alpha: float = Field(default=0.7, description="混合检索中向量检索权重")


class ChunkerConfig(BaseSettings):
    """文本分块配置"""
    chunk_size: int = Field(default=512, description="每个块的最大 token 数")
    chunk_overlap: int = Field(default=64, description="相邻块重叠 token 数")
    separator: str = Field(default="\n\n", description="分块分隔符")


class AgentConfig(BaseSettings):
    """Agent 相关配置"""
    max_iterations: int = Field(default=10, description="ReAct 最大循环次数")
    max_context_tokens: int = Field(default=128000, description="最大上下文 token 数")
    parallel_tool_calls: bool = Field(default=True, description="是否允许并行工具调用")
    thinking_enabled: bool = Field(default=True, description="是否启用思考工具")
    max_tool_output_size: int = Field(default=16384, description="工具输出最大字符数")


class WikiConfig(BaseSettings):
    """Wiki 模式配置"""
    granularity: str = Field(default="standard", description="提取粒度: focused/standard/exhaustive")
    max_concurrent_extractions: int = Field(default=4, description="最大并发提取数")
    chunk_batch_size: int = Field(default=5, description="每个批次处理的块数")
    debounce_seconds: float = Field(default=5.0, description="去抖动延迟秒数")
    wiki_dir: str = Field(default="./wiki_output", description="Wiki 输出目录")


class AppConfig(BaseSettings):
    """应用全局配置"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    wiki: WikiConfig = Field(default_factory=WikiConfig)

    # 数据存储
    data_dir: str = Field(default="./data", description="数据存储目录")
    vector_store_dir: str = Field(default="./data/vector_store", description="向量存储目录")
    graph_store_path: str = Field(default="./data/knowledge_graph.json", description="知识图谱存储路径")

    # API 服务
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)

    model_config = {"populate_by_name": True}


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """加载配置，优先级：环境变量 > YAML 配置文件 > 默认值"""
    config = AppConfig()

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if "llm" in yaml_data:
            for k, v in yaml_data["llm"].items():
                if hasattr(config.llm, k):
                    setattr(config.llm, k, v)
        if "retriever" in yaml_data:
            for k, v in yaml_data["retriever"].items():
                if hasattr(config.retriever, k):
                    setattr(config.retriever, k, v)
        if "chunker" in yaml_data:
            for k, v in yaml_data["chunker"].items():
                if hasattr(config.chunker, k):
                    setattr(config.chunker, k, v)
        if "agent" in yaml_data:
            for k, v in yaml_data["agent"].items():
                if hasattr(config.agent, k):
                    setattr(config.agent, k, v)
        if "wiki" in yaml_data:
            for k, v in yaml_data["wiki"].items():
                if hasattr(config.wiki, k):
                    setattr(config.wiki, k, v)

        # 顶层配置
        for k in ("data_dir", "vector_store_dir", "graph_store_path", "host", "port", "workers"):
            if k in yaml_data:
                setattr(config, k, yaml_data[k])

    return config
