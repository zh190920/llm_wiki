"""
Local RAG System - 主入口
基于 WeKnora 核心设计思想的本地 RAG 系统

启动方式:
    1. 命令行启动: python main.py
    2. 使用 uvicorn: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    3. 指定配置: python main.py --config config.yaml

环境变量:
    OPENAI_API_KEY   - OpenAI API Key (必需)
    OPENAI_BASE_URL  - OpenAI API 基础URL (可选，默认 https://api.openai.com/v1)
    CHAT_MODEL       - 聊天模型 (可选，默认 gpt-4o-mini)
    EMBEDDING_MODEL  - 嵌入模型 (可选，默认 text-embedding-3-small)
"""
import argparse
import logging
import os
import sys

import uvicorn

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import AppConfig, load_config
from api.server import create_app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Local RAG System")
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径 (默认: config.yaml)"
    )
    parser.add_argument(
        "--host", type=str, default=None,
        help="服务监听地址"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="服务监听端口"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="工作进程数"
    )
    args = parser.parse_args()

    # 加载配置
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"
    )
    config = load_config(config_path) if os.path.exists(config_path) else AppConfig()

    # 命令行参数覆盖
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.workers:
        config.workers = args.workers

    # 检查必要配置
    if not config.llm.api_key:
        logger.warning(
            "⚠️  OPENAI_API_KEY 未设置！请通过环境变量或配置文件设置。"
            "\n   export OPENAI_API_KEY=sk-xxxxx"
            "\n   或在 config.yaml 中设置 llm.api_key"
        )

    # 创建应用
    app = create_app(config)

    # 启动服务
    logger.info("=" * 60)
    logger.info("  Local RAG System - 本地 RAG 系统")
    logger.info("  基于 WeKnora 核心设计思想")
    logger.info("=" * 60)
    logger.info(f"  地址: http://{config.host}:{config.port}")
    logger.info(f"  文档: http://{config.host}:{config.port}/docs")
    logger.info(f"  模型: {config.llm.chat_model} / {config.llm.embedding_model}")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=1,  # 单 worker（多 worker 需要额外的状态共享机制）
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
