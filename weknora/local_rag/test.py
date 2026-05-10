from cli_qa import LocalQA
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

qa = LocalQA(
    api_key="",
    base_url="https://api.siliconflow.cn/v1",
    chat_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    embedding_model="BAAI/bge-m3",
    embedding_dim=1024,
)

# 加载文档
# qa.load_file("操作手册.pdf")
# qa.load_directory(r"E:\vs_git\manual")

# 设置别名
# qa._doc_router.set_aliases({
#     "设备A": "XX型设备操作手册",
#     "安全规程": "安全操作规程",
#     "SOP": "标准操作流程",
# })

# 第一步：构建知识图谱（必须先执行，图谱才会注入到检索管线）
# graph = qa.build_knowledge_graph()
# print(f"实体数: {graph['entities']}, 关系数: {graph['relationships']}")
# qa.save()

# 第二步：使用图谱增强检索的问答
# use_graph=True 启用三源RRF融合（向量+关键词+图谱）
answer = qa.ask("xxx", use_graph=False, deep=True)
print(answer)

# 深度模式 + 图谱增强
# answer = qa.ask("安全操作规程和设备维护有什么关系？", deep=True, use_graph=False)
# print(answer)

# 带来源的图谱增强问答
# result = qa.ask_with_sources("设备A的故障码E003和安全操作规程有什么关联？", use_graph=False)
# print(result["answer"])
# for src in result["sources"]:
#     print(f"  来源: {src['section']}, 相关度: {src['score']}, 匹配类型: {src['match_type']}")