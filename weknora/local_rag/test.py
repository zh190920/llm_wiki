import logging

from cli_qa import LocalQA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


qa = LocalQA(api_key="")

# 第一步：加载文档
qa.load_file(r"E:\vs_git\19011157-SC_A19《H5U&Easy系列可编程逻辑控制器编程手册》.pdf")
# qa.load_directory("./docs/")

# 第二步：构建知识图谱（构建后自动开启图增强检索）
# result = qa.build_graph()
result = qa.load_graph()
print(result)
# 输出: {"entities": 45, "relationships": 32, "graph_enabled": True, 
#        "message": "图谱构建完成: 45 个实体, 32 条关系，图增强检索已自动开启"}

# 开启
# qa.enable_graph_search(True)

# # 关闭
# qa.enable_graph_search(False)

# status = qa.status()
# print(status["graph"])
# {'entities': 45, 'relationships': 32, 'graph_enabled': True}

# 第三步：使用深度模式问答（图增强检索会自动生效）
answer = qa.ask("easy520是否支持fins", deep=True)

print("answer: ", answer)

# 手动控制图增强检索开关
qa.enable_graph_search(False)  # 关闭
qa.enable_graph_search(True)   # 重新开启

# 查看状态
status = qa.status()
print(status["graph"])  # {"entities": 45, "relationships": 32, "graph_enabled": True}