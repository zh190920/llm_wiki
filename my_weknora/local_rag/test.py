from cli_qa import LocalQA

# 初始化（传入 OpenAI 配置）
qa = LocalQA(
    api_key="",
    base_url="https://api.siliconflow.cn/v1",  # 可选
    chat_model="Qwen/Qwen3-30B-A3B-Instruct-2507",              # 可选
)

# 加载文档
qa.load_file(r"E:\vs_git\19011157-SC_A19《H5U&Easy系列可编程逻辑控制器编程手册》.pdf")
# qa.load_file("技术文档.md")
# qa.load_directory("./docs/")  # 也可加载整个目录

# 问答
# answer = qa.ask("opcua数据类型有哪些")
# print("answer1: ", answer)

# 带来源引用的问答
result = qa.ask_with_sources("opcua数据类型有哪些")
print("answer2: ", result["answer"])
for src in result["sources"]:
    print(f"  来源: {src['section']}, 相关度: {src['score']}")

# 交互式问答（终端对话）
qa.interactive()