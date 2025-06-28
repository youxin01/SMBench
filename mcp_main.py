from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import asyncio
import sys
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage

def extract_to_txt(response, output_file1: str, output_file2: str) -> None:
    """
    提取response中的有效信息并分别保存到两个文件
    
    Args:
        response: 包含消息的字典，应有"messages"键
        output_file1: 保存原始响应数据的文件名
        output_file2: 保存提取后的结构化信息的文件名
    """
    # 保存原始响应到第一个文件
    try:
        with open(output_file1, 'w', encoding='utf-8') as f:
            f.write(str(response))
        print(f"Successfully saved raw response to {output_file1}")
    except Exception as e:
        print(f"Error writing raw response to file: {e}")
        return

    # 提取结构化信息
    content_lines = []
    
    for msg in response["messages"]:
        # 处理用户消息（Prompt）
        if isinstance(msg, HumanMessage):
            content_lines.append(f"[USER PROMPT]\n{msg.content}\n\n")
        
        # 处理AI的工具调用
        elif isinstance(msg, AIMessage) and msg.additional_kwargs.get("tool_calls"):
            for call in msg.additional_kwargs["tool_calls"]:
                tool_call = call["function"]
                content_lines.append(
                    f"[FUNCTION CALL]\n"
                    f"Name: {tool_call.get('name', 'N/A')}\n"
                    f"Arguments: {tool_call.get('arguments', 'N/A')}\n\n"
                )
        # 处理AI的最终回答
        elif isinstance(msg, AIMessage):
            content_lines.append(f"[AI RESPONSE]\n{msg.content}\n\n")
    
    if not content_lines:
        print("No valid messages found in the response.")
        return
    
    # 保存提取后的信息到第二个文件
    try:
        with open(output_file2, "w", encoding="utf-8") as f:
            f.writelines(content_lines)
        print(f"Successfully saved extracted information to {output_file2}")
    except Exception as e:
        print(f"Error writing extracted information to file: {e}")

# # 1. 首先设置事件循环策略（必须在所有异步操作之前）
# if sys.platform == "win32":
#     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# 2. 初始化模型
model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key="",
    base_url="https://api.deepseek.com/v1",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接 math_server.py 的绝对路径
math_server_path = os.path.join(current_dir, "mcp_ver.py")

# 4. 配置服务器参数（使用绝对路径）
server_params = StdioServerParameters(
    command="python",
    args=[math_server_path],  # 使用绝对路径
)

system="""
你是一个数学建模专家，擅长调用tools来解决各种建模问题。
"""

async def main():
    try:
        print("正在连接服务器...")
        async with stdio_client(server_params) as (read, write):
            print("服务器连接成功，创建会话...")
            print(math_server_path)
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("正在加载工具...")
                tools = await load_mcp_tools(session)
                print(f"已加载工具: {[t.name for t in tools]}")

                agent = create_react_agent(model, tools)
                print("代理已创建，开始对话...")

                # 初始对话历史
                messages = [SystemMessage(content=system)]

                while True:
                    user_input = input("\n🧑 你说：").strip()
                    if user_input.lower() in {"exit", "quit"}:
                        print("✅ 对话结束")
                        break

                    # 加入用户消息
                    messages.append(HumanMessage(content=user_input))

                    # 转换为标准格式
                    raw_messages = []
                    for m in messages:
                        if isinstance(m, SystemMessage):
                            raw_messages.append({"role": "system", "content": m.content})
                        elif isinstance(m, HumanMessage):
                            raw_messages.append({"role": "user", "content": m.content})
                        elif isinstance(m, AIMessage):
                            raw_messages.append({"role": "assistant", "content": m.content})

                    # 调用 agent
                    response = await agent.ainvoke({"messages": raw_messages})

                    # 提取 AI 回复
                    reply = response['messages'][-1].content
                    print("\n🤖 AI 回答：")
                    print(reply)

                    ai_msg = response['messages'][-1]
                    messages.append(ai_msg)

                    # # 选做：保存每轮结果
                # print(response)
                extract_to_txt(response=response, output_file1="./original.txt", output_file2="./ex_result.txt")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise


# 5. 确保主程序入口正确
if __name__ == "__main__":
    print("启动主程序...")
    asyncio.run(main())