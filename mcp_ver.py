# math_server.py
from mcp.server.fastmcp import FastMCP
import argparse
from utils.log_util import logger
from pipelines.pipe import run_planner, run_rag, run_developer, run_executor
import os
from utils.utils1 import write_file

mcp = FastMCP("Modeling Problem")

@mcp.tool()
def solve_modeling_problem(question_path: str, type: str = "opt", agent: str = "deepseek", cover: bool = False, max_retries: int = 3) -> str:
    """
    自动化开发任务的主流程。

    Parameters:
    - question_path: str，问题文件的路径
    - type: str，问题类型，可选值"opt"(优化问题),"pre"(预测问题),"eval"(评价问题)，"basic"(基础问题,包括假设检验等)。
    - agent: str，使用的 LLM，如 "deepseek", "kimi", "doubao", "qwen32", "qwen72"。
    - cover: bool，是否强制覆盖重跑
    - max_retries: int，代码修复最大重试次数，默认为3.

    Returns:
    - str: 执行结束后的消息或结果路径。
    """
    logger.info("🚀 MCP Tool: Pipeline started...")

    plan_path = run_planner(question_path, agent, cover,type)
    func_list = run_rag(question_path, plan_path)
    dev_code_path,dev_sys,dev_user = run_developer(question_path, agent, cover,type, func_list)

    chat_message,out1=run_executor(question_path, agent, dev_code_path, func_list, max_retries,dev_sys,dev_user)
    write_file(os.path.join(os.path.dirname(question_path), f"message_{agent}.json"), chat_message)

    logger.info("✅ MCP Tool: Pipeline finished.")
    return out1

@mcp.tool()
def read_file(file_path: str):
    """
    读取指定文件的内容（支持 .txt, .csv, .md），并以字符串形式返回。

    Parameters:
    - file_path: 文件路径

    Returns:
    - 文件内容字符串
    """
    if file_path.endswith('txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    if file_path.endswith('csv'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    if file_path.endswith('md'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


if __name__ == "__main__":
    mcp.run(transport="stdio")