from utils.log_util import logger
from pipelines.pipe import run_planner, run_rag, run_developer, run_executor
from utils.utils1 import write_file
import os

def main():
    # ==== 手动设置参数 ====
    types = "prediction"  # 问题类型
    ty = "pre"  # 问题类型
    # number = os.listdir("./MMBench/prediction")
    number = ["p9"]  # 选择问题编号
    # agents = ["deepseek", "kimi", "doubao", "qwen32", "qwen72"]
    agents = ["doubao"]
    for i in number:
        for j in agents:
            logger.info(f"Processing {i} with agent {j}...")
            question_path = f"./MMBench/{types}/{i}/question.txt"
            agent = j  # 选择代理
            cover = False
            max_retries = 3
            problem_type = ty

            logger.info("Pipeline started...")

            # ==== 各阶段流程 ====
            plan_path = run_planner(question_path, agent, cover, problem_type)
            func_list = run_rag(question_path, plan_path)
            dev_code_path,dev_sys,dev_user = run_developer(question_path, agent, cover, problem_type, func_list)

            chat_message,out1=run_executor(question_path, agent, dev_code_path, func_list, max_retries,dev_sys,dev_user)
            write_file(os.path.join(os.path.dirname(question_path), f"message_{agent}.json"), chat_message)
            logger.info("Pipeline finished ✅")

if __name__ == "__main__":
    main()