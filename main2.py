from utils.log_util import logger
from pipelines.pipe import run_planner, run_rag, run_developer, run_executor
from utils.utils1 import write_file
import os

def main():
    # ==== 手动设置参数 ====
    number = os.listdir("./MMBench/prediction")
    # number = ["p1"]
    for i in number:
        question_path = f"./MMBench/prediction/{i}/question.txt"
        agent = "kimi"
        cover = False
        max_retries = 3
        problem_type = "pre"

        logger.info("Pipeline started...")

        # ==== 各阶段流程 ====
        plan_path = run_planner(question_path, agent, cover, problem_type)
        func_list = run_rag(question_path, plan_path)
        dev_code_path,dev_sys,dev_user = run_developer(question_path, agent, cover, problem_type, func_list)

        chat_message=run_executor(question_path, agent, dev_code_path, func_list, max_retries,dev_sys,dev_user)
        write_file(os.path.join(os.path.dirname(question_path), f"message_{agent}.json"), chat_message)
        logger.info("Pipeline finished ✅")

if __name__ == "__main__":
    main()