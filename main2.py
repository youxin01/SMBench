from utils.log_util import logger
from pipelines.pipe import run_planner, run_rag, run_developer, run_executor

def main():
    # ==== 手动设置参数 ====
    question_path = "./test_case/o7/question.txt"
    agent = "deepseek"
    cover = False
    max_retries = 3
    problem_type = "opt"

    logger.info("Pipeline started...")

    # ==== 各阶段流程 ====
    plan_path = run_planner(question_path, agent, cover, problem_type)
    func_list = run_rag(question_path, plan_path)
    dev_code_path = run_developer(question_path, agent, cover, problem_type, func_list)
    run_executor(question_path, agent, dev_code_path, func_list, max_retries)

    logger.info("Pipeline finished ✅")

if __name__ == "__main__":
    main()