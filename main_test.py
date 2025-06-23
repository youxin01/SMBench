from utils.log_util import logger
from pipelines.pipe import run_planner, run_rag, run_developer, run_executor
from utils.utils1 import write_file
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
# types =["prediction","evaluation","optimization","basic"]
# ty =["pre","eval","opt","basic"]

types = "optimization"
ty = "opt"
agents = ["deepseek", "kimi", "doubao", "qwen32", "qwen72"]
# agents = ["kimi"]  # 可以扩展为多个
path = f"./MMBench/{types}"
question_dirs = os.listdir(path)

def process(agent, j):
    import os  # 每个进程需重新导入
    from pipelines.pipe import run_planner, run_rag, run_developer, run_executor
    from utils.utils1 import write_file
    from utils.log_util import logger

    question_path = f"./MMBench/{types}/{j}/question.txt"
    cover = False
    max_retries = 3
    problem_type = ty

    logger.info(f"[{agent}] Processing {question_path}...")

    plan_path = run_planner(question_path, agent, cover, problem_type)
    func_list = run_rag(question_path, plan_path)
    dev_code_path, dev_sys, dev_user = run_developer(question_path, agent, cover, problem_type, func_list)

    chat_message = run_executor(question_path, agent, dev_code_path, func_list, max_retries, dev_sys, dev_user)
    output_path = os.path.join(os.path.dirname(question_path), f"message_{agent}.json")
    write_file(output_path, chat_message)

    logger.info(f"[{agent}] Finished {question_path} ✅")
    return output_path

if __name__ == "__main__":
   for q in question_dirs:
        logger.info(f"🧩 Starting batch for: {q}")
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process, agent, q) for agent in agents]

            for future in as_completed(futures):
                try:
                    result_path = future.result()
                    logger.info(f"✅ Completed: {result_path}")
                except Exception as e:
                    logger.error(f"❌ Error: {e}")
        
        # 可选：在两批之间加个短暂停顿
        time.sleep(2)