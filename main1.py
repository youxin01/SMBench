import argparse
from utils.log_util import logger
from pipelines.pipe import run_planner, run_rag, run_developer, run_executor
import os
from utils.utils1 import write_file
def main():
    parser = argparse.ArgumentParser(description="Open-source pipeline for auto development tasks")
    parser.add_argument("--question", default="./test_case/o7/question.txt",required=True, help="Path to the question file")
    parser.add_argument("--type", default="opt", help="Problem type")
    parser.add_argument("--agent", default="deepseek", help="LLM agent provider")
    parser.add_argument("--cover", action="store_true", help="Force re-run all steps")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries for code correction")
    
    args = parser.parse_args()

    logger.info("Pipeline started...")

    plan_path = run_planner(args.question, args.agent, args.cover, args.type)
    func_list = run_rag(args.question, plan_path)
    dev_code_path = run_developer(args.question, args.agent, args.cover, args.type, func_list)
    chat_message,out1=run_executor(args.question, args.agent, dev_code_path, func_list, args.max_retries)
    write_file(os.path.join(os.path.dirname(args.question_path), f"message_{args.agent}.json"), chat_message)
    logger.info("Pipeline finished ✅")


if __name__ == "__main__":
    main()