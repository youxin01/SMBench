import argparse
import os
from planner.tools import get_pla_user, get_plan
from planner.prompt import get_planer
from developer.prompt import get_developer
from developer.tools import get_dev_user
from utils.api import gpt_chat
from utils.utils1 import read_file, write_file, extract_code
from utils.rag import ChoromaDBManager
from utils.notebook_serializer import NotebookSerializer
from utils.local_interpreter import LocalCodeInterpreter
from utils.log_util import logger
import importlib


def run_planner(question_path: str, agent: str, cover: bool, problem_type: str) -> str:
    plan_path = os.path.join(os.path.dirname(question_path), f"plan_{agent}.txt")
    if not cover and os.path.exists(plan_path):
        logger.info(f"{plan_path} already exists, skipping planner step.")
        return plan_path
    
    logger.info("Running planner...")
    planer_prompt = get_planer(problem_type=problem_type)
    user_input = get_pla_user(ques=question_path, problem_type=problem_type)
    response = gpt_chat(sys=planer_prompt, user=user_input, provider=agent)
    write_file(plan_path, response)
    return plan_path


def run_rag(question_path: str, plan_path: str) -> list:
    db_path = os.path.join(os.path.dirname(question_path), "tool_db")
    exist= os.path.exists(db_path)
    chroma_db = ChoromaDBManager(db_path)

    if not exist:
        logger.info("tool_db does not exist. Initializing and storing tools...")
        chroma_db.store_tools_to_db(dir_path="./tool_doc_md")
    else:
        logger.info("tool_db already exists. Skipping tool storage.")

    plan = get_plan(str_path=plan_path)
    return chroma_db.get_all_tools(plan)


def run_developer(question_path: str, agent: str, cover: bool, problem_type: str, func_list: list) -> str:
    dev_path = os.path.join(os.path.dirname(question_path), f"dev_{agent}.txt")
    developer_prompt = get_developer(problem_type=problem_type, func=func_list)
    user_input = get_dev_user(question=question_path, problem_type=problem_type)
    if not cover and os.path.exists(dev_path):
        logger.info(f"{dev_path} already exists, skipping developer step.")
        return dev_path,developer_prompt,user_input

    logger.info("Running developer...")
    developer_prompt = get_developer(problem_type=problem_type, func=func_list)
    user_input = get_dev_user(question=question_path, problem_type=problem_type)
    response = gpt_chat(sys=developer_prompt, user=user_input, provider=agent)
    write_file(dev_path, response)
    return dev_path,developer_prompt,user_input


def code_header(funcs, code_interpreter):
    loaded_files = set()
    for func in funcs:
        if func["source_file"] not in loaded_files:
            module_name = f"tool_code.{func['source_file'].replace('.md', '')}"
            module = importlib.import_module(module_name)
            code_interpreter.execute_code(module.get_header())
        code_interpreter.execute_code(f"from {module_name} import {func['tool_name']}")
        loaded_files.add(func["source_file"])


def correct_code(file_path, code, error_message, agent,messages):
    critic_prompt = """
    下面是我运行的代码和我的报错信息：
    <code>{}</code>
    报错信息：
    <error>{}</error>
    请你帮我分析代码错误原因，并且给出修改后的代码，要求：
    1. 只需要在原本代码上进行修改，尽可能不增添新的代码。
    2. 返回的代码用```python开头和```结尾。
    """
    user = critic_prompt.format(code, error_message)
    messages.append({"role": "user", "content": user})
    response = gpt_chat(messages=messages, provider=agent)
    messages.append({"role": "assistant", "content": response})
    write_file(file_path, response)
    return response,messages


def run_executor(question_path: str, agent: str, dev_code_path: str, func_list: list, max_retries: int,dev_sys: str = None, dev_user: str = None):
    logger.info("Running code executor...")
    notebook = NotebookSerializer(work_dir=os.path.dirname(question_path),notebook_name=f"notebook_{agent}.ipynb")
    code_interpreter = LocalCodeInterpreter(work_dir="./", notebook_serializer=notebook, task_id="111")
    code_interpreter.initialize()
    code_header(func_list, code_interpreter)
    message=[]

    message.append({"role": "system", "content": dev_sys})
    message.append({"role": "user", "content": dev_user})
    with open(dev_code_path, "r") as f:
        md_str = f.read()
    message.append({"role": "assistant", "content": md_str})
    exec_code = extract_code(dev_code_path)
    if not exec_code:
        raise ValueError("extract_code failed. No executable code found.")

    retry_count, success = 0, False
    while not success and retry_count < max_retries:
        text, error, msg = code_interpreter.execute_code(exec_code[-1])
        if error:
            logger.warning(f"Code error on attempt {retry_count + 1}, retrying...")
            file2 = os.path.join(os.path.dirname(question_path), f"critic_{agent}_{retry_count + 1}.txt")
            corrected,message = correct_code(
                file_path=file2,
                code=exec_code[-1],
                error_message=msg,
                agent=agent,
                messages=message
            )
            exec_code = extract_code(file2)
            if not exec_code:
                raise ValueError("Correction failed: extract_code returned empty result.")
            retry_count += 1
        else:
            success = True
            logger.info("Code executed successfully.")

    code_interpreter.cleanup()
    return message
