from planner.tools import get_pla_user
from planner.prompt import get_planer
from utils.api import gpt_chat
from utils.utils1 import read_file,write_file,extract_code
import os
from planner.tools import get_plan
from utils.rag import ChoromaDBManager
from developer.prompt import get_developer
from developer.tools import get_dev_user
from utils.notebook_serializer import NotebookSerializer
from utils.local_interpreter import LocalCodeInterpreter
from utils.log_util import logger
import sys
import json
import importlib
import re

def correct_code(file,code,error_message,agent):
    critic_prompt ="""
    下面是我的代码和我的报错信息：
    <code>{}</code>
    报错信息：
    <error>{}</error>
    请你帮我分析代码错误原因，并且给出修改后的代码，要求：
    1. 在原本代码上进行修改，尽可能不增添新的代码。
    2. 返回的代码用```python开头和```结尾。
    """
    system ="You are a helpful assistant."
    user = critic_prompt.format(code,error_message)
    response = gpt_chat(sys=system,user=user,provider=agent)
    write_file(file_path=file,content=response)
    return response

def code_header(funcs,code_interpreter):
    for func in funcs:
        file = set()
        if func["source_file"] not in file:
            module_name = f"tool_code.{func['source_file'].replace('.md', '')}"
            module = importlib.import_module(module_name)
            header = module.get_header()
            tool = f"from tool_code.{func['source_file'].replace('.md', '')} import {func['tool_name']}"
            code_interpreter.execute_code(header)
            code_interpreter.execute_code(tool)
            file.add(func['source_file'])
    return 

if __name__ == "__main__":
    type1="opt"
    question="./test_case/o7/question.txt"
    agent ="deepseek"
    cover = False
    # planner环节
    print("planner start")
    planer = get_planer(problem_type=type1)
    info = get_pla_user(ques=question,problem_type=type1)
    if (not cover) and os.path.exists(os.path.join(os.path.dirname(question),f"plan_{agent}.txt")):
        print(f"plan_{agent}.txt already exists, skipping planner step.")
        response1 = read_file(os.path.join(os.path.dirname(question),f"plan_{agent}.txt"))
    else:
        print(f"plan_{agent}.txt does not exist, running planner step.")
        response1 = gpt_chat(sys=planer,user=info,provider="deepseek")
        write_file(os.path.join(os.path.dirname(question),f"plan_{agent}.txt"), response1)

    # RAG环节
    print("RAG start")
    if not os.path.exists(os.path.join(os.path.dirname(question), "tool_db")):
        chroma_db = ChoromaDBManager(os.path.join(os.path.dirname(question), "tool_db"))
        chroma_db.store_tools_to_db(dir_path="./tool_doc_md")
    else:
        print("tool_db already exists, skipping tool storage step.")
        chroma_db = ChoromaDBManager(os.path.join(os.path.dirname(question), "tool_db"))
    plan = get_plan(str_path=os.path.join(os.path.dirname(question),f"plan_{agent}.txt"))
    prepare_funcs=chroma_db.get_all_tools(plan)

    #developer环节
    print("developer start")
    if cover and os.path.exists(os.path.join(os.path.dirname(question),f"dev_{agent}.txt")):
        print(f"dev_{agent}.txt already exists, skipping developer step.")
        response2 = read_file(os.path.join(os.path.dirname(question),f"dev_{agent}.txt"))
    else:
        devloper_prompt = get_developer(problem_type=type1,func=prepare_funcs)
        user2 = get_dev_user(question=question,problem_type=type1)
        response2 = gpt_chat(sys=devloper_prompt,user=user2,provider=agent)
        write_file(os.path.join(os.path.dirname(question),f"dev_{agent}.txt"), response2)

    # 代码执行环节
    print("code execute start")
    notebook = NotebookSerializer("./")
    code_interpreter = LocalCodeInterpreter(work_dir="./",notebook_serializer=notebook,task_id="111")
    code_interpreter.initialize()
    code_header(prepare_funcs, code_interpreter)
    exec_code = extract_code(os.path.join(os.path.dirname(question),f"dev_{agent}.txt"))

    finish = True
    max_retries = 3
    retry_count = 0

    if not exec_code:
        raise ValueError("exec_code 为空，无法执行")

    while finish and retry_count < max_retries:
        text, error_occurred, error_message = code_interpreter.execute_code(exec_code[-1])
        if error_occurred:
            print(f"代码执行错误，正在尝试修正第{retry_count+1}次...")
            tmp_code = correct_code(os.path.join(os.path.dirname(question),f"critic_{agent}_{retry_count+1}.txt"),exec_code, error_message, agent)
            exec_code = extract_code(tmp_code)
            if not exec_code:
                raise ValueError("correct_code 后 extract_code 为空，无法继续执行")
            retry_count += 1
        else:
            finish = False
            print("执行完成")

    code_interpreter.cleanup()





