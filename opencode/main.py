from planner.tools import get_pla_user
from planner.prompt import get_planer
from utils.api import gpt_chat
from utils.utils1 import read_file,write_file,extract_functions,convert_str_function
import os
from planner.tools import get_plan
from utils.rag import ChoromaDBManager
from developer.prompt import get_developer
from developer.tools import get_dev_user
from utils.notebook_serializer import NotebookSerializer
from utils.local_interpreter import LocalCodeInterpreter
import sys
import json
import importlib
import re

def func_preprocess(functions):
    func_call = convert_str_function(functions[0])
    data_processed = func_call.split("\n")
    updated_lines = []

    train = 0
    test = 0

    for tmp in data_processed:
        if 'data="train_data"' in tmp:
            if train == 0:
                data_name = 'train_data'
            else:
                data_name = f'train_data{train}'
            output_var = f'train_data{train + 1}'
            tmp = tmp.replace('data="train_data"', f'data={data_name}')
            tmp = re.sub(r'^result\s*=', f'{output_var} =', tmp)
            train += 1
        elif 'data="test_data"' in tmp:
            if test == 0:
                data_name = 'test_data'
            else:
                data_name = f'test_data{test}'
            output_var = f'test_data{test + 1}'
            tmp = tmp.replace('data="test_data"', f'data={data_name}')
            tmp = re.sub(r'^result\s*=', f'{output_var} =', tmp)
            test += 1
        updated_lines.append(tmp)
    return updated_lines,train,test

def code_exe(prepare,type,functions,question):
    notebook = NotebookSerializer("./")
    code_interpreter = LocalCodeInterpreter(work_dir="./",notebook_serializer=notebook,task_id="111")
    code_interpreter.initialize()

    for func in prepare:
        file = set()
        if func["source_file"] not in file:
            module_name = f"tool_code.{func['source_file'].replace('.md', '')}"
            module = importlib.import_module(module_name)
            header = module.get_header()
            tool = f"from tool_code.{func['source_file'].replace('.md', '')} import {func['tool_name']}"
            code_interpreter.execute_code(header)
            code_interpreter.execute_code(tool)
            file.add(func['source_file'])
    if (len(functions) > 1) and (type in ["pre","eval"]):
        update_func,train_index,test_index = func_preprocess(functions)
        data_load ="""train_data = pd.read_csv("{train_path}")
        test_data = pd.read_csv("{test_path}")"""
        path1 = os.path.join(os.path.dirname(question),"train.csv")
        path2 = os.path.join(os.path.dirname(question),"test.csv")
        data_load=data_load.format(train_path=path1,test_path=path2)
        code_interpreter.execute_code(data_load)
        for i in update_func:
            code_interpreter.execute_code(i)
        model = convert_str_function(functions[1])
        model1=model.replace('"train_data"', f'train_data{train_index}').replace('"test_data"', f'test_data{test_index}')
        code_interpreter.execute_code(model1)
    else:
        model = convert_str_function(functions[0])
        code_interpreter.execute_code(model)
        code_interpreter.execute_code("print(result)")
    code_interpreter.cleanup()

if __name__ == "__main__":
    type1="opt"
    question="./test_case/o8/question.txt"
    agent ="qwen32"
    # planner环节
    print("planner start")
    planer = get_planer(problem_type=type1)
    info = get_pla_user(ques=question,problem_type=type1)
    # response1 = gpt_chat(sys=planer,user=info,provider="deepseek")
    # write_file(os.path.join(os.path.dirname(question),f"plan_{agent}.txt"), response1)
    response1 = read_file(os.path.join(os.path.dirname(question),f"plan_{agent}.txt"))

    # RAG环节
    print("RAG start")
    chroma_db = ChoromaDBManager(os.path.join(os.path.dirname(question), "tool_db"))
    chroma_db.store_tools_to_db(dir_path="./tool_doc_md")
    plan = get_plan(str_path=os.path.join(os.path.dirname(question),f"plan_{agent}.txt"))
    prepare_funcs=chroma_db.get_all_tools(plan)

    #developer环节
    print("developer start")
    devloper_prompt = get_developer(problem_type=type1,func=prepare_funcs)
    user2 = get_dev_user(question=question,problem_type=type1)
    # response2 = gpt_chat(sys=devloper_prompt,user=user2)
    # write_file(os.path.join(os.path.dirname(question),f"dev_{agent}.txt"), response2)
    response2 = read_file(os.path.join(os.path.dirname(question),f"dev_{agent}.txt"))

    # 代码执行环节
    print("code execute start")
    exec_func = extract_functions(os.path.join(os.path.dirname(question),f"dev_{agent}.txt"))
    code_exe(prepare=prepare_funcs,type=type1,functions=exec_func,question=question)

