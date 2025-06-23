from utils.utils1 import read_file, show_info, check_categorical_columns
import pandas as pd
import os
import re

def get_pla_user(problem_type,ques):
    if problem_type == 'pre':
        return info_pre(ques)
    elif problem_type == 'opt':
        return info_opt(ques)
    elif problem_type == 'eval':
        return info_eval(ques)
    elif problem_type == 'basic':
        return info_basic(ques)
    else:
        raise ValueError("Unknown problem type: {}".format(problem_type))

def get_plan(str_path):
    with open(str_path, "r", encoding="utf-8") as f:
        content = f.read()
    # 用正则表达式提取每个 STEP 段落内容
    steps = re.findall(r'## STEP \d+\n(.*?)(?=(?:\n## STEP \d+|\Z))', content, re.DOTALL)

    # 去除每段前后的空白符
    steps = [step.strip() for step in steps]
    return steps


def info_pre(question):
    ques = read_file(question)
    dir_path = os.path.dirname(question)
    file_names = os.listdir(dir_path)
    if "test.csv" in file_names:
        # 非时序问题
        train_path = os.path.join(os.path.dirname(question),"train.csv")
        test_path = os.path.join(os.path.dirname(question),"test.csv")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        info1 = show_info(train_data)
        info2 = show_info(test_data)
        info3 = check_categorical_columns(train_data, test_data)
        info4 = f"文件路径信息：\n训练集路径: '{train_path}'\n测试集路径: '{test_path}'"
        info = f"{ques}\n\n {info4}\n#训练集信息\n{info1}\n\n #测试集信息\n{info2}\n\n #以下列应当弃用由于此列当中的特征数目在两数据集当中不一致\n{info3}"
    else:
        # 时序类问题
        train_path = os.path.join(os.path.dirname(question),"train.csv")
        train_data = pd.read_csv(train_path)
        info1 = show_info(train_data)
        info2 = f"文件路径信息：\n训练集路径: '{train_path}'"
        info = f"{ques}\n\n {info2} \n#训练集信息\n{info1}\n\n"
    return info

def info_opt(ques):
    question = read_file(ques)
    user2 = f"以下是我的问题信息：\n{question}"
    return user2

def info_eval(question):
    dir_path = os.path.dirname(question)
    data_info=""
    csv_file = [p for p in os.listdir(dir_path) if p.endswith(".csv")]
    if len(csv_file)>0:
        for i in csv_file:
            tmp_path = os.path.join(dir_path,i)
            tmp_data = pd.read_csv(tmp_path)
            tmp_info = show_info(tmp_data)
            data_info += f"\n 【附件 {i} 的基本信息】 \n 路径为：{tmp_path}\n 介绍：{tmp_info}\n"

    # 问题信息
    ques = read_file(question)

    # 合并
    if len(data_info)>0:
        info_final = f"下面是我的问题:\n{ques},\n以及数据集介绍\n{data_info}"
    else:
        info_final = f"下面是我的问题:\n{ques}"
    user2 = info_final
    return user2

def info_basic(ques):
    question = read_file(ques)
    user2 = f"以下是我的问题信息：\n{question}"
    return user2
