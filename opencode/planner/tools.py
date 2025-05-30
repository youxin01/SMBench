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


def info_pre(ques):
    question = read_file(ques)
    train_path = os.join(os.path.dirname(ques),"train.csv")
    test_path = os.join(os.path.dirname(ques),"test.csv")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    info1 = show_info(train_data)
    info2 = show_info(test_data)
    info3 = check_categorical_columns(train_data, test_data)
    info = f"{question}\n\n #训练集信息\n{info1}\n\n #测试集信息\n{info2}\n\n #以下列应当弃用由于此列当中的特征数目在两数据集当中不一致\n{info3}"
    return info

def info_opt(ques):
    question = read_file(ques)
    user2 = f"以下是我的问题信息：\n{question}"
    return user2

def info_eval(ques):
    dir_path = os.path.dirname(ques)
    data_info=""
    csv_file = [p for p in os.listdir(dir_path) if p.endswith(".csv")]
    if len(csv_file)>0:
        for i in range(len(csv_file)):
            tmp_data = pd.read_csv(os.path.join(dir_path,csv_file[i]))
            tmp_info = show_info(tmp_data)
            data_info += f"\n 附件{csv_file[i]}的基本信息"
            data_info += tmp_info 
    # 问题信息
    question = read_file(ques)

    # 合并
    if len(data_info)>0:
        info_final = f"下面是我的问题:\n{question},\n以及数据集介绍\n{data_info}"
    else:
        info_final = f"下面是我的问题:\n{question}"
    user2 = info_final
    return user2

def info_basic(ques):
    question = read_file(ques)
    user2 = f"以下是我的问题信息：\n{question}"
    return user2
