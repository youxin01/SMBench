import pandas as pd
from io import StringIO
import re
import json
import ast

# 展示数据集的基本信息
def show_info(data: pd.DataFrame) -> None:
    # 捕获 data.info() 的输出
    info_buffer = StringIO()
    data.info(buf=info_buffer)
    info_str = info_buffer.getvalue()

    # 创建统一的字符串
    output = f"""
    ## 数据集的基本信息
    {info_str}

    ## 数据集的描述性统计信息
    {data.describe()}

    ## 数据集的前5行
    {data.head()}

    ## 数据集的后5行
    {data.tail()}

    ## 数据集的缺失值情况
    {data.isnull().sum()}

    ## 数据集的重复值情况
    {data.duplicated().sum()}

    ## 数据集的数据类型
    {data.dtypes}

    ## 数据集的形状
    {data.shape}

    ## 数据集的列名
    {list(data.columns)}

    ## 数据集的索引
    {data.index}
    """
    return output

# 检测差异的特征列融入promot
def check_categorical_columns(train, test):
    categorical_columns_test = test.select_dtypes(include=['object', 'category']).columns
    diff_col =[]
    for col in categorical_columns_test:
        train_categories = set(train[col].unique())
        test_categories = set(test[col].unique())

        if train_categories != test_categories:
            diff_col.append(col)
    return diff_col

# 提取选取的函数
def extract_functions(file_path):
    with open(file_path, "r") as f:
        md_str = f.read()
    pattern = r'```python\n(\[.*?\])\n```'
    matches = re.findall(pattern, md_str, re.DOTALL)

    parsed_results = []
    for mat in matches:
        fixed_mat = mat.replace("\n", "")
        fixed_mat = re.sub(r'\bfalse\b', 'False', fixed_mat)
        fixed_mat = re.sub(r'\btrue\b', 'True', fixed_mat)
        fixed_mat = re.sub(r'\bnull\b', 'None', fixed_mat)
        try:
            parsed_results.append(fixed_mat)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")

    return parsed_results

def extract_code(file_path):
    with open(file_path, "r") as f:
        md_str = f.read()
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, md_str, re.DOTALL)
    return [m.strip() for m in matches]

def convert_str_function(json_str):
    data = ast.literal_eval(json_str)
    results = []
    for item in data:
        name = item['name']
        args = item['args']
        arg_strs = []
        for k, v in args.items():
            # 保证字典等非字符串不被误转义为字符串
            if isinstance(v, str):
                arg_strs.append(f'{k}="{v}"')
            else:
                arg_strs.append(f"{k}={v}")
        arg_str = ", ".join(arg_strs)
        call_str = f'result = {name}({arg_str})'
        results.append(call_str)

    return "\n".join(results)

# 读取文件内容
def read_file(file_path: str):
    """
    Read the content of a file and return it as a string.
    """
    if file_path.endswith('txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    if file_path.endswith('csv'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    if file_path.endswith('md'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
        

def write_file(file_path: str, content):
    """
    Write content to a file based on its extension.
    
    Args:
        file_path: Path to the file to be written
        content: Content to write (string, list[str], or dict/list[dict])
    """
    ext = file_path.lower()

    if ext.endswith('.txt') or ext.endswith('.md'):
        with open(file_path, 'w', encoding='utf-8') as f:
            if isinstance(content, str):
                f.write(content)
            elif isinstance(content, list) and all(isinstance(x, str) for x in content):
                f.write('\n'.join(content))
            else:
                f.write(json.dumps(content, ensure_ascii=False, indent=2))
    elif ext.endswith('.csv'):
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            if isinstance(content, list) and all(isinstance(x, str) for x in content):
                f.writelines(content)
            else:
                f.write(str(content))
    elif ext.endswith('.json'):
        with open(file_path, 'w', encoding='utf-8') as f:
            if isinstance(content, str):
                f.write(content)
            else:
                f.write(json.dumps(content, ensure_ascii=False, indent=2))
    else:
        raise ValueError("Unsupported file format")