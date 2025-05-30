import pandas as pd
from io import StringIO
import re
import json

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
def extract_functions(md_str):
    pattern = r'```python\n(\[.*?\])\n```'
    matches = re.findall(pattern, md_str, re.DOTALL)

    parsed_results = []
    for mat in matches:
        fixed_mat = mat.replace("\n", "")
        fixed_mat = re.sub(r'\bFalse\b', 'false', fixed_mat)
        fixed_mat = re.sub(r'\bTrue\b', 'true', fixed_mat)
        fixed_mat = re.sub(r'\bNone\b', 'null', fixed_mat)
        try:
            parsed_results.append(json.loads(fixed_mat))
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")

    return parsed_results