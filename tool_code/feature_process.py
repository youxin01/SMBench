import pandas as pd
import numpy as np
from typing import List, Dict, Union
from functools import reduce
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def get_header():
    header = """import pandas as pd
import numpy as np
from typing import List, Dict, Union
from functools import reduce
from sklearn.preprocessing import LabelEncoder
import pickle
import os"""
    return header
# 特征编码（one-hot）
def one_hot_encode(data: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    all_col = data.columns
    if isinstance(columns, str):
        columns = [columns]
    columns = [col for col in columns if col in all_col]
    return pd.get_dummies(data, columns=columns)

def label_encode(
    data: pd.DataFrame,
    columns: Union[str, List[str], None],
    encoder_path: str = "./encoder_dir/label_encoder.pkl"
) -> pd.DataFrame:
    if columns is None:
        raise ValueError("columns 参数不能为空，请至少指定一个列名。")

    if isinstance(columns, str):
        columns = [columns]

    encoded_data = data.copy()
    label_encoders = {}

    for col in columns:
        le = LabelEncoder()
        encoded_data[col] = le.fit_transform(encoded_data[col])
        label_encoders[col] = le

    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)

    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoders, f)

    return encoded_data

def label_decode(
    data: Union[np.ndarray, pd.DataFrame],
    target_col: str,
    encoder_path: str = "./encoder_dir/label_encoder.pkl"
):
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

    with open(encoder_path, 'rb') as f:
        label_encoders = pickle.load(f)

    if target_col not in label_encoders:
        raise ValueError(f"No encoder found for column: {target_col}")

    le = label_encoders[target_col]

    if isinstance(data, pd.DataFrame):
        if target_col not in data.columns:
            raise ValueError(f"Column '{target_col}' not found in the DataFrame.")

        decoded_data = data.copy()
        decoded_data[target_col] = le.inverse_transform(decoded_data[target_col].astype(int))
        return decoded_data

    elif isinstance(data, np.ndarray):
        if data.ndim == 2 and data.shape[1] == 1:
            encoded = data[:, 0]
        elif data.ndim == 1:
            encoded = data
        else:
            raise ValueError("NumPy data must be shape (n,) or (n, 1).")

        decoded = le.inverse_transform(encoded)
        return pd.DataFrame({target_col: decoded})

    else:
        raise TypeError("Input must be a pandas DataFrame or a NumPy ndarray.")
# 映射函数
def map_func(data: pd.DataFrame, col_dict: dict):
    for col, func in col_dict.items():
        data[col] = data[col].map(func)
    return data

# 利用两个数据集产生的特征
def two_data_func(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    feature: str,
    col: list,
    func1: list,
    func2: str,
    group_by: list
) -> pd.DataFrame:
    # 参数校验
    if len(col) != 2 or len(func1) != 2:
        raise ValueError("参数格式错误: col和func1需2个元素，group_by需1个元素")

    # 定义聚合函数字典
    func_map = {
        'sum': 'sum',
        'mean': 'mean',
        'count': 'size',
        'max': 'max',
        'min': 'min'
    }

    # 处理聚合函数
    agg_funcs = []
    for f in func1:
        if isinstance(f, str):
            agg_funcs.append(func_map[f.lower()])
        else:
            agg_funcs.append(f)

    # 执行分组聚合
    agg1 = data1.groupby(group_by)[col[0]].agg(agg_funcs[0]).rename('x')
    agg2 = data2.groupby(group_by)[col[1]].agg(agg_funcs[1]).rename('y')

    # 合并结果（外连接填充0）
    merged = pd.merge(agg1, agg2, left_index=True, right_index=True, how='outer').fillna(0)

    # 表达式计算
    try:
        merged[feature] = merged.eval(func2.replace("x", "x").replace("y", "y"))
    except:
        raise ValueError(f"表达式解析失败: {func2}")

    return merged[feature].reset_index()

# 根据一列进行特征计算
def generate_single_feature(
    data: pd.DataFrame,
    feature: str,
    feature_config: dict,
    group_key: str = None
) -> pd.DataFrame:

    # 参数校验
    required_params = ['agg_funcs']
    if not all(k in feature_config for k in required_params):
        missing = [k for k in required_params if k not in feature_config]
        raise ValueError(f"缺失必要配置项: {missing}")

    # 初始化分组对象
    grouped = data.groupby(group_key) if group_key else data

    # 执行聚合计算
    agg_results = []
    for func in feature_config['agg_funcs']:
        # 解析函数参数格式：支持 (column, func) 或 func
        if isinstance(func, tuple):
            col, func_def = func
            result = grouped[col].agg(func_def)
        else:
            col = feature_config.get('column', data.columns[0])
            result = grouped[col].agg(func)
        
        agg_results.append(result.to_frame(f'{feature}_temp_{len(agg_results)}'))

    # 合并中间结果
    df_agg = pd.concat(agg_results, axis=1)
    
    # 执行后处理
    if 'post_process' in feature_config:
        final_result = feature_config['post_process'](df_agg)
        if isinstance(final_result, pd.Series):
            final_result = final_result.to_frame(feature)
    else:
        # 直接取第一个聚合结果
        final_result = df_agg.iloc[:, 0].to_frame(feature)

    # 重置索引保持数据对齐
    return final_result.reset_index() if group_key else final_result


# 合并多个特征表
def merge_features(features: list[pd.DataFrame], on: list[str], how: str = 'outer') -> pd.DataFrame:
    # 确保 on 是列表
    if isinstance(on, str):
        on = [on]

    # 参数校验，检查所有 DataFrame 是否包含 on 指定的列
    missing_dfs = [i for i, df in enumerate(features) if not all(col in df.columns for col in on)]
    if missing_dfs:
        raise ValueError(f"以下 DataFrame 缺少对齐列 {on}: {missing_dfs}")

    # 逐步合并，避免列名冲突
    def merge_func(left, right):
        return pd.merge(left, right, on=on, how=how, suffixes=(None, "_dup"))

    # 使用 reduce 依次合并所有特征表
    merged_df = reduce(merge_func, features)

    # 移除重复列（避免 "_dup" 后缀的列）
    duplicate_cols = [col for col in merged_df.columns if col.endswith('_dup')]
    merged_df.drop(columns=duplicate_cols, inplace=True)

    return merged_df


