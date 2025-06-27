import pandas as pd
import numpy as np
import networkx as nx

def get_header():
    header = """import pandas as pd
import numpy as np
import networkx as nx"""
    return header
# Entropy Weight Method(熵权法)
def ewn_analysis(
    data: pd.DataFrame,
    criteria: dict = None,
    index_col: str = None,
    positive_indicators: list = None,
    negative_indicators: list = None,
    enable_grade_evaluation: bool = False,
    grade_labels: list = None,
    level_ranges: list = None
) -> pd.DataFrame:
    
    # 数据预处理
    df = data.set_index(index_col).apply(pd.to_numeric, errors='coerce')
    indicators = positive_indicators + negative_indicators if (positive_indicators and negative_indicators) else df.columns.tolist()
    
    # 指标方向处理逻辑优化
    indicator_directions = {}
    # 优先使用明确指定的正负指标
    if positive_indicators or negative_indicators:
        for ind in indicators:
            indicator_directions[ind] = 1 if ind in positive_indicators else -1
    # 次选自动判断（根据criteria阈值结构）
    elif criteria:
        for ind in indicators:
            thresholds = sorted([v[0] for v in criteria[ind].values()])
            indicator_directions[ind] = 1 if thresholds[-1] > thresholds[0] else -1
    
    # 熵权法计算权重（优化标准化逻辑）
    def entropy_weight(df):
        # 正向化处理
        normalized = df.copy()
        for col in indicators:
            if indicator_directions.get(col, 1) == -1:
                normalized[col] = df[col].max() - df[col]
            else:
                normalized[col] = df[col] - df[col].min()
        
        # 标准化计算
        p = normalized / normalized.sum()
        p = p.replace(0, 1e-10)
        
        # 熵权计算
        k = 1 / np.log(len(df))
        entropy = -k * (p * np.log(p)).sum()
        weights = (1 - entropy) / (1 - entropy).sum()
        return weights
    
    for ind in indicators:
        if ind not in df.columns:
            raise ValueError(f"指标 {ind} 不存在于数据中")
        if df[ind].isnull().any():
            raise ValueError(f"指标 {ind} 包含 NaN 值，请处理后再进行熵权计算")
    weights = entropy_weight(df[indicators])
    print("指标权重：", weights.to_dict())
    # 构建评价结果
    results = []
    for idx, row in df.iterrows():
        record = {index_col: idx}
        
        # 经济效益专用输出
        record.update({
            '综合得分': sum(row[ind] * weights[ind] for ind in indicators)
        })
        
        # 条件判断：是否启用等级评价
        if enable_grade_evaluation and criteria:
            def evaluate_level(value, thresholds):
                for level, (lower, upper) in thresholds.items():
                    if lower <= value < upper:
                        return level
                return list(thresholds.keys())[-1]
            
            # 增加指标等级评价
            levels = {f'{ind}_等级': evaluate_level(row[ind], criteria[ind]) for ind in indicators}
            record.update(levels)
            
            # 综合等级判定
            if grade_labels and level_ranges:
                total_score = record['综合得分']
                print("综合得分：", total_score)
                final_level = grade_labels[np.digitize(total_score, level_ranges)-1]
                record['综合等级'] = final_level
        
        results.append(record)
    if enable_grade_evaluation:
        return pd.DataFrame(results).set_index(index_col)
    else:
        tmp = pd.DataFrame(results).set_index(index_col)
        final = data.copy()
        final['综合得分'] = tmp['综合得分'].values
        return final

# TOPSIS Method(层次分析法)
def topsis_analysis(data: pd.DataFrame,
                    index_col: str, 
                    types: dict, 
                    params: dict, 
                    weights: dict) -> np.ndarray:
    
    # 确定参与计算的指标，使用 weights 的 keys 作为计算指标
    features = list(weights.keys())
    
    # 检查所有指标是否都存在于 DataFrame 及对应字典中
    for col in features:
        if data[col].isnull().any():
            raise ValueError(f"输入数据列 '{col}' 中含有 NaN 值，处理或者不使用此特征")
        if col not in data.columns:
            raise ValueError(f"原始数据中缺少指标: {col}")
        if col not in types:
            raise ValueError(f"缺少指标 {col} 的类型定义")
        if col not in params:
            params[col] = None  # 没有参数则设为 None

    # 提取参与计算的矩阵，转换为 float 类型
    matrix = data[features].astype(float).to_numpy()
    n_samples, n_features = matrix.shape

    # STEP 1：正向化处理
    # 将所有指标转换为“值越大越好”的形式
    positive_matrix = np.copy(matrix)
    for j, col in enumerate(features):
        indicator_type = types[col]
        param = params[col]
        col_data = positive_matrix[:, j]
        
        if indicator_type == 0:  # 极小型指标：取 最大值 - 当前值
            max_val = np.max(col_data)
            positive_matrix[:, j] = max_val - col_data
        elif indicator_type == 1:  # 极大型指标：不变
            pass
        elif indicator_type == 2:  # 中间型指标：最佳值为 param
            if param is None:
                raise ValueError(f"中间型指标 {col} 需要提供最佳值参数")
            best_value = param
            deviation = np.abs(col_data - best_value)
            max_deviation = np.max(deviation)
            # 避免 max_deviation 为 0
            if max_deviation == 0:
                positive_matrix[:, j] = 1
            else:
                positive_matrix[:, j] = 1 - deviation / max_deviation
        elif indicator_type == 3:  # 区间型指标：param 为 [lower, upper]
            if param is None or len(param) != 2:
                raise ValueError(f"区间型指标 {col} 需要提供 [下限, 上限] 参数")
            lower, upper = param
            # 计算离区间上下界的偏离
            deviations = np.minimum(np.abs(col_data - lower), np.abs(col_data - upper))
            max_deviation = np.max(deviations)
            # 在区间内的赋值为 1，否则按偏离比例计算
            if max_deviation == 0:
                positive_matrix[:, j] = 1
            else:
                positive_matrix[:, j] = np.where((col_data >= lower) & (col_data <= upper), 
                                                   1, 
                                                   1 - deviations / max_deviation)
        else:
            raise ValueError(f"未知的指标类型: {indicator_type} for {col}")
    
    # STEP 2：标准化
    # 对每一列进行向量标准化：每列除以该列的 L2 范数
    norm_matrix = np.copy(positive_matrix)
    for j in range(n_features):
        col_norm = np.linalg.norm(norm_matrix[:, j])
        if col_norm != 0:
            norm_matrix[:, j] = norm_matrix[:, j] / col_norm
        else:
            norm_matrix[:, j] = 0

    # STEP 3：加权及 TOPSIS 计算
    # 构造权重向量，确保顺序与 features 一致
    weight_vector = np.array([weights[col] for col in features])
    weighted_matrix = norm_matrix * weight_vector

    # 理想正解（最优解）与理想负解（最劣解）
    ideal_positive = np.max(weighted_matrix, axis=0)
    ideal_negative = np.min(weighted_matrix, axis=0)

    # 计算各样本到理想正解和理想负解的欧式距离
    distance_positive = np.sqrt(np.sum((weighted_matrix - ideal_positive) ** 2, axis=1))
    distance_negative = np.sqrt(np.sum((weighted_matrix - ideal_negative) ** 2, axis=1))

    # TOPSIS 得分
    scores = distance_negative / (distance_positive + distance_negative)
    # 归一化得分（可选：使所有得分之和为1）
    normalized_score = scores / np.sum(scores)
    data = data.copy()
    data["TOPSIS得分"]=normalized_score

    return data[[index_col, "TOPSIS得分"]]

# Weighted Scoring Method(加权评分法)
def wsm_evaluation(data: pd.DataFrame, 
                     indicator_markers: dict, 
                     weight_allocation: dict, 
                     index_col: str,
                     normalization: str = 'minmax') -> pd.DataFrame:
    """
    计算加权评分。
    
    参数:
      data: 原始数据，包含各项指标（如变异系数、平均供货量、订单完成率、订单满足率等）
      indicator_markers: 指标标记字典，例如 {"变异系数": -1, "平均供货量": 1, "订单完成率": 1, "订单满足率": 1}
                         其中 1 表示正向指标（数值越大越好），-1 表示反向指标（数值越小越好）
      weight_allocation: 权重分配字典，例如 {"变异系数": 0.2, "平均供货量": 0.3, "订单完成率": 0.25, "订单满足率": 0.25}
      index_col: 用作索引的列名称（例如供应商ID）
      normalization: 归一化方法（目前仅支持 'minmax'）
    
    返回:
      包含 index_col 及加权评分的 DataFrame
    """
    # 复制数据，设置索引
    data = data.copy()
    data.set_index(index_col, inplace=True)
    for col in weight_allocation.keys():
        if col not in data.columns:
            raise ValueError(f"指标 {col} 不存在于数据中")
        if data[col].isnull().any():
            raise ValueError(f"指标 {col} 包含 NaN 值，请处理后再进行加权评分")
    normalized_df = pd.DataFrame(index=data.index)
    
    # 对每个指标进行归一化处理
    for col, weight in weight_allocation.items():
        values = data[col]
        min_val = values.min()
        max_val = values.max()
        # 如果该指标数值全相同，则归一化为 1
        if max_val == min_val:
            normalized = pd.Series(1, index=values.index)
        else:
            if normalization == 'minmax':
                # 根据指标标记进行归一化：正向指标 (1) 按 (x-min)/(max-min)，反向指标 (-1) 按 (max-x)/(max-min)
                if indicator_markers.get(col, 1) == 1:
                    normalized = (values - min_val) / (max_val - min_val)
                else:
                    normalized = (max_val - values) / (max_val - min_val)
            else:
                raise ValueError("目前只支持 'minmax' 归一化方法。")
        normalized_df[col] = normalized

    # 计算加权评分：各指标归一化后的值乘以对应权重后求和
    weighted_score = normalized_df.mul(pd.Series(weight_allocation)).sum(axis=1)
    
    result = weighted_score.to_frame('weighted_score')
    result.reset_index(inplace=True)
    
    return result

# PageRank 算法
def pagerank(graph:dict, alpha:float = 0.85):
    """
    使用 NetworkX 计算 PageRank
    :param graph: 字典，键为 (u, v) 表示有向边，值为权重
    :param alpha: 阻尼系数，通常取 0.85
    :return: PageRank 值的字典
    """
    G = nx.DiGraph()
    for (u, v), w in graph.items():
        G.add_edge(u, v, weight=w)

    return nx.pagerank(G, alpha=alpha)