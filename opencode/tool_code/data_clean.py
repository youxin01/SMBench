import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List,Optional

def get_header():
    header = """import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List,Optional"""
    return header

# 缺失值处理
def fill_missing_values(data: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None, strategy: str = 'mean') -> pd.DataFrame:
    processed_data = data.copy()
    columns_del = []
    if columns is None:
        columns = data.columns
    if strategy=='median' or strategy=='mean':
        columns_remain = [i for i in columns if pd.api.types.is_numeric_dtype(data[i])]
        columns_del = [ i for i in columns if i not in columns_remain]
        columns = columns_remain
        
    if isinstance(columns, str):
        columns = [columns]

    if strategy == 'auto':
        # 删除包含缺失值的行
        processed_data = data.dropna(subset=columns, axis=0)
    else:
        for col in columns:
            try:
                if strategy == 'mean':
                    processed_data[col] = data[col].fillna(data[col].mean())
                elif strategy == 'median':
                    processed_data[col] = data[col].fillna(data[col].median())
                elif strategy == 'mode':
                    processed_data[col] = data[col].fillna(data[col].mode()[0])
            except Exception as e:
                print(f"Error occurred while filling missing values for column '{col}': {e}")
        if len(columns_del)>0:
            for i in columns_del:
                processed_data[i] = data[i].fillna(data[i].mode()[0])
    return processed_data

# 去除缺失值较多的列
def remove_columns_with_missing_data(data: pd.DataFrame, thresh: float = 0.5, columns: Union[str, List[str]] = None) -> pd.DataFrame:

    if not 0 <= thresh <= 1:
        raise ValueError("thresh must be between 0 and 1")

    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        data_subset = data[columns]
    else:
        data_subset = data

    max_missing = int(thresh * len(data_subset))

    columns_to_keep = data_subset.columns[data_subset.isna().sum() < max_missing]

    if columns is not None:
        columns_to_keep = columns_to_keep.union(data.columns.difference(columns))

    return data[columns_to_keep]

# 离群点处理（zscore）
def detect_and_handle_outliers_zscore(data: pd.DataFrame, columns: Union[str, List[str]], threshold: float = 3.0, method: str = 'remove') -> pd.DataFrame:
    processed_data = data.copy()
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        mean = data[column].mean()
        std = data[column].std()
        z_scores = (data[column] - mean) / std

        if method == 'clip':
            # Define the bounds
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            # Apply clipping only to values exceeding the threshold
            processed_data.loc[z_scores > threshold, column] = upper_bound
            processed_data.loc[z_scores < -threshold, column] = lower_bound
        elif method == 'remove':
            processed_data = data[abs(z_scores) <= threshold]
        else:
            raise ValueError("Invalid method. Choose from 'clip' or 'remove'.")

    return processed_data

# 离群点处理（iqr）
def detect_and_handle_outliers_iqr(data: pd.DataFrame, columns: Union[str, List[str]], factor: float = 1.5, method: str = 'remove') -> pd.DataFrame:
    processed_data = data.copy()
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        if method == 'clip':
            processed_data[column] = data[column].clip(lower_bound, upper_bound)
        elif method == 'remove':
            processed_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        else:
            raise ValueError("Invalid method. Choose from 'clip' or 'remove'.")

    return processed_data

# 删除重复行
def remove_duplicates(data: pd.DataFrame, columns: Union[str, List[str]] = None, keep: str = 'first', inplace: bool = False) -> pd.DataFrame:
    try:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("The 'data' argument must be a pandas DataFrame.")
        
        if columns is not None and not isinstance(columns, (str, list)):
            raise TypeError("The 'columns' argument must be a string, list of strings, or None.")
        
        if keep not in ['first', 'last', False]:
            raise ValueError("The 'keep' argument must be 'first', 'last', or False.")
        
        if not isinstance(inplace, bool):
            raise TypeError("The 'inplace' argument must be a boolean.")

        if inplace:
            data.drop_duplicates(subset=columns, keep=keep, inplace=True)
            return data
        else:
            return data.drop_duplicates(subset=columns, keep=keep)
    except Exception as e:
        raise RuntimeError(f"Error occurred while removing duplicates: {e}")

# 删除列
def drop_column_from_datasets(
    dataset1: pd.DataFrame, 
    dataset2: pd.DataFrame, 
    column_name: Union[str, List[str]], 
) -> None:
    dataset1.drop(columns=column_name, inplace=True)
    dataset2.drop(columns=column_name, inplace=True)