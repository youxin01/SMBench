#### fill_missing_values
**Name:** fill_missing_values  
**Description:** 填充DataFrame中的缺失值，根据指定的列和填充策略完成缺失值的处理。  
**Applicable Situations:**  
- 数据集中存在缺失值时需要处理以确保数据的完整性。  
- 根据特定策略（如删除、均值填充等）处理缺失值。  
**Parameters:**  
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 包含缺失值需要处理的数据集。  

- `columns`:  
  - **Type:** `Optional[Union[str, List[str]]]`  
  - **Description:** 必须提供需要处理的列名,支持单个字符串或列表。如果是`None`，默认检查所有列 

- `strategy`:  
  - **Type:** `str`  
  - **Description:** 缺失值的处理策略。默认值为`'auto'`。  
    - `'auto'`: 删除包含缺失值的行。  
    - `'mean'`: 使用列的均值填充缺失值，只能用于数值类型的数据列。  
    - `'median'`: 使用列的中位数填充缺失值，只能用于数值类型的数据列。
    - `'mode'`: 使用列的众数填充缺失值。  

**Result:**  
- 返回一个新的DataFrame，其中指定列的缺失值已根据策略进行处理。  

**Notes:**  
- `columns`默认为`None`检查所有的列。
- `'auto'`策略，所有包含缺失值的行将被删除,一般不采用。 
- `'mean'`、`'median'`只能用于数值类型的数据列。 
-----
#### remove_columns_with_missing_data
**Name:** remove_columns_with_missing_data  
**Description:** 删除缺失值比例超过指定阈值的列。  

**Applicable Situations:**  
- 数据集中存在缺失值比例过高的列，影响建模效果或数据完整性时。  
- 希望在保留尽可能多的有用数据的同时去除不完整列。  

**Parameters:**  
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 待处理的数据集，包含需要检测缺失值的列。  

- `thresh`:  
  - **Type:** `float`  
  - **Description:** 缺失值比例的阈值，介于0到1之间，默认值为0.5。如果某列缺失值比例超过该阈值，则该列将被删除。  

- `columns`:  
  - **Type:** `Union[str, List[str]]`  
  - **Description:** 指定需要检测缺失值的列。如果为`None`，默认检查所有列。支持单列字符串或多列列表。  

**Result:**  
- 返回一个新的DataFrame。

**Notes:**  
- 参数`thresh`必须在`[0, 1]`范围内，否则会引发`ValueError`异常。  
- 如果提供了`columns`参数，则仅对指定列进行缺失值检测；其他列将自动保留。  
- 如果所有列的缺失值比例均超过阈值，返回结果可能为空DataFrame。  
-----
#### detect_and_handle_outliers_zscore
**Name:** detect_and_handle_outliers_zscore  
**Description:** 使用z-score方法检测并处理异常值，支持通过截取或删除异常值的方法提高数据质量。  

**Applicable Situations:**  
- 数据中包含数值型列，且可能存在极端值（异常值）影响分析或建模结果。  
- 希望对异常值进行处理，例如限制其范围或直接删除。  

**Parameters:**  
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 待处理的数据集，包含需要检测和处理异常值的列。  

- `columns`:  
  - **Type:** `Union[str, List[str]]`  
  - **Description:** 指定需要处理异常值的列名，支持单列字符串或多列列表。列必须为数值型。  

- `threshold`:  
  - **Type:** `float`  
  - **Description:** Z分数阈值，表示偏离均值的标准差数。绝对值大于阈值的样本将被视为异常值。默认值为3.0。  

- `method`:  
  - **Type:** `str`  
  - **Description:** 处理异常值的方式，可选`'clip'`或`'remove'`：  
    - `'clip'`: 将异常值截取到阈值范围内。  
    - `'remove'`: 删除包含异常值的样本。  

**Result:**  
- 返回处理后的DataFrame

**Notes:**  
- 指定的`columns`必须为数值型，否则将引发`ValueError`异常。  
- 如果选择`'clip'`方法，将对异常值进行限制而不删除样本。  
- 如果选择`'remove'`方法，返回的数据集中可能会删除多行。  
-----
#### detect_and_handle_outliers_iqr
**Name:** detect_and_handle_outliers_iqr  
**Description:** 使用四分位距法（IQR）检测并处理异常值，支持通过截取或删除异常值来提高数据质量。  

**Applicable Situations:**  
- 数据中包含数值型列，且可能存在异常值需要处理。  
- 希望通过统计学方法检测和处理异常值，例如限制其范围或直接移除。  

**Parameters:**  
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 待处理的数据集，包含需要检测和处理异常值的列。  

- `columns`:  
  - **Type:** `Union[str, List[str]]`  
  - **Description:** 需要处理异常值的列名，可以是单列字符串或多列列表。列必须为数值型。  

- `factor`:  
  - **Type:** `float`  
  - **Description:** 控制异常值范围的因子。默认为`1.5`，表示将范围设定为Q1 - 1.5 * IQR到Q3 + 1.5 * IQR之外的值为异常值。  

- `method`:  
  - **Type:** `str`  
  - **Description:** 处理异常值的方式，可选`'clip'`或`'remove'`：  
    - `'clip'`: 将异常值限制在四分位距范围内。  
    - `'remove'`: 删除包含异常值的样本。  

**Result:**  
- 返回处理后的DataFrame。

**Notes:**  
- 输入的`columns`必须为数值型，否则会引发`ValueError`异常。  
- `'clip'`方法会将数据中的异常值修改为指定范围的边界值。  
- `'remove'`方法会直接删除包含异常值的样本，可能会导致数据行数减少。  
-----
#### remove_duplicates
**Name:** remove_duplicates  
**Description:** 移除数据集中重复的数据，可按指定列处理，支持保留首个、最后一个或全部移除。  

**Applicable Situations:**  
- 数据中包含重复记录，需要清理以避免影响分析或模型训练。  
- 需要按特定列判断重复记录或全局处理重复数据。  

**Parameters:**  
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 待处理的数据集，包含可能存在重复记录的数据。  

- `columns`:  
  - **Type:** `Union[str, List[str]]`  
  - **Description:** 判断重复的依据列，可以是单列名称、列名列表。  

- `keep`:  
  - **Type:** `str`  
  - **Description:** 指定保留的重复记录，支持以下选项：  
    - `'first'`: 保留每组重复记录中的第一条（默认）。  
    - `'last'`: 保留每组重复记录中的最后一条。  
    - `False`: 删除所有重复记录。  

- `inplace`:  
  - **Type:** `bool`  
  - **Description:** 是否直接在原数据集上操作，默认值为`False`：  
    - `True`: 修改原数据，方法返回`None`。  
    - `False`: 不修改原数据，返回新的DataFrame。  

**Result:**  
- 返回处理后的DataFrame或直接修改原DataFrame。  

**Notes:**  
- 输入参数`columns`必须为字符串或字符串列表；若为`None`，按所有列判断重复性。  
- 如果`inplace=True`，原DataFrame将被修改，返回值为`None`。  
- 参数`keep`决定保留哪条重复记录，设置不符合要求时会抛出`ValueError`。  