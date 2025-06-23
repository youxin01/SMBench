#### one_hot_encode
**Name:** one_hot_encode  
**Description:** 对指定列进行独热编码（One-Hot Encoding），将分类数据转换为数值特征，方便模型输入。  

**Applicable Situations:**  
- 当数据中包含分类特征，需将其转换为机器学习模型能够处理的数值形式。  
- 需要减少模型对分类特征排序信息的误解。  

**Parameters:**  
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 输入数据集，包含需要编码的列。  

- `columns`:  
  - **Type:** `Union[str, List[str]]`  
  - **Description:** 需要进行独热编码的列名，可以是单列名称或列名列表。  

**Result:**  
- 返回处理后的DataFrame，其中指定的列被替换为独热编码生成的数值列。  

**Notes:**  
- 输入参数`columns`必须是字符串或字符串列表。  
- 对于分类列的每个类别，生成一个新列（列名格式为`原列名_类别名`）。  
- 独热编码会自动忽略原列中的数值特征，因此应确保选择的是分类列。  
-----

#### label_encode
**Name:** `label_encode`
**Description:** 对指定列进行标签编码（Label Encoding），将分类特征映射为整数值并保存编码器，用于后续还原或预测任务。

**Applicable Situations:**
- 当数据中包含分类特征，需要将其转换为整数值以供机器学习模型处理。
- 需要保存编码器以便后续在预测或部署阶段进行一致性转换。

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** 输入数据集，包含需要编码的列。

- `columns`:
  - **Type:** `Union[str, List[str], None]`
  - **Description:** 需要进行标签编码的列名，可以是单个列名或列名列表。

- `encoder_path`:

  - **Type:** `str`
  - **Description:** 用于保存所有列的 LabelEncoder 对象，默认为"./encoder_dir/label_encoder.pkl"。

**Result:**
- 返回处理后的 DataFrame，其中指定的列被替换为整数编码值。

**Notes:**
- 输入参数 `columns` 不能为空。若为字符串则自动转为列表。
- 每个被编码的列将使用独立的 `LabelEncoder` 处理。
- 所有编码器将作为字典保存到指定路径，用于 `label_decode` 解码。
-----

#### label_decode
**Name:** `label_decode`
**Description:** 使用已保存的 LabelEncoder 对预测结果或数据中的指定列进行标签解码，恢复原始的分类标签。

**Applicable Situations:**
- 需要将模型输出的整数预测结果转换回原始类别标签，用于结果展示或业务解释。
- 当原始训练阶段保存了 `LabelEncoder` 并希望在推理时使用相同的编码器解码。

**Parameters:**
- `data`:
  - **Type:** `Union[np.ndarray, pd.DataFrame]`
  - **Description:** 待解码的数据。若为 `DataFrame`，将对 `target_col` 这一列进行解码；若为 `ndarray`，表示预测的编码值。

- `target_col`:
  - **Type:** `str`
  - **Description:** 目标列名，用于确定使用哪个编码器进行解码，也将作为返回的列名。

- `encoder_path`:
  - **Type:** `str`
  - **Description:** 之前保存的编码器路径（由 `label_encode` 生成的 pickle 文件）。

**Result:**
- 返回处理后的`pd.DataFrame`。

**Notes:**
- 若输入为 `DataFrame`，必须包含 `target_col`。
- 若输入为 `ndarray`，将返回一个包含解码结果的单列 DataFrame。
- 编码器必须已通过 `label_encode` 保存，并在目标列名下可用。
-----

#### map_func
**Name:** map_func
**Description:** 基于字典映射的列数据转换函数，
**Applicable Situations:**  
- 需要对DataFrame中指定列进行批量数据转换  
- 需要实现多列不同映射规则的快速转换场景  

**Parameters:**  
- `data`:  
  - ​**Type:** `pd.DataFrame`  
  - ​**Description:** 需要处理的原数据表，必须包含`col_dict`中指定的所有列  

- `col_dict`:  
  - ​**Type:** `dict`  
  - ​**Description:** 列处理字典，支持两种格式：  
    - `{列名: 映射字典}`：直接替换值为字典对应值  
  - ​**Example:** `{"等级": {"A":90, "B":80},"是否违约": {"否": 0, "是": 1}}`  

**Result:**  
- 返回转换后的DataFrame，保留原始数据结构，仅修改指定列的数据   

**Example Call:**  
```python
processed_df = map_func(
    data=df_base,
    col_dict= {
    "信誉评级": {"A": 4, "B": 3, "C": 2, "D": 1},
    "是否违约": {"否": 0, "是": 1}
}
)
```
-----

#### two_data_func  
**Name:** two_data_func  
**Description:**  
基于两个表的列操作函数，支持通过指定列进行聚合计算，并结合数学表达式生成新特征。适用于特征工程中跨表、按列聚合的场景。支持分组（groupby）与非分组操作。

**Applicable Situations:**  
- 需对两个不同数据表中的列进行聚合后组合生成新特征  
- 使用表达式（如 `x - y`、`x / (y + 1)`）从列的聚合值中计算复合特征  
- 需要保留所有分组（使用外连接）并对缺失值自动填充为0，避免信息丢失  

**Parameters:**  
- `data1` / `data2`:  
  - ​**Type:** `pd.DataFrame`  
  - ​**Description:** 参与计算的两个源数据表，每行代表一个样本，列为特征  

- `feature`:  
  - ​**Type:** `str`  
  - ​**Description:** 最终生成的新特征名称（列名）  

- `col`:  
  - ​**Type:** `list[str]`  
  - ​**Description:** 两张表中参与聚合计算的列名，格式为 `[data1列名, data2列名]`，即**针对特定列的操作**  

- `func1`:  
  - ​**Type:** `list[Union[str, Callable]]`  
  - ​**Description:** 两个数据集对应列的聚合函数，格式为 `[data1聚合函数, data2聚合函数]`，支持字符串函数（如 "mean"）或自定义函数  

- `func2`:  
  - ​**Type:** `str`  
  - ​**Description:** 表达式字符串，用 `x` 表示第一个数据集的聚合结果，用 `y` 表示第二个数据集的聚合结果。例如 `"x / (y + 1)"`  

- `group_by`:  
  - ​**Type:** `list[str]` | `None`  
  - ​**Description:** 可选，若提供则按指定列进行分组聚合，否则对整表进行全局聚合  

**Result:**  
- 返回一个 `pd.DataFrame`，包含分组字段（如有）和生成的新特征列。空值会自动填充为0，确保数据完整性。

**Example Call:**  
基于“订单表”和“退款表”，计算每个客户的订单均价与退款总金额的差值：

```python
df_feature = two_data_func(
    data1=df_orders,
    data2=df_refunds,
    feature="净消费值",
    col=["订单金额", "退款金额"],
    func1=["mean", "sum"],
    func2="x - y",
    group_by=["客户ID"]
)
```
-----

#### generate_single_feature  
**Name:** generate_single_feature  
**Description:**  
基于单个数据表和配置字典，生成一个新特征。支持灵活的列级聚合操作，可选的分组处理，以及自定义后处理函数。适用于数据预处理或特征工程中对单列或多列特征进行组合聚合的场景。

**Applicable Situations:**  
- 需要按某一列或某几列聚合构造新特征  
- 支持对同一列使用多个聚合函数进行计算（如 `mean` 和 `std`）  
- 支持自定义 `post_process` 函数对多个聚合结果进行组合或逻辑处理  
- 可选分组键 `group_key` 支持分用户、分商品等粒度计算  

**Parameters:**  
- `data`:  
  - ​**Type:** `pd.DataFrame`  
  - ​**Description:** 原始数据表，每行为一个样本，列为特征字段  

- `feature`:  
  - ​**Type:** `str`  
  - ​**Description:** 最终生成的新特征名称  

- `feature_config`:  
  - ​**Type:** `dict`  
  - ​**Description:** 特征配置字典，支持以下字段：  
    - `agg_funcs`: 必填，聚合函数列表，可为字符串函数名、函数对象，或 (列名, 函数) 元组  
    - `column`: 可选，默认使用数据表第一列，指定需要聚合的列名  
    - `post_process`: 可选，对聚合结果 `df` 进行后处理的函数，返回单列或多列  

- `group_key`:  
  - ​**Type:** `str` | `None`  
  - ​**Description:** 分组字段，如 "客户ID"，若为 `None` 则对全体数据聚合  

**Result:**  
- 返回一个 `pd.DataFrame`，包含分组字段（如有）和最终生成的新特征列。自动处理索引对齐。

**Example Call:**  
计算每个客户的订单均值与标准差之和作为波动感知特征：

```python
feature_config = {
    "agg_funcs": ["mean", "std"],
    "column": "订单金额",
    "post_process": lambda df: df.sum(axis=1)
}

result_df = generate_single_feature(
    data=df_orders,
    feature="波动感知",
    feature_config=feature_config,
    group_key="客户ID"
)
```
-----
#### merge_features  
**Name:** merge_features  
**Description:**  
将多个包含相同主键的特征表合并为一个统一的特征表。支持自定义合并方式（如 outer/inner），并自动处理列名冲突。适用于特征工程中多个模块生成的特征表整合。

**Applicable Situations:**  
- 多个子模块或处理步骤分别生成特征表，需要统一合并  
- 合并时可能存在列名重复，需要自动冲突处理  
- 保留所有主键样本（通过 outer join）或仅保留交集样本（通过 inner join）  

**Parameters:**  
- `features`:  
  - ​**Type:** `list[pd.DataFrame]`  
  - ​**Description:** 待合并的特征表列表，每个表都应包含主键字段  

- `on`:  
  - ​**Type:** `list[str]` | `str`  
  - ​**Description:** 合并依据的主键列名，支持单列或多列  

- `how`:  
  - ​**Type:** `str`  
  - ​**Description:** 合并方式，默认为 `'outer'`，可选 `'inner'`, `'left'`, `'right'` 等  

**Result:**  
- 返回合并后的 `pd.DataFrame`，包含所有特征列和主键列。若存在列名冲突，将自动去除重复列（后缀为 `_dup`）。

**Example Call:**  
合并三个子特征表 `feat1`、`feat2`、`feat3`，保留所有客户样本：

```python
merged_df = merge_features(
    features=[feat1, feat2, feat3],
    on="客户ID",
    how="outer"
)
```