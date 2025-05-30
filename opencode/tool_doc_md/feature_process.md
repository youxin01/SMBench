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
**Name:** label_encode  
**Description:** 对指定列进行标签编码（Label Encoding），将分类特征映射为整数值, 为分类任务准备。  

**Applicable Situations:**  
- 当数据中包含分类特征，需要将其转换为整数值以供机器学习模型处理。  
- 标签之间具有隐式顺序（如排名数据）时优先使用。  

**Parameters:**  
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 输入数据集，包含需要编码的列。  

- `columns`:  
  - **Type:** `Union[str, List[str]]`  
  - **Description:** 需要进行标签编码的列名，可以是单列名称或列名列表。

**Result:**  
- 返回处理后的DataFrame，其中指定的列被替换为整数编码的列。  

**Notes:**  
- 输入参数`columns`必须是字符串或字符串列表。  
- 标签编码会将分类列中的每个类别映射到唯一的整数值，范围为`[0, n_classes-1]`。  
- 对于未处理的分类数据，应优先检查是否有缺失值或不合理类别值。
-----

#### map_func_col
**Name:** map_func_col  
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

#### two_data_func_col
**Name:** two_data_func_col  
**Description:** 基于双数据集跨表特征生成函数，通过分组聚合和表达式计算生成复合特征  
**Applicable Situations:**  
- 需要结合两个数据集的聚合结果进行特征工程  
- 支持外连接合并后填充缺失值，避免数据丢失  
- 需通过数学表达式（如x/y）组合聚合值的场景  

**Parameters:**  
- `data1`/`data2`:  
  - ​**Type:** `pd.DataFrame`  
  - ​**Description:** 参与计算的两个原始数据表,两个数据集都是，行为样本，列为特征。

- `feature`:  
  - ​**Type:** `str`  
  - ​**Description:** 最终生成的特征列名称  

- `col`:  
  - ​**Type:** `list`  
  - ​**Description:** 两个原始数据表当中参加计算的列名，格式为[data1列名, data2列名]  

- `func1`:  
  - ​**Type:** `list`  
  - ​**Description:** 第一步操作的聚合函数列表，格式为[data1聚合方法, data2聚合方法]，支持sum/mean/count/max/min或自定义函数  ，例如["sum","sum"]。

- `func2`:  
  - ​**Type:** `str`  
  - ​**Description:** 第二部特征计算表达式，使用x代表data1聚合结果，y代表data2聚合结果（如"x/y"）  

- `group_by`:  
  - ​**Type:** `list`  
  - ​**Description:** 分组字段列表（如["客户ID"]），  

**Result:**  
- 返回包含分组字段和新建特征列的DataFrame，缺失值自动填充0  

**Example Call:** 
比如我要计算**净现金流**这个特征,要用销售额-采购额，便可以使用下述对两个数据集进行操作。
```python
feature_df = two_data_func_col(
    data1=df_input,
    data2=df_output,
    feature="净现金流",
    col=["金额", "金额"],
    func1=["sum", "sum"],
    func2="x-y",
    group_by=["客户ID"]
)
```
-----

#### generate_single_feature_col
**Name:** generate_single_feature  
**Description:** 单数据集特征生成函数，支持多聚合方式组合及后处理计算  
**Applicable Situations:**  
- 需要从单数据集生成复杂聚合特征  
- 需组合多个中间聚合结果进行二次计算  
- 支持分组/全量两种计算模式  

**Parameters:**  
- `data`:  
  - ​**Type:** `pd.DataFrame`  
  - ​**Description:** 原始数据表，行为样本，列为特征

- `feature`:  
  - ​**Type:** `str`  
  - ​**Description:** 最终生成的特征列名称  

- `feature_config`:  
  - ​**Type:** `dict`  
  - ​**Description:** 特征配置字典，必须包含：  
    - `agg_funcs`: 聚合函数列表，元素可为函数名字符串或(列名,函数)元组  
    - `post_process`: 可选，接受中间结果DataFrame并返回最终Series的后处理函数  

- `group_key`:  
  - ​**Type:** `str`  
  - ​**Description:** 分组列名，为空时进行全量计算  

**Result:**  
- 返回包含分组字段（如有）和新建特征列的DataFrame，索引自动重置  

**Example Call:**  
```python
feature_df = generate_single_feature_col(
    data=df_log,
    feature="登录频次",
    feature_config={
        "agg_funcs": [("登录时间", "count"), ("设备类型", lambda x: x.nunique())],
        "post_process": lambda df: df.iloc[:,0] / df.iloc[:,1]
    },
    group_key="用户ID"
)
```
-----

#### apply_feature_row
**Name:** apply_feature_row  
**Description:** 借用apply函数的行特征生成函数，对每一行进行操作获得特征 
**Applicable Situations:**  
- 需要对选定数值列进行列方向统计量计算  
- 需生成与索引列强关联的复合特征（如变异系数）  
- 本函数针对的是需要对行进行操作得到特征的数据集

**Parameters:**  
- `data`:  
  - ​**Type:** `pd.DataFrame`  
  - ​**Description:** 原始输入数据，需包含配置中索引列和可计算的数值列  

- `feature`:  
  - ​**Type:** `str`  
  - ​**Description:** 生成的特征列名称  

- `feature_config`:  
  - ​**Type:** `dict`  
  - ​**Description:** 特征配置字典，包括：  
    - `without_column`：需保留但排除计算的索引列列表  
    - `apply_funcs`：函数列表，每个元素为（函数对象, axis参数）元组  

**Result:**  
- 返回包含原始索引列和新特征列的DataFrame 

**Example Call:**  
```python
Supply_Stability = apply_feature_row(
    data=supply,
    feature="变异系数",
    feature_config={
        'without_column': ['供应商ID','材料分类'],
        'apply_funcs': [
            (lambda x: x.std() / x.mean() if x.mean() > 0 else 0, 'axis=1')
        ]
    }
)
```
-----

#### Order_Fulfillment_cal
**Name:** Order_Fulfillment_cal  
**Description:** 双数据集订单满足率动态计算函数，通过逐行对比订单与供应数据生成评估指标  
**Applicable Situations:**  
- 需要对比供应商实际供应量与订单量的匹配程度  
- 需排除标识列后对数值列进行行级动态计算  
- 存在零值订单需要特殊处理的业务场景  

**Parameters:**  
- `order1`/`supply2`:  
  - ​**Type:** `pd.DataFrame`  
  - ​**Description:** 订单数据表与供应数据表，必须包含相同的索引列（without_col）  

- `without_col`:  
  - ​**Type:** `list`  
  - ​**Description:** 标识列列表（如供应商ID等），计算时会被保留但排除在数值计算外  

- `feature`:  
  - ​**Type:** `str`  
  - ​**Description:** 生成的特征列名称（如"订单满足率"）  

**Result:**  
- 返回包含标识列和新建特征列的DataFrame，满足率取值范围[0,1]的小数值  

**Example Call:**  
```python
Order_Fulfillment = Order_Fulfillment_cal(
    order=order_data,
    supply=supply_data,
    without_col=['供应商ID', '材料分类'],
    feature="订单满足率"
)
```