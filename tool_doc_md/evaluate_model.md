#### ewn_analysis
**Name:** ewn_analysis  
**Description:** 基于熵权法的综合评价函数，适用于企业经济效益等多指标评价场景，支持指标权重计算与等级划分。  
**Applicable Situations:**  
- 需要综合多指标进行客观评价（如企业经济效益、区域发展水平评估等）。    

**Parameters:**  
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 必须包含所有待评价指标列和索引列的数据集。  

- `criteria`:  
  - **Type:** `dict`  
  - **Description:** 等级阈值标准字典，格式为 `{指标: {等级: (下限, 上限), ...}, ...}`。  
  - **Example:** `{"利润率": {"优": (0.2, 1.0), "中": (0.1, 0.2)} }`  

- `index_col`:  
  - **Type:** `str`  
  - **Description:** 索引的列名（如企业ID、地区名称等）。  

- `positive_indicators`:  
  - **Type:** `list`  
  - **Description:** 明确的正向指标列表（值越大越优，如利润率）。  

- `negative_indicators`:  
  - **Type:** `list`  
  - **Description:** 明确的负向指标列表（值越小越优，如成本率）。  

- `enable_grade_evaluation`:  
  - **Type:** `bool`  
  - **Description:** 是否启用等级评价功能，默认关闭。启用时需要提供`criteria`。  

- `grade_labels`:  
  - **Type:** `list`  
  - **Description:** 综合等级标签列表，需与`level_ranges`配合使用，如 `['差','中','好']`。  

- `level_ranges`:  
  - **Type:** `list`  
  - **Description:** 综合得分分界点，格式为升序列表，如 `[0.3, 0.6]` 表示：  
    - 得分 <0.3 → 第1级，0.3≤得分<0.6 → 第2级，得分≥0.6 → 第3级  

**Result:**  
- 返回DataFrame

**Notes:**  
1. 启用等级评价时，当今当题目给出**criteria**才考虑。 
3. `level_ranges`的分界点数量应为`grade_labels长度-1`（如5级标签需4个分界点）   

**Example Call:**  
```python
result = ewn_analysis(
    data=data_eval,
    criteria=criteria_dict,
    index_col="企业ID",
    positive_indicators=["利润率", "研发投入"],
    negative_indicators=["负债率"],
    enable_grade_evaluation=True,
    grade_labels=["差", "中", "优"],
    level_ranges=[0.4, 0.7]
)
```
-----

#### topsis_analysis
**Name:** topsis_analysis  
**Description:** 基于TOPSIS的多指标决策分析函数，支持极小型、极大型、中间型、区间型4种指标类型，输出归一化得分（值越大代表越优）。  
**Applicable Situations:**  
- 需要处理多类型指标（如同时存在效益型、成本型、区间最优型等）的决策问题  
- 需通过客观权重进行综合评价（如供应商选择、方案排序等）  
- 要求结果输出为可直接比较的归一化得分  

**Parameters:**  
- `data`:  
  - ​**Type:** `pd.DataFrame`  
  - ​**Description:** 原始数据矩阵，每行为一个样本，每列为一个指标。

- `index_col`:  
  - ​**Type:** `str`  
  - ​**Description:** 标识样本的索引列名（如"企业ID"）。

- `types`:  
  - ​**Type:** `dict`  
  - ​**Description:** 指标类型字典，键为指标名，值为类型编码：  
    - `0`: 极小型（值越小越好，如成本）  
    - `1`: 极大型（值越大越好，如收益）  
    - `2`: 中间型（距离指定最佳值越近越好）  
    - `3`: 区间型（在指定区间内最优）  
  - ​**Example:** `{"能耗":0, "产量":1, "pH值":2, "温度":3}`  

- `params`:  
  - ​**Type:** `dict`  
  - ​**Description:** 特殊指标参数：  
    - 中间型指标：需提供最佳值（如`{"pH值":6.5}`）  
    - 区间型指标：需提供`[下限, 上限]`（如`{"温度":[20,25]}`）  
    - 其他类型：设为`None`  
  - ​**Example:** `{"能耗":None, "产量":None, "pH值":6.5, "温度":[20,25]}`  

- `weights`:  
  - ​**Type:** `dict`  
  - ​**Description:** 指标权重字典，权重总和建议为1。  
  - ​**Example:** `{"能耗":0.2, "产量":0.3, "pH值":0.3, "温度":0.2}`  

**Result:**  
- 返回包含以下列的DataFrame：  
  - `index_col`: 保留的索引列  
  - `TOPSIS得分`: 归一化后的综合评价得分（0-1之间，越大越好）  

**Example Call:**  
```python
result = topsis_analysis(
    data=data_eval,
    index_col="企业ID",
    types={
        "变异系数":0, 
        "供货量":1,
        "合格率":2,
        "响应时间":3
    },
    params={
        "变异系数":None,
        "供货量":None,
        "合格率":0.98,       # 中间型最佳值
        "响应时间":[2,5]      # 区间型最优范围
    },
    weights={
        "变异系数":0.3,
        "供货量":0.2,
        "合格率":0.3,
        "响应时间":0.2
    }
)
```
-----

#### wsm_evaluation
**Name:** wsm_evaluation  
**Description:** 基于加权评分模型的评价方法，支持正反向指标的归一化处理与自定义权重分配，输出综合评价得分。  
**Applicable Situations:**  
- 需要结合正/反向指标进行综合评分（如供应商评估、方案优选等）  
- 需自定义指标权重及方向性的评分场景  
- 要求简单直观的线性加权评分模型  

**Parameters:**  
- `data`:  
  - ​**Type:** `pd.DataFrame`  
  - ​**Description:** 包含原始指标数据的二维表格，每行为一个样本，每列为一个指标。

- `indicator_markers`:  
  - ​**Type:** `dict`  
  - ​**Description:** 指标方向标记字典：  
    - `1`表示正向指标（值越大越好）  
    - `-1`表示反向指标（值越小越好）  
  - ​**Example:** `{"成本":-1, "产量":1}`  

- `weight_allocation`:  
  - ​**Type:** `dict`  
  - ​**Description:** 指标权重分配字典，键为指标名，值为权重值（建议总和为1）  
  - ​**Example:** `{"成本":0.3, "产量":0.7}`  

- `index_col`:  
  - ​**Type:** `str`  
  - ​**Description:** 标识样本的唯一索引列名（如"供应商ID"）  

- `normalization`:  
  - ​**Type:** `str`  
  - ​**Description:** 归一化方法，当前仅支持：  
    - `'minmax'`: 极差归一化（默认）  

**Result:**  
- 返回包含以下列的DataFrame：  
  - `index_col`: 保留的索引列  
  - `weighted_score`: 加权综合得分（0-1之间，越大越好）  

**Example Call:**  
```python
result = weighted_scoring(
    data=data_eval,
    indicator_markers={
    "成本": -1,
    "交货准时率": 1,
    "合格率": 1,
    "响应速度": 1
},
    weight_allocation={
    "成本": 0.3,
    "交货准时率": 0.25,
    "合格率": 0.25,
    "响应速度": 0.2
},
    index_col='供应商ID'
)
```