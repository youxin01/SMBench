#### hypothesis_test
**Name:** hypothesis_test  
**Description:** 执行各种统计假设检验，包括参数检验、非参数检验和多重比较校正  
**Applicable Situations:**  
- 比较组间均值差异（t检验、ANOVA）  
- 非参数数据比较（Mann-Whitney、Wilcoxon等）  
- 多重比较校正（Bonferroni、FDR）  
- 科学实验数据分析  

**Parameters:**  
- `data`:  
  - **Type:** `pandas.DataFrame`  
  - **Description:** 包含待分析数据的DataFrame（多重比较检验不需要）  
- `method`:  
  - **Type:** `str`  
  - **Description:** 指定检验方法（必需参数），可用选项见下方  
- `**kwargs`:  
  - **Type:** `dict`  
  - **Description:** 不同检验方法所需的特定参数  

**Supported Methods:**  
- `"one_sample_t"`: 单样本t检验（需要`col`, `popmean`参数）  
- `"two_sample_t"`: 独立样本t检验（需要`col1`, `col2`, `equal_var`参数）  
- `"paired_t"`: 配对t检验（需要`col1`, `col2`参数）  
- `"anova"`: 单因素方差分析（需要`dependent`, `factor`参数）  
- `"tukey"`: Tukey HSD事后检验（需要`dependent`, `factor`参数）  
- `"dunnett"`: Dunnett检验（需要`dependent`, `factor`, `control`参数）  
- `"mann_whitney"`: Mann-Whitney U检验（需要`col1`, `col2`参数）  
- `"wilcoxon"`: Wilcoxon符号秩检验（需要`col1`, `col2`参数）  
- `"kruskal"`: Kruskal-Wallis检验（需要`group_col`, `value_col`参数）  
- `"bonferroni"`: Bonferroni校正（需要`pvals`参数）  
- `"fdr"`: FDR校正（需要`pvals`参数）  

**Result:**  
- 返回一个字典，包含检验统计量、p值及相关统计信息，具体内容取决于检验方法  

**Example Call:**  
```python
# 独立样本t检验示例
result = hypothesis_test(
    data=df,
    method="two_sample_t",
    col1="group_A",
    col2="group_B"
)
```
-----

#### distribution_test
**Name:** distribution_test  
**Description:** 执行数据分布检验，判断数据是否符合特定分布
**Applicable Situations:**  
- 正态性检验（Shapiro-Wilk检验）  
- 任意分布拟合检验（Kolmogorov-Smirnov检验）  
- 数据预处理前的分布验证  

**Parameters:**  
- `data`:  
  - **Type:** `pandas.DataFrame`  
  - **Description:** 包含待检验数据的DataFrame  
- `col`:  
  - **Type:** `str`  
  - **Description:** 需要检验的数据列名  
- `method`:  
  - **Type:** `str`  
  - **Description:** 检验方法（默认"shapiro"），可选："shapiro"或"ks"  
- `**kwargs`:  
  - **Type:** `dict`  
  - **Description:** 不同检验方法所需的特定参数  

**Supported Methods:**  
- `"shapiro"`: Shapiro-Wilk正态性检验（无额外参数）  
- `"ks"`: Kolmogorov-Smirnov检验（需要`dist`参数指定理论分布名称）  

**Additional Parameters for KS Test:**  
- `dist`: 理论分布名称（如"norm", "expon"等）  
- `args`: 分布参数元组（可选）  
- `loc`: 位置参数（默认0）  
- `scale`: 尺度参数（默认1）  
- `alternative`: 检验方向（默认"two-sided"）  

**Result:**  
- 返回一个字典，包含检验统计量和p值  

**Example Call:**  
```python
# 正态性检验
result = distribution_test(
    data=df,
    col="height",
    method="shapiro"
)
```