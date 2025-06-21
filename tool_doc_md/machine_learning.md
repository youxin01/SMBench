#### train_xgboost_regressor
**Name:** train_xgboost_regressor  
**Description:** 训练xgboost模型用于回归任务，预测新数据，并输出模型在测试集上的表现。  
**Applicable Situations:**  
- 适用于回归任务，训练数据为带有目标列的DataFrame。
- 需要预测新的数据并返回相应结果。

**Parameters:**
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 训练的数据集，包括特征和目标列。

- `target`:  
  - **Type:** `string`
  - **Description:** 目标列名称，模型需要预测的输出。

- `new_data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 新数据集，用于模型预测。

- `test_size`:  
  - **Type:** `float`  
  - **Description:** 测试集比例，默认为0.2。

- `params`:  
  - **Type:** `dict`  
  - **Description:** XGBoost模型的超参数字典，以下为具体参数举例解释
    ```python
    params = {
        'objective': 'reg:squarederror',  # 指定学习任务和相应的损失函数。'reg:squarederror' 表示使用平方误差作为回归任务的损失函数。
        'max_depth': 6,                   # 树的最大深度。
        'eta': 0.3,                       # 学习率（步长）。控制每次更新模型权重时的步长大小。较小的值会使模型训练更慢，但可能更精确。
        'subsample': 0.8,                 # 训练每棵树时使用的样本比例。设置为0.8表示每次只使用80%的样本进行训练，有助于防止过拟合。
        'colsample_bytree': 0.8           # 训练每棵树时使用的特征比例。设置为0.8表示每次只使用80%的特征进行训练，有助于防止过拟合。
    }
    ```

- `num_boost_round`:  
  - **Type:** `int`  
  - **Description:** xgboost训练的轮次，默认为100。

- `with_label`:  
  - **Type:** `bool`  
  - **Description:** 是否将预测结果与`new_data`一起返回，默认为False。

**Required:**  
- `data`  
- `target`  
- `new_data`
- `params`

**Result:**  
- 返回预测结果。如果`with_label`为True，则返回包含预测值的新数据；否则，仅返回预测值。

**Notes:**  
- `params`支持用户自定义；若未传入，函数会使用默认超参数配置。  
-----

#### train_xgboost_classifier
**Name:** train_xgboost_classifier  
**Description:** 训练XGBoost模型用于多分类任务，对新数据进行预测，并输出模型在测试集上的表现。  

**Applicable Situations:**  
- 适用于多分类任务，训练数据为带有目标列的DataFrame。  
- 需要预测新数据的分类结果并返回相应输出。  

**Parameters:**  
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 训练的数据集，包括特征和目标列。  

- `target`:  
  - **Type:** `string`  
  - **Description:** 目标列名称，模型需要预测的类别。  

- `new_data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 新数据集，用于模型预测。  

- `test_size`:  
  - **Type:** `float`  
  - **Description:** 测试集比例，默认为0.2。  

- `params`:  
  - **Type:** `dict`  
  - **Description:** XGBoost模型的超参数字典，以下为具体参数举例解释：
    ```python
    params = {
        'objective': 'multi:softprob',  # 多分类概率输出
        'num_class': 类别数量,           # 根据目标列自动计算类别数量
        'max_depth': 6,                 # 树的最大深度
        'eta': 0.1,                     # 学习率
        'subsample': 0.8,               # 每次训练使用的样本比例
        'colsample_bytree': 0.8,        # 每次训练使用的特征比例
        'eval_metric': ['mlogloss', 'merror']  # 多分类评估指标
    }
    ```

- `num_boost_round`:  
  - **Type:** `int`  
  - **Description:** xgboost训练的轮次，默认为100。  

- `with_label`:  
  - **Type:** `bool`  
  - **Description:** 是否将预测结果与`new_data`一起返回，默认为False。  

**Required:**  
- `data`  
- `target`  
- `new_data`  

**Result:**  
- 返回预测结果：  
  - 如果`with_label`为True，则返回包含预测类别的`new_data`；  
  - 否则，仅返回预测类别列表。  

**Notes:**  
- `params`支持用户自定义；若未传入，函数会使用默认超参数配置。  
- 函数输出模型在测试集上的准确率，帮助用户评估模型性能。  
-----

#### train_lightgbm_regressor
**Name:** train_lightgbm_regressor  
**Description:** 训练LightGBM模型用于回归任务，对新数据进行预测，并输出模型在测试集上的表现（使用RMSE作为评估指标）。  

**Applicable Situations:**  
- 适用于回归任务，训练数据为带有目标列的DataFrame。  
- 需要预测新数据的连续值输出并返回相应结果。  

**Parameters:**  
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 训练的数据集，包括特征和目标列。  

- `target`:  
  - **Type:** `string`  
  - **Description:** 目标列名称，模型需要预测的输出值。  

- `new_data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 新数据集，用于模型预测。  

- `test_size`:  
  - **Type:** `float`  
  - **Description:** 测试集比例，默认为0.2。  

- `params`:  
  - **Type:** `dict`  
  - **Description:** LightGBM模型的超参数字典，以下为具体参数举例解释：
    ```python
    params = {
        'objective': 'regression',  # 指定回归任务
        'metric': 'rmse',          # 使用RMSE（均方根误差）作为评估指标
        'max_depth': 6,            # 树的最大深度
        'learning_rate': 0.3,      # 学习率
        'subsample': 0.8,          # 每次训练使用的样本比例
        'colsample_bytree': 0.8    # 每次训练使用的特征比例
    }
    ```

- `num_boost_round`:  
  - **Type:** `int`  
  - **Description:** LightGBM训练的轮次，默认为100。  

- `with_label`:  
  - **Type:** `bool`  
  - **Description:** 是否将预测结果与`new_data`一起返回，默认为False。  

**Required:**  
- `data`  
- `target`  
- `new_data`  

**Result:**  
- 返回预测结果：  
  - 如果`with_label`为True，则返回包含预测值的`new_data`；  
  - 否则，仅返回预测值列表。  

**Notes:**  
- `params`支持用户自定义；若未传入，函数会使用默认超参数配置。  
- 函数输出模型在测试集上的RMSE（均方根误差），帮助用户评估模型性能。  
-----

#### train_lightgbm_classifier
**Name:** train_lightgbm_classifier  
**Description:** 训练LightGBM模型用于分类任务，对新数据进行预测，并输出模型在测试集上的表现（使用Accuracy作为评估指标）。  

**Applicable Situations:**  
- 适用于分类任务，训练数据为带有目标列的DataFrame。  
- 需要预测新数据的类别标签输出并返回相应结果。  

**Parameters:**  
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 训练的数据集，包括特征和目标列。  

- `target`:  
  - **Type:** `string`  
  - **Description:** 目标列名称，模型需要预测的类别标签。  

- `new_data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 新数据集，用于模型预测。  

- `test_size`:  
  - **Type:** `float`  
  - **Description:** 测试集比例，默认为0.2。  

- `params`:  
  - **Type:** `dict`  
  - **Description:** LightGBM模型的超参数字典，以下为具体参数举例解释：
    ```python
    params = {
        'objective': 'multiclass',          # 指定分类任务
        'metric': 'multi_logloss',         # 使用多分类对数损失作为评估指标
        'num_class': len(np.unique(y_train)),  # 类别数量
        'max_depth': 6,                    # 树的最大深度
        'learning_rate': 0.1,              # 学习率
        'subsample': 0.8,                  # 每次训练使用的样本比例
        'colsample_bytree': 0.8            # 每次训练使用的特征比例
    }
    ```

- `num_boost_round`:  
  - **Type:** `int`  
  - **Description:** LightGBM训练的轮次，默认为100。  

- `with_label`:  
  - **Type:** `bool`  
  - **Description:** 是否将预测结果与`new_data`一起返回，默认为False。  

**Required:**  
- `data`  
- `target`  
- `new_data`  

**Result:**  
- 返回预测结果：  
  - 如果`with_label`为True，则返回包含预测类别的`new_data`；  
  - 否则，仅返回预测类别的列表。  

**Notes:**  
- `params`支持用户自定义；若未传入，函数会使用默认超参数配置。  
- 函数输出模型在测试集上的Accuracy（准确率），帮助用户评估模型性能。  
-----

#### train_knn_classifier
**Name:** train_knn_classifier  
**Description:** 使用K近邻（KNN）算法训练分类模型，预测新数据类别，并输出模型在测试集上的表现（使用Accuracy作为评估指标）。  

**Applicable Situations:**  
- 适用于分类任务，训练数据为带有目标列的DataFrame。  
- 需要预测新数据的类别标签输出并返回相应结果。  

**Parameters:**  
- `data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 训练的数据集，包括特征和目标列。  

- `target`:  
  - **Type:** `string`  
  - **Description:** 目标列名称，模型需要预测的类别标签。  

- `new_data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 新数据集，用于模型预测。  

- `test_size`:  
  - **Type:** `float`  
  - **Description:** 测试集比例，默认为0.2。  

- `n_neighbors`:  
  - **Type:** `int`  
  - **Description:** KNN算法中的近邻数量，默认为3。  

- `with_label`:  
  - **Type:** `bool`  
  - **Description:** 是否将预测结果与`new_data`一起返回，默认为False。  

**Required:**  
- `data`  
- `target`  
- `new_data`  

**Result:**  
- 返回预测结果：  
  - 如果`with_label`为True，则返回包含预测类别的`new_data`；  
  - 否则，仅返回预测类别的列表。  

**Notes:**  
- 函数确保输入数据是连续的NumPy数组以提高性能。  
- 输出模型在测试集上的Accuracy（准确率），帮助用户评估模型性能。  

-----
#### svm_classify
**Name:** svm_classify  
**Description:** 执行支持向量机(SVM)分类  
**Applicable Situations:**  
- 高维数据分类  
- 小样本数据集  
- 需要清晰决策边界的情况  
- 处理线性/非线性可分数据（通过核函数）  

**Parameters:**  
- `data`:  
  - **Type:** `pandas.DataFrame`  
  - **Description:** 包含特征和标签的数据集  
- `independent`:  
  - **Type:** `str` or `list`  
  - **Description:** 特征列名（单个特征传入str，多个特征传入list）  
- `dependent`:  
  - **Type:** `str`  
  - **Description:** 标签列名  
- `**kwargs`:  
  - **Type:** `dict`  
  - **Description:** 传递给SVC的额外参数（如kernel, C, gamma等）  

**Result:**  
返回包含以下内容的字典：  
- `model`: 训练好的SVM模型  
- `y_pred`: 预测标签  
- `accuracy`: 准确率  
- `report`: 分类报告（包含precision/recall/f1-score等）  

**Example Call:**  
```python
result = svm_classify(
    data=df,
    independent=["feature1", "feature2"],
    dependent="label",
    kernel="rbf",
    C=1.0
)
```
-----

#### logistic_classify
**Name:** logistic_classify  
**Description:** 执行逻辑回归进行二分类  
**Applicable Situations:**  
- 二分类或多分类问题  
- 需要概率输出的情况  
- 解释特征重要性（系数分析）  
- 线性可分数据  

**Parameters:**  
- `data`:  
  - **Type:** `pandas.DataFrame`  
  - **Description:** 包含特征和标签的数据集  
- `independent`:  
  - **Type:** `str` or `list`  
  - **Description:** 特征列名（单个特征传入str，多个特征传入list）  
- `dependent`:  
  - **Type:** `str`  
  - **Description:** 标签列名  
- `**kwargs`:  
  - **Type:** `dict`  
  - **Description:** 传递给LogisticRegression的额外参数（如penalty, C, solver等）  

**Result:**  
返回包含以下内容的字典：  
- `model`: 训练好的逻辑回归模型  
- `y_pred`: 预测标签  
- `accuracy`: 准确率  
- `report`: 分类报告  

**Example Call:**  
```python
result = logistic_classify(
    data=df,
    independent="age",
    dependent="purchased",
    penalty="l2",
    solver="lbfgs"
)
```
-----

#### knn_classify
**Name:** knn_classify  
**Description:** 执行K最近邻(KNN)分类  
**Applicable Situations:**  
- 非线性决策边界  
- 小到中等规模数据集  
- 需要惰性学习（无需显式训练）  
- 特征尺度相似的情况  

**Parameters:**  
- `data`:  
  - **Type:** `pandas.DataFrame`  
  - **Description:** 包含特征和标签的数据集  
- `independent`:  
  - **Type:** `str` or `list`  
  - **Description:** 特征列名（单个特征传入str，多个特征传入list）  
- `dependent`:  
  - **Type:** `str`  
  - **Description:** 标签列名  
- `**kwargs`:  
  - **Type:** `dict`  
  - **Description:** 传递给KNeighborsClassifier的额外参数（如n_neighbors, weights等）  

**Result:**  
返回包含以下内容的字典：  
- `model`: 训练好的KNN模型  
- `y_pred`: 预测标签  
- `accuracy`: 准确率  
- `report`: 分类报告  

**Example Call:**  
```python
result = knn_classify(
    data=df,
    independent=["height", "weight"],
    dependent="gender",
    n_neighbors=5,
    weights="distance"
)
```

-----
#### linear_regression
**Name:** linear_regression  
**Description:** 执行线性回归分析（简单或多元），返回模型对象和关键统计指标  
**Applicable Situations:**  
- 探究自变量与因变量的线性关系  
- 预测建模和趋势分析  
- 需要量化变量间关系的科学研究  

**Parameters:**  
- `data`:  
  - **Type:** `pandas.DataFrame`  
  - **Description:** 包含自变量和因变量的数据集  
- `independent`:  
  - **Type:** `str` or `list`  
  - **Description:** 自变量列名（单个变量传入str，多个变量传入list）  
- `dependent`:  
  - **Type:** `str`  
  - **Description:** 因变量列名  

**Result:**  
返回包含以下内容的字典：  
- `model`: statsmodels回归模型对象  
- `summary`: 模型统计摘要（文本格式）  
- `r2`: 决定系数（R-squared）  
- `adj_r2`: 调整后的决定系数  
- `pvalues`: 各系数的p值（字典格式）  
- `residuals`: 模型残差序列  

**Example Call:**  
```python
# 简单线性回归示例
result = linear_regression(
    data=df,
    independent="age",
    dependent="income"
)
```
-----

#### polynomial_regression
**Name:** polynomial_regression  
**Description:** 执行单变量多项式回归分析  
**Applicable Situations:**  
- 数据呈现非线性但可多项式化的关系  
- 需要捕捉变量间的曲线关系  
- 探索特征的高阶效应  

**Parameters:**  
- `data`:  
  - **Type:** `pandas.DataFrame`  
  - **Description:** 包含自变量和因变量的数据集  
- `independent`:  
  - **Type:** `str` or `list`  
  - **Description:** 自变量列名（仅使用第一个变量）  
- `dependent`:  
  - **Type:** `str`  
  - **Description:** 因变量列名  
- `degree`:  
  - **Type:** `int`  
  - **Description:** 多项式阶数（默认=2）  

**Result:**  
返回包含以下内容的字典：  
- `model`: 拟合的线性回归模型  
- `poly_features`: 多项式特征转换器  
- `degree`: 使用的多项式阶数  
- `r2`: 决定系数（R-squared）  
- `y_pred`: 模型预测值  

**Example Call:**  
```python
result = polynomial_regression(
    data=df,
    independent="temperature",
    dependent="yield",
    degree=3
)
```
-----

#### exponential_regression
**Name:** exponential_regression  
**Description:** 执行单变量指数回归分析  
**Applicable Situations:**  
- 数据呈现指数增长/衰减模式  
- 因变量必须为正值  
- 生物学/经济学中的增长模型  

**Parameters:**  
- `data`:  
  - **Type:** `pandas.DataFrame`  
  - **Description:** 包含自变量和因变量的数据集  
- `independent`:  
  - **Type:** `str` or `list`  
  - **Description:** 自变量列名（仅使用第一个变量）  
- `dependent`:  
  - **Type:** `str`  
  - **Description:** 因变量列名（必须>0）  

**Result:**  
返回包含以下内容的字典：  
- `model`: 对数空间的线性模型  
- `a`: 指数系数（y = a*exp(bx)）  
- `b`: 指数增长率  
- `r2`: 对数空间的决定系数  
- `y_pred`: 原始空间的预测值  

**Example Call:**  
```python
result = exponential_regression(
    data=df,
    independent="time",
    dependent="bacteria_count"
)
```
-----

#### power_regression
**Name:** power_regression  
**Description:** 执行单变量幂律回归分析  
**Applicable Situations:**  
- 数据呈现幂律关系（y = a*x^b）  
- 自变量和因变量都必须为正值  
- 物理/工程中的缩放关系分析  

**Parameters:**  
- `data`:  
  - **Type:** `pandas.DataFrame`  
  - **Description:** 包含自变量和因变量的数据集  
- `independent`:  
  - **Type:** `str` or `list`  
  - **Description:** 自变量列名（必须>0）  
- `dependent`:  
  - **Type:** `str`  
  - **Description:** 因变量列名（必须>0）  

**Result:**  
返回包含以下内容的字典：  
- `model`: 对数空间的线性模型  
- `a`: 比例系数  
- `b`: 幂指数  
- `r2`: 原始空间的决定系数  
- `y_pred`: 模型预测值  

**Example Call:**  
```python
result = power_regression(
    data=df,
    independent="body_mass",
    dependent="metabolic_rate"
)
```
-----

#### ransac_regression
**Name:** ransac_regression  
**Description:** 执行RANSAC鲁棒回归（单变量）  
**Applicable Situations:**  
- 数据包含显著离群值  
- 需要稳健的回归估计  
- 异常检测和去除  

**Parameters:**  
- `data`:  
  - **Type:** `pandas.DataFrame`  
  - **Description:** 包含自变量和因变量的数据集  
- `independent`:  
  - **Type:** `str` or `list`  
  - **Description:** 自变量列名（仅使用第一个变量）  
- `dependent`:  
  - **Type:** `str`  
  - **Description:** 因变量列名  

**Result:**  
返回包含以下内容的字典：  
- `model`: RANSAC回归模型  
- `inlier_mask`: 内点布尔掩码  
- `r2`: 内点数据的决定系数  
- `y_pred`: 模型预测值  

**Example Call:**  
```python
result = ransac_regression(
    data=df,
    independent="feature",
    dependent="target"
)
```
-----

#### train_lstm_model
**Name:** train_lstm_model  
**Description:** 使用LSTM模型进行时间序列预测，返回指定长度的未来预测值，并输出模型在测试集上的表现（使用MSE作为评估指标）。  

**Applicable Situations:**  
- 适用于单变量时间序列预测任务。  
- 需要预测未来指定时间步长的数值。  

**Parameters:**  
- `train_data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 训练数据集，包含目标列的时间序列数据。  

- `new_row`:  
  - **Type:** `int`  
  - **Description:** 需要预测的未来时间步长度。  

- `target_col`:  
  - **Type:** `string`  
  - **Description:** 目标列名称，模型需要预测的数值列。  

- `seq_length`:  
  - **Type:** `int`  
  - **Description:** LSTM输入序列的时间步长度。  

- `split`:  
  - **Type:** `float`  
  - **Description:** 测试集比例，默认为0.2。  

- `num_epochs`:  
  - **Type:** `int`  
  - **Description:** 训练轮数，默认为10。  

- `input_size`:  
  - **Type:** `int`  
  - **Description:** LSTM输入特征维度，默认为1（单变量）。  

- `hidden_size`:  
  - **Type:** `int`  
  - **Description:** LSTM隐藏层维度，默认为64。  

- `output_size`:  
  - **Type:** `int`  
  - **Description:** 输出维度，默认为1。  

- `num_layers`:  
  - **Type:** `int`  
  - **Description:** LSTM堆叠层数，默认为3。  

- `learning_rate`:  
  - **Type:** `float`  
  - **Description:** 学习率，默认为0.001。  

- `criterion_fun`:  
  - **Type:** `Callable`  
  - **Description:** 损失函数，默认为`nn.MSELoss()`。  

**Required:**  
- `train_data`  
- `new_row`  
- `target_col`  
- `seq_length`  

**Result:**  
- 返回`new_row`长度的预测值（NumPy数组），数值已反归一化到原始范围。  

**Notes:**  
- 数据会自动归一化到[0,1]范围，预测结果会自动反归一化。  
- 输出模型在测试集上的MSE（均方误差），帮助用户评估模型性能。  
-----

#### train_arima_model
**Name:** train_arima_model  
**Description:** 使用ARIMA模型进行时间序列预测，自动选择最优参数并返回指定长度的未来预测值，输出模型在测试集上的表现（使用MSE作为评估指标）。  

**Applicable Situations:**  
- 适用于单变量时间序列预测任务  
- 支持季节性时间序列预测（需指定seasonal_period）  
- 需要预测未来指定时间步长的数值  

**Parameters:**  
- `train_data`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 训练数据集，包含目标列的时间序列数据  

- `target_col`:  
  - **Type:** `string`  
  - **Description:** 目标列名称，模型需要预测的数值列  

- `new_row`:  
  - **Type:** `int`  
  - **Description:** 需要预测的未来时间步长度  

- `split`:  
  - **Type:** `float`  
  - **Description:** 测试集比例，默认为0.2  

- `seasonal`:  
  - **Type:** `bool`  
  - **Description:** 是否使用季节性ARIMA模型，默认为False  

- `seasonal_period`:  
  - **Type:** `int`  
  - **Description:** 季节性周期长度（当seasonal=True时必须提供）  

**Required:**  
- `train_data`  
- `target_col`  
- `new_row`  

**Result:**  
- 返回`new_row`长度的预测值（NumPy数组，shape为(new_row,1)），数值已反归一化到原始范围  

**Notes:**  
- 使用auto_arima自动选择最优(p,d,q)(P,D,Q)参数组合  
- 数据会自动归一化到[0,1]范围，预测结果会自动反归一化  
- 输出模型在测试集上的MSE（均方误差）和模型摘要  
- 当seasonal=True时，必须提供seasonal_period参数  


