#### solve_lp
**Name:** solve_lp  
**Description:** 用于简单线性规划问题，线性规划求解函数。
**Applicable Situations:**  
- 需要求解目标函数最小化的线性规划问题  

**Parameters:**  
- `c`:  
  - ​**Type:** `array-like`  
  - ​**Description:** 目标函数系数向量，维度需与变量数一致  

- `A_ub`:  
  - ​**Type:** `2D array-like | None`  
  - ​**Description:** 不等式约束系数矩阵（默认None），每行对应一个约束条件  

- `b_ub`:  
  - ​**Type:** `array-like | None`  
  - ​**Description:** 不等式约束右侧常数向量（默认None）  

- `A_eq`:  
  - ​**Type:** `2D array-like | None`  
  - ​**Description:** 等式约束系数矩阵（默认None）  

- `b_eq`:  
  - ​**Type:** `array-like | None`  
  - ​**Description:** 等式约束右侧常数向量（默认None）  

- `bounds`:  
  - ​**Type:** `list of tuples | None`  
  - ​**Description:** 变量边界列表（默认None），每个元组表示变量下界和上界  
  - ​**Example:** `[(0, None), (0, 5)]` 表示x₁≥0，x₂∈[0,5]  

**Result:**  
- 成功时返回包含最优解x和最优值fun的元组  
- 失败时返回包含错误信息的字符串  

**Example Call:**  
```python
# 求解 min -x1 + 4x2 
# s.t.  x1 + x2 <= 5
#       2x1 + x2 = 4
#       x1 ∈ [0,3], x2 ≥ 0
x_opt, f_opt = solve_lp(
    c=[-1, 4],
    A_ub=[[1, 1]],
    b_ub=[5],
    A_eq=[[2, 1]],
    b_eq=[4],
    bounds=[(0, 3), (0, None)]
)
```
-----

#### solve_ilp
**Name:** solve_ilp  
**Description:** 用于整数规划问题，整数规划求解函数，支持混合整数约束。
**Applicable Situations:**   
- 整数线性规划

**Parameters:**  
- `c`:  
  - ​**Type:** `array-like`  
  - ​**Description:** 目标函数系数向量  

- `A_ub`:  
  - ​**Type:** `2D array-like | None`  
  - ​**Description:** 不等式约束矩阵（默认None）  

- `b_ub`:  
  - ​**Type:** `array-like | None`  
  - ​**Description:** 不等式约束右侧向量（默认None）  

- `A_eq`:  
  - ​**Type:** `2D array-like | None`  
  - ​**Description:** 等式约束矩阵（默认None）  

- `b_eq`:  
  - ​**Type:** `array-like | None`  
  - ​**Description:** 等式约束右侧向量（默认None）  

- `bounds`:  
  - ​**Type:** `list of tuples | None`  
  - ​**Description:** 变量边界列表（默认None），例如 `[(0,5), (1,10)]`  

**Result:**  
- 成功时返回元组 `(x, obj)`，包含整数最优解和目标值  
- 失败时返回错误描述字符串  

**Example Call:**  
```python
# 求解 min 3x1 + 4x2 
# s.t. x1 + 2x2 <= 7, x1, x2 ∈ {0,1,2,...}
solution, obj_value = solve_ilp(
    c=[3, 4],
    A_ub=[[1, 2]],
    b_ub=[7],
    bounds=[(0, None), (0, None)]
)
```
-----

#### solve_nlp
**Name:** solve_nlp  
**Description:** 用于非线性规划的函数问题，支持通用非线性优化。
**Applicable Situations:**  
- 目标函数或约束为非线性的优化问题  
- 工程参数优化、非线性曲线拟合等场景  

**Parameters:**  
- `fun`:  
  - ​**Type:** `callable`  
  - ​**Description:** 目标函数，格式 `fun(x) -> float`  

- `x0`:  
  - ​**Type:** `array-like`  
  - ​**Description:** 初始猜测解  

- `constraints`:  
  - ​**Type:** `list of dict | None`  
  - ​**Description:** 约束列表（默认None），每个字典定义一种约束，支持两种格式：  
    ◦ **不等式约束**: `{"type": "ineq", "fun": callable}` → 约束条件为 `fun(x) >= 0`  
    ◦ **等式约束**: `{"type": "eq", "fun": callable}` → 约束条件为 `fun(x) = 0`  
  - ​**Note:** 若约束返回多维数组（如多条件约束），需确保`fun`返回一维数组  

- `bounds`:  
  - ​**Type:** `list of tuples | None`  
  - ​**Description:** 变量边界列表（默认None）  

- `method`:  
  - ​**Type:** `str`  
  - ​**Description:** 求解器类型（默认'SLSQP'），可选 'trust-constr' 等  

**Result:**  
- 成功时返回元组 `(x, fun)`，包含最优解和目标值  
- 失败时返回错误描述字符串  

**Example Call:**  
```python
# 求解 min x^2 + y^2 
# s.t. x + y >= 1
result = solve_nlp(
    fun=lambda x: x[0]**2 + x[1]**2,
    x0=[0, 0],
    constraints=[{
        "type": "ineq",
        "fun": lambda x: x[0] + x[1] - 1  # 转换为 x0 + x1 -1 >= 0
    }]
)
```
-----

#### solve_qp
**Name:** solve_qp  
**Description:** 二次规划求解函数，支持凸二次优化，解决二次规划问题。
**Applicable Situations:**  
- 投资组合优化、最小二乘问题等二次目标场景  

**Parameters:**  
- `Q`:  
  - ​**Type:** `2D array-like`  
  - ​**Description:** 二次项系数矩阵（需对称）
   - ​**Notice:** 对应标准形式 `0.5*x^T Q x + c^T x`。若原问题未自带1/2系数，请手动缩放矩阵。

- `c`:  
  - ​**Type:** `array-like`  
  - ​**Description:** 线性项系数向量  

- `A`:  
  - ​**Type:** `2D array-like | None`  
  - ​**Description:** 不等式约束矩阵（默认None）  

- `b`:  
  - ​**Type:** `array-like | None`  
  - ​**Description:** 不等式约束右侧向量（默认None）  

- `A_eq`:  
  - ​**Type:** `2D array-like | None`  
  - ​**Description:** 等式约束矩阵（默认None）  

- `b_eq`:  
  - ​**Type:** `array-like | None`  
  - ​**Description:** 等式约束右侧向量（默认None）  

- `bounds`:  
  - ​**Type:** `list of tuples | None`  
  - ​**Description:** 变量边界列表（默认None）  

**Result:**  
- 成功时返回元组 `(x, obj)`，包含最优解和目标值  
- 失败时返回错误描述字符串  

**Example Call:**  
```python
# 求解 min 0.5x² + y² + x + y
# s.t. x + y <= 3
x_opt, f_opt = solve_qp(
    Q=[[1, 0], [0, 2]],
    c=[1, 1],
    A=[[1, 1]],
    b=[3]
)
```
-----

#### solve_qcqp
**Name:** solve_qcqp  
**Description:** 用于二次约束凸规划问题，二次约束凸规划求解函数  
**Applicable Situations:**  
- 带二次约束的凸优化问题  
- 结构设计优化、鲁棒控制等场景  

**Parameters:**  
- `c`:  
  - ​**Type:** `array-like`  
  - ​**Description:** 目标函数线性项系数  

- `A`:  
  - ​**Type:** `2D array-like | None`  
  - ​**Description:** 线性不等式约束矩阵（默认None）  

- `b`:  
  - ​**Type:** `array-like | None`  
  - ​**Description:** 线性不等式约束右侧向量（默认None）  

- `quad_constraints`:  
  - ​**Type:** `list of tuples | None`  
  - ​**Description:** 二次约束列表，每个元素为 `(Q, l, r)`，对应 `x^T Q x + l^T x <= r`  

**Result:**  
- 成功时返回元组 `(x, obj)`，包含最优解和目标值  
- 失败时返回错误描述字符串  

**Example Call:**  
```python
# 求解 min x + y 
# s.t. x² + y² <= 1
solution = solve_qcqp(
    c=[1, 1],
    quad_constraints=[(np.eye(2), [0, 0], 1)]
)
```
-----

#### solve_zop
**Name:** solve_zop  
**Description:** 用于0-1整数规划问题，用于求解零一整数规划 。
**Applicable Situations:**  
- 需要变量取0或1的组合优化问题  

**Parameters:**  
- `c`:  
  - ​**Type:** `array-like`  
  - ​**Description:** 目标函数系数向量  

- `A`:  
  - ​**Type:** `2D array-like | None`  
  - ​**Description:** 不等式约束矩阵（默认None）  

- `b`:  
  - ​**Type:** `array-like | None`  
  - ​**Description:** 不等式约束右侧向量（默认None）  

**Result:**  
- 成功时返回元组 `(x, obj)`，包含0-1解和目标值  
- 失败时返回错误描述字符串  

**Example Call:**  
```python
# 求解 min 3x1 + 2x2 
# s.t. x1 + x2 >= 1
binary_solution = solve_zop(
    c=[3, 2],
    A=[[-1, -1]],  # 转换为 -x1 -x2 <= -1
    b=[-1]
)
```
-----

#### solve_socp
**Name:** solve_socp  
**Description:** 二阶锥规划求解函数  
**Applicable Situations:**  
- 带二阶锥约束的优化问题  
- 鲁棒优化、金融风险控制等场景  

**Parameters:**  
- `c`:  
  - ​**Type:** `array-like`  
  - ​**Description:** 目标函数系数向量  

- `A`:  
  - ​**Type:** `2D array-like | None`  
  - ​**Description:** 线性不等式约束矩阵（默认None）  

- `b`:  
  - ​**Type:** `array-like | None`  
  - ​**Description:** 线性不等式约束右侧向量（默认None）  

- `socp_constraints`:  
  - ​**Type:** `list of tuples | None`  
  - ​**Description:** 二阶锥约束列表，每个元素为 `(A_socp, b_socp, c_socp, d_socp)`，对应 `||A_socp*x + b_socp||₂ <= c_socp^T x + d_socp`  

**Result:**  
- 成功时返回元组 `(x, obj)`，包含最优解和目标值  
- 失败时返回错误描述字符串  

**Example Call:**  
```python
# 求解 min x 
# s.t. ||(y, z)||₂ <= x
socp_solution = solve_socp(
    c=[1, 0, 0],
    socp_constraints=[(
        [[0, 1, 0], [0, 0, 1]],  # A_socp
        [0, 0],                   # b_socp
        [1, 0, 0],               # c_socp
        0                         # d_socp
    )]
)
```
-----

#### solve_mulob
**Name:** solve_mulob 
**Description:** 多目标优化求解函数，使用线性加权法将多目标转化为单目标优化  
**Applicable Situations:**  
• 需要权衡多个线性目标的场景（如成本与收益平衡）  
• 支持最大化/最小化目标混合优化  

**Parameters:**  
• `c_list`:  
  • ​**Type:** `list of array-like`  
  • ​**Description:** 目标函数系数列表，每个元素对应一个目标函数的系数数组  

• `senses`:  
  • ​**Type:** `list of str`  
  • ​**Description:** 优化方向列表，每个元素取'min'或'max'  
  • ​**Notice:** 若存在最大化目标，函数内部自动转换为等效最小化形式  

• `weights`:  
  • ​**Type:** `list of float`  
  • ​**Description:** 各目标函数的归一化权重  
  • ​**Constraint:** 权重之和必须严格等于1  

• `A_ub`:  
  • ​**Type:** `2D array-like | None`  
  • ​**Description:** 不等式约束矩阵（默认None），形式为 `A_ub * x <= b_ub`  

• `b_ub`:  
  • ​**Type:** `array-like | None`  
  • ​**Description:** 不等式约束右侧向量（默认None）  

• `A_eq`:  
  • ​**Type:** `2D array-like | None`  
  • ​**Description:** 等式约束矩阵（默认None），形式为 `A_eq * x == b_eq`  

• `b_eq`:  
  • ​**Type:** `array-like | None`  
  • ​**Description:** 等式约束右侧向量（默认None）  

• `bounds`:  
  • ​**Type:** `list of tuples | None`  
  • ​**Description:** 变量边界列表，每个元组形式为 `(lb, ub)`（默认None）  

**Result:**  
• 成功时返回元组 `(x, objs)`，包含最优解和各目标函数值  
• 失败时返回错误描述字符串  

**Example Call:**  
```python
# 优化目标：min(2x + 3y) 与 max(5x + 4y) 加权平衡
# 约束：x + y <= 10, x >= 0, y >= 0
x_opt, objs = solve_multi_objective(
    c_list=[[2, 3], [5, 4]], 
    senses=['min', 'max'],
    weights=[0.6, 0.4],
    A_ub=[[1, 1]],
    b_ub=[10],
    bounds=[(0, None), (0, None)]
)
```
-----

#### multidimensional_knapsack
**Name:** multidimensional_knapsack  
**Description:** 多维背包问题求解器，支持精确整数规划和贪心启发式算法，适用于多资源约束下的物品选择问题  
**Applicable Situations:**  
- 资源受限的物品选择问题  
- 投资组合、项目筛选等多维约束优化场景  
- 工程优化和调度决策  

**Parameters:**  
- `items`:  
  - **Type:** `pd.DataFrame`  
  - **Description:** 包含物品数据的数据框，必须包含 `'value'` 列及各维度资源列  
- `constraints`:  
  - **Type:** `dict`  
  - **Description:** 各维度资源约束，如 `{'cpu': 100, 'memory': 512}`，默认为小于等于约束  
- `method`:  
  - **Type:** `str`  
  - **Description:** 求解方法，支持 `'exact'`（精确整数规划）和 `'greedy'`（贪心启发式算法），默认 `'exact'`  
- `maximize`:  
  - **Type:** `bool`  
  - **Description:** 是否最大化价值，默认为 `True`  
- `solver`:  
  - **Type:** `str`  
  - **Description:** 整数规划求解器（当 `method='exact'` 时有效），默认 `'PULP_CBC_CMD'`  
- `time_limit`:  
  - **Type:** `int`  
  - **Description:** 求解时间限制（秒），默认 `300`  

**Result:**  
- 返回一个字典，包含：  
  - `selected_items`: 选中物品的索引列表  
  - `total_value`: 选中物品的总价值  
  - `resource_usage`: 各维度资源的使用量  
  - `status`: 求解状态（如 `optimal`、`feasible` 或其他状态描述）  

**Example Call:**  
```python
result = multidimensional_knapsack(
    items=pd.DataFrame({
        'value': [10, 20, 30],
        'cpu': [1, 2, 3],
        'memory': [100, 200, 300]
    }),
    constraints={'cpu': 5, 'memory': 500},
    method='exact',
    maximize=True
)
```
-----

#### quadratic_assignment
**Name:** quadratic_assignment  
**Description:** 二次指派问题求解器 (QAP)，支持精确枚举和模拟退火算法，用于最小化设施之间流量与距离乘积的成本  
**Applicable Situations:**  
- 设施布局和分配问题  
- 优化生产、物流或通信网络中的位置安排  
- 运筹优化与组合决策场景  

**Parameters:**  
- `distance_matrix`:  
  - **Type:** `list`  
  - **Description:** 表示位置间距离的方阵，尺寸为 (n x n)  
- `flow_matrix`:  
  - **Type:** `list`  
  - **Description:** 表示设施间流量的方阵，尺寸为 (n x n)  
- `method`:  
  - **Type:** `str`  
  - **Description:** 求解方法，支持 `'exact'`（精确枚举，仅适用于 n <= 12）和 `'simulated_annealing'`（模拟退火），默认 `'simulated_annealing'`  
- `time_limit`:  
  - **Type:** `int`  
  - **Description:** 最大运行时间（秒），默认 `60`  
- `initial_temp`:  
  - **Type:** `float`  
  - **Description:** 模拟退火初始温度，默认 `1000`  
- `cooling_rate`:  
  - **Type:** `float`  
  - **Description:** 模拟退火的冷却率（建议范围 0.8~0.99），默认 `0.95`  
- `seed`:  
  - **Type:** `int | None`  
  - **Description:** 随机种子，用于结果复现  

**Result:**  
- 返回一个字典，包含：  
  - `assignment`: 设施到位置的分配结果（映射位置索引到设施索引）  
  - `total_cost`: 总成本（目标函数值）  
  - `runtime`: 求解运行时间（秒）  
  - `history`: 成本变化历史（仅针对启发式算法，如模拟退火）  
  - `status`: 求解状态（如 `optimal`、`feasible` 或 `timeout`）  

**Example Call:**  
```python
result = quadratic_assignment(
    distance_matrix=[[0, 2, 3],[2, 0, 1],
                              [3, 1, 0]],
    flow_matrix=np.array([[0, 5, 2],
                          [5, 0, 3],
                          [2, 3, 0]]),
    method='simulated_annealing',
    time_limit=60,
    initial_temp=1000,
    cooling_rate=0.95,
    seed=42
)
```