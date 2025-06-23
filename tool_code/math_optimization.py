from scipy.optimize import linprog
import cvxpy as cp
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from itertools import permutations
import pulp
import random
import math
import time

def get_header():
    header ="""from scipy.optimize import linprog
import cvxpy as cp
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from itertools import permutations
import pulp
import random
import math
import time"""
    return header
# 线性规划求解函数
def solve_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """
    求解线性规划问题:
      minimize: c^T x
      subject to: A_ub x <= b_ub, A_eq x = b_eq, 以及变量界限 bounds
    参数:
      c     : 目标函数系数向量
      A_ub  : 不等式约束系数矩阵
      b_ub  : 不等式约束右侧向量
      A_eq  : 等式约束系数矩阵
      b_eq  : 等式约束右侧向量
      bounds: 每个变量的取值范围（例如 [(0, None), (0, None)] 表示非负）
    返回:
      x: 最优解
      fun: 最优目标函数值
    """
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        return res.x, res.fun
    else:
        return f"Optimization failed: {res.message}"
    
# 整数线性规划求解函数
def solve_ilp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """
    求解整数线性规划问题:
      minimize: c^T x
      subject to: A_ub * x <= b_ub, A_eq * x = b_eq, 以及变量的界限 bounds
    参数:
      c      : 目标函数系数向量
      A_ub   : 不等式约束系数矩阵
      b_ub   : 不等式约束右侧向量
      A_eq   : 等式约束系数矩阵
      b_eq   : 等式约束右侧向量
      bounds : 每个变量的取值范围，例如 [(0, 10), (0, 5)] 表示两个变量的界限
    返回:
      x      : 最优整数解
      obj    : 目标函数值
      status : 求解状态信息
    """
    c = np.array(c)
    n = len(c)
    x = cp.Variable(n, integer=True)  # 整数变量
    constraints = []

    # 处理不等式约束 A_ub * x <= b_ub
    if A_ub is not None and b_ub is not None:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        constraints.append(A_ub @ x <= b_ub)

    # 处理等式约束 A_eq * x = b_eq
    if A_eq is not None and b_eq is not None:
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        constraints.append(A_eq @ x == b_eq)

    # 处理变量界限 bounds
    if bounds is not None:
        for i, bound in enumerate(bounds):
            lb, ub = bound
            if lb is not None:
                constraints.append(x[i] >= lb)
            if ub is not None:
                constraints.append(x[i] <= ub)

    # 定义目标函数
    objective = cp.Minimize(c.T @ x)
    
    # 设置问题并求解
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if prob.status in ["optimal", "optimal_inaccurate"]:
        return x.value, prob.value
    else:
        return f"Optimization failed: {prob.status}"

# 非线性规划求解函数
def solve_nlp(fun, x0, constraints=None, bounds=None, method='SLSQP'):
    """
    求解非线性规划问题:
      minimize: fun(x)
    参数:
      fun: 目标函数，接受 x（数组）并返回标量值
      x0: 初始解
      constraints: 约束条件列表（字典形式，参见 minimize 文档）
      bounds: 每个变量的取值范围
      method: 求解方法，默认 'SLSQP'
    返回:
      x: 最优解
      fun: 最优目标函数值
    """
    res = minimize(fun, x0, method=method, bounds=bounds, constraints=constraints)
    if res.success:
        return res.x, res.fun
    else:
        return f"Optimization failed: {res.message}"

# 二次规划求解函数
def solve_qp(Q, c, A=None, b=None, A_eq=None, b_eq=None, bounds=None):
    """
    求解二次规划问题:
      minimize: 0.5 * x^T Q x + c^T x
      subject to: 线性不等式和等式约束，以及变量界限
    参数:
      Q     : 二次项矩阵（对称正定为佳）
      c     : 线性项系数向量
      A, b  : 不等式约束 A x <= b
      A_eq, b_eq: 等式约束 A_eq x = b_eq
      bounds: 每个变量的取值范围，格式为 [(lb, ub), ...]
    返回:
      x: 最优解
      obj: 目标函数值
    """
    Q = np.array(Q)
    c = np.array(c)
    n = len(c)
    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x)
    constraints = []
    if A is not None and b is not None:
        A = np.array(A)
        b = np.array(b)
        constraints.append(A @ x <= b)
    if A_eq is not None and b_eq is not None:
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        constraints.append(A_eq @ x == b_eq)
    if bounds is not None:
        for i, bound in enumerate(bounds):
            lb, ub = bound
            if lb is not None:
                constraints.append(x[i] >= lb)
            if ub is not None:
                constraints.append(x[i] <= ub)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status in ["optimal", "optimal_inaccurate"]:
        return x.value, prob.value
    else:
        return f"Optimization failed: {prob.status}"
    
# 二次约束凸规划求解函数
def solve_qcqp(c, A=None, b=None, quad_constraints=None):
    """
    求解带二次约束的凸规划问题:
      minimize: c^T x
      subject to: A x <= b 和若干二次约束
    参数:
      c: 目标函数线性项系数向量
      A, b: 线性不等式约束 A x <= b
      quad_constraints: 列表，每个元素为 (Q, l, r)，表示二次约束: x^T Q x + l^T x <= r
    返回:
      x: 最优解
      obj: 目标函数值
    """
    c = np.array(c)
    n = len(c)
    x = cp.Variable(n)
    objective = cp.Minimize(c.T @ x)
    constraints = []
    if A is not None and b is not None:
        A = np.array(A)
        b = np.array(b)
        constraints.append(A @ x <= b)
    if quad_constraints is not None:
        for Q, l, r in quad_constraints:
            Q = np.array(Q)
            l = np.array(l)
            constraints.append(cp.quad_form(x, Q) + l.T @ x <= r)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status in ["optimal", "optimal_inaccurate"]:
        return x.value, prob.value
    else:
        return f"Optimization failed: {prob.status}"
    
# 零一规划求解函数
def solve_zop(c, A=None, b=None):
    """
    求解零一规划问题:
      minimize: c^T x
      subject to: A x <= b, x ∈ {0, 1}
    参数:
      c: 目标函数系数向量
      A, b: 约束 A x <= b
    返回:
      x: 最优解（0/1 向量）
      obj: 目标函数值
    """
    c = np.array(c)
    n = len(c)
    x = cp.Variable(n, integer=True)
    constraints = [x >= 0, x <= 1]
    if A is not None and b is not None:
        A = np.array(A)
        b = np.array(b)
        constraints.append(A @ x <= b)
    prob = cp.Problem(cp.Minimize(c.T @ x), constraints)
    prob.solve()
    if prob.status in ["optimal", "optimal_inaccurate"]:
        return x.value, prob.value
    else:
        return f"Optimization failed: {prob.status}"
    
# 二阶锥规划求解函数
def solve_socp(c, A=None, b=None, socp_constraints=None):
    """
    求解二阶锥规划问题:
      minimize: c^T x
      subject to: A x <= b 和二阶锥约束
    参数:
      c: 目标函数系数向量
      A, b: 线性不等式约束 A x <= b
      socp_constraints: 列表，每个元素为 (A_socp, b_socp, c_socp, d_socp)
                        表示约束: || A_socp*x + b_socp ||_2 <= c_socp^T x + d_socp
    返回:
      x: 最优解
      obj: 目标函数值
    """
    c = np.array(c)
    n = len(c)
    x = cp.Variable(n)
    objective = cp.Minimize(c.T @ x)
    constraints = []
    if A is not None and b is not None:
        A = np.array(A)
        b = np.array(b)
        constraints.append(A @ x <= b)
    if socp_constraints is not None:
        for A_socp, b_socp, c_socp, d_socp in socp_constraints:
            A_socp = np.array(A_socp)
            b_socp = np.array(b_socp)
            c_socp = np.array(c_socp)
            constraints.append(cp.SOC(d_socp + c_socp.T @ x, A_socp @ x + b_socp))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status in ["optimal", "optimal_inaccurate"]:
        return x.value, prob.value
    else:
        return f"Optimization failed: {prob.status}"
    
def solve_mulob(c_list, senses, weights, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """
    使用线性加权法求解多目标优化问题:
      minimize: weights[0]*f1(x) + weights[1]*f2(x) + ... + weights[n]*fn(x)
      subject to: A_ub * x <= b_ub, A_eq * x = b_eq, 以及变量的界限 bounds
    
    参数:
      c_list  : 目标函数系数列表，每个元素是一个系数数组
      senses  : 优化方向列表，'min'或'max'
      weights : 各目标函数的权重列表
      A_ub    : 不等式约束系数矩阵
      b_ub    : 不等式约束右侧向量
      A_eq    : 等式约束系数矩阵
      b_eq    : 等式约束右侧向量
      bounds  : 每个变量的取值范围
    
    返回:
      x       : 最优解
      objs    : 各目标函数值列表
    """
    # 验证输入
    if len(c_list) != len(senses) or len(c_list) != len(weights):
        raise ValueError("目标函数、优化方向和权重数量必须一致")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("权重之和必须为1")
    
    # 确定变量数量
    if bounds is not None:
        n = len(bounds)
    elif A_ub is not None:
        n = A_ub.shape[1] if hasattr(A_ub, 'shape') else len(A_ub[0])
    elif A_eq is not None:
        n = A_eq.shape[1] if hasattr(A_eq, 'shape') else len(A_eq[0])
    else:
        raise ValueError("无法确定变量数量，请提供bounds、A_ub或A_eq")
    
    x = cp.Variable(n)
    constraints = []
    
    # 处理不等式约束 A_ub * x <= b_ub
    if A_ub is not None and b_ub is not None:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        constraints.append(A_ub @ x <= b_ub)
    
    # 处理等式约束 A_eq * x = b_eq
    if A_eq is not None and b_eq is not None:
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        constraints.append(A_eq @ x == b_eq)
    
    # 处理变量界限 bounds
    if bounds is not None:
        for i, bound in enumerate(bounds):
            lb, ub = bound
            if lb is not None:
                constraints.append(x[i] >= lb)
            if ub is not None:
                constraints.append(x[i] <= ub)
    
    # 构建加权目标函数
    weighted_objectives = []
    for c, sense, weight in zip(c_list, senses, weights):
        c = np.array(c)
        if sense == 'max':
            weighted_objectives.append(weight * (-c.T @ x))  # 最大化转为最小化
        else:
            weighted_objectives.append(weight * (c.T @ x))
    
    combined_objective = cp.sum(weighted_objectives)
    
    # 设置问题并求解
    prob = cp.Problem(cp.Minimize(combined_objective), constraints)
    prob.solve()
    
    if prob.status in ["optimal", "optimal_inaccurate"]:
        # 计算各目标函数值
        obj_values = []
        for c, sense in zip(c_list, senses):
            c = np.array(c)
            if sense == 'max':
                obj_values.append(c.T @ x.value)
            else:
                obj_values.append(c.T @ x.value)
        
        return x.value, obj_values
    else:
        return f"Optimization failed: {prob.status}"

# 多维背包问题
def multidimensional_knapsack(
    items: pd.DataFrame,
    constraints: dict,
    method: str = 'exact',
    maximize: bool = True,
    solver: str = 'PULP_CBC_CMD',
    time_limit: int = 300
) -> dict:
    """
    多维背包问题求解器
    
    参数：
    items : pd.DataFrame
        物品数据框，必须包含'value'列和各维度资源列
    constraints : dict
        各维度的资源约束，如 {'cpu': 100, 'memory': 512},默认都是小于等于约束
    method : str (可选)
        求解方法：'exact'（精确算法），'greedy'（贪心算法）
    maximize : bool (可选)
        是否最大化价值，默认为True
    solver : str (可选)
        整数规划求解器（仅当method='exact'时有效）
    time_limit : int (可选)
        求解时间限制（秒）
    
    返回：
    dict 包含：
        - selected_items: 选中物品索引列表
        - total_value: 总价值
        - resource_usage: 各维度资源使用量
        - status: 求解状态
    """
    
    # 参数校验
    required_columns = ['value'] + list(constraints.keys())
    if not all(col in items.columns for col in required_columns):
        missing = [col for col in required_columns if col not in items.columns]
        raise ValueError(f"缺少必要列: {missing}")
    
    if any(items[col].lt(0).any() for col in required_columns):
        raise ValueError("所有数值列必须为非负值")
    
    # 精确解法（整数规划）
    if method == 'exact':
        prob = pulp.LpProblem("MKP", pulp.LpMaximize if maximize else pulp.LpMinimize)
        
        # 创建二进制决策变量
        x = pulp.LpVariable.dicts('x', items.index, cat='Binary')
        
        # 设置目标函数
        prob += pulp.lpSum(items.loc[i, 'value'] * x[i] for i in items.index)
        
        # 添加资源约束
        for dim, limit in constraints.items():
            prob += pulp.lpSum(items.loc[i, dim] * x[i] for i in items.index) <= limit
            
        # 求解问题
        try:
            prob.solve(solver=getattr(pulp, solver)(msg=0, timeLimit=time_limit))
        except pulp.PulpSolverError:
            raise ValueError("求解器配置错误")
        
        # 处理结果
        if pulp.LpStatus[prob.status] == 'Optimal':
            selected = [i for i in items.index if x[i].value() == 1]
            status = 'optimal'
        else:
            selected = []
            status = pulp.LpStatus[prob.status].lower()
    
    # 贪心算法
    elif method == 'greedy':
        # 计算价值密度（考虑多维度归一化）
        items_copy = items.copy()
        dimensions = constraints.keys()
        
        # 归一化资源消耗并计算综合密度
        items_copy['density'] = items.apply(
            lambda row: row['value'] / sum(row[dim]/constraints[dim] for dim in dimensions), 
            axis=1
        )
        
        # 按密度降序排序
        sorted_items = items_copy.sort_values('density', ascending=False)
        
        selected = []
        remaining = constraints.copy()
        
        for idx, row in sorted_items.iterrows():
            if all(row[dim] <= remaining[dim] for dim in dimensions):
                selected.append(idx)
                for dim in dimensions:
                    remaining[dim] -= row[dim]
        
        status = 'feasible' if selected else 'infeasible'
    
    else:
        raise ValueError("支持的解法: 'exact' 或 'greedy'")
    
    # 计算结果指标
    total_value = items.loc[selected, 'value'].sum() if selected else 0
    resource_usage = {dim: items.loc[selected, dim].sum() for dim in constraints.keys()}
    
    return {
        'selected_items': selected,
        'total_value': total_value,
        'resource_usage': resource_usage,
        'status': status
    }

# 二次指派问题求解器
def quadratic_assignment(
    distance_matrix: list,
    flow_matrix: list,
    method: str = 'simulated_annealing',
    time_limit: int = 60,
    initial_temp: float = 1000,
    cooling_rate: float = 0.95,
    seed: int = None
) -> dict:
    """
    二次指派问题求解器 (QAP)
    
    参数：
    distance_matrix : 位置间距离矩阵 (n x n)
    flow_matrix : 设施间流量矩阵 (n x n)
    method : 求解方法 ['exact', 'simulated_annealing']
    time_limit : 最大运行时间（秒）
    initial_temp : 模拟退火初始温度
    cooling_rate : 退火冷却率 (0.8~0.99)
    seed : 随机种子
    
    返回：
    dict 包含：
        - assignment: 设施到位置的分配 (位置索引 -> 设施索引)
        - total_cost: 总成本
        - runtime: 运行时间（秒）
        - history: 成本变化历史（仅启发式算法）
        - status: 状态描述
    """
    
    # 参数校验
    if not isinstance(distance_matrix, np.ndarray):
        distance_matrix = np.array(distance_matrix)
    if not isinstance(flow_matrix, np.ndarray):
        flow_matrix = np.array(flow_matrix)
    n = distance_matrix.shape[0]
    if n != flow_matrix.shape[0] or n != distance_matrix.shape[1] or n != flow_matrix.shape[1]:
        raise ValueError("输入矩阵必须是相同大小的方阵")
    
    if not (np.all(distance_matrix >= 0) and np.all(flow_matrix >= 0)):
        raise ValueError("矩阵元素必须为非负值")
    
    # 精确解法（仅适用于n <= 12）
    if method == 'exact':
        start_time = time.time()
        best_cost = float('inf')
        best_assignment = None
        
        for perm in permutations(range(n)):
            if time.time() - start_time > time_limit:
                break
                
            cost = 0
            for i in range(n):
                for j in range(n):
                    cost += flow_matrix[i][j] * distance_matrix[perm[i]][perm[j]]
            
            if cost < best_cost:
                best_cost = cost
                best_assignment = perm
        
        runtime = time.time() - start_time
        return {
            'assignment': dict(zip(range(n), best_assignment)),
            'total_cost': best_cost,
            'runtime': runtime,
            'status': 'optimal' if best_assignment else 'timeout'
        }
    
    # 模拟退火算法（适用于n > 12）
    elif method == 'simulated_annealing':
        def calculate_cost(perm):
            return np.sum(flow_matrix * distance_matrix[np.ix_(perm, perm)])
        
        random.seed(seed)
        current = list(range(n))
        random.shuffle(current)
        current_cost = calculate_cost(current)
        best = current.copy()
        best_cost = current_cost
        history = [current_cost]
        temp = initial_temp
        start_time = time.time()
        
        while temp > 1e-5 and (time.time() - start_time) < time_limit:
            # 生成邻域解
            i, j = random.sample(range(n), 2)
            neighbor = current.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
            # 计算成本变化
            neighbor_cost = calculate_cost(neighbor)
            delta = neighbor_cost - current_cost
            
            # 接受准则
            if delta < 0 or math.exp(-delta / temp) > random.random():
                current = neighbor
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost
            
            history.append(best_cost)
            temp *= cooling_rate
        
        return {
            'assignment': dict(zip(range(n), best)),
            'total_cost': best_cost,
            'runtime': time.time() - start_time,
            'history': history,
            'status': 'feasible'
        }
    else:
        raise ValueError("支持的解法: 'exact' 或 'simulated_annealing'")