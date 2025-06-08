import networkx as nx

def get_header():
    header = """import networkx as nx"""
    return header
# dijkstra 算法
def dijkstra(edges: dict, start: str, end: str = None, directed: bool = False):
    """
    使用 NetworkX 实现 Dijkstra 算法，支持无向图和有向图。

    :param edges: 字典，表示图的边，键是元组 (u, v)，值是边的权重
    :param start: 起点节点
    :param end: 终点节点（可选），如果未提供，默认返回从 start 到所有节点的最短路径
    :param directed: 是否是有向图（默认 False，即无向图）
    :return: 如果指定 end，返回最短路径长度；否则返回所有节点的最短路径距离
    """
    # 创建图（无向或有向）
    graph = nx.DiGraph() if directed else nx.Graph()
    
    # 添加边
    for (u, v), weight in edges.items():
        graph.add_edge(u, v, weight=weight)
    
    # 如果指定了终点，计算 start → end 的最短路径
    if end:
        try:
            return nx.dijkstra_path_length(graph, start, end)
        except nx.NetworkXNoPath:
            return f"No path from {start} to {end}"
    
    # 否则，返回 start 到所有可达节点的最短路径
    return nx.single_source_dijkstra_path_length(graph, start)

# 最小生成树 (MST) 算法
def mst(edges:dict, algorithm:str = 'prim'):
    """
    计算最小生成树 (MST)，支持 Prim 和 Kruskal 算法。

    :param edges: 字典，键为 (u, v) 表示边，值为对应的权重
    :param algorithm: 选择计算 MST 的算法，'prim' 或 'kruskal'（默认 'prim'）
    :return: (mst_edges, total_weight)
             mst_edges 为 MST 的边列表，每个元素形如 (u, v, {'weight': w})
             total_weight 为 MST 的总权重
    """
    # 构造无向图
    G = nx.Graph()
    for (u, v), weight in edges.items():
        G.add_edge(u, v, weight=weight)
    
    # 计算最小生成树
    if algorithm not in ['prim', 'kruskal']:
        raise ValueError("algorithm 必须是 'prim' 或 'kruskal'")
    
    mst = nx.minimum_spanning_tree(G, algorithm=algorithm)
    
    # 计算 MST 的总权重
    total_weight = sum(data['weight'] for _, _, data in mst.edges(data=True))
    mst_edges = list(mst.edges(data=True))
    
    return mst_edges, total_weight

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

# 顶点着色问题
def greedy_coloring(graph:dict):
    """
    使用贪心算法解决顶点着色问题
    :param graph: 字典，表示图，键为节点，值为该节点的邻接节点列表
    :return: 字典，键为节点，值为该节点的颜色
    """
    # 存储每个顶点的颜色
    colors = {}
    
    # 遍历所有顶点，给每个顶点分配一个颜色
    for node in graph:
        # 找出所有相邻节点的颜色
        neighbor_colors = {colors[neighbor] for neighbor in graph[node] if neighbor in colors}
        
        # 分配一个最小的颜色（从1开始）
        color = 1
        while color in neighbor_colors:
            color += 1
        
        # 给当前节点着色
        colors[node] = color
    num_colors = len(set(colors.values()))  # 计算使用的颜色数量
    return colors,num_colors

# 最大流问题
def max_flow_recruitment(graph:dict, source:str, sink:str):
    """
    使用 Edmonds-Karp 算法解决最大流问题。
    :param graph: 字典形式的图，表示每条边的容量
    :param source: 源点
    :param sink: 汇点
    :return: 最大流量
    """
    # 创建有向图
    G = nx.DiGraph()

    # 添加边到图中
    for (u, v), capacity in graph.items():
        G.add_edge(u, v, capacity=capacity)
    
    # 使用 NetworkX 内建的最大流函数
    flow_value, flow_dict = nx.maximum_flow(G, source, sink)
    
    return flow_value, flow_dict