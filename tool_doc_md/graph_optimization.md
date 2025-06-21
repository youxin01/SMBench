#### dijkstra
**Name:** dijkstra  
**Description:** Dijkstra 算法，计算从起点到终点或所有节点的最短路径。
**Applicable Situations:**  
- 寻找图中两个节点之间的最短路径  
- 单源最短路径问题  
- 网络路由和路径优化等场景  

**Parameters:**  
- `edges`:  
  - **Type:** `dict`  
  - **Description:** 图的边字典，键为 `(start, end)` 的元组，值为对应边的权重  
- `start`:  
  - **Type:** `str`  
  - **Description:** 起始节点  
- `end`:  
  - **Type:** `str | None`  
  - **Description:** 终点节点（可选）；若提供，则返回从 `start` 到 `end` 的最短路径距离，否则返回从 `start` 到所有节点的最短路径字典  
- `directed`:
  - **Type:** `bool | None`  
  - **Description:** 是否是有向图（默认 False，即无向图）。

**Result:**  
- 若指定 `end` 节点，返回从 `start` 到 `end` 的最短路径距离；若路径不存在，则返回错误描述字符串；  
- 若不指定 `end`，返回一个字典，包含从 `start` 到各节点的最短路径距离  

**Example Call:**  
```python
result = dijkstra(
    edges={("A", "B"): 1, ("B", "C"): 2, ("A", "C"): 4},
    start="A",
    end="C"
)
```
-----

#### mst
**Name:** mst  
**Description:** 计算图的最小生成树 (MST)
**Applicable Situations:**  
- 网络设计与优化  
- 电网、通信网络等连接问题  
- 边权重最小化需求场景  

**Parameters:**  
- `edges`:  
  - **Type:** `dict`  
  - **Description:** 图的边字典，键为 `(u, v)` 的元组，值为对应的边权重  
- `algorithm`:  
  - **Type:** `str`  
  - **Description:** 计算 MST 的算法选择，取值 `'prim'` 或 `'kruskal'`（默认 `'prim'`）  

**Result:**  
- 返回一个元组 `(mst_edges, total_weight)`  
  - `mst_edges`: MST 的边列表，每个边的格式为 `(u, v, {'weight': w})`  
  - `total_weight`: MST 的总权重  

**Example Call:**  
```python
mst_edges, total_weight = mst(
    edges={("A", "B"): 1, ("B", "C"): 2, ("A", "C"): 4},
    algorithm='prim'
)
```
-----

#### pagerank
**Name:** pagerank  
**Description:** 计算图的 PageRank 值，用于衡量节点的重要性。
**Applicable Situations:**  
- 网页链接分析  
- 社交网络影响力评估  
- 任意有向图的节点排序与重要性判定  

**Parameters:**  
- `graph`:  
  - **Type:** `dict`  
  - **Description:** 有向图的边字典，键为 `(u, v)` 的元组，值为边的权重  
- `alpha`:  
  - **Type:** `float`  
  - **Description:** 阻尼系数（默认 0.85），控制随机跳转的概率  

**Result:**  
- 返回一个字典，其中键为节点，值为对应的 PageRank 值  

**Example Call:**  
```python
pagerank_values = pagerank(
    graph={("A", "B"): 1, ("B", "C"): 1, ("C", "A"): 1},
    alpha=0.85
)
```
-----

#### greedy_coloring
**Name:** greedy_coloring  
**Description:** 使用贪心算法解决图的顶点着色问题，尽量减少所需颜色数 。
**Applicable Situations:**  
- 图着色问题  
- 资源分配与调度  
- 避免相邻节点冲突的场景  

**Parameters:**  
- `graph`:  
  - **Type:** `dict`  
  - **Description:** 图的邻接字典，键为节点，值为该节点的邻接节点列表  

**Result:**  
- 返回一个元组 `(colors, num_colors)`  
  - `colors`: 节点着色字典，键为节点，值为分配的颜色  
  - `num_colors`: 使用的颜色总数  

**Example Call:**  
```python
colors, num_colors = greedy_coloring(
    graph={
        "A": ["B", "C"],
        "B": ["A", "C"],
        "C": ["A", "B"]
    }
)
```
-----

#### max_flow_recruitment
**Name:** max_flow_recruitment  
**Description:** 使用 Edmonds-Karp 算法解决最大流问题，确定流网络中可能传输的最大流量。
**Applicable Situations:**  
- 流网络问题  
- 资源分配与运输优化  

**Parameters:**  
- `graph`:  
  - **Type:** `dict`  
  - **Description:** 表示图的边字典，键为 `(u, v)` 的元组，值为边的容量  
- `source`:  
  - **Type:** `str`  
  - **Description:** 流网络的源点  
- `sink`:  
  - **Type:** `str`  
  - **Description:** 流网络的汇点  

**Result:**  
- 返回一个元组 `(flow_value, flow_dict)`  
  - `flow_value`: 网络的最大流量  
  - `flow_dict`: 每条边上流量的分配情况  

**Example Call:**  
```python
flow_value, flow_dict = max_flow_recruitment(
    graph={("S", "A"): 10, ("A", "B"): 5, ("B", "T"): 10, ("S", "C"): 15, ("C", "T"): 10},
    source="S",
    sink="T"
)
```