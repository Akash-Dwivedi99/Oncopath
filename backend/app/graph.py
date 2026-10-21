from collections import deque
from itertools import combinations

import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix

GRAPH_FIELDS = [
    "cancer_type", "tumor_stage", "molecular_subtype", "smoking_status", "m_stage",
    "er_status", "pr_status", "her2_status", "braf_v600e", "alk_rearranged",
    "lymphovascular_invasion", "histologic_grade",
]


def build_graph(frame: pd.DataFrame, node_limit: int = 70) -> nx.Graph:
    fields = [name for name in GRAPH_FIELDS if name in frame.columns]
    node_counts: dict[tuple[str, str], int] = {}
    edge_counts: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}
    for row in frame[fields].fillna("Unknown").astype(str).to_dict("records"):
        values = [(field, value.strip()) for field, value in row.items()
                  if value.strip() and value.strip().casefold() not in {"unknown", "nan", "none"}]
        for item in values:
            node_counts[item] = node_counts.get(item, 0) + 1
        for left, right in combinations(values, 2):
            if left[0] == right[0]:
                continue
            key = tuple(sorted((left, right)))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    selected = set(sorted(node_counts, key=node_counts.get, reverse=True)[:node_limit])
    graph = nx.Graph()
    for (field, value), count in node_counts.items():
        if (field, value) in selected:
            node_id = f"{field}={value}"
            graph.add_node(node_id, field=field, value=value, count=count)
    minimum_weight = max(3, int(len(frame) * 0.008))
    for (left, right), weight in edge_counts.items():
        if left in selected and right in selected and weight >= minimum_weight:
            source, target = f"{left[0]}={left[1]}", f"{right[0]}={right[1]}"
            graph.add_edge(source, target, weight=weight, distance=1 / weight)
    return graph


def graph_summary(graph: nx.Graph) -> dict:
    if not graph:
        return {"nodes": 0, "edges": 0, "components": 0, "top_centrality": []}
    if graph.number_of_edges() == 0:
        return {"nodes": graph.number_of_nodes(), "edges": 0,
                "components": graph.number_of_nodes(), "top_centrality": []}
    pagerank = nx.pagerank(graph, weight="weight")
    centrality = nx.degree_centrality(graph)
    top = sorted(graph.nodes, key=lambda node: pagerank[node], reverse=True)[:12]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": nx.number_connected_components(graph),
        "top_centrality": [{"node": node, "degree": graph.degree(node),
                            "degree_centrality": centrality[node], "pagerank": pagerank[node],
                            "count": graph.nodes[node]["count"]} for node in top],
    }


def graph_payload(graph: nx.Graph) -> dict:
    nodes = [{"id": node, **attrs} for node, attrs in graph.nodes(data=True)]
    edges = [{"source": left, "target": right, **attrs}
             for left, right, attrs in graph.edges(data=True)]
    return {"nodes": nodes, "edges": edges, "summary": graph_summary(graph)}


def sparse_adjacency(graph: nx.Graph) -> dict:
    """Return CSR storage stats without serializing the matrix itself."""
    nodes = list(graph.nodes)
    matrix = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", format="csr", dtype="float32")
    matrix = csr_matrix(matrix)
    return {"nodes": nodes, "shape": list(matrix.shape), "nonzero_entries": int(matrix.nnz),
            "format": "CSR", "sparse_bytes": int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes),
            "dense_float32_bytes": int(matrix.shape[0] * matrix.shape[1] * 4)}


def traverse(graph: nx.Graph, start: str, method: str = "bfs") -> list[str]:
    if start not in graph:
        raise KeyError(f"Unknown graph node: {start}")
    method = method.lower()
    if method not in {"bfs", "dfs"}:
        raise ValueError("method must be bfs or dfs")
    pending = deque([start])
    seen, order = {start}, []
    while pending:
        current = pending.popleft() if method == "bfs" else pending.pop()
        order.append(current)
        neighbors = sorted(graph.neighbors(current))
        for neighbor in neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return order


def shortest_path(graph: nx.Graph, source: str, target: str) -> dict:
    if source not in graph or target not in graph:
        raise KeyError("Choose source and target nodes that exist in the current graph.")
    path = nx.shortest_path(graph, source, target, weight="distance")
    return {"path": path, "hops": len(path) - 1,
            "cost": sum(graph[a][b]["distance"] for a, b in zip(path, path[1:]))}


def communities(graph: nx.Graph) -> list[dict]:
    if not graph:
        return []
    if graph.number_of_edges() == 0:
        return [{"community": index + 1, "size": 1, "nodes": [node]}
                for index, node in enumerate(sorted(graph.nodes))]
    groups = nx.community.greedy_modularity_communities(graph, weight="weight")
    return [{"community": index + 1, "size": len(group), "nodes": sorted(group)}
            for index, group in enumerate(sorted(groups, key=len, reverse=True))]
