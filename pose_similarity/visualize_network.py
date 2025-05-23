import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.manifold import MDS
import umap
import hdbscan
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import plotly.graph_objects as go
from tqdm import tqdm
from .preprocess import normalize_pose
from .metrics import procrustes_distance


def load_json(json_path: Path) -> List[dict]:
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_embeddings(data: List[dict]) -> Tuple[np.ndarray, List[int]]:
    embeddings = []
    ids = []
    for item in data:
        try:
            vec, _ = normalize_pose(item)
            embeddings.append(vec)
            ids.append(item["id"])
        except ValueError:
            continue
    return np.array(embeddings), ids


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    n = embeddings.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in tqdm(range(n), desc="Distance matrix"):
        for j in range(i + 1, n):
            d = procrustes_distance(embeddings[i], embeddings[j])
            D[i, j] = D[j, i] = d
    return np.nan_to_num(D, nan=1000.0, posinf=1000.0, neginf=1000.0)


def cluster_hdbscan(D: np.ndarray, min_cluster_size=5, min_samples=3) -> np.ndarray:
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed'
    )
    return model.fit_predict(D)


def cluster_agglomerative(D: np.ndarray, n_clusters: int) -> np.ndarray:
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        linkage='average'
    )
    return model.fit_predict(D)


def cluster_data(D: np.ndarray, method: str, min_cluster_size: int = 5, min_samples: int = 3, n_clusters: int = None) -> np.ndarray:
    if method == "hdbscan":
        return cluster_hdbscan(D, min_cluster_size, min_samples)
    elif method == "agglomerative":
        if n_clusters is None:
            n_clusters = int(np.sqrt(D.shape[0]))
        return cluster_agglomerative(D, n_clusters)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def embed_positions(D: np.ndarray, method: str, n_neighbors: int, min_dist: float) -> np.ndarray:
    if method == "umap":
        reducer = umap.UMAP(
            metric='precomputed',
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42
        )
        return reducer.fit_transform(D)
    else:
        model = MDS(
            n_components=2,
            dissimilarity='precomputed',
            random_state=42
        )
        return model.fit_transform(D)


def build_graph(ids: List[int], D: np.ndarray, positions: np.ndarray,
                labels: np.ndarray, edge_mode: str, threshold: float, k: int) -> nx.Graph:
    G = nx.Graph()
    for i, id_ in enumerate(ids):
        G.add_node(id_, pos=positions[i], cluster=int(labels[i]))

    n = len(ids)
    if edge_mode == "knn":
        for i in range(n):
            neighbors = np.argsort(D[i])[1:k+1]
            for j in neighbors:
                w = 1.0 / (D[i, j] + 1e-3)
                G.add_edge(ids[i], ids[j], weight=w)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] < threshold:
                    w = 1.0 / (D[i, j] + 1e-3)
                    G.add_edge(ids[i], ids[j], weight=w)
    return G


def create_network_visualization(G: nx.Graph, output_path: Path) -> None:
    pos = nx.spring_layout(G, weight='weight', seed=42, iterations=500)
    for node in G.nodes:
        G.nodes[node]['pos'] = pos[node]

    node_x, node_y, node_color, node_id, node_size, node_symbol = [], [], [], [], [], []
    for node, data in G.nodes(data=True):
        x, y = data['pos']
        cluster = data['cluster']
        node_x.append(x)
        node_y.append(y)
        node_color.append(cluster)
        node_id.append(node)
        node_size.append(15 if cluster != -1 else 8)
        node_symbol.append("circle" if cluster != -1 else "x")

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = G.nodes[u]['pos']
        x1, y1 = G.nodes[v]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        opacity=0.15
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_id,
        hoverinfo='text',
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=node_size,
            symbol=node_symbol,
            colorbar=dict(
                thickness=15,
                title="Cluster",
                xanchor='left'
            )
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Pose Similarity Network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=40, l=40, r=40, t=40),
                        plot_bgcolor='white',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    fig.write_html(str(output_path.with_suffix(".html")))
    print(f"Saved: {output_path.with_suffix('.html')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("network.html"))
    parser.add_argument("--embed", choices=["umap", "mds"], default="umap")
    parser.add_argument("--edge-mode", choices=["knn", "threshold"], default="knn")
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--cluster-method", choices=["hdbscan", "agglomerative"], default="hdbscan")
    parser.add_argument("--n-clusters", type=int)
    args = parser.parse_args()

    data = load_json(args.json_path)
    embeddings, ids = compute_embeddings(data)
    D = compute_distance_matrix(embeddings)
    labels = cluster_data(D, args.cluster_method, args.min_cluster_size, args.min_samples, args.n_clusters)
    positions = embed_positions(D, args.embed, args.n_neighbors, args.min_dist)
    G = build_graph(ids, D, positions, labels, args.edge_mode, args.threshold, args.k)
    create_network_visualization(G, args.output)


if __name__ == "__main__":
    main()
