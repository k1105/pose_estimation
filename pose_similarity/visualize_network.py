import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.manifold import MDS
import umap
import hdbscan
import networkx as nx
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from .preprocess import normalize_pose
from .metrics import pose_distance, procrustes_distance


def load_json(json_path: Path) -> List[dict]:
    """JSONファイルを読み込む"""
    print(f"Loading JSON file: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")
    return data


def calculate_distances_for_record(
    target_id: int,
    target_idx: int,
    data: List[dict],
    id_to_idx: Dict[int, int]
) -> Tuple[int, Dict[int, float]]:
    """
    1つのレコードに対する距離を計算
    """
    distances = {}
    # ターゲットの姿勢を正規化
    target_pose = next(r for r in data if r["id"] == target_id)
    try:
        target_flat, _ = normalize_pose(target_pose)
    except ValueError as e:
        print(f"Warning: Failed to normalize pose for ID {target_id}: {e}")
        return target_idx, distances
    
    for other_record in data:
        other_id = other_record["id"]
        if other_id == target_id:
            continue
            
        try:
            # 比較対象の姿勢を正規化
            other_flat, _ = normalize_pose(other_record)
            
            # 距離を計算
            # distance = pose_distance(target_flat, other_flat)
            distance =  procrustes_distance(target_flat, other_flat)
            if not np.isinf(distance):
                distances[other_id] = distance
            
        except ValueError as e:
            print(f"Warning: Failed to normalize pose for ID {other_id}: {e}")
            continue
    
    return target_idx, distances


def normalize_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """
    距離行列を正規化
    """
    # 無限大とNaNを大きな値に置き換え
    distance_matrix = np.nan_to_num(distance_matrix, nan=1000.0, posinf=1000.0, neginf=1000.0)
    
    # 最大値を100に正規化
    max_distance = np.max(distance_matrix)
    if max_distance > 0:
        distance_matrix = distance_matrix / max_distance * 100
    
    return distance_matrix


def create_distance_matrix(data: List[dict]) -> np.ndarray:
    """
    距離行列を作成（並列処理版）
    data: JSONデータ
    """
    print("Creating distance matrix...")
    n = len(data)
    distance_matrix = np.zeros((n, n))
    
    # 各IDのインデックスを取得
    id_to_idx = {record["id"]: idx for idx, record in enumerate(data)}
    
    # 並列処理で距離を計算
    with ProcessPoolExecutor() as executor:
        # 部分関数を作成
        calc_func = partial(
            calculate_distances_for_record,
            data=data,
            id_to_idx=id_to_idx
        )
        
        # 進捗バーを表示しながら並列処理
        futures = []
        for i, record in enumerate(data):
            futures.append(executor.submit(calc_func, record["id"], i))
        
        # 結果を収集
        for future in tqdm(futures, total=n, desc="Calculating distances"):
            target_idx, distances = future.result()
            for other_id, distance in distances.items():
                other_idx = id_to_idx[other_id]
                distance_matrix[target_idx, other_idx] = distance
                distance_matrix[other_idx, target_idx] = distance  # 対称行列
    
    # 距離行列を正規化
    distance_matrix = normalize_distance_matrix(distance_matrix)
    
    print("Distance matrix creation completed")
    return distance_matrix


def create_network_visualization(
    json_path: Path,
    output_path: Path,
    distance_threshold: float = 50.0,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    embed_method: str = "mds",
    n_neighbors: int = 15,
    min_dist: float = 0.1
) -> None:
    """
    ネットワーク可視化を作成（インタラクティブ版）
    json_path: JSONファイルのパス
    output_path: 出力HTMLファイルのパス
    distance_threshold: エッジを描画する距離の閾値
    min_cluster_size: クラスターの最小サイズ
    min_samples: コアポイントを定義するのに必要な最小近傍点数
    embed_method: 埋め込み手法 ("mds" or "umap")
    n_neighbors: UMAPのneighbors数
    min_dist: UMAPの最小距離
    """
    # JSONファイルを読み込む
    data = load_json(json_path)
    
    # 距離行列を作成
    distance_matrix = create_distance_matrix(data)
    
    print(f"Calculating 2D coordinates using {embed_method.upper()}...")
    # 2次元座標を計算
    if embed_method.lower() == "mds":
        embedder = MDS(
            n_components=2,
            dissimilarity='precomputed',
            random_state=42,
            max_iter=300,
            eps=1e-6,
            n_init=4
        )
        positions = embedder.fit_transform(distance_matrix)
    else:  # umap
        embedder = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='precomputed',
            random_state=42
        )
        positions = embedder.fit_transform(distance_matrix)
    
    print(f"{embed_method.upper()} calculation completed")
    
    print("Performing clustering...")
    # HDBSCANでクラスタリング
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',
        cluster_selection_epsilon=distance_threshold
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)
    unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # ノイズを除外
    noise_points = np.sum(cluster_labels == -1)
    print(f"Clustering completed: {unique_clusters} clusters found, {noise_points} noise points")
    
    print("Creating network graph...")
    # ネットワークグラフを作成
    G = nx.Graph()
    
    # ノードを追加
    for i, record in enumerate(data):
        G.add_node(
            record["id"],
            pos=positions[i],
            cluster=cluster_labels[i]
        )
    
    # エッジを追加（距離が閾値未満のペアのみ）
    edge_distances = []
    edge_pairs = []
    for i in tqdm(range(len(data)), desc="Adding edges"):
        for j in range(i + 1, len(data)):
            if distance_matrix[i, j] < distance_threshold:
                G.add_edge(
                    data[i]["id"],
                    data[j]["id"],
                    weight=distance_matrix[i, j]
                )
                edge_distances.append(distance_matrix[i, j])
                edge_pairs.append((data[i]["id"], data[j]["id"]))
    
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Plotlyでの可視化用のデータを準備
    edge_x = []
    edge_y = []
    for edge in edge_pairs:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # エッジのトレースを作成
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        opacity=0.2
    )
    
    # ノードの位置を取得
    node_x = []
    node_y = []
    node_ids = []
    node_clusters = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_ids.append(node)
        node_clusters.append(G.nodes[node]['cluster'])
    
    # ノードのトレースを作成（ノイズポイントは別のマーカーで表示）
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_ids,
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_clusters,
            size=[15 if c != -1 else 10 for c in node_clusters],  # ノイズポイントは小さく
            symbol=[0 if c != -1 else 4 for c in node_clusters],  # ノイズポイントは異なる形状
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Cluster',
                    side='right'
                ),
                xanchor='left'
            )
        )
    )
    
    # レイアウトを作成
    layout = go.Layout(
        title=f'Pose Network Visualization<br>{unique_clusters} clusters, {noise_points} noise points, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    # 図を作成
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    
    # HTMLファイルとして保存
    output_path = output_path.with_suffix('.html')
    print(f"Saving visualization to {output_path}...")
    fig.write_html(str(output_path))
    print(f"Interactive visualization saved to: {output_path}")


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(
            description="Create an interactive network visualization of pose similarities"
        )
        parser.add_argument("json_path", type=Path, help="Path to the JSON file")
        parser.add_argument("--output", type=Path, default=Path("network.html"),
                          help="Output HTML file path (default: network.html)")
        parser.add_argument("--threshold", type=float, default=50.0,
                          help="Distance threshold for edges (default: 50.0)")
        parser.add_argument("--min-cluster-size", type=int, default=5,
                          help="Minimum cluster size for HDBSCAN (default: 5)")
        parser.add_argument("--min-samples", type=int, default=3,
                          help="Minimum samples for HDBSCAN core points (default: 3)")
        parser.add_argument("--embed", type=str, default="mds",
                          help="Embedding method (default: mds)")
        parser.add_argument("--n-neighbors", type=int, default=15,
                          help="Number of neighbors for UMAP (default: 15)")
        parser.add_argument("--min-dist", type=float, default=0.1,
                          help="Minimum distance for UMAP (default: 0.1)")
        args = parser.parse_args()
    
    create_network_visualization(
        args.json_path,
        args.output,
        args.threshold,
        args.min_cluster_size,
        args.min_samples,
        args.embed,
        args.n_neighbors,
        args.min_dist
    )


if __name__ == "__main__":
    main() 