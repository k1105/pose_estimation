from typing import List, Dict, Any, Tuple, Union
import numpy as np

from .preprocess import normalize_pose
from .metrics import pose_distance, procrustes_distance

Pose = Dict[str, Any]
Dataset = List[Pose]


def build_embeddings(data: Dataset) -> Dict[int, np.ndarray]:
    """id → 正規化ベクトル (34,) の辞書を作成"""
    return {p["id"]: normalize_pose(p)[0] for p in data}


def find_top_n_similar(target_id: int, data: Dataset, n: int = 5) -> List[Tuple[int, float]]:
    """
    target_id と最も近い (＝距離が最小) 上位n件のレコードを返す
    戻り値: [(id, distance), ...] のリスト
    """
    emb = build_embeddings(data)
    if target_id not in emb:
        raise KeyError(f"id {target_id} not found")

    tgt = emb[target_id]
    distances = []

    for pid, vec in emb.items():
        if pid == target_id:
            continue
        d = pose_distance(tgt, vec)
        # d = procrustes_distance(tgt, vec)
        distances.append((pid, d))
    
    # 距離でソートして上位n件を返す
    distances.sort(key=lambda x: x[1])
    return distances[:n]


def find_most_similar(target_id: int, data: Dataset) -> Tuple[Union[int, None], float]:
    """
    target_id と最も近い (＝距離が最小) 他レコードを返す
    戻り値: (best_id, distance)  見つからなければ (None, inf)
    """
    results = find_top_n_similar(target_id, data, n=1)
    if not results:
        return None, float("inf")
    return results[0]

