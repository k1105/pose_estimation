import numpy as np


def pose_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    2 つの正規化済みベクトル (34,) の距離を計算。
    有効キーポイントが 4 個未満なら np.inf を返す。
    """
    valid = ~np.isnan(vec1) & ~np.isnan(vec2)
    valid_pairs = valid.reshape(17, 2).all(axis=1)
    if valid_pairs.sum() < 4:
        return float("inf")

    a = vec1.reshape(17, 2)[valid_pairs]
    b = vec2.reshape(17, 2)[valid_pairs]
    return float(np.linalg.norm(a - b, axis=1).mean())

def procrustes_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """回転・スケール・平行移動を除いた Procrustes 距離"""
    pts1 = vec1.reshape(17, 2)
    pts2 = vec2.reshape(17, 2)
    valid = ~(np.isnan(pts1).any(axis=1) | np.isnan(pts2).any(axis=1))
    if valid.sum() < 4:
        return float("inf")

    a = pts1[valid]         # shape = (K,2)
    b = pts2[valid]

    # 1) 中心化
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)

    # 2) スケールを揃える（フロベニウスノルム）
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    a /= norm_a
    b /= norm_b

    # 3) 最適回転 (Orthogonal Procrustes)
    #    R = U Vᵀ  where  aᵀ b = U Σ Vᵀ
    u, _, vt = np.linalg.svd(a.T @ b)
    r = u @ vt
    a_rot = a @ r           # 回転後

    return float(np.linalg.norm(a_rot - b) / np.sqrt(valid.sum()))
