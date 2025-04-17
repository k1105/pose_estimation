import numpy as np
from typing import Dict, Tuple, Any

Pose = Dict[str, Any]


def normalize_pose(pose: Pose) -> Tuple[np.ndarray, np.ndarray]:
    """
    bbox を基準に 1000×1000 スケールへ正規化。
    戻り値:
        flat  : shape=(34,), 欠損は np.nan
        mask17: shape=(17,), True=有効キーポイント
    """
    x1, y1, x2, y2 = pose["bbox"]
    w, h = x2 - x1, y2 - y1
    pts = np.asarray(pose["keypoints"], dtype=np.float32)        # (17,2)
    mask = ~np.isclose(pts, 0).all(axis=1)                      # (17,)
    if w == 0 or h == 0:
        raise ValueError(f"Invalid bbox: {pose['bbox']}")
    pts[mask] = (pts[mask] - [x1, y1]) / [w, h] * 1000          # 0‑1000
    flat = pts.reshape(-1)                                      # (34,)
    flat[~np.repeat(mask, 2)] = np.nan
    return flat, mask

