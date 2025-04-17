from pathlib import Path
import json
from typing import List, Dict, Any, Union

Pose = Dict[str, Any]
Dataset = List[Pose]


def load(path: Union[str, Path]) -> Dataset:
    """JSON ファイルを読み込んでリストを返す"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"JSON not found: {p}")
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    # バリデーション（最低限）
    if not isinstance(data, list):
        raise ValueError("Top‑level JSON must be a list")
    return data

