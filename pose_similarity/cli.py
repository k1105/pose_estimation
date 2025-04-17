import argparse
from pathlib import Path
from .loader import load
from .finder import find_top_n_similar


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pose_similarity",
        description="指定 id の姿勢に最も類似するレコードを検索します。",
    )
    parser.add_argument("json_path", type=Path, help="対象 JSON ファイルへの相対/絶対パス")
    parser.add_argument("--id", type=int, required=True, help="検索対象の id")
    parser.add_argument("--top", type=int, default=5, help="表示する上位件数 (デフォルト: 5)")
    args = parser.parse_args()

    data = load(args.json_path)
    results = find_top_n_similar(args.id, data, n=args.top)
    
    if not results:
        print("有効な比較対象が見つかりませんでした。")
    else:
        # 対象のidの画像ファイル名を取得
        target_image = next((p["image_name"] for p in data if p["id"] == args.id), "unknown")
        print(f"id={args.id} 画像={target_image} に類似する上位{len(results)}件:")
        
        for rank, (best_id, dist) in enumerate(results, 1):
            # 各結果の画像ファイル名を取得
            image = next((p["image_name"] for p in data if p["id"] == best_id), "unknown")
            print(f"{rank}位: id={best_id:02d} 画像={image} 距離={dist:.2f}")


if __name__ == "__main__":
    main()

