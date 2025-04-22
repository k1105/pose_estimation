import json
import argparse
from pathlib import Path
from typing import List, Set
import shutil
from .finder import find_top_n_similar


def load_json(json_path: Path) -> List[dict]:
    """JSONファイルを読み込む"""
    with open(json_path, 'r') as f:
        return json.load(f)


def find_image_path(base_dir: Path, image_name: str) -> Path:
    """画像ファイルのパスを探す"""
    # 拡張子を除いたファイル名を取得
    stem = Path(image_name).stem
    # 画像ファイルを探す
    for img_path in base_dir.rglob(f"{stem}.jpg"):
        return img_path
    raise FileNotFoundError(f"Image not found: {image_name}")


def create_chain_images(json_path: Path, start_id: int, count: int, output_dir: Path) -> None:
    """
    連鎖的に類似する姿勢の画像をコピーする
    json_path: JSONファイルのパス
    start_id: 開始ID
    count: 生成する画像の枚数
    output_dir: 出力ディレクトリのパス
    """
    # JSONファイルを読み込む
    data = load_json(json_path)
    
    # 使用済みIDを記録
    used_ids: Set[int] = {start_id}
    
    # 画像ファイルのパスを探す
    base_dir = json_path.parent / "image"
    
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 現在のID
    current_id = start_id
    
    for i in range(count):
        # 現在のIDの画像情報を取得
        current_record = next((r for r in data if r["id"] == current_id), None)
        if current_record is None:
            raise ValueError(f"ID {current_id} not found in JSON")
        
        # 画像のパスを取得
        image_path = find_image_path(base_dir, current_record["image_name"])
        
        # 出力ファイル名を生成（連番とIDを含める）
        output_filename = f"{i+1:03d}_id{current_id}.jpg"
        output_path = output_dir / output_filename
        
        # 画像をコピー
        shutil.copy2(image_path, output_path)
        print(f"Copied image {i+1}/{count}: {output_filename}")
        
        # 次のIDを探す
        if i < count - 1:  # 最後の画像でなければ次のIDを探す
            similar_records = find_top_n_similar(current_id, data, n=10)
            for next_id, _ in similar_records:
                if next_id not in used_ids:
                    current_id = next_id
                    used_ids.add(current_id)
                    break
            else:
                raise ValueError(f"Could not find enough unique similar poses starting from ID {start_id}")
    
    print(f"Chain images saved to: {output_dir}")


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(
            description="Create a chain of similar pose images"
        )
        parser.add_argument("json_path", type=Path, help="Path to the JSON file")
        parser.add_argument("--id", type=int, required=True, help="Starting ID")
        parser.add_argument("--count", type=int, required=True, help="Number of images to generate")
        parser.add_argument("--output", type=Path, default=Path("chain_images"), 
                            help="Output directory path (default: chain_images)")
        args = parser.parse_args()
    
    create_chain_images(args.json_path, args.id, args.count, args.output)


if __name__ == "__main__":
    main() 