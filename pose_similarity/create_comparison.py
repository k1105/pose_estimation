import cv2
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import numpy as np
from .finder import find_top_n_similar


def load_json(json_path: Path) -> List[dict]:
    """JSONファイルを読み込む"""
    with open(json_path, 'r') as f:
        return json.load(f)


def find_image_path(base_dir: Path, image_name: str) -> Path:
    """画像ファイルのパスを探す"""
    # 拡張子を除いたファイル名を取得
    stem = Path(image_name).stem
    # ハイフンで区切られた部分をディレクトリパスとして扱う
    parts = stem.split('-')
    if len(parts) > 1:
        # 最後の部分をファイル名として、それ以外をディレクトリパスとして扱う
        dir_path = Path('/'.join(parts[:-1]))
        file_name = parts[-1]
    else:
        # ハイフンがない場合は通常のファイル名として扱う
        dir_path = Path("")
        file_name = stem

    # 画像ファイルを探す
    input_dir = base_dir.parent.parent / "input_images"  # out ディレクトリと同じ階層の input_images を参照
    target_path = input_dir / dir_path / f"{file_name}.jpg"
    
    if target_path.exists():
        return target_path
    raise FileNotFoundError(f"Image not found: {image_name} (searched at {target_path})")


def create_comparison_image(json_path: Path, target_id: int, output_path: Path) -> None:
    """
    検索結果の画像を1枚にまとめる
    json_path: JSONファイルのパス
    target_id: 検索対象のID
    output_path: 出力画像のパス
    """
    # JSONファイルを読み込む
    data = load_json(json_path)
    
    # 検索対象の画像情報を取得
    target_record = next((r for r in data if r["id"] == target_id), None)
    if target_record is None:
        raise ValueError(f"ID {target_id} not found in JSON")
    
    # 画像ファイルのパスを探す
    base_dir = json_path.parent
    target_image_path = find_image_path(base_dir, target_record["image_name"])
    
    # 出力ディレクトリを作成
    output_dir = output_path.parent / f"compare_{target_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 元画像をコピー
    shutil.copy2(target_image_path, output_dir / f"original_{target_id}.jpg")
    
    # 画像を読み込む
    target_image = cv2.imread(str(target_image_path))
    if target_image is None:
        raise FileNotFoundError(f"Failed to load image: {target_image_path}")
    
    # 類似度でソートされた画像を取得
    similar_records = find_top_n_similar(target_id, data, n=5)
    
    # 検索結果を表示
    print(f"id={target_id} 画像={target_record['image_name']} に類似する上位{len(similar_records)}件:")
    for rank, (best_id, dist) in enumerate(similar_records, 1):
        record = next((r for r in data if r["id"] == best_id), None)
        if record:
            print(f"{rank}位: id={best_id:02d} 画像={record['image_name']} 距離={dist:.2f}")
            # 類似画像をコピー
            similar_image_path = find_image_path(base_dir, record["image_name"])
            shutil.copy2(similar_image_path, output_dir / f"similar_{rank}_{best_id}.jpg")
    
    # バウンディングボックスから人物を切り出し
    def crop_person(image, bbox):
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
    
    # ターゲット画像から人物を切り出し
    target_bbox = target_record["bbox"]
    target_person = crop_person(target_image, target_bbox)
    if target_person.size == 0:
        raise ValueError("Failed to crop target person")
    
    # 切り出した人物画像のサイズを取得
    height, width = target_person.shape[:2]
    
    # 余白の設定
    padding = 20  # 画像間の余白
    target_padding = 40  # ターゲット画像の上下の余白
    text_height = 90  # テキスト領域の高さ（30から90に増加）
    
    # 結果画像のサイズを計算（3列×3行 + 余白）
    result_width = width * 3 + padding * 4  # 3列 + 余白
    result_height = height * 3 + target_padding * 2 + padding * 2 + text_height * 3  # 3行 + 余白 + テキスト領域
    result_image = np.zeros((result_height, result_width, 3), dtype=np.uint8)
    
    # テキストを追加する関数
    def add_text_with_background(img, text, position, font_scale, thickness):
        # テキストのサイズを取得
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        # テキストの背景領域を計算
        text_width = text_size[0] + 40  # 余白を増加
        text_height = text_size[1] + 20  # 余白を増加
        x1 = position[0] - 20  # 余白を増加
        y1 = position[1] - text_size[1] - 10  # 余白を増加
        x2 = x1 + text_width
        y2 = y1 + text_height
        
        # 背景を描画（半透明の黒）
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        # テキストを描画
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # テキストの設定
    font_scale = 2.7 * (width / 1920)  # フォントスケールを3倍に（0.9から2.7に）
    thickness = max(3, int(3 * width / 1920))  # 線の太さも増加
    
    # ターゲット画像を中央に配置
    x_center = (result_width - width) // 2
    y_center = target_padding + text_height
    result_image[y_center:y_center+height, x_center:x_center+width] = target_person
    
    # ターゲット画像のラベル
    label = f"Target (ID: {target_id})"
    text_x = x_center + width // 2
    text_y = target_padding + text_height - 5
    add_text_with_background(result_image, label, (text_x, text_y), font_scale, thickness)
    
    # 類似画像を3枚ずつ2行に配置
    for i, (best_id, dist) in enumerate(similar_records, 1):
        if best_id == target_id:
            continue
            
        # 対応するレコードを探す
        record = next((r for r in data if r["id"] == best_id), None)
        if record is None:
            continue
            
        image_path = find_image_path(base_dir, record["image_name"])
        similar_image = cv2.imread(str(image_path))
        if similar_image is None:
            print(f"Failed to load image: {image_path}")
            continue
            
        # バウンディングボックスから人物を切り出し
        similar_bbox = record["bbox"]
        similar_person = crop_person(similar_image, similar_bbox)
        if similar_person.size == 0:
            print(f"Failed to crop person from image: {image_path}")
            continue
            
        # アスペクト比を保持してリサイズ
        h, w = similar_person.shape[:2]
        scale = min(width/w, height/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_person = cv2.resize(similar_person, (new_w, new_h))
        
        # 画像の配置位置を計算（3列×2行）
        row = (i - 1) // 3  # 0 or 1
        col = (i - 1) % 3   # 0, 1, or 2
        y_offset = height * (row + 1) + target_padding * 2 + padding * (row + 1) + text_height * (row + 1)
        x_offset = width * col + padding * (col + 1)
        
        # 画像を中央に配置
        y_center = y_offset + (height - new_h) // 2
        x_center = x_offset + (width - new_w) // 2
        result_image[y_center:y_center+new_h, x_center:x_center+new_w] = resized_person
        
        # 類似画像のラベル
        label = f"#{i} (ID: {best_id}, dist: {dist:.2f})"
        text_x = x_offset + width // 2
        text_y = y_offset - 5
        add_text_with_background(result_image, label, (text_x, text_y), font_scale, thickness)
    
    # 最終的な出力画像を1280x1280以内にリサイズ
    max_size = 2048
    h, w = result_image.shape[:2]
    scale = min(max_size/w, max_size/h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    result_image = cv2.resize(result_image, (new_w, new_h))
    
    # 画像を保存
    output_path = output_dir / "comparison.jpg"
    cv2.imwrite(str(output_path), result_image)
    print(f"Comparison image saved to: {output_path}")
    print(f"All images are saved in: {output_dir}")


def get_random_ids(data: List[dict], count: int = 10) -> List[int]:
    """ランダムにIDを選択する"""
    import random
    all_ids = [record["id"] for record in data]
    return random.sample(all_ids, min(count, len(all_ids)))


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(
            description="Create a comparison image from pose similarity search results"
        )
        parser.add_argument("json_path", type=Path, help="Path to the JSON file")
        parser.add_argument("--id", type=int, help="Target ID to search for")
        parser.add_argument("--random", action="store_true", help="Select random IDs")
        parser.add_argument("--count", type=int, default=10, help="Number of random IDs to select (default: 10)")
        parser.add_argument("--output", type=Path, default=Path("comparison.jpg"), 
                            help="Output image path (default: comparison.jpg)")
        args = parser.parse_args()
    
    # 引数のバリデーション
    if not args.random and args.id is None:
        parser.error("Either --id or --random must be specified")
    
    # JSONファイルを読み込む
    data = load_json(args.json_path)
    
    if args.random:
        # ランダムモードの場合
        target_ids = get_random_ids(data, args.count)
        for target_id in target_ids:
            output_path = Path(f"comparison{target_id}.jpg")
            print(f"\nProcessing ID: {target_id}")
            create_comparison_image(args.json_path, target_id, output_path)
    else:
        # 通常モードの場合
        create_comparison_image(args.json_path, args.id, args.output)


if __name__ == "__main__":
    main() 