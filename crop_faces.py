import argparse
import json
from pathlib import Path
import cv2
import os
from tqdm import tqdm
import concurrent.futures

def crop_face(image_path, bbox, margin_percent=20):
    """
    画像から顔をクロップする
    
    Args:
        image_path (Path): 入力画像のパス
        bbox (dict): バウンディングボックスの情報 {'x': int, 'y': int, 'width': int, 'height': int}
        margin_percent (float): バウンディングボックスに対するマージンの割合（%）
    
    Returns:
        numpy.ndarray: クロップされた画像
    """
    # 画像を読み込む
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # 画像のサイズを取得
    height, width = image.shape[:2]
    
    # マージンを計算
    margin_x = int(bbox['width'] * margin_percent / 100)
    margin_y = int(bbox['height'] * margin_percent / 100)
    
    # クロップ領域を計算
    x1 = max(0, bbox['x'] - margin_x)
    y1 = max(0, bbox['y'] - margin_y)
    x2 = min(width, bbox['x'] + bbox['width'] + margin_x)
    y2 = min(height, bbox['y'] + bbox['height'] + margin_y)
    
    # 画像をクロップ
    cropped = image[y1:y2, x1:x2]
    return cropped

def process_face(face_info, input_dir, output_dir, margin_percent=20):
    """
    1つの顔の情報を処理し、クロップ画像を保存する
    
    Args:
        face_info (dict): 顔の検出情報
        input_dir (Path): 入力ディレクトリのパス
        output_dir (Path): 出力ディレクトリのパス
        margin_percent (float): バウンディングボックスに対するマージンの割合（%）
    """
    try:
        # 入力画像のパスを構築
        image_path = input_dir / face_info['file_path']
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return
        
        # 顔をクロップ
        cropped = crop_face(image_path, face_info['bbox'], margin_percent)
        if cropped is None:
            return
        
        # 出力ディレクトリを作成
        output_path = output_dir / face_info['file_path']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ファイル名を生成（元のファイル名に顔のIDを追加）
        stem = output_path.stem
        suffix = output_path.suffix
        new_filename = f"{stem}_face{face_info['id']}{suffix}"
        output_path = output_path.parent / new_filename
        
        # 画像を保存
        cv2.imwrite(str(output_path), cropped)
        
    except Exception as e:
        print(f"Error processing face {face_info['id']} in {face_info['file_path']}: {str(e)}")

def process_detections(json_file, input_dir, output_dir, margin_percent=20, max_workers=4):
    """
    検出結果のJSONファイルを読み込み、顔のクロップ画像を生成する
    
    Args:
        json_file (str): 検出結果のJSONファイルのパス
        input_dir (str): 入力ディレクトリのパス
        output_dir (str): 出力ディレクトリのパス
        margin_percent (float): バウンディングボックスに対するマージンの割合（%）
        max_workers (int): 並列処理のワーカー数
    """
    # JSONファイルを読み込む
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 入力ディレクトリと出力ディレクトリをPathオブジェクトに変換
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 並列処理で顔を処理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_face, face_info, input_dir, output_dir, margin_percent)
            for face_info in data['faces']
        ]
        
        # プログレスバーを表示
        for _ in tqdm(concurrent.futures.as_completed(futures), 
                     total=len(futures), 
                     desc="Cropping faces"):
            pass

def main():
    parser = argparse.ArgumentParser(description='Crop faces from images based on detection results')
    parser.add_argument('json_file', type=str, help='JSON file containing face detection results')
    parser.add_argument('input_dir', type=str, help='Input directory containing original images')
    parser.add_argument('--output_dir', type=str, default='cropped_faces',
                      help='Output directory for cropped faces (default: cropped_faces)')
    parser.add_argument('--margin', type=float, default=20,
                      help='Crop margin percentage around face (default: 20)')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of worker threads (default: 4)')
    
    args = parser.parse_args()
    
    # 入力ファイルとディレクトリの存在確認
    if not os.path.isfile(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # 顔のクロップ処理を実行
    process_detections(args.json_file, args.input_dir, args.output_dir,
                      args.margin, args.workers)

if __name__ == "__main__":
    main() 