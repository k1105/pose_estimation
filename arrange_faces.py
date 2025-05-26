import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os

def calculate_scale_and_position(image_size, bbox, target_face_size=300):
    """
    画像のスケールと配置位置を計算する
    
    Args:
        image_size (tuple): 元画像のサイズ (width, height)
        bbox (dict): バウンディングボックスの情報 {'x': int, 'y': int, 'width': int, 'height': int}
        target_face_size (int): 目標とする顔のサイズ（最長辺）
    
    Returns:
        tuple: (scale_factor, center_x, center_y)
    """
    # バウンディングボックスの最長辺を取得
    face_size = max(bbox['width'], bbox['height'])
    
    # スケール係数を計算
    scale = target_face_size / face_size
    
    # スケール後の画像サイズを計算
    scaled_width = int(image_size[0] * scale)
    scaled_height = int(image_size[1] * scale)
    
    # 顔の中心位置を計算（スケール後）
    face_center_x = int((bbox['x'] + bbox['width'] / 2) * scale)
    face_center_y = int((bbox['y'] + bbox['height'] / 2) * scale)
    
    # キャンバス（4096x4096）の中心に顔が来るように配置位置を計算
    canvas_center = 4096 // 2
    pos_x = canvas_center - face_center_x
    pos_y = canvas_center - face_center_y
    
    return scale, pos_x, pos_y

def arrange_image(image_path, bbox, output_path, target_face_size=300):
    """
    画像を配置して保存する
    
    Args:
        image_path (Path): 入力画像のパス
        bbox (dict): バウンディングボックスの情報
        output_path (Path): 出力画像のパス
        target_face_size (int): 目標とする顔のサイズ（最長辺）
    """
    try:
        # 画像を読み込む
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return
        
        # 画像のサイズを取得
        height, width = image.shape[:2]
        
        # スケールと配置位置を計算
        scale, pos_x, pos_y = calculate_scale_and_position((width, height), bbox, target_face_size)
        
        # 画像をリサイズ
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)
        resized = cv2.resize(image, (scaled_width, scaled_height))
        
        # 新しいキャンバスを作成（白背景）
        canvas = np.ones((4096, 4096, 3), dtype=np.uint8) * 255
        
        # 画像を配置
        # 画像がキャンバスからはみ出る場合は、はみ出る部分を切り取る
        x1 = max(0, pos_x)
        y1 = max(0, pos_y)
        x2 = min(4096, pos_x + scaled_width)
        y2 = min(4096, pos_y + scaled_height)
        
        # 画像の配置範囲を計算
        img_x1 = max(0, -pos_x)
        img_y1 = max(0, -pos_y)
        img_x2 = img_x1 + (x2 - x1)
        img_y2 = img_y1 + (y2 - y1)
        
        # 画像をキャンバスに配置
        canvas[y1:y2, x1:x2] = resized[img_y1:img_y2, img_x1:img_x2]
        
        # 画像を保存
        cv2.imwrite(str(output_path), canvas)
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def process_detections(json_file, input_dir, output_dir, target_face_size=300, min_face_size=0, emotion=None, emotion_threshold=0.90):
    """
    検出結果のJSONファイルを読み込み、画像を配置する
    
    Args:
        json_file (str): 検出結果のJSONファイルのパス
        input_dir (str): 入力ディレクトリのパス
        output_dir (str): 出力ディレクトリのパス
        target_face_size (int): 目標とする顔のサイズ（最長辺）
        min_face_size (int): 処理対象とする顔の最小サイズ（最長辺）
        emotion (str): フィルタリングする感情の種類
        emotion_threshold (float): 感情の閾値
    """
    # JSONファイルを読み込む
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 入力ディレクトリと出力ディレクトリをPathオブジェクトに変換
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 感情でフィルタリングし、yaw角でソート
    filtered_faces = []
    for face_info in data['faces']:
        if emotion and face_info.get('emotions', {}).get(emotion, 0) < emotion_threshold:
            continue
        filtered_faces.append(face_info)
    
    # yaw角でソート
    filtered_faces.sort(key=lambda x: abs(x.get('direction', {}).get('yaw', 0)))
    
    # 各画像を処理
    for idx, face_info in enumerate(tqdm(filtered_faces, desc="Arranging images")):
        # バウンディングボックスの最長辺をチェック
        face_size = max(face_info['bbox']['width'], face_info['bbox']['height'])
        if face_size < min_face_size:
            continue
        
        # 入力画像のパスを構築
        image_path = input_dir / face_info['file_path']
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            continue
        
        # 出力パスを生成（ソート順とIDを含む）
        output_path = output_dir / f"{idx:04d}_{face_info['id']}{image_path.suffix}"
        
        # 画像を配置
        arrange_image(image_path, face_info['bbox'], output_path, target_face_size)

def visualize_bbox(image_path, bbox, output_path=None):
    """
    画像にバウンディングボックスを重畳して表示・保存する
    
    Args:
        image_path (Path): 入力画像のパス
        bbox (dict): バウンディングボックスの情報
        output_path (Path, optional): 出力画像のパス。Noneの場合は表示のみ
    """
    try:
        # 画像を読み込む
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return
        
        # バウンディングボックスを描画
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 出力パスが指定されている場合は保存
        if output_path:
            cv2.imwrite(str(output_path), image)
        else:
            # 画像を表示
            cv2.imshow('Image with Bounding Box', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def visualize_by_id(json_file, input_dir, target_id, output_dir=None):
    """
    指定したIDの画像にバウンディングボックスを重畳して表示・保存する
    
    Args:
        json_file (str): 検出結果のJSONファイルのパス
        input_dir (str): 入力ディレクトリのパス
        target_id (int): 対象のID
        output_dir (str, optional): 出力ディレクトリのパス。Noneの場合は表示のみ
    """
    # JSONファイルを読み込む
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 指定されたIDの情報を探す
    target_face = None
    for face_info in data['faces']:
        if face_info['id'] == target_id:
            target_face = face_info
            break
    
    if target_face is None:
        print(f"Error: Face with ID {target_id} not found")
        return
    
    # 入力ディレクトリをPathオブジェクトに変換
    input_dir = Path(input_dir)
    
    # 入力画像のパスを構築
    image_path = input_dir / target_face['file_path']
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    # 出力パスを設定
    output_path = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"bbox_{target_id}{image_path.suffix}"
    
    # バウンディングボックスを重畳して表示・保存
    visualize_bbox(image_path, target_face['bbox'], output_path)

def main():
    parser = argparse.ArgumentParser(description='Arrange images with faces centered on a 4096x4096 canvas')
    parser.add_argument('json_file', type=str, help='JSON file containing face detection results')
    parser.add_argument('input_dir', type=str, help='Input directory containing original images')
    parser.add_argument('--output_dir', type=str, default='arranged_faces',
                      help='Output directory for arranged images (default: arranged_faces)')
    parser.add_argument('--face_size', type=int, default=300,
                      help='Target face size in pixels (default: 300)')
    parser.add_argument('--min_face_size', type=int, default=0,
                      help='Minimum face size in pixels to process (default: 0)')
    parser.add_argument('--emotion', type=str, choices=['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral'],
                      help='Filter images by emotion type')
    parser.add_argument('--emotion_threshold', type=float, default=0.90,
                      help='Threshold for emotion filtering (default: 0.90)')
    parser.add_argument('--visualize_id', type=int,
                      help='Visualize bounding box for a specific face ID')
    parser.add_argument('--visualize_output', type=str,
                      help='Output directory for visualization (if not specified, image will be displayed)')
    
    args = parser.parse_args()
    
    # 入力ファイルとディレクトリの存在確認
    if not os.path.isfile(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # 特定のIDの可視化が指定されている場合
    if args.visualize_id is not None:
        visualize_by_id(args.json_file, args.input_dir, args.visualize_id, args.visualize_output)
        return
    
    # 画像の配置処理を実行
    process_detections(args.json_file, args.input_dir, args.output_dir,
                      args.face_size, args.min_face_size,
                      args.emotion, args.emotion_threshold)

if __name__ == "__main__":
    main() 