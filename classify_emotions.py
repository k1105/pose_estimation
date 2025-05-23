import argparse
from feat import Detector
import os
import pandas as pd
import shutil
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
import torch
from image_cropper import create_cropper
import cv2
import numpy as np

def draw_pose_axes(image, face_box, pose):
    """
    画像上に顔の向きを表す3つの軸を描画する
    face_box: [x1, y1, x2, y2] 形式の顔のバウンディングボックス
    pose: [pitch, roll, yaw] 形式の回転角（度）
    """
    # 顔の中心点を計算
    x1, y1, x2, y2 = face_box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # 軸の長さを顔のサイズに基づいて計算
    axis_length = int((x2 - x1) / 2)
    
    # 回転行列を計算（ラジアンに変換）
    pitch = np.radians(pose[0])
    roll = np.radians(pose[1])
    yaw = np.radians(pose[2])
    
    # 回転行列
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    # 3D軸の点を定義
    axis_points = np.float32([[axis_length, 0, 0],
                             [0, -axis_length, 0],
                             [0, 0, -axis_length]]).reshape(-1, 3)
    
    # 回転を適用
    axis_points = np.dot(axis_points, R.T)
    
    # 2D投影（簡易的な投影）
    axis_points_2d = axis_points[:, :2]
    
    # 軸の始点（顔の中心）
    center = np.array([center_x, center_y])
    
    # 各軸を描画
    # X軸（赤）
    end_point_x = tuple(map(int, center + axis_points_2d[0]))
    cv2.line(image, (center_x, center_y), end_point_x, (0, 0, 255), 2)
    
    # Y軸（緑）
    end_point_y = tuple(map(int, center + axis_points_2d[1]))
    cv2.line(image, (center_x, center_y), end_point_y, (0, 255, 0), 2)
    
    # Z軸（青）
    end_point_z = tuple(map(int, center + axis_points_2d[2]))
    cv2.line(image, (center_x, center_y), end_point_z, (255, 0, 0), 2)
    
    # 回転角度をテキストで表示
    text = f"Pitch: {pose[0]:.1f}° Yaw: {pose[2]:.1f}° Roll: {pose[1]:.1f}°"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return image

def convert_image_path(image_path: Path, base_dir: Path) -> str:
    """
    画像パスを新しいファイル名形式に変換
    例: 'dir1/dir2/filename.jpg' -> 'dir1-dir2-filename.jpg'
    """
    # パスの各部分を取得
    parts = list(image_path.relative_to(base_dir).parts)
    # 拡張子を除いたファイル名を取得
    filename = image_path.stem
    # ディレクトリ名とファイル名を結合（リストとして結合）
    new_name = "-".join(parts[:-1] + [filename])
    # 元の拡張子を追加
    return f"{new_name}{image_path.suffix}"

def get_emotion_columns():
    """感情の列名を取得"""
    return ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

def create_emotion_dirs(base_dir):
    """感情ごとのディレクトリを作成"""
    emotion_dirs = {}
    for emotion in get_emotion_columns():
        dir_path = os.path.join(base_dir, emotion)
        os.makedirs(dir_path, exist_ok=True)
        emotion_dirs[emotion] = dir_path
    return emotion_dirs

def process_image(image_path, detector, emotion_dirs, base_dir, face_confidence=0.5, emotion_confidence=0.5, min_box_size=0, cropper=None):
    """
    1枚の画像を処理し、最も強い感情に基づいて分類する
    
    Args:
        image_path (Path): 入力画像のパス
        detector (Detector): pyFeatの検出器
        emotion_dirs (dict): 感情ごとのディレクトリパス
        base_dir (Path): 基準となる入力ディレクトリ
        face_confidence (float): 顔検出の信頼度閾値
        emotion_confidence (float): 感情の信頼度閾値
        min_box_size (int): バウンディングボックスの最小サイズ（高さまたは幅の大きい方）
        cropper (ImageCropper): 画像クロップ処理を行うオブジェクト
    """
    try:
        # 画像を読み込む
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return

        # 画像から顔と表情を検出
        with torch.no_grad():  # 勾配計算を無効化
            result = detector.detect_image(str(image_path))
        
        # FaceScoreでフィルタリング
        if 'FaceScore' in result.columns:
            result = result[result['FaceScore'] >= face_confidence]
        
        if len(result) == 0:
            return
        
        # 姿勢推定を実行
        pose_result = detector.detect_facepose(image)
        
        # 各顔の感情を分析
        for i in range(len(result)):
            # バウンディングボックスのサイズをチェック
            if 'FaceRectWidth' in result.columns and 'FaceRectHeight' in result.columns:
                width = result.iloc[i]['FaceRectWidth']
                height = result.iloc[i]['FaceRectHeight']
                if max(width, height) < min_box_size:
                    continue
            
            # 感情の値を取得
            emotions = {emotion: result.iloc[i][emotion] 
                       for emotion in get_emotion_columns() 
                       if emotion in result.columns}
            
            if not emotions:
                continue
            
            # 最も強い感情を特定
            max_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_value = max_emotion
            
            # 感情の信頼度が閾値を下回る場合はスキップ
            if emotion_value < emotion_confidence:
                continue
            
            # 新しいファイル名形式に変換
            new_filename = convert_image_path(image_path, base_dir)
            
            # yawの値を取得（最初の顔の姿勢を使用）
            yaw = pose_result['poses'][0][0][2] if pose_result and 'poses' in pose_result else 0.0
            
            # 新しいファイル名を作成（yawの値と感情の確度を先頭に付与）
            new_filename = f"yaw{yaw:.2f}_{emotion_value:.2f}_{new_filename}"
            
            # コピー先のパスを作成
            dest_path = os.path.join(emotion_dirs[emotion_name], new_filename)
            
            # バウンディングボックスを取得
            if cropper is not None and 'FaceRectX' in result.columns and 'FaceRectY' in result.columns:
                x = result.iloc[i]['FaceRectX']
                y = result.iloc[i]['FaceRectY']
                width = result.iloc[i]['FaceRectWidth']
                height = result.iloc[i]['FaceRectHeight']
                bbox = (x, y, width, height)
                
                # 画像をクロップして保存
                cropped_image = cropper.crop_image(str(image_path), bbox)
                cv2.imwrite(dest_path, cropped_image)
            else:
                # クロップ処理を行わない場合は元の画像をコピー
                shutil.copy2(image_path, dest_path)
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()  # スタックトレースを出力

def process_directory(input_dir, output_dir, face_confidence=0.5, emotion_confidence=0.5, min_box_size=0, margin_percent=20, max_workers=4):
    """
    ディレクトリ内の画像を処理し、感情ごとに分類する
    
    Args:
        input_dir (str): 入力ディレクトリのパス
        output_dir (str): 出力ディレクトリのパス
        face_confidence (float): 顔検出の信頼度閾値
        emotion_confidence (float): 感情の信頼度閾値
        min_box_size (int): バウンディングボックスの最小サイズ（高さまたは幅の大きい方）
        margin_percent (float): バウンディングボックスに対するマージンの割合（%）
        max_workers (int): 並列処理のワーカー数
    """
    # タイムスタンプ付きの出力ディレクトリを作成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Face detection confidence threshold: {face_confidence}")
    print(f"Emotion confidence threshold: {emotion_confidence}")
    print(f"Minimum bounding box size: {min_box_size}")
    print(f"Crop margin: {margin_percent}%")
    
    emotion_dirs = create_emotion_dirs(output_dir)
    
    # 検出器の初期化
    detector = Detector(
        face_detector='retinaface',
        face_detector_kwargs={'confidence_threshold': face_confidence},
        batch_size=1,
        output_size=(224, 224),
        device='cpu'  # CPUを使用
    )
    
    # クロッパーの初期化
    cropper = create_cropper(margin_percent) if margin_percent > 0 else None
    
    # 入力ディレクトリをPathオブジェクトに変換
    input_dir = Path(input_dir)
    
    # 画像ファイルのリストを取得（再帰的に）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    # 並列処理で画像を処理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_image, image_path, detector, emotion_dirs, input_dir,
                          face_confidence, emotion_confidence, min_box_size, cropper)
            for image_path in image_files
        ]
        
        # プログレスバーを表示
        for _ in tqdm(concurrent.futures.as_completed(futures), 
                     total=len(futures), 
                     desc="Processing images"):
            pass

def main():
    parser = argparse.ArgumentParser(description='Classify images by detected emotions')
    parser.add_argument('input_dir', type=str, help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, default='out',
                      help='Output directory for classified images (default: out)')
    parser.add_argument('--face_confidence', type=float, default=0.9,
                      help='Face detection confidence threshold (0.0-1.0, default: 0.9)')
    parser.add_argument('--emotion_confidence', type=float, default=0.9,
                      help='Emotion confidence threshold (0.0-1.0, default: 0.9)')
    parser.add_argument('--min_box_size', type=int, default=0,
                      help='Minimum bounding box size (larger of width or height, default: 0)')
    parser.add_argument('--margin', type=float, default=20,
                      help='Crop margin percentage around face (default: 20)')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of worker threads (default: 4)')
    
    args = parser.parse_args()
    
    # 入力ディレクトリの存在確認
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # 画像の処理を実行
    process_directory(args.input_dir, args.output_dir, 
                     args.face_confidence, args.emotion_confidence,
                     args.min_box_size, args.margin, args.workers)

if __name__ == "__main__":
    main() 