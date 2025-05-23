import argparse
from feat import Detector
import os
import json
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import torch
import cv2
import numpy as np

def get_emotion_columns():
    """感情の列名を取得"""
    return ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

def process_image(image_path, detector, base_dir, face_confidence=0.5):
    """
    1枚の画像を処理し、顔の位置、向き、感情を検出する
    
    Args:
        image_path (Path): 入力画像のパス
        detector (Detector): pyFeatの検出器
        base_dir (Path): 基準となる入力ディレクトリ
        face_confidence (float): 顔検出の信頼度閾値
    
    Returns:
        list: 検出された顔の情報のリスト
    """
    try:
        # 画像を読み込む
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return []

        # 画像から顔と表情を検出
        with torch.no_grad():  # 勾配計算を無効化
            result = detector.detect_image(str(image_path))
        
        # FaceScoreでフィルタリング
        if 'FaceScore' in result.columns:
            result = result[result['FaceScore'] >= face_confidence]
        
        if len(result) == 0:
            return []
        
        # 姿勢推定を実行
        pose_result = detector.detect_facepose(image)
        
        # 相対パスを取得
        rel_path = str(image_path.relative_to(base_dir))
        
        # 各顔の情報を収集
        faces = []
        for i in range(len(result)):
            face_info = {
                'id': i,
                'file_path': rel_path,
                'bbox': {
                    'x': int(result.iloc[i]['FaceRectX']),
                    'y': int(result.iloc[i]['FaceRectY']),
                    'width': int(result.iloc[i]['FaceRectWidth']),
                    'height': int(result.iloc[i]['FaceRectHeight'])
                },
                'confidence': float(result.iloc[i]['FaceScore']) if 'FaceScore' in result.columns else None
            }
            
            # 姿勢情報を追加
            if pose_result and 'poses' in pose_result and i < len(pose_result['poses']):
                pitch, roll, yaw = pose_result['poses'][i][0]
                face_info['direction'] = {
                    'yaw': float(yaw),
                    'roll': float(roll),
                    'pitch': float(pitch)
                }
            
            # 感情情報を追加
            emotions = {emotion: float(result.iloc[i][emotion]) 
                       for emotion in get_emotion_columns() 
                       if emotion in result.columns}
            face_info['emotions'] = emotions
            
            faces.append(face_info)
        
        return faces
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def process_directory(input_dir, output_file, face_confidence=0.5, max_workers=4):
    """
    ディレクトリ内の画像を処理し、顔の検出結果をJSONファイルに出力する
    
    Args:
        input_dir (str): 入力ディレクトリのパス
        output_file (str): 出力JSONファイルのパス
        face_confidence (float): 顔検出の信頼度閾値
        max_workers (int): 並列処理のワーカー数
    """
    # 検出器の初期化
    detector = Detector(
        face_detector='retinaface',
        face_detector_kwargs={'confidence_threshold': face_confidence},
        batch_size=1,
        output_size=(224, 224),
        device='cpu'  # CPUを使用
    )
    
    # 入力ディレクトリをPathオブジェクトに変換
    input_dir = Path(input_dir)
    
    # 画像ファイルのリストを取得（再帰的に）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    all_faces = []
    
    # 並列処理で画像を処理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_image, image_path, detector, input_dir, face_confidence)
            for image_path in image_files
        ]
        
        # プログレスバーを表示しながら結果を収集
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="Processing images"):
            faces = future.result()
            all_faces.extend(faces)
    
    # 結果をJSONファイルに出力
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'faces': all_faces}, f, ensure_ascii=False, indent=2)
    
    print(f"Detection results saved to: {output_file}")
    print(f"Total faces detected: {len(all_faces)}")

def main():
    parser = argparse.ArgumentParser(description='Detect faces, poses, and emotions in images')
    parser.add_argument('input_dir', type=str, help='Input directory containing images')
    parser.add_argument('--output_file', type=str, default='face_detections.json',
                      help='Output JSON file path (default: face_detections.json)')
    parser.add_argument('--face_confidence', type=float, default=0.9,
                      help='Face detection confidence threshold (0.0-1.0, default: 0.9)')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of worker threads (default: 4)')
    
    args = parser.parse_args()
    
    # 入力ディレクトリの存在確認
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # 画像の処理を実行
    process_directory(args.input_dir, args.output_file, 
                     args.face_confidence, args.workers)

if __name__ == "__main__":
    main() 