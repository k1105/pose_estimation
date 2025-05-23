import argparse
from feat import Detector
import matplotlib.pyplot as plt
import os
import pandas as pd

def detect_face_emotion(image_path, face_confidence=0.5):
    """
    指定された画像から顔と表情を検出する関数
    
    Args:
        image_path (str): 入力画像のパス
        face_confidence (float): 顔検出の信頼度閾値（0.0-1.0）
    """
    # 検出器の初期化
    detector = Detector()
    
    # 画像の存在確認
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    try:
        # 画像から顔と表情を検出
        result = detector.detect_image(image_path)
        
        # FaceScoreでフィルタリング
        if 'FaceScore' in result.columns:
            result = result[result['FaceScore'] >= face_confidence]
        
        # 検出結果の表示
        print("\nDetection Results:")
        print(f"Number of faces detected: {len(result)}")
        
        # 各顔の表情分析結果を表示
        for i in range(len(result)):
            print(f"\nFace {i+1}:")
            
            # 顔検出の信頼度を表示
            if 'FaceScore' in result.columns:
                conf = result.iloc[i]['FaceScore']
                print(f"Face detection score: {conf:.2f}")
            
            # Action Unitsの表示
            print("Action Units (AUs):")
            au_columns = [col for col in result.columns if col.startswith('AU')]
            if au_columns:
                for au in au_columns:
                    value = result.iloc[i][au]
                    print(f"  {au}: {value:.2f}")
            else:
                print("  No AU data available")
            
            # 感情の表示
            print("\nEmotions:")
            emotion_columns = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
            emotion_columns = [col for col in emotion_columns if col in result.columns]
            if emotion_columns:
                for emotion in emotion_columns:
                    value = result.iloc[i][emotion]
                    print(f"  {emotion}: {value:.2f}")
            else:
                print("  No emotion data available")
        
        # 検出結果の可視化
        result.plot_detections()
        plt.show()
        
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        print("\nDebug information:")
        print(f"Result type: {type(result)}")
        if isinstance(result, pd.DataFrame):
            print("\nAvailable columns:")
            print(result.columns.tolist())
            print("\nFirst row data:")
            print(result.iloc[0] if len(result) > 0 else "No data available")

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='Detect faces and emotions in an image using pyFeat')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--confidence', type=float, default=0.5,
                      help='Face detection confidence threshold (0.0-1.0, default: 0.5)')
    
    args = parser.parse_args()
    
    # 顔と表情の検出を実行
    detect_face_emotion(args.image_path, args.confidence)

if __name__ == "__main__":
    main() 