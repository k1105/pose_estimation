import cv2
import numpy as np
from feat import Detector
import argparse

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

def estimate_face_pose(image_path):
    # 画像の読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")

    # pyFeatの検出器を初期化
    detector = Detector()

    # 顔の検出
    faces = detector.detect_faces(image)
    
    if len(faces) == 0:
        print("顔が検出されませんでした。")
        return None
    
    # 姿勢推定
    result = detector.detect_facepose(image)
    
    print("\n検出された顔の情報:")
    print(f"facesの型: {type(faces)}")
    print(f"facesの内容: {faces}")
    print(f"\n姿勢推定結果:")
    print(f"resultの型: {type(result)}")
    print(f"resultの内容: {result}")
    
    if not result or 'poses' not in result or not result['poses']:
        print("姿勢推定に失敗しました。")
        return None
    
    # 最初に検出された顔の姿勢を取得
    pose = result['poses'][0][0]  # 最初の画像の最初の顔の姿勢
    face_box = result['faces'][0][0][:4]  # 最初の顔のバウンディングボックス
    
    # 画像上に姿勢を可視化
    vis_image = image.copy()
    vis_image = draw_pose_axes(vis_image, face_box, pose)
    
    # 可視化結果を保存
    output_path = image_path.rsplit('.', 1)[0] + '_pose.jpg'
    cv2.imwrite(output_path, vis_image)
    print(f"\n可視化結果を保存しました: {output_path}")
    
    return {
        'pitch': pose[0],  # 上下の回転（度）
        'yaw': pose[2],    # 左右の回転（度）
        'roll': pose[1]    # 傾き（度）
    }

def main():
    parser = argparse.ArgumentParser(description='顔の姿勢推定を行うスクリプト')
    parser.add_argument('image_path', help='入力画像のパス')
    args = parser.parse_args()

    try:
        pose = estimate_face_pose(args.image_path)
        if pose:
            print("\n顔の姿勢推定結果:")
            print(f"上下の回転 (Pitch): {pose['pitch']:.2f}度")
            print(f"左右の回転 (Yaw): {pose['yaw']:.2f}度")
            print(f"傾き (Roll): {pose['roll']:.2f}度")
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main() 