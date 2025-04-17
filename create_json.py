import cv2
import os
import json
from ultralytics import YOLO
from datetime import datetime

# YOLOv11xモデルをロード（バウンディングボックス用）
bbox_model = YOLO("yolo11x.pt")

# YOLOv11x-poseモデルをロード（キーポイント用）
pose_model = YOLO("yolo11x-pose.pt")

# 入力画像ディレクトリ
input_folder = 'input_images'

# タイムスタンプ付きの出力フォルダを作成
timestamp = datetime.now().strftime('%y%m%d%H%M%S')
output_base = os.path.join('out', timestamp)
image_output_folder = os.path.join(output_base, 'image')
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(output_base, exist_ok=True)

# 全ての結果を格納するリスト
all_results = []
record_id = 1

# input_imagesフォルダ内のすべての画像を処理
for img_file in os.listdir(input_folder):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):  # 対象の画像ファイルのみ処理
        img_path = os.path.join(input_folder, img_file)

        # 画像を読み込む
        img = cv2.imread(img_path)
        if img is None:
            print(f"{img_file} は読み込めませんでした。スキップします。")
            continue

        # ファイル名を取得し、拡張子を除外
        image_filename = os.path.splitext(img_file)[0]

        # 1. バウンディングボックスの推論を実行
        bbox_results = bbox_model(img)

        # 2. ポーズ（キーポイント）の推論を実行
        pose_results = pose_model(img)

        # 画像のコピーを作成（描画用）
        img_draw = img.copy()

        # 各バウンディングボックスに対して処理
        for i, result in enumerate(bbox_results):
            boxes = result.boxes
            if len(boxes) > 0:  # データが存在するか確認
                # 信頼度の高いバウンディングボックスのみを処理
                valid_boxes = []
                for box in boxes:
                    confidence = float(box.conf[0])
                    if confidence >= 0.85:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # バウンディングボックスの中心座標を計算
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        valid_boxes.append((box, x1, y1, x2, y2, center_x, center_y, confidence))

                # ポーズデータの中心座標を計算
                pose_centers = []
                if len(pose_results) > i:
                    for pose in pose_results[i].keypoints:
                        if len(pose.xy) > 0 and len(pose.xy[0]) > 0:
                            # ポーズの中心座標を計算（有効なキーポイントの平均）
                            valid_points = [pt for pt in pose.xy[0] if pt[0] > 0 and pt[1] > 0]
                            if valid_points:
                                center_x = int(sum(pt[0] for pt in valid_points) / len(valid_points))
                                center_y = int(sum(pt[1] for pt in valid_points) / len(valid_points))
                                pose_centers.append((pose, center_x, center_y))

                # 各バウンディングボックスに対して最も近いポーズを探す
                for box_data in valid_boxes:
                    box, x1, y1, x2, y2, box_center_x, box_center_y, confidence = box_data
                    
                    # 最も近いポーズを探す
                    min_dist = float('inf')
                    best_pose = None
                    best_pose_center = None
                    
                    for pose, pose_center_x, pose_center_y in pose_centers:
                        # ユークリッド距離を計算
                        dist = ((box_center_x - pose_center_x) ** 2 + (box_center_y - pose_center_y) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_pose = pose
                            best_pose_center = (pose_center_x, pose_center_y)

                    # 有効なポーズが見つかった場合のみ処理
                    if best_pose is not None:
                        # キーポイント情報を取得
                        keypoints = []
                        has_valid_pose = False
                        for kpt in best_pose.xy[0]:
                            x, y = int(kpt[0]), int(kpt[1])
                            keypoints.append([x, y])
                            if x > 0 and y > 0:
                                cv2.circle(img_draw, (x, y), 5, (0, 0, 255), -1)
                                has_valid_pose = True

                        if has_valid_pose:
                            # バウンディングボックスを画像に描画
                            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f'ID:{record_id} Person {confidence:.2f}'
                            cv2.putText(img_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                            # 結果をリストに追加
                            result_data = {
                                "id": record_id,
                                "image_name": img_file,
                                "bbox": [x1, y1, x2, y2],
                                "keypoints": keypoints
                            }
                            all_results.append(result_data)
                            record_id += 1

        # 描画した画像を保存
        image_output_path = os.path.join(image_output_folder, f"{image_filename}.jpg")
        cv2.imwrite(image_output_path, img_draw)
        print(f"{img_file} の処理が完了しました。")

# 全ての結果を1つのJSONファイルに保存
json_output_path = os.path.join(output_base, f"{timestamp}.json")
with open(json_output_path, 'w') as json_file:
    json.dump(all_results, json_file, indent=2)

print(f"全ての結果が {json_output_path} に保存されました。")
