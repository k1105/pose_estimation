import os
import json

# 入力JSONディレクトリ
input_json_folder = 'out/json'

# 出力JSONファイル
output_json_file = 'out/normalized_json/hitomoji_data.json'
os.makedirs('out/normalized_json', exist_ok=True)

# 全てのJSONデータを格納するリスト
all_data = []

# フォルダ内のすべてのJSONファイルを処理
for json_file in os.listdir(input_json_folder):
    if json_file.endswith('.json'):
        input_json_path = os.path.join(input_json_folder, json_file)
        
        # JSONデータを読み込む
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # bboxの情報
        x1, y1, x2, y2 = data["bbox"]

        # bboxの幅と高さを計算
        width = x2 - x1
        height = y2 - y1

        # スケーリング係数を計算
        scale = 1000 / max(width, height)

        # bboxの新しい座標 (左上を (0, 0) に揃える)
        new_x1, new_y1 = 0, 0
        new_x2 = int((x2 - x1) * scale)
        new_y2 = int((y2 - y1) * scale)

        # bboxの更新
        data["bbox"] = [new_x1, new_y1, new_x2, new_y2]

        # キーポイントの更新
        new_keypoints = []
        for kpt in data["keypoints"]:
            new_x = int((kpt[0] - x1) * scale)
            new_y = int((kpt[1] - y1) * scale)
            new_keypoints.append([new_x, new_y])

        data["keypoints"] = new_keypoints

        # 各ファイルの正規化されたデータをリストに追加
        all_data.append(data)

# 正規化されたすべてのデータを1つのファイルに保存
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

print(f"すべてのJSONファイル情報が {output_json_file} に保存されました。")
