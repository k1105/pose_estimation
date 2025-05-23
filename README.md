# Pose Similarity Tool

姿勢の類似度を計算し、類似ポーズの検索や可視化を行うツールです。

## 前提条件

- Python 3.8 以上
- YOLOv8 モデル
  - `yolo11x.pt`（バウンディングボックス検出用）
  - `yolo11x-pose.pt`（キーポイント検出用）

## セットアップ

1. 必要なパッケージのインストール:

```bash
pip install -r requirements.txt
```

2. 入力画像の準備:

- `input_images` ディレクトリを作成
- 解析したい画像を配置（サブディレクトリ可）
- 対応フォーマット: jpg, jpeg, png

## 使用方法

### 0. ポーズデータの作成

最初に、画像からポーズを検出して JSON ファイルを作成します。

```bash
python create_json.py
```

- 処理内容:

  - `input_images` 内の全画像に対して人物検出とポーズ推定を実行
  - 検出された人物ごとに一意の ID を付与
  - バウンディングボックスとキーポイント座標を記録
  - 可視化用の画像を生成（バウンディングボックス、キーポイント、ID 表示）

- 出力:

  - `out/YYMMDDHHMMSS/` ディレクトリに結果を保存
    - `YYMMDDHHMMSS.json`: ポーズデータ（検索用）
    - `image/`: 可視化画像

- 設定:
  - 人物検出の信頼度閾値: 0.85
  - 画像サイズに応じて自動的にマーカーサイズを調整

## 主な機能

### 1. 類似ポーズの検索

指定した ID の姿勢に類似する姿勢を検索します。

```bash
python -m pose_similarity search <json_path> --id <検索対象ID> [--top <表示件数>]

# 例：ID 10 の姿勢に類似する上位5件を表示
python -m pose_similarity search out/250417235650/250417235650.json --id 10 --top 5
```

### 2. 姿勢の比較画像生成

指定した ID の姿勢と類似する姿勢を並べて表示する画像を生成します。

```bash
python -m pose_similarity compare <json_path> --id <検索対象ID> [--output <出力ファイル>]

# 例：ID 10 の姿勢との比較画像を生成
python -m pose_similarity compare out/250417235650/250417235650.json --id 10 --output comparison.jpg

# ランダムに選択した姿勢の比較
python -m pose_similarity compare out/250417235650/250417235650.json --random --count 10
```

### 3. 連鎖的な類似画像生成

指定した ID から始めて、連鎖的に類似する姿勢を選択して画像を生成します。

```bash
python -m pose_similarity chain <json_path> --id <開始ID> --count <生成枚数> [--output <出力ディレクトリ>]

# 例：ID 10 から始めて12枚の連鎖画像を生成
python -m pose_similarity chain out/250417235650/250417235650.json --id 10 --count 12
```

### 4. ネットワーク可視化

姿勢の類似関係をネットワークとして可視化します。

```bash
python -m pose_similarity network <json_path> [オプション]

# 基本的な使用例
python -m pose_similarity network out/250417235650/250417235650.json

# UMAPを使用した可視化
python -m pose_similarity network out/250417235650/250417235650.json --embed umap

# 詳細なパラメータ設定
python -m pose_similarity network out/250417235650/250417235650.json \
    --embed umap \
    --n-neighbors 15 \
    --min-dist 0.1 \
    --min-cluster-size 5 \
    --min-samples 3 \
    --threshold 50.0
```

#### ネットワーク可視化のオプション

- `--embed`: 次元削減手法の選択（"mds" または "umap"）
- `--n-neighbors`: UMAP の近傍点数（デフォルト: 15）
- `--min-dist`: UMAP の最小距離（デフォルト: 0.1）
- `--min-cluster-size`: クラスターの最小サイズ（デフォルト: 5）
- `--min-samples`: HDBSCAN のコアポイント定義用の最小サンプル数（デフォルト: 3）
- `--threshold`: エッジを描画する距離の閾値（デフォルト: 50.0）
- `--output`: 出力ファイル名（デフォルト: network.html）

#### クラスタリング手法の選択

2 つのクラスタリング手法が利用可能です：

1. HDBSCAN（デフォルト）: 密度ベースのクラスタリング

   ```bash
   python -m pose_similarity network out/250417235650/250417235650.json \
       --cluster-method hdbscan \
       --min-cluster-size 5 \
       --min-samples 3 \
       --threshold 50.0
   ```

2. Agglomerative: クラスター数を指定可能な階層的クラスタリング
   ```bash
   python -m pose_similarity network out/250417235650/250417235650.json \
       --cluster-method agglomerative \
       --n-clusters 10
   ```

クラスター数を直接制御したい場合は、Agglomerative クラスタリングを使用してください。
クラスター数を指定しない場合は、データ数の平方根が使用されます。

#### クラスタリングパラメータの調整

HDBSCAN のパラメータは以下のように動作します：

- `--min-cluster-size`: 各クラスターに必要な最小ポイント数

  - 大きくすると：小さなクラスターが除外される
  - 小さくすると：小さなクラスターも認識される

- `--min-samples`: コアポイントを定義するための最小近傍点数

  - 大きくすると：密度の高い領域のみがクラスターとして認識される
  - 小さくすると：疎な領域もクラスターとして認識される

- `--threshold`: エッジを描画する距離の閾値
  - 大きくすると：より遠いポイント間も接続され、クラスターが統合されやすい
  - 小さくすると：近いポイント間のみが接続され、クラスターが分割されやすい

クラスター数を調整するための設定例：

```bash
# より多くのクラスターを得る場合
python -m pose_similarity network out/250417235650/250417235650.json \
    --threshold 30.0 \
    --min-cluster-size 5 \
    --min-samples 2

# より少ないクラスターを得る場合
python -m pose_similarity network out/250417235650/250417235650.json \
    --threshold 70.0 \
    --min-cluster-size 10 \
    --min-samples 5
```

注意：これらのパラメータは相互に影響し合うため、データセットに応じて適切な値を探す必要があります。

## 注意事項

- 姿勢データは正規化されて処理されます（1000×1000 スケール）
- 類似度の計算には、有効なキーポイントのみを使用します
- ネットワーク可視化はインタラクティブな HTML 形式で出力され、ブラウザで操作可能です
  - ズームイン/アウト
  - ドラッグによる移動
  - ノードへのホバーで ID 表示
  - クラスター別の色分け表示
