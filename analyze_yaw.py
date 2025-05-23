import argparse
import os
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_yaw_from_filename(filename):
    """
    ファイル名からyaw角を抽出する
    
    Args:
        filename (str): ファイル名（例: 'yaw-79.56_0.95_ANTALYA_24-2024-04-image.jpg'）
    
    Returns:
        float: yaw角（度）。抽出できない場合はNone
    """
    # yaw{値}_ のパターンにマッチ
    match = re.match(r'yaw(-?\d+\.?\d*)_', filename)
    if match:
        return float(match.group(1))
    return None

def create_yaw_histogram(directory, bin_size=5):
    """
    ディレクトリ内の画像ファイルからyaw角を抽出し、ヒストグラムを作成する
    
    Args:
        directory (str): 画像ファイルが格納されたディレクトリ
        bin_size (int): ヒストグラムのビンサイズ（度）
    """
    # yaw角を格納するリスト
    yaw_angles = []
    
    # ディレクトリ内の全ファイルを再帰的に検索
    for file_path in Path(directory).rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
            yaw = extract_yaw_from_filename(file_path.name)
            if yaw is not None:
                yaw_angles.append(yaw)
    
    if not yaw_angles:
        print("No yaw angles found in filenames.")
        return
    
    # データの統計情報を計算
    yaw_array = np.array(yaw_angles)
    mean_yaw = np.mean(yaw_array)
    std_yaw = np.std(yaw_array)
    min_yaw = np.min(yaw_array)
    max_yaw = np.max(yaw_array)
    
    # ヒストグラムのビンの範囲を設定
    bin_edges = np.arange(min_yaw - bin_size/2, max_yaw + bin_size, bin_size)
    
    # ヒストグラムを作成
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(yaw_angles, bins=bin_edges, edgecolor='black')
    
    # グラフの設定
    plt.title(f'Distribution of Yaw Angles (Bin Size: {bin_size}°)')
    plt.xlabel('Yaw Angle (degrees)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # 統計情報を表示
    stats_text = f'Mean: {mean_yaw:.1f}°\nStd: {std_yaw:.1f}°\nMin: {min_yaw:.1f}°\nMax: {max_yaw:.1f}°\nTotal: {len(yaw_angles)}'
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 各ビンの値を表示
    for i in range(len(n)):
        if n[i] > 0:  # 値が0より大きいビンのみ表示
            plt.text(bins[i] + bin_size/2, n[i],
                    f'{int(n[i])}',
                    ha='center', va='bottom')
    
    # グラフを保存
    output_path = os.path.join(directory, 'yaw_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to: {output_path}")
    
    # 詳細な統計情報を表示
    print("\nDetailed Statistics:")
    print(f"Total images analyzed: {len(yaw_angles)}")
    print(f"Mean yaw angle: {mean_yaw:.1f}°")
    print(f"Standard deviation: {std_yaw:.1f}°")
    print(f"Minimum yaw angle: {min_yaw:.1f}°")
    print(f"Maximum yaw angle: {max_yaw:.1f}°")
    
    # 度数分布を表示
    print("\nDistribution by angle range:")
    for i in range(len(n)):
        if n[i] > 0:  # 値が0より大きいビンのみ表示
            print(f"{bins[i]:.1f}° to {bins[i+1]:.1f}°: {int(n[i])} images")

def main():
    parser = argparse.ArgumentParser(description='Analyze yaw angles from image filenames')
    parser.add_argument('directory', type=str, help='Directory containing images with yaw angles in filenames')
    parser.add_argument('--bin_size', type=int, default=5,
                      help='Bin size for histogram in degrees (default: 5)')
    
    args = parser.parse_args()
    
    # ディレクトリの存在確認
    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found: {args.directory}")
        return
    
    # ヒストグラムを作成
    create_yaw_histogram(args.directory, args.bin_size)

if __name__ == "__main__":
    main() 