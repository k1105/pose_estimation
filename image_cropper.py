import cv2
import numpy as np
from pathlib import Path
import os

class ImageCropper:
    def __init__(self, margin_percent=20, target_size=(4096, 4096)):
        """
        画像クロップ処理を行うクラス
        
        Args:
            margin_percent (float): バウンディングボックスに対するマージンの割合（%）
            target_size (tuple): リサイズ後の画像サイズ (width, height)
        """
        self.margin_percent = margin_percent
        self.target_size = target_size

    def crop_image(self, image_path, bbox):
        """
        画像をクロップする
        
        Args:
            image_path (str): 入力画像のパス
            bbox (tuple): (x, y, width, height) 形式のバウンディングボックス
        
        Returns:
            numpy.ndarray: クロップされた画像
        """
        # 画像を読み込む
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めませんでした: {image_path}")

        x, y, width, height = bbox
        
        # マージンを計算
        margin_x = int(width * self.margin_percent / 100)
        margin_y = int(height * self.margin_percent / 100)
        
        # バウンディングボックスの中心を計算
        bbox_center_x = int(x + width / 2)
        bbox_center_y = int(y + height / 2)
        
        # 正方形のサイズを決定（長い辺に合わせる）
        square_size = max(width + 2 * margin_x, height + 2 * margin_y)
        
        # 中心を基準に正方形の領域を計算
        square_x1 = int(bbox_center_x - square_size // 2)
        square_y1 = int(bbox_center_y - square_size // 2)
        square_x2 = int(square_x1 + square_size)
        square_y2 = int(square_y1 + square_size)
        
        # 画像の端に達した場合の調整
        if square_x1 < 0:
            square_x1 = 0
            square_x2 = int(square_size)
        elif square_x2 > image.shape[1]:
            square_x2 = int(image.shape[1])
            square_x1 = int(square_x2 - square_size)
        
        if square_y1 < 0:
            square_y1 = 0
            square_y2 = int(square_size)
        elif square_y2 > image.shape[0]:
            square_y2 = int(image.shape[0])
            square_y1 = int(square_y2 - square_size)
        
        # 画像をクロップ
        cropped = image[square_y1:square_y2, square_x1:square_x2]
        
        # 画像をリサイズ
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return resized

    def crop_and_save(self, image, bbox, output_path):
        """
        画像をクロップして保存する
        
        Args:
            image: 入力画像（numpy配列）または画像パス
            bbox (tuple): バウンディングボックス (x, y, width, height)
            output_path (str): 出力ファイルのパス
        
        Returns:
            str: 保存されたファイルのパス
        """
        # 画像をクロップ
        cropped = self.crop_image(image, bbox)
        
        # 出力ディレクトリを作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 画像を保存
        cv2.imwrite(output_path, cropped)
        
        return output_path

def create_cropper(margin_percent=20, target_size=(4096, 4096)):
    """
    クロッパーのインスタンスを作成する
    
    Args:
        margin_percent (float): バウンディングボックスに対するマージンの割合（%）
        target_size (tuple): リサイズ後の画像サイズ (width, height)
    
    Returns:
        ImageCropper: クロッパーのインスタンス
    """
    return ImageCropper(margin_percent, target_size) 