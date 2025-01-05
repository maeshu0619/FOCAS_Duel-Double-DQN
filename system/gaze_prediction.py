import os
import pandas as pd
import numpy as np
import random

def gaze_data(directory_path, total_timesteps, video_center=(960, 540)):
    # ディレクトリ内のCSVファイルを取得
    files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("指定されたディレクトリにCSVファイルが存在しません。")

    gaze_coordinates = []
    
    while len(gaze_coordinates) < total_timesteps:
        # ランダムなファイルを選択
        random_file = random.choice(files)
        file_path = os.path.join(directory_path, random_file)
        
        # CSVファイルを読み込む
        data = pd.read_csv(file_path, header=None)
        
        # 3, 4列目（0-indexedの2, 3列目）を取得
        if data.shape[1] < 4:
            raise ValueError(f"ファイル {random_file} に期待する列が存在しません。")

        x_column = data.iloc[:, 2]
        y_column = data.iloc[:, 3]
        
        # NaNを動画の中心座標で置き換え
        x_column = data.iloc[:, 2].fillna(video_center[0]).astype(int)
        y_column = data.iloc[:, 3].fillna(video_center[1]).astype(int)

        
        # 座標をタプルのリストに変換
        coordinates = list(zip(x_column, y_column))
        
        # 収集したデータを追加
        remaining_steps = total_timesteps - len(gaze_coordinates)
        gaze_coordinates.extend(coordinates[:remaining_steps])

    return gaze_coordinates
