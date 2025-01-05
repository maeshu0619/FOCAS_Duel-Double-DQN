from stable_baselines3.common.logger import configure, Logger, TensorBoardOutputFormat
import datetime
import os
import random
import math

# ログを記録するためのファイル名の準備
def file_setup(mode, current_time):
    if mode == 0:
        mode_name = "0_ABR"
    elif mode == 1:
        mode_name = "1_FOCAS"
    elif mode == 2:
        mode_name = "2_A-FOCAS"

    # サブディレクトリまで作成
    log_dir = f"log/{mode_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # ログファイルのセットアップ
    log_file_path = f"{log_dir}/A-{current_time}.txt"
    log_file = open(log_file_path, "w")
    
    # デバッグ内容のログファイルのセットアップ
    debug_log_file_path = f"{log_dir}/B-{current_time}.txt"
    debug_log_file = open(debug_log_file_path, "w")

    # TensorBoard用の設定
    tensorboard_log_dir = f"training/{mode_name}_{current_time}"
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    logger = Logger(folder=tensorboard_log_dir, output_formats=[
        TensorBoardOutputFormat(tensorboard_log_dir)
    ])

    return log_file, debug_log_file, logger

# 解像度情報を持つビットレートリストと解像度リストを返す関数。
def extract_bitrate_and_resolution(bitrate_to_resolution):
    bitrate_list = list(bitrate_to_resolution.keys())
    resolution_list = list(bitrate_to_resolution.values())
    return bitrate_list, resolution_list
