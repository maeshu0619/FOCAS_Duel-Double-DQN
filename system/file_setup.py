from stable_baselines3.common.logger import configure, Logger, TensorBoardOutputFormat
import datetime
import os
import random
import math

# ログを記録するためのファイル名の準備
def file_setup(mode, train_or_test, current_time, latency_file, network_file):
    if mode == 0:
        mode_name = "ABR"
    elif mode == 1:
        mode_name = "FOCAS"
    elif mode == 2:
        mode_name = "A-FOCAS"
    
    if train_or_test == 0:
        log_name = "log_train"
    elif train_or_test == 1:
        log_name = "log_test"

    # サブディレクトリまで作成
    log_dir = f"{log_name}/{mode_name}/{latency_file}/{network_file}/{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    
    # ログファイルのセットアップ
    log_file_path = f"{log_dir}/StreamingInfo.txt"
    log_file = open(log_file_path, "w")
    
    # デバッグ内容のログファイルのセットアップ
    debug_log_file_path = f"{log_dir}/DebugInfo.txt"
    debug_log_file = open(debug_log_file_path, "w")

    # TensorBoard用の設定
    tensorboard_log_dir = f"tensor_log/{mode_name}_{latency_file}_{network_file}_{current_time}"
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
