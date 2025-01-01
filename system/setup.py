from stable_baselines3.common.logger import configure, Logger, TensorBoardOutputFormat
import datetime
import os

def file_setup(mode):
    if mode == 0:
        mode_name = "0_ABR_"
    elif mode == 1:
        mode_name = "1_FOCAS_"
    elif mode == 2:
        mode_name = "2_A-FOCAS_"
    # ログファイルのセットアップ
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"log/{mode_name}{current_time}.txt"
    os.makedirs("log", exist_ok=True)
    log_file = open(log_file_path, "w")

    # TensorBoard用の設定
    tensorboard_log_dir = f"training/{current_time}"
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    logger = Logger(folder=tensorboard_log_dir, output_formats=[
        TensorBoardOutputFormat(tensorboard_log_dir)
    ])

    return log_file, logger

def extract_bitrate_and_resolution(bitrate_to_resolution):
    """
    解像度情報を持つビットレートリストと解像度リストを返す関数。

    Args:
        bitrate_to_resolution (dict): ビットレートと解像度のマッピング辞書。

    Returns:
        tuple: 
            - bitrate_list (list): ビットレートのリスト。
            - resolution_list (list): 解像度 (横, 縦) のリスト。
    """
    bitrate_list = list(bitrate_to_resolution.keys())
    resolution_list = list(bitrate_to_resolution.values())
    return bitrate_list, resolution_list
