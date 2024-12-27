from stable_baselines3.common.logger import configure, Logger, TensorBoardOutputFormat
import datetime
import os

def file_setup():
    # ログファイルのセットアップ
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"log/{current_time}.txt"
    os.makedirs("log", exist_ok=True)
    log_file = open(log_file_path, "w")

    # TensorBoard用の設定
    tensorboard_log_dir = f"training/{current_time}"
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    logger = Logger(folder=tensorboard_log_dir, output_formats=[
        TensorBoardOutputFormat(tensorboard_log_dir)
    ])

    return log_file, logger