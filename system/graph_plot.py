import matplotlib.pyplot as plt
import os
import datetime

import matplotlib.pyplot as plt
import os
import numpy as np

# 標準偏差からグラフの縦軸のおおよその範囲を決定する
def calculate_axis_limits(data, margin=1.5):
    mean = np.mean(data)
    std = np.std(data)
    y_min = mean - margin * std
    y_max = mean + margin * std
    return y_min, y_max

def generate_training_plot(mode, cdf_file_path, reward_history, q_values, bandwidth_legacy, bitrate_legacy):
    # 出力フォルダの作成
    os.makedirs(cdf_file_path, exist_ok=True)
    
    # 完全なファイルパスの設定
    output_file = os.path.join(cdf_file_path, "training_plot.png")
    
    # グラフ設定
    plt.figure(figsize=(14, 10))

    # 報酬履歴のプロット
    plt.subplot(2, 2, 1)
    y_min, y_max = calculate_axis_limits(reward_history)
    plt.plot(reward_history, color="blue", label="Reward History")
    plt.title("Reward History")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    # Q値の履歴のプロット
    plt.subplot(2, 2, 2)
    y_min, y_max = calculate_axis_limits(q_values)
    plt.plot(q_values, color="green", label="Q Values")
    plt.title("Q Values")
    plt.xlabel("Step")
    plt.ylabel("Q Value")
    plt.grid(True)
    plt.legend()

    # 帯域幅履歴のプロット
    plt.subplot(2, 2, 3)
    y_min, y_max = calculate_axis_limits(bandwidth_legacy)
    plt.plot(bandwidth_legacy, color="red", label="Bandwidth Legacy")
    plt.title("Bandwidth Legacy")
    plt.xlabel("Step")
    plt.ylabel("Bandwidth (Mbps)")
    plt.grid(True)
    plt.legend()

    # ビットレート履歴のプロット
    plt.subplot(2, 2, 4)
    y_min, y_max = calculate_axis_limits(bitrate_legacy)
    plt.plot(bitrate_legacy, color="purple", label="Bitrate Legacy")
    plt.title("Bitrate Legacy")
    plt.xlabel("Step")
    plt.ylabel("Bitrate (Mbps)")
    plt.grid(True)
    plt.legend()

    # レイアウト調整と保存
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def generate_cdf_plot(mode, cdf_file_path, reward_history, quality_legacy, jitter_t_legacy, jitter_s_legacy, rebuffer_legacy):
    # 出力フォルダの作成
    os.makedirs(cdf_file_path, exist_ok=True)
    
    # (1) QoEの画像を生成
    output_qoe_file = os.path.join(cdf_file_path, "qoe_cdf_plot.png")
    plt.figure(figsize=(7, 5))
    sorted_reward = sorted(reward_history)
    cdf = [(i + 1) / len(sorted_reward) for i in range(len(sorted_reward))]
    y_min, y_max = calculate_axis_limits(sorted_reward)
    plt.plot(sorted_reward, cdf, color="blue", label="CDF of QoE")
    plt.title("CDF of QoE", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_qoe_file)
    plt.close()

    # (2) 他4つのグラフを1枚にまとめた画像を生成
    output_combined_file = os.path.join(cdf_file_path, "qoe_info_cdf_plot.png")
    data = {
        "Quality": sorted(quality_legacy),
        "Temporal Jitter": sorted(jitter_t_legacy),
        "Spatial Jitter": sorted(jitter_s_legacy),
        "Rebuffering": sorted(rebuffer_legacy),
    }
    plt.figure(figsize=(14, 10))
    colors = ["green", "red", "purple", "orange"]  # グラフの色

    for idx, (label, sorted_data) in enumerate(data.items(), start=1):
        cdf = [(i + 1) / len(sorted_data) for i in range(len(sorted_data))]
        plt.subplot(2, 2, idx)
        y_min, y_max = calculate_axis_limits(sorted_data)
        plt.plot(sorted_data, cdf, color=colors[idx - 1], label=f"CDF of {label}")
        plt.title(f"CDF of {label}", fontsize=14)
        plt.xlabel("Value", fontsize=12)
        plt.ylabel("CDF", fontsize=12)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_combined_file)
    plt.close()
