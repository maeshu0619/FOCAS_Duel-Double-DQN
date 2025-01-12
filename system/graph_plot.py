import matplotlib.pyplot as plt
import os
import numpy as np

# 標準偏差からグラフの縦軸のおおよその範囲を決定する
def calculate_axis_limits(data):
    if not isinstance(data, (list, np.ndarray)) or len(data) == 0:
        print("Warning: Data is empty or invalid. Returning default axis limits.")
        return 0, 1  # デフォルトの軸の範囲

    if all(d is None for d in data):
        print("Warning: Data contains only None values. Returning default axis limits.")
        return 0, 1

    # None を除外
    valid_data = [d for d in data if d is not None]
    if len(valid_data) == 0:
        print("Warning: After filtering None, data is empty. Returning default axis limits.")
        return 0, 1

    mean = np.mean(valid_data)
    std = np.std(valid_data)
    return mean - 2 * std, mean + 2 * std


def generate_training_plot(mode, graph_file_path, 
                           latency_file, network_file, 
                           reward_ave_history, error_late_per, error_buffer_per, bandwidth_ave_legacy, bitrate_legacy):
    
    # 出力フォルダの作成
    os.makedirs(graph_file_path, exist_ok=True)
    
    # 完全なファイルパスの設定
    output_file = os.path.join(graph_file_path, "training_plot.png")
    
    # グラフ設定
    plt.figure(figsize=(14, 10))

    # 平均報酬履歴のプロット
    plt.subplot(2, 2, 1)
    y_min, y_max = calculate_axis_limits(reward_ave_history)
    plt.plot(reward_ave_history, color="blue", label="Reward History")
    plt.title(f"Reward ({network_file})")
    plt.xlabel("episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    # Error Rates のプロット (error_late_per, error_buffer_per)
    plt.subplot(2, 2, 2)
    y_min, y_max = calculate_axis_limits(error_late_per + error_buffer_per)
    plt.plot(error_late_per, color="orange", label="Latency Constraint")
    plt.plot(error_buffer_per, color="cyan", label="Rebuffering Penalty")
    plt.title(f"Latency constraint violation and Rebuffering penalty probability\n ({network_file})")
    plt.xlabel("episode")
    plt.ylabel("Percentage (%)")
    plt.grid(True)
    plt.legend()

    # 平均帯域幅履歴のプロット
    plt.subplot(2, 2, 3)
    y_min, y_max = calculate_axis_limits(bandwidth_ave_legacy)
    plt.plot(bandwidth_ave_legacy, color="red", label="Bandwidth Average Legacy")
    plt.title(f"Bandwidth ({network_file})")
    plt.xlabel("episode")
    plt.ylabel("Bandwidth (kbps)")
    plt.grid(True)
    plt.legend()

    # ビットレート履歴のプロット
    plt.subplot(2, 2, 4)
    plt.plot(bitrate_legacy, color="purple", label="Bitrate Legacy")
    plt.title(f"Selected Bitrate ({network_file})")
    plt.xlabel("episode")
    plt.ylabel("Bitrate (kbps)")
    plt.grid(True)
    plt.legend()

    # レイアウト調整と保存
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def generate_cdf_plot(mode, graph_file_path, 
                      latency_file, network_file, 
                      reward_history, quality_legacy, jitter_t_legacy, jitter_s_legacy, rebuffer_legacy):
    # 出力フォルダの作成
    os.makedirs(graph_file_path, exist_ok=True)
    
    # (1) QoEの画像を生成
    output_qoe_file = os.path.join(graph_file_path, "qoe_cdf_plot.png")
    plt.figure(figsize=(7, 5))
    sorted_reward = sorted(reward_history)
    cdf = [(i + 1) / len(sorted_reward) for i in range(len(sorted_reward))]
    plt.plot(sorted_reward, cdf, color="blue", label="CDF of QoE")
    plt.title(f"CDF of QoE ({network_file})", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_qoe_file)
    plt.close()

    # (2) 他4つのグラフを1枚にまとめた画像を生成
    output_combined_file = os.path.join(graph_file_path, "qoe_info_cdf_plot.png")
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
        plt.plot(sorted_data, cdf, color=colors[idx - 1], label=f"CDF of {label}")
        plt.title(f"CDF of {label} ({network_file})", fontsize=14)
        plt.xlabel("Value", fontsize=12)
        plt.ylabel("CDF", fontsize=12)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_combined_file)
    plt.close()
