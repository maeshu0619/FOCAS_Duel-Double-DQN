import os
import numpy as np

COOKED_TRACE_FOLDER = './cooked_traces/'
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
PACKET_PAYLOAD_PORTION = 0.95

class BandwidthSimulator:
    def __init__(self, cooked_trace_folder=COOKED_TRACE_FOLDER):
        self.all_cooked_bw, self.all_file_names = self.load_trace(cooked_trace_folder)
        self.reset()

    def load_trace(self, cooked_trace_folder):
        cooked_files = os.listdir(cooked_trace_folder)
        all_cooked_bw = []
        all_file_names = []
        for cooked_file in cooked_files:
            file_path = os.path.join(cooked_trace_folder, cooked_file)
            cooked_bw = []
            with open(file_path, 'r') as f:
                for line in f:
                    parse = line.split()
                    try:
                        bandwidth = float(parse[1]) * 10**3  # 帯域幅を取得し10^3倍
                        if bandwidth > 1.0:  # 帯域幅が1以下の場合は無視
                            cooked_bw.append(bandwidth)
                    except (ValueError, IndexError):
                        continue
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(cooked_file)
        return all_cooked_bw, all_file_names

    def reset(self):
        # ランダムにトレースを選択
        self.trace_idx = np.random.randint(len(self.all_cooked_bw))
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        self.time_idx = 0  # トレース内の現在の時間インデックス

    def simulate_total_timesteps(self, total_timesteps):
        """
        合計total_timesteps秒分の帯域幅情報を取得。
        """
        accumulated_bandwidths = []

        while len(accumulated_bandwidths) < total_timesteps:
            if self.time_idx >= len(self.cooked_bw):
                # 次のトレースをランダムに選択
                self.reset()

            # 現在の帯域幅を取得
            current_bw = self.cooked_bw[self.time_idx]
            if current_bw > 1.0:  # 帯域幅が1以下の場合は無視
                accumulated_bandwidths.append(current_bw)
            self.time_idx += 1

        # ノイズを適用
        accumulated_bandwidths = [
            bw * PACKET_PAYLOAD_PORTION * np.random.uniform(NOISE_LOW, NOISE_HIGH)
            for bw in accumulated_bandwidths
        ]

        return accumulated_bandwidths
