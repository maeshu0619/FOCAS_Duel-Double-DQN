import numpy as np
import math
import os
import datetime
from gym import Env
from gym.spaces import Box, Discrete
from environment.merton import MertonBandwidthSimulator

DEFAULT_BITRATE_INDEX = 0
a = 1.0
b = 4.3
c = 1.0

class AFOCASDQN(Env):
    def __init__(self, mu, sigma, a, b, lamda, base_bw, 
                 VIDEO_BIT_RATE, episode_length):
        self.simulator = MertonBandwidthSimulator(episode_length, mu, sigma, a, b, lamda, base_bw)
        self.episode_length = episode_length
        self.bandwidth_series = self.simulator.series_bw
        self.current_frame = 0
        self.current_quality = DEFAULT_BITRATE_INDEX
        self.buffer_size = 0.0  # 秒
        self.VIDEO_BIT_RATE = VIDEO_BIT_RATE
        self.last_quality = DEFAULT_BITRATE_INDEX

        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=1, shape=(3,), dtype=np.float32)

    def _calculate_playing_reward(self, bit_rate):
        """再生報酬の計算"""
        return math.log(bit_rate)

    def _calculate_rebuffering_penalty(self, chunk_size, bandwidth):
        """リバッファリングペナルティの計算"""
        rebuffer_time = max((chunk_size / bandwidth) - self.buffer_size, 0)
        self.buffer_size = max(self.buffer_size - rebuffer_time + 4, 0)  # バッファ更新
        return -rebuffer_time

    def _calculate_smoothness_penalty(self, current_quality, last_quality):
        """スムーズネスペナルティの計算"""
        return -abs(self.VIDEO_BIT_RATE[current_quality] - self.VIDEO_BIT_RATE[last_quality])

    def _log_transmission_rate(self, bandwidth, bitrate):
        """
        伝送レートとビットレートをログファイルに記録する。

        Args:
            bandwidth (float): 現在の伝送レート (bps)。
            bitrate (float): 現在選択されたビットレート (bps)。
        """
        # デバッグの瞬間の日時をファイル名に反映
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"transmission_log_{timestamp}.txt")

        with open(log_file, "a") as f:
            f.write(f"Frame: {self.current_frame}, Bandwidth: {bandwidth:.2f} bps, Bitrate: {bitrate} bps\n")

    def step(self, action):
        if self.current_frame >= len(self.bandwidth_series):
            raise IndexError(f"Current frame index {self.current_frame} exceeds bandwidth series length {len(self.bandwidth_series)}")

        # アクションによるビットレートの変更
        if action == 0 and self.current_quality > 0:
            self.current_quality -= 1
        elif action == 2 and self.current_quality < len(self.VIDEO_BIT_RATE) - 1:
            self.current_quality += 1

        # 現在の帯域幅と選択されたビットレート
        current_bandwidth = self.bandwidth_series[self.current_frame]
        selected_bitrate = self.VIDEO_BIT_RATE[self.current_quality]
        chunk_size = selected_bitrate * 4 / 8  # チャンクサイズ (Bytes)

        # 各報酬要素を計算
        playing_reward = self._calculate_playing_reward(selected_bitrate)
        rebuffering_penalty = self._calculate_rebuffering_penalty(chunk_size, current_bandwidth)
        smoothness_penalty = self._calculate_smoothness_penalty(self.current_quality, self.last_quality)

        # 総合報酬
        reward = (a * playing_reward +
                  b * rebuffering_penalty +
                  c * smoothness_penalty)
        
        # ログ記録
        self._log_transmission_rate(current_bandwidth, selected_bitrate)

        self.last_quality = self.current_quality  # 前回のビットレートを更新

        done = self.current_frame + 1 >= self.episode_length
        if not done:
            self.current_frame += 1

        return self._get_obs(), reward, done, {}

    def reset(self):
        self.current_frame = 0
        self.current_quality = DEFAULT_BITRATE_INDEX
        self.last_quality = DEFAULT_BITRATE_INDEX
        self.buffer_size = 0.0
        self.bandwidth_series = self.simulator._generate_series(self.episode_length)

        return self._get_obs()

    def _get_obs(self):
        """
        現在の観測値を返す
        """
        if self.current_frame >= len(self.bandwidth_series):
            print(f"[ERROR] Current frame {self.current_frame} exceeds bandwidth series length {len(self.bandwidth_series)}")
            return np.array([0, 0, 0])  # エラー時はデフォルト値を返す
        
        return np.array([
            self.bandwidth_series[self.current_frame] / 1e6,  # 帯域幅をMbpsに正規化
            self.buffer_size / 10,  # バッファサイズを正規化
            self.current_quality / len(self.VIDEO_BIT_RATE)  # ビットレートインデックスを正規化
        ])
