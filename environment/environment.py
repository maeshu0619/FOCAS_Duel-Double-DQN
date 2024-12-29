import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from tqdm import tqdm
from system.calculater import compute_average, qoe_cal, rate_cal
from system.setup import file_setup

class VideoStreamingEnv(Env):
    def __init__(self, total_timesteps, video_length: int, segment_duration: int, video_fps: int):
        super(VideoStreamingEnv, self).__init__()
        self.total_timesteps = total_timesteps
        self.video_length = video_length
        self.segment_duration = segment_duration
        self.num_segments = video_length // segment_duration
        self.video_fps = video_fps

        self.bit_rates = [300, 750, 1200, 1850, 2850, 4300]  # kbps
        self.last_bit_rate_index = 0

        # セグメントの長さ毎に更新
        self.bitrate_legacy = []
        self.bitrate_legacy.append(self.bit_rates[self.last_bit_rate_index])
        self.segment_data = self.segment_duration * self.bit_rates[self.last_bit_rate_index]
        self.segmnet_cnt = 0 # セグメント選択(ビットレート選択、4ステップごとに記録)
        self.bandwidth_legacy = [] # 今までシミュレートした伝送レート履歴
        #self.T_send = []
        self.video_st = True

        # 正規分布を用いた伝送レートのシミュレーション
        self.mean_bandwidth = 3000  # 平均帯域幅（kbps）
        self.std_bandwidth = 1000  # 帯域幅の標準偏差（kbps）
        
        self.max_bandwidth_history = int(self.total_timesteps / self.segment_duration)

        # 状態空間: [現在の帯域幅, 前回選択したビットレート]
        self.observation_space = Box(
            low=0,
            high=np.inf,
            shape=(self.max_bandwidth_history + 1,),  # 最大帯域幅履歴 + 現在のビットレート
            dtype=np.float32
        )

        # 状態の更新
        #next_state = np.array(self.bandwidth_legacy + [self.bitrate_legacy[self.last_bit_rate_index]], dtype=np.float32)

        # 行動空間: 解像度を「上げる」「そのまま」「下げる」の3択
        self.action_space = Discrete(3)

        # ログファイルのセットアップ
        self.log_file, self.logger = file_setup()

        # プログレスバーのセットアップ
        self.total_timesteps = None
        self.progress_bar = None

        self.time_in_video = 0 # 100ステップごとに初期化
        self.time_in_segment = 0 # 4ステップごとに初期化

        self.time_in_training = 0 # 初期化しない

    def reset(self):
        #self.bandwidth_legacy = []  # 帯域幅履歴をリセット
        #self.bitrate_legacy = [self.bit_rates[0]]  # 初期化
        #self.time_in_video = 0
        #self.segmnet_cnt = 0
        #bandwidth = rate_cal(self.time_in_video)
        #self.video_st = True

        self.time_in_segment = 0
        # 状態をゼロパディング
        padded_bandwidth_legacy = self.bandwidth_legacy + [0] * (self.max_bandwidth_history - len(self.bandwidth_legacy))
        state = np.array(padded_bandwidth_legacy + [self.bitrate_legacy[0]], dtype=np.float32)

        done = False
        
        if self.progress_bar:
            self.progress_bar.n = 0
            self.progress_bar.last_print_n = 0
            self.progress_bar.refresh()
            
        self.log_file.write(f"--- state reset ---\n")
        return state

    def step(self, action):
        # 帯域幅の変化をシミュレーション
        bandwidth = rate_cal(self.time_in_video)
        self.bandwidth_legacy.append(bandwidth)
        reward = 0
                
        if self.time_in_segment == 0:
            if self.video_st == True:
                reward = 0
                pass
            else:
                if action == 0:  # 解像度を下げる
                    self.last_bit_rate_index = max(0, self.last_bit_rate_index - 1)
                elif action == 2:  # 解像度を上げる
                    self.last_bit_rate_index = min(len(self.bit_rates)-1, self.last_bit_rate_index + 1)  # 最大値は選択可能インデックス以下
                else:
                    pass
                          
            self.bitrate_legacy.append(self.bit_rates[self.last_bit_rate_index])

            # 選択されたビットレートにおけるセグメントデータ量の計算
            self.segment_data = self.segment_duration * self.bitrate_legacy[self.segmnet_cnt]
            
            # QoE計算
            reward = qoe_cal(self.time_in_video, self.segmnet_cnt, self.bitrate_legacy, self.bandwidth_legacy)
            if self.time_in_video == 0:
                self.log_file.write(f"segmnet num(init): {self.segmnet_cnt}, segment data:{self.segment_data}, reward: {reward}\n")
            else:
                self.log_file.write(f"segmnet num({self.time_in_video-4}~{self.time_in_video-1}): {self.segmnet_cnt}, segment data:{self.segment_data}, reward: {reward}\n")
            
        # ログに記録
        self.log_file.write(f"frame:{self.time_in_video}, bandwidth:{bandwidth:.2f}, streamed rate:{self.bitrate_legacy[self.segmnet_cnt]}\n")
        

        # TensorBoard用のカスタム記録
        self.logger.record("Reward/QoE", reward)
        self.logger.record("Selected Bitrate", self.bitrate_legacy[self.segmnet_cnt])
        self.logger.record("Bandwidth", bandwidth)
        # 任意の横軸（例えば秒単位、またはセグメント単位）
        self.logger.dump(self.time_in_training)

        # プログレスバーの更新
        if self.progress_bar:
            self.progress_bar.update(1)

        # 状態の更新
        self.time_in_training += 1
        self.time_in_segment += 1
        self.time_in_video += 1

        # セグメント終了時の処理
        if self.time_in_segment == self.segment_duration:
            self.time_in_segment = 0
            self.segmnet_cnt += 1

        # ビデオ終了時の処理
        if self.time_in_video >= self.video_length:
            self.time_in_video = 0
            self.bandwidth_legacy = []  # 帯域幅履歴をリセット
            self.bitrate_legacy = [self.bit_rates[0]]  # 初期化
            self.segmnet_cnt = 0
            self.video_st = True
            reward = 0
            done = True
        else:
            done = False

        # 状態をゼロパディング
        padded_bandwidth_legacy = self.bandwidth_legacy + [0] * (self.max_bandwidth_history - len(self.bandwidth_legacy))
        next_state = np.array(padded_bandwidth_legacy + [self.bitrate_legacy[-1]], dtype=np.float32)

        self.video_st = False

        return next_state, reward, done, {}

    def set_total_timesteps(self, total_timesteps):
        """
        総ステップ数を設定し、プログレスバーを初期化する。
        """
        self.total_timesteps = total_timesteps
        self.progress_bar = tqdm(total=total_timesteps, desc="Training Progress")

    def render(self, mode='human'):
        pass

    def __del__(self):
        """
        環境が破棄されるときにロガーを終了し、ログファイルを閉じる。
        """
        if hasattr(self, 'logger') and self.logger:
            self.logger = None
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
