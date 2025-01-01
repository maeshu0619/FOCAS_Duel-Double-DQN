import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from tqdm import tqdm
from system.calculater import compute_average, qoe_cal, rate_cal
from system.setup import file_setup, extract_bitrate_and_resolution
from system.gaze_prediction import gaze_data

log_cnt = 0

class VideoStreamingEnv(Env):
    def __init__(self, mode, total_timesteps, max_steps_per_episode, latency_constraint):
        super(VideoStreamingEnv, self).__init__()
        self.mode = mode
        self.total_timesteps = total_timesteps
        self.max_steps_per_episode = max_steps_per_episode
        self.latency_constraint = latency_constraint

        self.bitrate_to_resolution = {
            200: (144, 256),   # 超低解像度
            1000: (480, 640),  # SD解像度
            2500: (720, 1280), # HD解像度
            4300: (1080, 1920), # フルHD解像度
            8000: (2160, 3840)  # UHD解像度（4K）
        }
        # ResBlock1ピクセル当たり一層通過させるのに必要な計算時間と、それによって何倍解像度が向上するかの情報
        self.resblock_time = 0.2 * 10**(-8)
        self.resblock_quality = 1.2
        self.resblock_info = (self.resblock_time, self.resblock_quality)

        self.bitrate_list, self.resolution_list = extract_bitrate_and_resolution(self.bitrate_to_resolution)
        self.depth_fovea = [7,8,9]
        self.depth_blend = [4,5,6]
        self.depth_peri = [1,2,3]
        self.size_fovea = [4/27, 6/27] # 縦の長さと半径の比
        self.size_blend = [8/27, 10/27] # 縦の長さと半径の比

        self.gaze_coordinates = [] # 視線情報の取得
        self.video_st = True # 動画開始

        self.quality_vc_legacy = [] # 報酬の動画品質の履歴
        self.jitter_t_legacy = [] # 報酬の時間ジッタの履歴
        self.jitter_s_legacy = [] # 報酬の空間ジッタの履歴

        self.bitrate_legacy = [] # 選択された品質の履歴
        self.resolution_legacy = [] # 選択された動画サイズの履歴
        self.size_legacy = [] # 選択された領域サイズの履歴
        self.depth_legacy = [] # 選択された深さの履歴
        self.bandwidth_legacy = [] # シミュレートした帯域幅の履歴
        
        self.q_values = []  # Q値の履歴を保持
        self.action_history = [] # 行動の履歴
        self.reward_history = [] # 報酬の履歴

        self.max_bandwidth_history = min(total_timesteps, 100) 
        # 状態空間: [現在の帯域幅, 前回選択したビットレート]
        self.observation_space = Box(
            low=0,
            high=np.inf,
            shape=(self.max_bandwidth_history + 1,),  # 最大帯域幅履歴 + 現在のビットレート
            dtype=np.float32
        )
        
        # 視線情報の取得
        directory_path = "UD_UHD_EyeTrakcing_Videos/Gaze_Data/HD"
        self.gaze_coordinates = gaze_data(directory_path, total_timesteps, video_center=(960, 540))
        print("Gaze Cordinate cathed")

        if mode == 0:
            self.action_space = Discrete(5)
        elif mode == 1:
            self.action_space = Discrete(108)
        elif mode == 2:
            self.action_space = Discrete(540)


        # ログファイルのセットアップ
        self.log_file, self.logger = file_setup()

        # プログレスバーのセットアップ
        self.total_timesteps = None
        self.progress_bar = None

        self.time_in_video = 0 # 100ステップごとに初期化
        self.time_in_training = 0 # 初期化しない

    def reset(self):        
        # 状態をゼロパディング
        padded_bandwidth_legacy = self.bandwidth_legacy + [0] * (self.max_bandwidth_history - len(self.bandwidth_legacy))
        state = np.array(padded_bandwidth_legacy + [self.bitrate_legacy[0]], dtype=np.float32)

        done = False
            
        self.log_file.write(f"--- state reset ---\n")
        return state

    def step(self, action):
        # 帯域幅の変化をシミュレーション
        bandwidth, r = rate_cal(self.time_in_video)
        self.bandwidth_legacy.append(bandwidth)
        reward = 0

        if self.mode == 0:
            self.last_bit_rate_index = action
            self.bitrate_legacy.append(self.bitrate_list[self.last_bit_rate_index])
        elif self.mode == 1:
            size_fovea_index = (action // 5) % 2
            size_blend_index = (action // (5 * 2)) % 2
            depth_fovea_index = (action // (5 * 2 * 2)) % 3
            depth_blend_index = (action // (5 * 2 * 2 * 3)) % 3
            depth_peri_index = action // (5 * 2 * 2 * 3 * 3)
        elif self.mode == 2:
            bitrate_index = action % 5
            size_fovea_index = (action // 5) % 2
            size_blend_index = (action // (5 * 2)) % 2
            depth_fovea_index = (action // (5 * 2 * 2)) % 3
            depth_blend_index = (action // (5 * 2 * 2 * 3)) % 3
            depth_peri_index = action // (5 * 2 * 2 * 3 * 3)
            print(f'{size_fovea_index}, {size_blend_index}, {depth_fovea_index}, {depth_blend_index}, {depth_peri_index}')
            
            self.bitrate_legacy.append(self.bitrate_list[bitrate_index])
            self.resolution_legacy.append(self.resolution_list[bitrate_index])
            self.size_legacy.append[int(self.size_fovea[size_fovea_index] * self.resolution_list[bitrate_index][0]), 
                         int(self.size_blend[size_blend_index] * self.resolution_list[bitrate_index][0])]
            self.depth_legacy.append[self.depth_fovea[depth_fovea_index], 
                          self.depth_blend[depth_blend_index], 
                          self.depth_peri[depth_peri_index]]

                       
        self.bitrate_legacy.append(self.bitrate_list[self.last_bit_rate_index])

        # QoE計算
        quality, jitter_t, jitter_s, reward = qoe_cal(self.mode, self.time_in_video, self.bitrate_legacy, self.resolution_legacy, 
                                    self.bitrate_list, self.resolution_list,
                                    self.bandwidth_legacy, self.resblock_info, self.gaze_coordinates, 
                                    self.size_legacy, self.depth_legacy, self.latency_constraint)
        
        self.action_history.append(action)
        self.reward_history.append(reward)

        self.quality_vc_legacy.append(quality)
        self.jitter_t_legacy.append(jitter_t)
        self.jitter_s_legacy.append(jitter_s)

        # 次のターゲットQ値を計算して保存
        target_q_value = reward + self.compute_discounted_future_reward()
        self.q_values.append(target_q_value)

        # ログに記録
        self.log_file.write(f"frame:{self.time_in_video}, bandwidth:{bandwidth:.2f}, distance: {r}, streamed rate:{self.bitrate_legacy[self.time_in_video]}\n")
        self.log_file.write(
            f"Step {self.time_in_training}: Action={action}, Reward={reward:.2f}, Target Q={target_q_value:.2f}\n"
        )
        
        # TensorBoard用のカスタム記録
        self.logger.record("Reward/QoE", reward)
        self.logger.record("Selected Bitrate", self.bitrate_legacy[self.time_in_video])
        self.logger.record("Bandwidth", bandwidth)
        self.logger.dump(self.time_in_training)

        # 状態の更新
        self.time_in_training += 1
        self.time_in_video += 1
        
        done = (self.time_in_video >= self.max_steps_per_episode)  # ビデオ長に達したら終了
        print(f'environment time_in_training: {self.time_in_training}')

        # 状態をゼロパディング
        padded_bandwidth_legacy = self.bandwidth_legacy + [0] * (self.max_bandwidth_history - len(self.bandwidth_legacy))
        next_state = np.array(padded_bandwidth_legacy + [self.bitrate_legacy[-1]], dtype=np.float32)

        self.video_st = False

        # info辞書にQ値を追加
        info = {
            "q_values": self.q_values[-1]  # 最新のQ値を追加
        }
        
        return next_state, reward, done, info

    def compute_discounted_future_reward(self, gamma=0.99):
        """
        割引率gammaを用いて、将来の割引報酬を計算。
        """
        discounted_reward = 0.0
        for idx, r in enumerate(reversed(self.reward_history)):
            discounted_reward += (gamma ** idx) * r
        return discounted_reward
    
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
