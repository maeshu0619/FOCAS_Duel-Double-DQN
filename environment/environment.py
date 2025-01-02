import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from tqdm import tqdm
from system.calculater import compute_average, qoe_cal, rate_cal
from system.setup import file_setup, extract_bitrate_and_resolution, A_FOCAS_assign_parameters
from system.gaze_prediction import gaze_data
from model.model import Actor

log_cnt = 0

class VideoStreamingEnv(Env):
    def __init__(self, mode, total_timesteps, max_steps_per_episode, latency_constraint):
        super(VideoStreamingEnv, self).__init__()
        self.mode = mode
        self.total_timesteps = total_timesteps
        self.max_steps_per_episode = max_steps_per_episode
        self.latency_constraint = latency_constraint
        
        self.actor = Actor(mode)

        self.bitrate_to_resolution = {
            200: (135, 245),   # 超低解像度
            1000: (270, 490),  # SD解像度
            2500: (540, 980), # HD解像度
            4300: (1080, 1920), # フルHD解像度
            # 8000: (2160, 3840)  # UHD解像度（4K）
        }
        self.max_block = 10 # ResBlockの最大個数
        self.max_scale = 9

        # ResBlock1ピクセル当たり一層通過させるのに必要な計算時間と、それによって何倍解像度が向上するかの情報
        self.resblock_time = 5.2389490086734666 * 10**(-8)
        self.resblock_quality = 1.148698354997035
        self.resblock_info = (self.resblock_time, self.resblock_quality)

        self.bitrate_list, self.resolution_list = extract_bitrate_and_resolution(self.bitrate_to_resolution)
        self.depth_list = [1,2,3,4,5,6,7,8,9,10] # 深度リスト
        self.size_list = [x / 18 for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] # サイズリスト

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
        self.bitrate_legacy.append(self.bitrate_list[0])  # 初期ビットレートを設定
        
        self.q_values = []  # Q値の履歴を保持
        self.action_history = [] # 行動の履歴
        self.reward_history = [] # 報酬の履歴

        self.max_bandwidth_history = min(total_timesteps, 100) 
        
        # 視線情報の取得
        directory_path = "UD_UHD_EyeTrakcing_Videos/Gaze_Data/HD"
        self.gaze_coordinates = gaze_data(directory_path, total_timesteps, video_center=(960, 540))
        print("Gaze Cordinate cathed\n")

        if self.mode == 0:
            self.action_space = Discrete(4)
            self.observation_space = Box(
                low=0,
                high=np.inf,
                shape=(2,), # 現在の帯域幅，選択された品質
                dtype=np.float32
            )
        elif self.mode == 1:
            self.action_space = Discrete(5400)
        elif self.mode == 2:
            self.action_space = Discrete(21600)
            self.observation_space = Box(
                low=0,
                high=np.inf,
                shape=(2,), # 現在の帯域幅，選択された品質
                dtype=np.float32
            )

        # ログファイルのセットアップ
        self.log_file, self.logger = file_setup(self.mode)

        # プログレスバーのセットアップ
        self.total_timesteps = None
        self.progress_bar = None

        self.time_in_training = 0 # 初期化しない
        self.steps_per_episode = 0 # エピソード終了時に初期化

    def reset(self):
        self.steps_per_episode = 0

        # 帯域幅履歴とビットレート履歴をリセット
        self.bandwidth_legacy = []  # 帯域幅の履歴を初期化
        self.bitrate_legacy = [self.bitrate_list[0]]  # 初期ビットレートを設定

        state = np.array([0, self.bitrate_legacy[0]], dtype=np.float32)  # 初期帯域幅は0
        episode_fin = False
        done = False
            
        self.log_file.write(f"--- state reset ---\n")
        return state

    def step(self, action):
        # 帯域幅の変化をシミュレーション
        bandwidth, r = rate_cal(self.steps_per_episode) # 帯域幅とその時
        self.bandwidth_legacy.append(bandwidth)
        reward = 0

        if self.mode == 0:
            if self.steps_per_episode != 0:
                self.last_bit_rate_index = action # 選択された動画品質
                self.bitrate_legacy.append(self.bitrate_list[self.last_bit_rate_index])
        elif self.mode == 1:
            pass
        elif self.mode == 2:
            print(f'action: {action}')
            bitrate_index, size_fovea_index, size_blend_index, depth_fovea_index, depth_blend_index, depth_peri_index = A_FOCAS_assign_parameters(action)
            print(f'''bitrate_index: {bitrate_index}, size_fovea_index: {size_fovea_index}, size_blend_index: {size_blend_index}, depth_fovea_index: {depth_fovea_index}, depth_blend_index: {depth_blend_index}, depth_peri_index: {depth_peri_index}''')
            print(f'gaze_y: {self.gaze_coordinates[self.time_in_training][0]}, gaze_x: {self.gaze_coordinates[self.time_in_training][1]}')
            
            self.bitrate_legacy.append(self.bitrate_list[bitrate_index])
            self.resolution_legacy.append(self.resolution_list[bitrate_index])
            self.size_legacy.append([int(self.size_list[size_fovea_index] * self.resolution_list[bitrate_index][0]), 
                         int(self.size_list[size_blend_index] * self.resolution_list[bitrate_index][0])])
            self.depth_legacy.append([self.depth_list[depth_fovea_index], 
                          self.depth_list[depth_blend_index], 
                          self.depth_list[depth_peri_index]])

        # QoE計算
        quality, jitter_t, jitter_s, reward , episode_fin= qoe_cal(self.mode, self.steps_per_episode, self.bitrate_legacy, self.resolution_legacy, 
                                                                self.bitrate_list, self.resolution_list, self.quality_vc_legacy, 
                                                                self.bandwidth_legacy, self.resblock_info, self.gaze_coordinates, 
                                                                self.size_legacy, self.depth_legacy, self.latency_constraint)
        
        # 無効な行動を記録
        if episode_fin:
            self.actor.add_invalid_action(action)

        # 報酬の内訳の保存
        self.quality_vc_legacy.append(quality)
        self.jitter_t_legacy.append(jitter_t)
        self.jitter_s_legacy.append(jitter_s)

        # 報酬と行動の保存
        self.action_history.append(action)
        self.reward_history.append(reward)

        # 次のターゲットQ値を計算して保存
        target_q_value = reward + self.compute_discounted_future_reward()
        self.q_values.append(target_q_value)
        print(f'reward: {reward},self.compute_discounted_future_reward(): {self.compute_discounted_future_reward()},target_q_value: {target_q_value}')

        # ログに記録
        self.log_file.write(
            f"Step {self.time_in_training}({self.steps_per_episode+1}/{self.max_steps_per_episode}): Action={action}, Reward={reward:.2f}, Target Q={target_q_value:.2f}\n"
        )           
        self.log_file.write(
            f"     bandwidth: {bandwidth:.2f}, streamed rate: {self.bitrate_legacy[self.steps_per_episode]}, quality: {quality}, jitter_t: {jitter_t}, jitter_s: {jitter_s}\n"
        )
        
        # TensorBoard用のカスタム記録
        self.logger.record("Reward/QoE", reward)
        self.logger.record("Selected Bitrate", self.bitrate_legacy[self.steps_per_episode])
        self.logger.record("Bandwidth", bandwidth)
        self.logger.dump(self.time_in_training)

        # 状態の更新
        self.steps_per_episode += 1
        self.time_in_training += 1
        
        done = (self.steps_per_episode >= self.max_steps_per_episode)  # ビデオ長に達したら終了
        print(f'environment time_in_training: {self.time_in_training}')

        # 状態を更新
        state = np.array([self.bandwidth_legacy[-1], self.bitrate_legacy[-1]], dtype=np.float32)

        self.video_st = False

        # info辞書にQ値を追加
        info = {
            "q_values": self.q_values[-1]  # 最新のQ値を追加
        }
        
        return state, reward, done, info

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
