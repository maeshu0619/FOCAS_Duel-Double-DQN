import numpy as np
import os
from gym import Env
from gym.spaces import Discrete, Box
from tqdm import tqdm
import datetime
from system.calculater import qoe_cal, rate_cal
from system.setup import file_setup, extract_bitrate_and_resolution
from system.gaze_prediction import gaze_data
from system.graph_plot import generate_cdf_plot, generate_training_plot
from system.action_generate import focas_combination, a_focas_combination
from system.load_trace import BandwidthSimulator
from model.model import Actor

log_cnt = 0

class VideoStreamingEnv(Env):
    def __init__(self, mode, total_timesteps, max_steps_per_episode, latency_constraint):
        super(VideoStreamingEnv, self).__init__()
        self.mode = mode
        self.total_timesteps = total_timesteps
        self.max_steps_per_episode = max_steps_per_episode
        self.latency_constraint = latency_constraint
        
        self.target_q_value = None
        self.info = {}

        if mode == 0:
            mode_name = "0_ABR"
            print(f'action length is 4')
        elif mode == 1:
            mode_name = "1_FOCAS"
            self.action_comb = focas_combination() # 行動範囲の組み合わせ
            self.focas_bitrate_index = 0
            print(f'action length is {len(self.action_comb)}, focas video resolution index: {self.focas_bitrate_index}')
        elif mode == 2:
            mode_name = "2_A-FOCAS"
            self.action_comb = a_focas_combination() # 行動範囲の組み合わせ
            print(f'action length is {len(self.action_comb)}')
        output_file="graph_plots"
        self.current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 現在の時刻を取得    
        self.cdf_file_path = os.path.join(output_file, f"{mode_name}/{self.current_time}") # 出力フォルダの作成

        if self.mode == 0 or 2:
            # 帯域幅シミュレート
            self.simulator = BandwidthSimulator('./cooked_traces/')
            self.bandwidth_list = self.simulator.simulate_total_timesteps(self.max_steps_per_episode)

        # 行動
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
        self.depth_fovea_list = [6,7,8,9,10] # フォビア深度リスト
        self.depth_blend_list = [3,4,5,6,7] # ブレンド深度リスト
        self.depth_peri_list = [1,2,3,4,5] # 周辺深度リスト
        self.size_list = [x / 12 for x in [0,1,2,3,4,5]] # サイズリスト

        self.gaze_coordinates = [] # 視線情報の取得
        self.video_st = True # 動画開始

        self.quality_vc_legacy = [] # 報酬の動画品質の履歴
        self.jitter_t_legacy = [] # 報酬の時間ジッタの履歴
        self.jitter_s_legacy = [] # 報酬の空間ジッタの履歴
        self.rebuffer_legacy = [] # 報酬のリバッファリングペナルティ

        self.bitrate_legacy = [] # 選択された品質の履歴
        self.resolution_legacy = [] # 選択された動画サイズの履歴
        self.size_legacy = [] # 選択された領域サイズの履歴
        self.depth_legacy = [] # 選択された深さの履歴
        self.bandwidth_legacy = [] # シミュレートした帯域幅の履歴
        self.bitrate_legacy.append(self.bitrate_list[0])  # 初期ビットレートを設定
        
        self.q_values = []  # Q値の履歴を保持
        self.action_history = [] # 行動の履歴
        self.reward_history = [] # 報酬の履歴
        
        # 視線情報の取得
        self.directory_path = "UD_UHD_EyeTrakcing_Videos/Gaze_Data/HD"
        self.gaze_coordinates = gaze_data(self.directory_path, max_steps_per_episode, video_center=(960, 540))
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
            self.action_space = Discrete(1200)
            self.observation_space = Box(
                low=0,
                high=np.inf,
                shape=(3,), # 選択された品質、視線座標
                dtype=np.float32
            )
        elif self.mode == 2:
            self.action_space = Discrete(4800)
            self.observation_space = Box(
                low=0,
                high=np.inf,
                shape=(4,), # 現在の帯域幅，選択された品質，現在の視線座標
                dtype=np.float32
            )

        # ログファイルのセットアップ
        self.log_file, self.logger = file_setup(self.mode)

        self.time_in_training = 0 # 初期化しない
        self.steps_per_episode = 0 # エピソード終了時に初期化

    def reset(self):
        self.steps_per_episode = 0

        if self.mode == 0:
            self.bandwidth_list = self.simulator.simulate_total_timesteps(self.max_steps_per_episode)
            state = np.array([self.bandwidth_legacy[0], self.bitrate_legacy[0]], dtype=np.float32)
        elif self.mode == 1:
            self.gaze_coordinates = gaze_data(self.directory_path, self.max_steps_per_episode, video_center=(960, 540))
            state = np.array([self.bitrate_legacy[0], self.gaze_coordinates[0][0], self.gaze_coordinates[0][1]], dtype=np.float32)
        elif self.mode == 2:
            self.gaze_coordinates = gaze_data(self.directory_path, self.max_steps_per_episode, video_center=(960, 540))
            self.bandwidth_list = self.simulator.simulate_total_timesteps(self.max_steps_per_episode)
            state = np.array([self.bandwidth_legacy[0], self.bitrate_legacy[0], self.gaze_coordinates[0][0], self.gaze_coordinates[0][1]], dtype=np.float32)

        episode_fin = False
        done = False
            
        self.log_file.write(f"--- state reset ---\n")
        return state

    def step(self, action):
        reward = 0

        if self.mode == 0:
            # 帯域幅の変化をシミュレーション
            self.bandwidth_legacy.append(self.bandwidth_list[self.steps_per_episode])

            print(f'action: {action}')
            self.last_bit_rate_index = action # 選択された動画品質
            self.bitrate_legacy.append(self.bitrate_list[self.last_bit_rate_index])
        elif self.mode == 1:
            # 帯域幅の変化をシミュレーション
            self.bandwidth_legacy.append(None)

            print(f'action: {action}')
            # 行動の数値を各情報に割り当てる
            size_fovea_index = self.action_comb[action][0]
            size_blend_index = self.action_comb[action][1]
            depth_fovea_index = self.action_comb[action][2]
            depth_blend_index = self.action_comb[action][3]
            depth_peri_index = self.action_comb[action][4]
            print(f'size_fovea_index: {size_fovea_index}, size_blend_index: {size_blend_index}, depth_fovea_index: {depth_fovea_index}, depth_blend_index: {depth_blend_index}, depth_peri_index: {depth_peri_index}')
            print(f'gaze_y: {self.gaze_coordinates[self.time_in_training][0]}, gaze_x: {self.gaze_coordinates[self.time_in_training][1]}')
            
            self.bitrate_legacy.append(self.bitrate_list[self.focas_bitrate_index])
            self.resolution_legacy.append(self.resolution_list[self.focas_bitrate_index])
            self.size_legacy.append([int(self.size_list[size_fovea_index] * self.resolution_list[self.focas_bitrate_index][0]), 
                         int(self.size_list[size_blend_index] * self.resolution_list[self.focas_bitrate_index][0])])
            self.depth_legacy.append([self.depth_fovea_list[depth_fovea_index], 
                          self.depth_blend_list[depth_blend_index], 
                          self.depth_peri_list[depth_peri_index]])
        elif self.mode == 2:
            # 帯域幅の変化をシミュレーション
            self.bandwidth_legacy.append(self.bandwidth_list[self.steps_per_episode])

            print(f'action: {action}')
            # 行動の数値を各情報に割り当てる
            bitrate_index = self.action_comb[action][0]
            size_fovea_index = self.action_comb[action][1]
            size_blend_index = self.action_comb[action][2]
            depth_fovea_index = self.action_comb[action][3]
            depth_blend_index = self.action_comb[action][4]
            depth_peri_index = self.action_comb[action][5]
            print(f'bitrate_index: {bitrate_index}, size_fovea_index: {size_fovea_index}, size_blend_index: {size_blend_index}, depth_fovea_index: {depth_fovea_index}, depth_blend_index: {depth_blend_index}, depth_peri_index: {depth_peri_index}')
            print(f'gaze_y: {self.gaze_coordinates[self.time_in_training][0]}, gaze_x: {self.gaze_coordinates[self.time_in_training][1]}')
            
            self.bitrate_legacy.append(self.bitrate_list[bitrate_index])
            self.resolution_legacy.append(self.resolution_list[bitrate_index])
            self.size_legacy.append([int(self.size_list[size_fovea_index] * self.resolution_list[bitrate_index][0]), 
                         int(self.size_list[size_blend_index] * self.resolution_list[bitrate_index][0])])
            self.depth_legacy.append([self.depth_fovea_list[depth_fovea_index], 
                          self.depth_blend_list[depth_blend_index], 
                          self.depth_peri_list[depth_peri_index]])

        # QoE計算
        quality, jitter_t, jitter_s, rebuffer, reward , episode_fin= qoe_cal(self.mode, self.steps_per_episode, self.time_in_training, self.bitrate_legacy, self.resolution_legacy, 
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
        self.rebuffer_legacy.append(rebuffer)

        # 報酬と行動の保存
        self.action_history.append(action)
        self.reward_history.append(reward)

        if self.time_in_training >= 60:
            if self.time_in_training % 5 == 0:
                self.target_q_value = reward + self.compute_discounted_future_reward()
                self.q_values.append(self.target_q_value)

                # info辞書にQ値を追加
                self.info = {
                    "q_values": self.q_values[-1]  # 最新のQ値を追加
                }

        print(f'reward: {reward}, discounted future reward: {self.compute_discounted_future_reward()},target q value: {self.target_q_value}')
        
        # ログに記録
        if self.target_q_value is not None:
            self.log_file.write(
                f"Step {self.time_in_training+1}({self.steps_per_episode+1}/{self.max_steps_per_episode}): Action={action}, Reward={reward:.2f}, Target Q={self.target_q_value:.2f}\n"
            )
        else:
            self.log_file.write(
                f"Step {self.time_in_training+1}({self.steps_per_episode+1}/{self.max_steps_per_episode}): Action={action}, Reward={reward:.2f}, Target Q=None\n"
            )         
        if self.bandwidth_legacy[self.steps_per_episode] == None:
            self.log_file.write(
                f"     bandwidth: None, streamed rate: {self.bitrate_legacy[self.steps_per_episode]}, quality: {quality}, jitter_t: {jitter_t}, jitter_s: {jitter_s}\n"
            )
        else:
            self.log_file.write(
                f"     bandwidth: {self.bandwidth_legacy[self.steps_per_episode]:.2f}, streamed rate: {self.bitrate_legacy[self.steps_per_episode]}, quality: {quality}, jitter_t: {jitter_t}, jitter_s: {jitter_s}\n"
            )
        
        # TensorBoard用のカスタム記録
        self.logger.record("Reward/QoE", reward)
        self.logger.record("Selected Bitrate", self.bitrate_legacy[self.steps_per_episode])
        self.logger.record("Bandwidth", self.bandwidth_legacy[self.steps_per_episode])
        self.logger.dump(self.time_in_training)

        # 状態の更新
        self.steps_per_episode += 1
        self.time_in_training += 1
        
        done = (self.steps_per_episode >= self.max_steps_per_episode)  # ビデオ長に達したら終了

        # 状態を更新
        state = np.array([self.bandwidth_legacy[-1], self.bitrate_legacy[-1]], dtype=np.float32)

        if self.mode == 0:
            state = np.array([self.bandwidth_legacy[-1], self.bitrate_legacy[-1]], dtype=np.float32)
        elif self.mode == 1:
            state = np.array([self.bandwidth_legacy[-1], self.gaze_coordinates[-1][0], self.gaze_coordinates[-1][1]], dtype=np.float32)
        elif self.mode == 2:
            state = np.array([self.bandwidth_legacy[-1], self.bitrate_legacy[-1], self.gaze_coordinates[-1][0], self.gaze_coordinates[-1][1]], dtype=np.float32)

        self.video_st = False
        print(f'current step is {self.time_in_training+1} / {self.total_timesteps+1}\n')

        if self.time_in_training == self.total_timesteps:
            generate_training_plot(self.mode, self.cdf_file_path, self.reward_history, self.q_values, self.bandwidth_legacy, self.bitrate_legacy,)
            generate_cdf_plot(self.mode, self.cdf_file_path, self.reward_history, self.quality_vc_legacy, self.jitter_t_legacy, self.jitter_s_legacy, self.rebuffer_legacy)
            self.log_file.write(f'CDF plot done')

        return state, reward, done, self.info

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
