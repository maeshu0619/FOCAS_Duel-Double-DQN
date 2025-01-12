import numpy as np
import os
from gym import Env
from gym.spaces import Discrete, Box
from tqdm import tqdm
import datetime
from system.qoe_cal import qoe_cal, ave_cal
from system.file_setup import file_setup, extract_bitrate_and_resolution
from system.gaze_prediction import gaze_data
from system.graph_plot import generate_cdf_plot, generate_training_plot
from system.action_generate import focas_combination, a_focas_combination
from system.rate_simu import simulate_transmission_rate
from system.load_trace import BandwidthSimulator
from model.model import Actor

class VideoStreamingEnv(Env):
    def __init__(self, mode, train_or_test, latency_file, network_file, 
                 total_timesteps, max_steps_per_episode, latency_constraint, fps, 
                 mu, sigma_ratio, base_band):
        super(VideoStreamingEnv, self).__init__()
        self.mode = mode
        self.train_or_test = train_or_test
        self.latency_file = latency_file
        self.network_file = network_file
        self.total_timesteps = total_timesteps
        self.max_steps_per_episode = max_steps_per_episode
        self.latency_constraint = latency_constraint #* fps
        self.mu = mu
        self.sigma_ratio = sigma_ratio
        self.base_band = base_band
        
        self.target_q_value = None

        if mode == 0:
            mode_name = "ABR"
            print(f'action length is 4')
        elif mode == 1:
            mode_name = "FOCAS"
            self.action_comb = focas_combination() # 行動範囲の組み合わせ
            self.focas_bitrate_index = 1
            print(f'action length is {len(self.action_comb)}, focas video resolution index: {self.focas_bitrate_index}')
        elif mode == 2:
            mode_name = "A-FOCAS"
            self.action_comb = a_focas_combination() # 行動範囲の組み合わせ
            print(f'action length is {len(self.action_comb)}')
        if self.train_or_test == 0:
            output_file="graph_train"
        elif self.train_or_test == 1:
            output_file="graph_test"
        self.current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 現在の時刻を取得    
        self.graph_file_path = os.path.join(output_file, f"{mode_name}/{self.latency_file}/{self.network_file}/{self.current_time}") # 出力フォルダの作成

        # レイテンシ制約超過のエラー比率計算
        self.error_late = 0
        self.error_late_per = []
        # 伝送レート超過のエラー比率計算
        self.error_buffer = 0
        self.error_buffer_per = []

        self.min_dis = 200
        self.max_dis = 1000
        self.distance = np.random.uniform(self.min_dis, self.max_dis)

        # 帯域幅シミュレート
        self.bandwidth_list = simulate_transmission_rate(self.distance, self.max_steps_per_episode, self.mu, self.sigma_ratio, self.base_band)

        # 行動
        self.actor = Actor(mode)

        self.bitrate_to_resolution = {
            500: (540, 960),
            1000: (750, 1333),
            2000: (1080, 1920),
            4000: (1500, 2666)
            #8000: (2160, 3840)
        }
        self.max_block = 10 # ResBlockの最大個数
        self.max_scale = 9 # 各領域サイズ数

        # ResBlock1ピクセル当たり一層通過させるのに必要な計算時間と、それによって何倍解像度が向上するかの情報
        self.resblock_time = 2.525720165e-8 #2.525720165e-8 # 1.078679591e-9 # 4.487906e-8
        self.other_time = 0 #0.04784066667 # 0.0167315
        self.resblock_quality = 1.148698354997035
        self.resblock_info = (self.resblock_time, self.other_time, self.resblock_quality)

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
        '''
        if mode == 1:
            self.bitrate_legacy.append(self.bitrate_list[self.focas_bitrate_index])  # 初期ビットレートを設定
        else:
            self.bitrate_legacy.append(self.bitrate_list[0])  # 初期ビットレートを設定
        '''
        self.q_values = []  # Q値の履歴を保持
        self.action_history = [] # 行動の履歴
        self.reward_history = [] # 報酬の履歴
        self.reward_ave_history = [] # 各エピソードの報酬の平均の履歴
        self.bandwidth_ave_history = [] # 各エピソードの伝送レートの平均の履歴
        
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
        self.log_file, self.debug_log, self.logger = file_setup(self.mode, self.train_or_test, self.current_time, self.latency_file, self.network_file)

        self.time_in_training = 0 # 初期化しない
        self.steps_per_episode = 0 # エピソード終了時に初期化

    def reset(self):
        self.bitrate_legacy = []
        
        if self.steps_per_episode != 0:
            # レイテンシ制約超過記録
            self.error_late_per.append(100*self.error_late/self.steps_per_episode)
            self.error_late = 0
            # 帯域幅超過記録
            self.error_buffer_per.append(100*self.error_buffer/self.steps_per_episode)
            self.error_buffer = 0

        self.steps_per_episode = 0

        if self.mode == 0:
            self.bitrate_legacy.append(self.bitrate_list[0])  # 初期ビットレートを設定
            self.gaze_coordinates = gaze_data(self.directory_path, self.max_steps_per_episode, video_center=(960, 540))        
            self.distance = np.random.uniform(self.min_dis, self.max_dis)
            self.bandwidth_list = simulate_transmission_rate(self.distance, self.max_steps_per_episode, self.mu, self.sigma_ratio, self.base_band)
            state = np.array([self.bandwidth_list[0], self.bitrate_legacy[0]], dtype=np.float32)
        elif self.mode == 1:
            self.bitrate_legacy.append(self.bitrate_list[self.focas_bitrate_index])  # 初期ビットレートを設定
            self.gaze_coordinates = gaze_data(self.directory_path, self.max_steps_per_episode, video_center=(960, 540))  
            self.distance = np.random.uniform(self.min_dis, self.max_dis)
            self.bandwidth_list = simulate_transmission_rate(self.distance, self.max_steps_per_episode, self.mu, self.sigma_ratio, self.base_band)
            state = np.array([self.bitrate_legacy[0], self.gaze_coordinates[0][0], self.gaze_coordinates[0][1]], dtype=np.float32)
        elif self.mode == 2:
            self.bitrate_legacy.append(self.bitrate_list[0])  # 初期ビットレートを設定
            self.gaze_coordinates = gaze_data(self.directory_path, self.max_steps_per_episode, video_center=(960, 540))        
            self.distance = np.random.uniform(self.min_dis, self.max_dis)
            self.bandwidth_list = simulate_transmission_rate(self.distance, self.max_steps_per_episode, self.mu, self.sigma_ratio, self.base_band)
            state = np.array([self.bandwidth_list[0], self.bitrate_legacy[0], self.gaze_coordinates[0][0], self.gaze_coordinates[0][1]], dtype=np.float32)

        action_invalid_judge = False
        done = False
            
        self.log_file.write(f"--- state reset ---\n")
        self.debug_log.write(f"--- state reset ---\n")
        return state

    def step(self, action, goal_reward):
        self.debug_log.write(f'current step is {self.time_in_training+1} / {self.total_timesteps+1}\n')
        reward = 0

        if self.mode == 0:
            # 帯域幅の変化をシミュレーション
            self.bandwidth_legacy.append(self.bandwidth_list[self.steps_per_episode])
            self.resolution_legacy.append(self.resolution_list[action])

            self.debug_log.write(f'action: {action}\n')
            self.last_bit_rate_index = action # 選択された動画品質
            if self.steps_per_episode != 0:
                self.bitrate_legacy.append(self.bitrate_list[self.last_bit_rate_index])
            self.debug_log.write(f'bitrate: {self.bitrate_list[self.last_bit_rate_index]}\n')
            self.debug_log.write(f'gaze (y,x): ({self.gaze_coordinates[self.steps_per_episode][0]}, {self.gaze_coordinates[self.steps_per_episode][1]})\n')
        elif self.mode == 1:
            # 帯域幅の変化をシミュレーション
            self.bandwidth_legacy.append(self.bandwidth_list[self.steps_per_episode])

            self.debug_log.write(f'action: {action}\n')
            # 行動の数値を各情報に割り当てる
            size_fovea_index = self.action_comb[action][0]
            size_blend_index = self.action_comb[action][1]
            depth_fovea_index = self.action_comb[action][2]
            depth_blend_index = self.action_comb[action][3]
            depth_peri_index = self.action_comb[action][4]
            self.debug_log.write(
                f'size_fovea_index: {size_fovea_index}, size_blend_index: {size_blend_index}, depth_fovea_index: {depth_fovea_index}, depth_blend_index: {depth_blend_index}, depth_peri_index: {depth_peri_index}\n'
                f'gaze (y,x): ({self.gaze_coordinates[self.steps_per_episode][0]}, {self.gaze_coordinates[self.steps_per_episode][1]})\n'
            )
            
            if self.steps_per_episode != 0:
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

            self.debug_log.write(f'action: {action}\n')
            # 行動の数値を各情報に割り当てる
            bitrate_index = self.action_comb[action][0]
            size_fovea_index = self.action_comb[action][1]
            size_blend_index = self.action_comb[action][2]
            depth_fovea_index = self.action_comb[action][3]
            depth_blend_index = self.action_comb[action][4]
            depth_peri_index = self.action_comb[action][5]
            self.debug_log.write(
                f'size_fovea_index: {size_fovea_index}, size_blend_index: {size_blend_index}, depth_fovea_index: {depth_fovea_index}, depth_blend_index: {depth_blend_index}, depth_peri_index: {depth_peri_index}\n'
                f'gaze (y,x): ({self.gaze_coordinates[self.steps_per_episode][0]}, {self.gaze_coordinates[self.steps_per_episode][1]})\n'
            )

            if self.steps_per_episode != 0:
                self.bitrate_legacy.append(self.bitrate_list[bitrate_index])
            self.resolution_legacy.append(self.resolution_list[bitrate_index])
            self.size_legacy.append([int(self.size_list[size_fovea_index] * self.resolution_list[bitrate_index][0]), 
                         int(self.size_list[size_blend_index] * self.resolution_list[bitrate_index][0])])
            self.depth_legacy.append([self.depth_fovea_list[depth_fovea_index], 
                          self.depth_blend_list[depth_blend_index], 
                          self.depth_peri_list[depth_peri_index]])

        # QoE計算
        quality, jitter_t, jitter_s, rebuffer, reward, all_cal_time, action_invalid_judge= qoe_cal(self.mode, self.steps_per_episode, self.time_in_training, self.bitrate_legacy, self.resolution_legacy, 
                                                                self.bitrate_list, self.resolution_list, self.quality_vc_legacy, 
                                                                self.bandwidth_legacy, self.resblock_info, self.gaze_coordinates, 
                                                                self.size_legacy, self.depth_legacy, self.latency_constraint, self.debug_log)
        
        # 無効な行動を記録
        if action_invalid_judge:
            self.error_late += 1 # レイテンシ制約違反回数を記録
            self.actor.add_invalid_action(action)

        if rebuffer > 0:
            self.error_buffer += 1 # 伝送レート超過回数を記録

        # 報酬の内訳の保存
        self.quality_vc_legacy.append(quality)
        self.jitter_t_legacy.append(jitter_t)
        self.jitter_s_legacy.append(jitter_s)
        self.rebuffer_legacy.append(rebuffer)

        # 報酬と行動の保存
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # ログに記録
        self.log_file.write(
            f"Step {self.time_in_training+1}({self.steps_per_episode+1}/{self.max_steps_per_episode}): Action={action}, Reward={reward:.2f}\n"
                f"     bandwidth: {self.bandwidth_legacy[self.steps_per_episode]:.2f}, streamed rate: {self.bitrate_legacy[self.steps_per_episode]}, quality: {quality}, jitter_t: {jitter_t}, jitter_s: {jitter_s}, rebuffer: {rebuffer}\n"
        )
        
        # TensorBoard用のカスタム記録
        self.logger.record("Reward/QoE", reward)
        self.logger.record("Selected Bitrate", self.bitrate_legacy[self.steps_per_episode])
        self.logger.record("Bandwidth", self.bandwidth_legacy[self.steps_per_episode])
        self.logger.dump(self.time_in_training)

        # 状態の更新
        self.steps_per_episode += 1
        self.time_in_training += 1
        
        done = (np.mean(self.reward_history) >= goal_reward)  # 目標報酬に達したら終了

        # 状態を更新 # 正規化が必要？
        if self.mode == 0:
            state = np.array([self.bandwidth_legacy[-1], self.bitrate_legacy[-1]], dtype=np.float32)
        elif self.mode == 1:
            state = np.array([self.bandwidth_legacy[-1], self.gaze_coordinates[-1][0], self.gaze_coordinates[-1][1]], dtype=np.float32)
        elif self.mode == 2:
            state = np.array([self.bandwidth_legacy[-1], self.bitrate_legacy[-1], self.gaze_coordinates[-1][0], self.gaze_coordinates[-1][1]], dtype=np.float32)

        self.video_st = False

        if self.steps_per_episode == self.max_steps_per_episode:
            self.reward_ave_history.append(ave_cal(self.reward_history, self.max_steps_per_episode))
            self.bandwidth_ave_history.append(ave_cal(self.bandwidth_legacy, self.max_steps_per_episode))
            self.debug_log.write(
                f'Average Reward per Erisode: {self.reward_ave_history[-1]}'
                f'Average Bandwdth per Erisode: {self.bandwidth_ave_history[-1]}'
            )

        if self.time_in_training == self.total_timesteps:
            generate_training_plot(self.mode, self.graph_file_path, 
                                   self.latency_file, self.network_file, 
                                   self.reward_ave_history, self.error_late_per, self.error_buffer_per, self.bandwidth_ave_history, self.bitrate_legacy)
            generate_cdf_plot(self.mode, self.graph_file_path, 
                                   self.latency_file, self.network_file, 
                                   self.reward_history, self.quality_vc_legacy, self.jitter_t_legacy, self.jitter_s_legacy, self.rebuffer_legacy)

        self.debug_log.write(f"\n")
        return state, reward, all_cal_time, done
    
    def __del__(self):
        if hasattr(self, 'logger') and self.logger:
            self.logger = None
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
