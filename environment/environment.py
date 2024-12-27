import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from tqdm import tqdm
from system.calculater import compute_average, qoe_cal
from system.setup import file_setup

class VideoStreamingEnv(Env):
    def __init__(self, video_length: int, segment_duration: int, video_fps: int):
        super(VideoStreamingEnv, self).__init__()
        self.video_length = video_length
        self.segment_duration = segment_duration
        self.num_segments = video_length // segment_duration
        self.video_fps = video_fps

        self.bit_rates = [300, 750, 1200, 1850, 2850, 4300]  # kbps
        self.current_segment = 0

        # セグメントの長さ毎に更新
        self.bitrate_legacy = []
        self.segment_data = self.segment_duration * self.bit_rates[self.current_segment]
        self.segmnet_cnt = 0 # セグメント選択(ビットレート選択、4ステップごとに記録)
        self.R_t = [] # 今までシミュレートした伝送レート履歴
        self.R_t_box = [] # 今までシミュレートした平均伝送レートの履歴
        self.T_send = []
        self.video_st = True

        self.rate_loss_penalty = 0 # 選択された解像度が平均伝送レートを下回らなかった場合のペナルティ

        # 正規分布を用いた伝送レートのシミュレーション
        self.mean_bandwidth = 3000  # 平均帯域幅（kbps）
        self.std_bandwidth = 1000  # 帯域幅の標準偏差（kbps）

        # 状態空間: [現在の帯域幅, 前回選択したビットレート]
        self.observation_space = Box(
            low=0,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        )

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
        """
        環境のリセット
        Returns:
            np.ndarray: 初期状態
        """
        self.current_segment = 0
        bandwidth = np.random.normal(self.mean_bandwidth, self.std_bandwidth)
        reward = 0  # reward の初期値を設定
        if self.time_in_training != 100:
            self.time_in_segment = 0
        if self.time_in_segment == 0:
            self.last_bit_rate_index = 0  # 初期ビットレートのインデックス

        #self.log_file.write("Timestep,Bandwidth,Selected Bitrate\n")  # ログヘッダー
        if self.progress_bar:
            self.progress_bar.n = 0
            self.progress_bar.last_print_n = 0
            self.progress_bar.refresh()
        return np.array([bandwidth, self.bit_rates[self.last_bit_rate_index]], dtype=np.float32)

    def step(self, action):
        # 帯域幅の変化をシミュレーション
        bandwidth = np.random.normal(self.mean_bandwidth, self.std_bandwidth)
        reward = 0  # reward の初期値を設定
        
        # セグメント送信期間中の平均伝送レートは、そのセグメントが送信される時間無いの伝送レートを用いて計算される
        self.R_t.append(bandwidth)
        
        if self.time_in_segment == 0:
            if self.video_st == True:
                self.bitrate_legacy.append(self.bit_rates[self.current_segment])
                self.log_file.write(f"T_send:0, segment_data:{self.bit_rates[0]*self.segment_duration}, R_t:0\n")
                pass
            else:
                # 積分によって平均伝送レートを計算する
                R_t_ave = compute_average(self.R_t, self.time_in_training, self.segment_duration)
                selected_index = -1  # 初期値として不適切な値を設定
                for index, bit_rate in enumerate(self.bit_rates):
                    if R_t_ave >= bit_rate:
                        selected_index = index  # 条件を満たしたインデックスを保存
                        break
                    else:
                        continue

                # 選択可能なインデックスは `selected_index` 以下に制限
                if selected_index != -1:  # 有効なインデックスが選ばれた場合
                    max_index = selected_index  # インデックスの上限を設定

                    # `last_bit_rate_index` の選択ロジック
                    if action == 0:  # 解像度を下げる
                        self.last_bit_rate_index = max(0, self.last_bit_rate_index - 1)
                        if self.last_bit_rate_index > max_index:
                            self.rate_loss_penalty = 1 # 選択された解像度が平均伝送レートを下回らなかった場合のペナルティ（仮）
                    elif action == 2:  # 解像度を上げる
                        self.last_bit_rate_index = min(max_index, self.last_bit_rate_index + 1)  # 最大値は選択可能インデックス以下
                    else:
                        pass
                else:
                    self.log_file.write(f"No valid bit_rate index found.\n")

                self.R_t_box.append(R_t_ave)
                R_t_ave = 0
                          
            self.bitrate_legacy.append(self.bit_rates[self.last_bit_rate_index])

            # 選択されたビットレートにおけるセグメントデータ量の計算
            self.segment_data = self.segment_duration * self.bitrate_legacy[self.segmnet_cnt]

            # セグメントの送信に必要な時間＝セグメントのデータ量/セグメント送信期間中の平均伝送レート
            self.T_send.append(self.segment_data / self.R_t[self.time_in_training])
            
            # QoE計算
            reward = qoe_cal(self.time_in_training, self.segmnet_cnt, self.bitrate_legacy, self.bit_rates[self.last_bit_rate_index], self.rate_loss_penalty)

            if self.video_st == True:
                self.log_file.write(f"segmnet_cnt: {self.segmnet_cnt}, T_send:{self.T_send[self.segmnet_cnt]}, segment_data:{self.segment_data}, reward: {reward}\n")
            else:
                #print(f'self.segmnet_cnt: {self.segmnet_cnt}, len R_t_box: {len(self.R_t_box)}, len T_send: {len(self.T_send)}, reward: {reward}')
                self.log_file.write(f"segmnet_cnt: {self.segmnet_cnt}, T_send:{self.T_send[self.segmnet_cnt]}, segment_data:{self.segment_data}, R_t_ave:{self.R_t_box[self.segmnet_cnt-1]}, reward: {reward}\n")

            #ペナルティの初期化
            if self.rate_loss_penalty != 0:
                self.rate_loss_penalty = 0

        # ログに記録
        if self.time_in_segment == 0:
            self.log_file.write(f"frame:{self.time_in_video}, bandwidth:{bandwidth:.2f}, streamed rate:{self.bitrate_legacy[self.current_segment]}\n")
        else:
            self.log_file.write(f"frame:{self.time_in_video}, bandwidth:{bandwidth:.2f},\n")

        # TensorBoard用のカスタム記録
        if self.time_in_training % 4 == 0:
            #print(f'self.segmnet_cnt: {self.segmnet_cnt}, len(self.bitrate_legacy): {len(self.bitrate_legacy)}')
            self.logger.record("Reward/QoE", reward)
            self.logger.record("Selected Bitrate", self.bitrate_legacy[self.segmnet_cnt])
        self.logger.record("Bandwidth", bandwidth)
        # 任意の横軸（例えば秒単位、またはセグメント単位）
        self.logger.dump(self.time_in_training)

        # プログレスバーの更新
        if self.progress_bar:
            self.progress_bar.update(1)


        done = self.time_in_video >= 100  # 動画の長さを100ステップに設定
        next_state = np.array([bandwidth, self.bitrate_legacy[self.current_segment]], dtype=np.float32)

        # 状態の更新
        self.time_in_training += 1
        self.time_in_segment += 1
        self.time_in_video += 1

        if self.time_in_segment == self.segment_duration and self.video_st != True:
            #print(f'time_in_segment: {self.time_in_segment}, time_in_video: {self.time_in_video}')
            self.time_in_segment = 0
            if self.time_in_video != self.video_length:
                self.segmnet_cnt += 1
                
        if self.time_in_video == 100:
            self.video_st = True
            self.time_in_video = 0
        elif self.time_in_video == 1:
            self.video_st = False
        else:
            pass

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
        if self.logger:
            self.logger = None
        if self.log_file:
            self.log_file.close()
