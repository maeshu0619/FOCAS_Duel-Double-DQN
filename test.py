import argparse
import time
from environment.environment import VideoStreamingEnv
from model.model import DqnAgent
from system.time_cal import debug_time
import numpy as np

def main(mode, latency, network):
    train_or_test = 1

    # レイテンシ制約
    if latency == 15: # 強い制約
        latency_constraint = 15 * 10**(-3)
        latency_file = "15ms"
    elif latency == 16: # 並みの制約
        latency_constraint = 20 * 10**(-3)
        latency_file = "16ms"
    elif latency == 17: # 並みの制約
        latency_constraint = 20 * 10**(-3)
        latency_file = "17ms"
    elif latency == 20: # 並みの制約
        latency_constraint = 20 * 10**(-3)
        latency_file = "20ms"
    elif latency == 25: # 弱い制約
        latency_constraint = 25 * 10**(-3)
        latency_file = "25ms"
        
    # 通信環境
    if network == 0: # 悪質な通信環境
        mu = 750
        sigma_ratio = 0.5
        base_band = 1.5e6
        network_file = "low transmission rate"
    elif network == 1: # 並みの通信環境
        mu = 1500
        sigma_ratio = 0.1
        base_band = 3e6
        network_file = "moderate transmission rate"
    elif network == 2: # 良質な通信環境
        mu = 3000
        sigma_ratio = 0.05
        base_band = 20e6
        network_file = "high transmission rate"
    elif network == 3: # ランダムな通信環境
        mu = 200
        sigma_ratio = 0.1
        base_band = 10e6
        network_file = ""

    if mode == 0:
        model_path = f"trainedmodel/{latency_file}/{network_file}/dqn_ABR_model.h5"
        print("\nTesting Mode 0: ABR")
    elif mode == 1:
        model_path = f"trainedmodel/{latency_file}/{network_file}/dqn_FOCAS_model.h5"
        print("\nTesting Mode 1: FOCAS")
    elif mode == 2:
        model_path = f"trainedmodel/{latency_file}/{network_file}/dqn_A-FOCAS_model.h5"
        print("\nTesting Mode 2: Adaptive-FOCAS")
    else:
        raise ValueError("Invalid mode. Please specify 0, 1, or 2.")

    # パラメータ設定
    fps = 30
    num_episodes = 15  # テストは1エピソードのみ
    max_steps_per_episode = 60
    total_timesteps = num_episodes*max_steps_per_episode

    # 環境とエージェントの初期化
    env = VideoStreamingEnv(mode, train_or_test, latency_file, network_file, 
                            total_timesteps, max_steps_per_episode, latency_constraint, fps, 
                            mu, sigma_ratio, base_band)
    agent = DqnAgent(env, mode)

    agent.load(model_path)

    training_cnt = 0
    goal_reward = 1000

    reward_legacy = []
    late_ave = 0

    # テストの実行
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}")
        state = env.reset()
        reward_per_episode = 0

        for step in range(max_steps_per_episode):
            training_cnt += 1
            if training_cnt % 10 == 0:
                #print(f'step {training_cnt} / {total_timesteps}')
                pass

            action = agent.actor.get_action(state, episode, agent.q_network, train_or_test)

            next_state, reward, late, done = env.step(action, goal_reward)

            late_ave += late
            #print(late)

            reward_per_episode += reward # 各ステップの報酬の合計
            state = next_state # 状態の取得

            if done: # トレーニング終了条件
                print("Episode finished early due to 'done' condition.")
                break
        
        reward_per_episode /= max_steps_per_episode # 各ステップの報酬平均
        reward_legacy.append(reward_per_episode) # 各ステップの報酬平均履歴記録
        reward_per_episode = 0 # リセット

        print(f"Reward for Episode {episode + 1}")
        print(f"Total: {reward_legacy[episode]:.2f}")

    ave_reward = 0
    for i in range(num_episodes):
        ave_reward += reward_legacy[i] # 各エピソードの報酬の合計
        print(f'Total Reward per Episode{i+1}: {reward_legacy[i]:.2f}')
    ave_reward /= num_episodes # 各エピソードの報酬の平均

    print(f"\nAverage Reward: {ave_reward:.2f}")
    print(f"Latency Average: {late_ave/training_cnt}")

    if mode == 0:
        print(f'\nABR training on {latency_file} constraint and {network_file} is done')
    elif mode == 1:
        print(f'\nFOCAS training on {latency_file} constraint and {network_file} is done')
    elif mode == 2:
        print(f'\nA-FOCAS training on {latency_file} constraint and {network_file} is done')

if __name__ == "__main__":
    start_time = time.time()  # 計測開始時刻
    
    parser = argparse.ArgumentParser(description="Train the model with specified mode and options.")
    parser.add_argument("--mode", type=int, required=True, help="Specify the mode for training. (e.g., 0, 1, or 2)")
    parser.add_argument("--late", type=int, choices=[15, 16, 17, 20, 25], default=25,
                        help="Set the latency constraint in milliseconds. Choices are 15, 20, 25. Default is 25.")
    parser.add_argument("--net", type=int, choices=[0, 1, 2, 3], default=1,
                        help="Specify the network condition. Options are: 0 (poor), 1 (average), 2 (good). Default is 1.")

    args = parser.parse_args()

    main(mode=args.mode, latency=args.late, network=args.net)
    
    end_time = time.time()  # 計測終了時刻

    formatted_time = debug_time(end_time - start_time)  # デバッグの計算時間の処理
    print(f"Test completed in: {formatted_time}")
