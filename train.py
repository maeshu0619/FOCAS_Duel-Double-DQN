import argparse
import time
from environment.environment import VideoStreamingEnv
from model.model import DqnAgent
import numpy as np
from system.time_cal import debug_time

def main(mode, latency, network):
    train_or_test = 0

    learning_rate = 0.001

    if mode == 0:
        print("\nMode 0: Training the model with ABR.")
        num_episodes = 250
        max_steps_per_episode = 60
        goal_reward = 1000
    elif mode == 1:
        print("\nMode 1: Training the model with FOCAS.")
        '''
        if latency < 17:
            learning_rate = 0.01
            num_episodes = 200 #400 # 500
        else:
            learning_rate = 0.001
            num_episodes = 250
        '''
        if latency == 15:
            num_episodes = 150 #400 # 500
        elif latency <= 17:
            num_episodes = 200 #400 # 500
        else:
            num_episodes = 250 #400 # 500
            
        num_episodes = 750 #400 # 500
        learning_rate = 0.0001
        max_steps_per_episode = 60 # 40
        goal_reward = 1000
    elif mode == 2:
        print("\nMode 2: Training the model with Adaptive-FOCAS.")
        if latency < 20:
            num_episodes = 200 #400 # 500
        else:
            num_episodes = 300
        max_steps_per_episode = 60 #60 # 40
        goal_reward = 1000

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
    '''
    if mode == 1 or mode == 2:
        if latency == 20 or latency == 25:
            num_episodes = 400
    '''
        
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

    q_update_gap = 10 # Q値を更新する頻度
    fps = 30

    # エージェントの初期化
    buffer_size = 50000
    batch_size = 32
    gamma = 0.99

    # トレーニング
    reward_log = []
    training_cnt = 0

    late_ave = 0

    total_timesteps = num_episodes*max_steps_per_episode # 合計ステップ数
    

    # https://qiita.com/sugulu_Ogawa_ISID/items/bc7c70e6658f204f85f9
    env = VideoStreamingEnv(mode, train_or_test, latency_file, network_file, 
                            total_timesteps, max_steps_per_episode, latency_constraint, fps, 
                            mu, sigma_ratio, base_band)
    agent = DqnAgent(env, mode, learning_rate=learning_rate, buffer_size=buffer_size)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            if training_cnt % 10 == 0:
                print(f'step {training_cnt} / {total_timesteps}')

            training_cnt += 1

            action = agent.actor.get_action(state, episode, agent.q_network, train_or_test)
            next_state, reward, late, done = env.step(action, goal_reward)

            late_ave += late

            agent.memory.add((state, action, reward, next_state, done))

            if agent.memory.len() >= batch_size and training_cnt % q_update_gap == 0:
                agent.q_network.replay(agent.memory, batch_size, gamma, agent.target_q_network)

            state = next_state
            total_reward += reward

            if done:
                print("Episode finished early due to 'done' condition.")
                break

        reward_log.append(total_reward)
        print(f"Latency Average: {late_ave/max_steps_per_episode} s")
        late_ave = 0
        
        # ターゲットネットワークの更新
        if episode % 10 == 0:
            agent.target_q_network.model.set_weights(agent.q_network.model.get_weights())

        # 終了条件
        if np.mean(reward_log[-10:]) >= goal_reward*10:
            print("--- Environment solved! ---")
            break

    # モデルの保存

    if mode == 0:
        agent.save(f"trainedmodel/{latency_file}/{network_file}/dqn_ABR_model.h5")
        print(f'\nABR training on {latency_file} constraint and {network_file} is done')
    elif mode == 1:
        agent.save(f"trainedmodel/{latency_file}/{network_file}/dqn_FOCAS_model.h5")
        print(f'\nFOCAS training on {latency_file} constraint and {network_file} is done')
    elif mode == 2:
        agent.save(f"trainedmodel/{latency_file}/{network_file}/dqn_A-FOCAS_model.h5")
        print(f'\nA-FOCAS training on {latency_file} constraint and {network_file} is done')




# https://qiita.com/sugulu_Ogawa_ISID/items/bc7c70e6658f204f85f9を参照
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
    formatted_time = debug_time(end_time-start_time) # デバッグの計算時間の処理
    print(formatted_time)

    
