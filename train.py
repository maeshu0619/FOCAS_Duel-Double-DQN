
import argparse
import time
from environment.environment import VideoStreamingEnv
from model.model import DqnAgent
import numpy as np
from system.time_calculate import debug_time
from tqdm import tqdm

def main(mode):
    if mode == 0 or mode == None:
        print("\nMode 0: Training the model with ABR.")
        num_episodes = 5
        max_steps_per_episode = 60
        goal_reward = 800
    elif mode == 1:
        print("\nMode 1: Training the model with FOCAS.")
        num_episodes = 5
        max_steps_per_episode = 60
        goal_reward = 800
    elif mode == 2:
        print("\nMode 2: Training the model with Adaptive-FOCAS.")
        num_episodes = 5
        max_steps_per_episode = 60
        goal_reward = 800

    total_timesteps = 5000

    # エージェントの初期化
    learning_rate = 0.0001
    buffer_size = 10000
    batch_size = 32
    gamma = 0.99

    # トレーニング
    reward_log = []
    training_cnt = 0

    total_timesteps = num_episodes*max_steps_per_episode 
    
    latency_constraint = 25 * 10**(-3) # レイテンシ制約

    # https://qiita.com/sugulu_Ogawa_ISID/items/bc7c70e6658f204f85f9
    env = VideoStreamingEnv(mode, total_timesteps, max_steps_per_episode, latency_constraint)
    agent = DqnAgent(env, mode, learning_rate=learning_rate, buffer_size=buffer_size)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            training_cnt += 1
            print(f'step {training_cnt} / {total_timesteps}')

            action = agent.actor.get_action(state, episode, agent.q_network)
            next_state, reward, done = env.step(action)

            agent.memory.add((state, action, reward, next_state, done))

            if agent.memory.len() >= batch_size and agent.memory.len() % 10 == 0:
                agent.q_network.replay(agent.memory, batch_size, gamma, agent.target_q_network)

            state = next_state
            total_reward += reward

            if done:
                #print('Loop is broken because of done')
                continue

        reward_log.append(total_reward)

        # ターゲットネットワークの更新
        if episode % 10 == 0:
            agent.target_q_network.model.set_weights(agent.q_network.model.get_weights())

        # 終了条件
        if np.mean(reward_log[-10:]) >= goal_reward:
            #print("--- Environment solved! ---")
            continue

    # モデルの保存
    if mode == 0:
        agent.save("trainedmodel/dqn_0_ABR_model.h5")
    elif mode == 1:
        agent.save("trainedmodel/dqn_1_FOCAS_model.h5")
    elif mode == 2:
        agent.save("trainedmodel/dqn_2_A-FOCAS_model.h5")




# https://qiita.com/sugulu_Ogawa_ISID/items/bc7c70e6658f204f85f9を参照
if __name__ == "__main__":
    start_time = time.time()  # 計測開始時刻
    parser = argparse.ArgumentParser(description="Train the model with specified mode.")
    parser.add_argument("--mode", type=int, required=True, help="Specify the mode for training. (e.g., 0 or 1)")
    args = parser.parse_args()

    main(args.mode)
    end_time = time.time()  # 計測終了時刻

    formatted_time = debug_time(end_time-start_time) # デバッグの計算時間の処理
    print(formatted_time)

    
