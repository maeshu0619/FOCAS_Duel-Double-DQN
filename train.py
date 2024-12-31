from environment.environment import VideoStreamingEnv
from model.model import DqnAgent
import numpy as np

# https://qiita.com/sugulu_Ogawa_ISID/items/bc7c70e6658f204f85f9を参照
if __name__ == "__main__":
    # 環境の初期化
    total_timesteps = 5000

    # エージェントの初期化
    learning_rate = 0.0001
    buffer_size = 10000
    batch_size = 32
    gamma = 0.99

    # トレーニング
    num_episodes = 10
    max_steps_per_episode = 10
    goal_reward = 2000
    reward_log = []

    env = VideoStreamingEnv(num_episodes*max_steps_per_episode, video_length=100, segment_duration=4, video_fps=30)
    agent = DqnAgent(env, learning_rate=learning_rate, buffer_size=buffer_size)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        print(f'episode: {episode}')

        for step in range(max_steps_per_episode):
            action = agent.actor.get_action(state, episode, agent.q_network)
            next_state, reward, done, _ = env.step(action)
            print(f'step; {step}')

            agent.memory.add((state, action, reward, next_state, done))
            if agent.memory.len() > batch_size:
                agent.q_network.replay(agent.memory, batch_size, gamma, agent.target_q_network)

            state = next_state
            total_reward += reward
            print(f'reward: {reward}')

            if done:
                print('Loop is broken because of done')
                break

        reward_log.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward}\n")

        # ターゲットネットワークの更新
        if episode % 10 == 0:
            agent.target_q_network.model.set_weights(agent.q_network.model.get_weights())

        # 終了条件
        if np.mean(reward_log[-10:]) >= goal_reward:
            print("Environment solved!")
            break

    # モデルの保存
    agent.save("dqn_model.h5")
