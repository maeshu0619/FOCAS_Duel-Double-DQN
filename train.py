from environment.environment import AFOCASDQN
from model.model import DqnAgent
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import os

def train():
    mu = 0.5
    sigma = 0.1
    a = 0.02
    b = 0.01
    lamda = 0.5
    base_bw = 1e6  # 基本帯域幅（Mbps）
    episode_length = 100  # フレーム数
    VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps

    # 環境の初期化
    env = AFOCASDQN(mu, sigma, a, b, lamda, base_bw, VIDEO_BIT_RATE, episode_length)
    agent = DqnAgent(env)

    log_dir = os.path.join("runs", f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)

    total_timesteps = 50000
    pbar = tqdm(total=total_timesteps, desc="Training Progress")
    episode_rewards = []

    obs = env.reset()
    episode_reward = 0

    for step in range(total_timesteps):
        # エージェントがアクションを予測
        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        if done:
            # エピソード終了時に報酬を記録
            episode_rewards.append(episode_reward)
            writer.add_scalar("Episode Reward", episode_reward, len(episode_rewards))
            obs = env.reset()
            episode_reward = 0

        # トレーニングステップごとの記録
        writer.add_scalar("Training Steps", step, step)

        # プログレスバーの更新
        pbar.update(1)

    pbar.close()
    writer.close()
    agent.save("dqn_adaptive_focas_model")
    print("Training completed and model saved!")

if __name__ == "__main__":
    train()
