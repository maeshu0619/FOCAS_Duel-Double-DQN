from environment.environment import VideoStreamingEnv
from model.model import DqnAgent
from tqdm import tqdm

if __name__ == "__main__":
    # 総トレーニングステップ数を指定
    total_timesteps = 5000

    # 環境の初期化
    env = VideoStreamingEnv(video_length=100, segment_duration=4, video_fps=30)

    # エージェントの初期化
    agent = DqnAgent(env)

    # プログレスバーを初期化
    with tqdm(total=total_timesteps, desc="Training Progress") as pbar:
        # トレーニングを実行しつつ進捗を更新
        for _ in range(total_timesteps):
            agent.model.learn(total_timesteps=1, log_interval=100)
            pbar.update(1)

    print("Training completed!")

    # モデルを保存
    agent.save("dqn_adaptive_focas_model")
