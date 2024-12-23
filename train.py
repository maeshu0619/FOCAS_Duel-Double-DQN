from model.model import DQNAgent
from environment.environment import FoveatedSRRenderingEnv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import os
import warnings

# Configuration
resolution_time = {
    'fovea': 2.0,
    'blend': 1.0,
    'peripheral': 0.5
}

latency_constraints = {
    'high': 50,
    'medium': 20,
    'low': 10
}

transmission_environments = {
    'poor': 20,
    'average': 50,
    'good': 80
}

def train(latency, transmission):
    # Initialize environment and agent
    env = FoveatedSRRenderingEnv(
        transmission_rate=transmission_environments[transmission],
        latency_constraint=latency_constraints[latency],
        resolution_time=resolution_time
    )
    agent = DQNAgent(env)

    # TensorBoard setup
    timenow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs", f"train_{timenow}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")

    # Training setup
    print(f"Training with latency: {latency}, transmission: {transmission}")
    total_timesteps = 20000
    eval_interval = 1000  # Record data every 1000 steps

    # Single tqdm progress bar
    pbar = tqdm(total=total_timesteps, desc="Training Progress", leave=False)

    for step in range(1, total_timesteps + 1):
        agent.model.learn(total_timesteps=1, reset_num_timesteps=False)

        # Log metrics to TensorBoard and tqdm
        if step % eval_interval == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress warnings temporarily
                ep_reward = agent.evaluate(env, n_eval_episodes=3)[0]
            
            exploration_rate = agent.model.exploration_rate

            # Write to TensorBoard
            writer.add_scalar('Reward/Episode', ep_reward, step)
            writer.add_scalar('Exploration/Rate', exploration_rate, step)

            # Update tqdm bar with metrics
            pbar.set_postfix({
                "Reward": ep_reward,
                "Exploration Rate": exploration_rate
            })

        pbar.update(1)

    # Finalize training
    pbar.close()
    agent.save("dqn_foveated_model")
    writer.close()
    print("Training complete. Model saved and logs written.")

if __name__ == "__main__":
    # Parameters
    selected_latency = 'medium'
    selected_transmission = 'average'

    train(selected_latency, selected_transmission)
