from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

class DqnAgent:
    """
    A wrapper class for training and interacting with a DQN agent using Stable Baselines3.

    Args:
        env (Env): The Gym environment to train and evaluate on.
        learning_rate (float, optional): The learning rate for the DQN model. Defaults to 0.0001.
        buffer_size (int, optional): The size of the replay buffer used by the DQN model. Defaults to 1000000.
    """

    def __init__(self, env, learning_rate: float = 0.0001, buffer_size: int = 1000000):
        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
        )

    def train(self, total_timesteps: int = 50000) -> None:
        """
        Trains the DQN agent on the specified environment for a given number of timesteps.

        Args:
            total_timesteps (int, optional): The total number of timesteps to train for. Defaults to 50000.
        """

        self.model.learn(total_timesteps=total_timesteps)

    def save(self, file_name: str) -> None:
        """
        Saves the trained DQN model to a file.

        Args:
            file_name (str): The name of the file to save the model to.
        """

        self.model.save(file_name)

    def load(self, file_name: str) -> None:
        """
        Loads a pre-trained DQN model from a file.

        Args:
            file_name (str): The name of the file containing the model to load.
        """

        self.model.load(file_name)

    def evaluate(self, env, n_eval_episodes: int = 10) -> tuple[float, float]:
        """
        Evaluates the performance of the DQN agent on the specified environment.

        Args:
            env (Env): The Gym environment to evaluate on.
            n_eval_episodes (int, optional): The number of evaluation episodes to run. Defaults to 10.

        Returns:
            tuple[float, float]: A tuple containing the mean reward and standard deviation of the reward across the evaluation episodes.
        """

        mean_reward, std_reward = evaluate_policy(
            self.model, env, n_eval_episodes=n_eval_episodes
        )

        return mean_reward, std_reward

    def predict(self, observation: np.ndarray) -> int:
        """
        Predicts the best action for a given observation using the DQN agent.

        Args:
            observation (np.ndarray): The observation from the environment.

        Returns:
            int: The predicted action to take.
        """

        action, _ = self.model.predict(observation, deterministic=True)

        return action
