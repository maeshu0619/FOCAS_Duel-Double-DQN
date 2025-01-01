import gymnasium
import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
from collections import deque

# 損失関数の定義
# 損失関数にhuber関数を使用します
# 参考: https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)

# Qネットワークの定義
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=5, hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)

    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, self.model.input_shape[1]))
        targets = np.zeros((batch_size, self.model.output_shape[1]))
        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        for i in range(batch_size):
            state_b, action_b, reward_b, next_state_b, done_b = states[i], actions[i], rewards[i], next_states[i], dones[i]
            inputs[i:i + 1] = state_b
            target = reward_b

            if not done_b:
                retmainQs = self.model.predict(next_state_b[np.newaxis])[0]
                next_action = np.argmax(retmainQs)
                target = reward_b + gamma * targetQN.model.predict(next_state_b[np.newaxis])[0][next_action]

            targets[i] = self.model.predict(state_b[np.newaxis])[0]
            targets[i][action_b] = target

        # verbose=0で進捗バーを抑制
        self.model.fit(inputs, targets, epochs=1, verbose=0)



# Experience Replayの実装
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        samples = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*samples)  # 各要素を分解
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def len(self):
        return len(self.buffer)

# Actorの実装
class Actor:
    def get_action(self, state, episode, mainQN):
        epsilon = max(0.1, 1.0 - episode * 0.01)  # 初期値1.0で減少を緩やかに
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1, 2])  # 行動をランダム選択
        else:
            retTargetQs = mainQN.model.predict(state[np.newaxis])[0]
            action = np.argmax(retTargetQs)
        return action


class DqnAgent:
    """
    A wrapper class for training and interacting with a DQN agent using Stable Baselines3.

    Args:
        env (Env): The Gym environment to train and evaluate on.
        learning_rate (float, optional): The learning rate for the DQN model. Defaults to 0.0001.
        buffer_size (int, optional): The size of the replay buffer used by the DQN model. Defaults to 1000000.
    """

    def __init__(self, env, learning_rate: float = 0.0001, buffer_size: int = 1000000):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.q_network = QNetwork(
            learning_rate=learning_rate,
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            hidden_size=128
        )
        self.target_q_network = QNetwork(
            learning_rate=learning_rate,
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            hidden_size=128
        )
        self.memory = Memory(max_size=buffer_size)
        self.actor = Actor()

    def train(self, total_timesteps: int = 50000, batch_size: int = 64, gamma: float = 0.99):
        """
        報酬履歴を活用したQ値のターゲットを計算してトレーニング。
        """
        obs = self.env.reset()
        for step in range(total_timesteps):
            action = self.actor.get_action(obs, step, self.q_network)

            # 環境に対するステップ実行
            next_obs, reward, done, _ = self.env.step(action)
            self.memory.add((obs, action, reward, next_obs, done))

            # 学習を開始
            if self.memory.len() > batch_size:
                self.q_network.replay(self.memory, batch_size, gamma, self.target_q_network)

            obs = next_obs
            if done:
                obs = self.env.reset()

    def predict(self, observation: np.ndarray) -> int:
        """
        Predicts the best action for a given observation using the Q-network.

        Args:
            observation (np.ndarray): The observation from the environment.

        Returns:
            int: The predicted action to take.
        """
        q_values = self.q_network.model.predict(observation[np.newaxis])[0]
        return np.argmax(q_values)

    def save(self, file_name: str) -> None:
        self.q_network.model.save(file_name)

    def load(self, file_name: str) -> None:
        from tensorflow.keras.models import load_model
        self.q_network.model = load_model(file_name, custom_objects={"huberloss": huberloss})
