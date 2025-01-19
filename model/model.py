import gymnasium
import gym
import os
import numpy as np
from system.multi_replay import process_replay
from multiprocessing import Pool
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import backend as K
from collections import deque
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from system.learning_rate import lr_scheduler

# 損失関数の定義
# 損失関数にhuber関数を使用します
# 参考: https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
@tf.keras.utils.register_keras_serializable()
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)

# Qネットワークの定義
class QNetwork:
    def __init__(self, mode, learning_rate, state_size, action_size, hidden_size=10):
        dropout_rate=0.2
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dropout(dropout_rate))  # ドロップアウトを追加
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dropout(dropout_rate))  # ドロップアウトを追加
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
        self.lr_callback = LearningRateScheduler(lr_scheduler)

    def replay(self, memory, batch_size, gamma, targetQN):
        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        # 一括予測
        mainQs = self.model.predict(states)
        nextMainQs = self.model.predict(next_states)
        targetQs = targetQN.model.predict(next_states)


        for i in range(batch_size):
            if dones[i]:
                mainQs[i][actions[i]] = rewards[i]
            else:
                max_next_action = np.argmax(nextMainQs[i])
                mainQs[i][actions[i]] = rewards[i] + gamma * targetQs[i][max_next_action]

        # 学習率スケジューラーを適用して一括トレーニング
        self.model.fit(states, mainQs, epochs=1, verbose=0, callbacks=[self.lr_callback])



# Experience Replayの実装
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        samples = np.array([self.buffer[i] for i in idx], dtype=object)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.vstack(states),  # 各状態を一括で取得
            np.array(actions),
            np.array(rewards),
            np.vstack(next_states),
            np.array(dones)
        )

    def len(self):
        return len(self.buffer)

# Actorの実装
invalid_actions = set()  # 無効な行動を記録する
class Actor:
    def __init__(self, mode):
        global invalid_actions
        self.mode = mode
        self.action_history = []  # 行動履歴を初期化
        if mode == 0:
            self.action_space = 4
        elif mode == 1:
            self.action_space = 1200
        elif mode == 2:
            self.action_space = 4800
        else:
            raise ValueError("Invalid mode. Mode must be 0, 1, or 2.")

    def add_invalid_action(self, action):
        invalid_actions.add(action)

    def get_action(self, state, episode, mainQN, test=False):
        if test:
            epsilon = 0.0  # テストモードではεをゼロに固定
        else:
            epsilon = max(0.1, 1.0 - episode * 0.005)  # εを減少させる
        
        valid_actions = list(set(range(self.action_space)) - invalid_actions)  # 有効な行動のみ
            
        # 未選択の行動を優先
        unexplored_actions = [a for a in valid_actions if a not in self.action_history]
        if unexplored_actions:
            action = np.random.choice(unexplored_actions)
        elif np.random.rand() < epsilon:
            # ランダムに行動を選択
            action = np.random.choice(valid_actions)
        else:
            # Q値に基づいて行動を選択
            retTargetQs = mainQN.model.predict(state[np.newaxis])[0]
            retTargetQs[list(invalid_actions)] = -np.inf  # 無効な行動のQ値を無限小に設定
            action = np.argmax(retTargetQs)

        self.action_history.append(action)  # 行動履歴を更新
        return action



class DqnAgent:
    def __init__(self, env, mode, learning_rate: float = 0.0001, buffer_size: int = 1000000):
        self.env = env
        self.mode = mode
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.q_network = QNetwork(
            mode=self.mode, 
            learning_rate=learning_rate,
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            hidden_size=128
        )
        self.target_q_network = QNetwork(
            mode=self.mode, 
            learning_rate=learning_rate,
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            hidden_size=128
        )
        self.memory = Memory(max_size=buffer_size)
        self.actor = Actor(self.mode)

    def train(self, total_timesteps: int = 50000, batch_size: int = 32, gamma: float = 0.99):
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
        q_values = self.q_network.model.predict(observation[np.newaxis])[0]
        return np.argmax(q_values)

    def save(self, file_name: str) -> None:
        self.q_network.model.save(file_name)

    def load(self, file_name: str) -> None:
        from tensorflow.keras.models import load_model
        self.q_network.model = load_model(file_name, custom_objects={"huberloss": huberloss})
