from multiprocessing import Pool

def process_replay(args):
    """
    並列処理用の関数。
    Replay に必要な各サンプルを処理する。
    """
    state, action, reward, next_state, done, gamma, targetQN = args
    target = reward
    if not done:
        # ターゲットQ値を計算
        target = reward + gamma * np.max(targetQN.model.predict(next_state[np.newaxis])[0])
    return state, action, target
