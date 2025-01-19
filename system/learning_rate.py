from tensorflow.keras.callbacks import LearningRateScheduler

# 学習深度を調査して学習率を変動させる
def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 50
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr