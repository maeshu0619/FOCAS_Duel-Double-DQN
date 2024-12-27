import numpy as np

# 平均伝送レートの計算関数
def compute_average(R_t, step, duration):
    """
    start_timeからend_timeの間でR_t配列を積分し、平均値を算出する。
    """
    # スライスの開始と終了インデックスを計算
    start_index = max(0, step - duration)  # 負の値を防ぐ
    end_index = step  # 現在のステップまで

    # R_t の指定範囲を抽出
    selected_R_t = R_t[start_index:end_index]

    # 時間範囲を計算（抽出されたデータのインデックスに基づく）
    time_range = np.arange(start_index, end_index)

    # 積分を計算（NumPyのtrapzを使用）
    if len(selected_R_t) > 1:
        integral_value = np.trapz(selected_R_t, time_range)
        average_value = integral_value / duration
    else:
        average_value = 0  # デフォルト値（データが不足している場合）

    return average_value

#　QoEの計算関数
def qoe_cal(cal_cnt, segmnet_cnt, bitrate_legacy, last_bit_rates, rate_loss_penalty):
    alpha = 10
    beta = 2
    gamma = 2
    
    # 平均映像品質の計算
    Q_vc = alpha * np.log(1 + last_bit_rates)

    # 時間ジッタの計算
    if cal_cnt == 0:
        S_t = 0
    else:
        S_t = beta * abs(last_bit_rates - bitrate_legacy[segmnet_cnt-1])

    # 空間ジッタの計算
    # S_s = gammma * 

    reward = Q_vc - S_t - rate_loss_penalty

    return reward
