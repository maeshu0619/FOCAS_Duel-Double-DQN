import numpy as np
import math
import random

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
def qoe_cal(cal_cnt, segmnet_cnt, bitrate_legacy, last_bit_rates, R_t):
    alpha = 10
    beta = 2
    gamma = 2
    
    # 平均映像品質の計算
    Q_vc = alpha * np.log(1 + last_bit_rates)
    if R_t[cal_cnt] < last_bit_rates:
        Q_vc = 0

    # 時間ジッタの計算
    if cal_cnt == 0:
        S_t = 0
    else:
        S_t = beta * abs(last_bit_rates - bitrate_legacy[segmnet_cnt-1])

    # 空間ジッタの計算
    # S_s = gammma * 

    reward = Q_vc - S_t

    return reward

c = 3.0 * 10**8
f = 2.4 * 10**9
ramda = c / f
No = -100
Pt = 100 * 10**(-3)
B = 20 * 10**6
PI = 3.1415926535

mu = 0
sigma = 2

last_distance = -1

# サーバとクライアントの距離を乱数で計算する関数
def distance(time_in_video):
    global last_distance
    if time_in_video == 0:
        last_distance = random.randint(1, 1000)
    else:
        min_dis = last_distance - 20
        max_dis = last_distance + 20
        if min_dis < 0:
            min_dis = 0
        elif max_dis > 1000:
            max_dis = 1000
        last_distance = random.randint(min_dis, max_dis)

    return last_distance


# 受信電力の計算関数
def pr_cal(time_in_video):
    r = distance(time_in_video)
        
    Pr = Pt * (ramda/(4*PI*r))**2
    L_add = np.random.normal(mu, sigma)

    Pr = Pr * 10**(-L_add/10)
    return Pr

# 伝送レートの計算関数
def rate_cal(time_in_video):
    Pr = pr_cal(time_in_video)

    rate = B * math.log(1+Pr/No, 2)
    return rate
