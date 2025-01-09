import numpy as np

# パラメータ
P_t = 0.1 # 送信電力 (W) - 増加
G_t = 1 # 送信アンテナゲイン
G_r = 1 # 受信アンテナゲイン
frequency = 2.4e9  # 周波数 (Hz, 2.4 GHz)
c = 3e8    # 光速 (m/s)
wavelength = c / frequency  # 波長 (m)
noise_power = 1e-6  # ノイズ電力 (W)

# フリスの法則による受信電力計算
def friss_law(distance):
    return P_t * G_t * G_r * (wavelength / (4 * np.pi * distance)) ** 2

# シャノンの定理による伝送レート計算
def shannon_capacity(snr, bandwidth):
    return bandwidth * np.log2(1 + snr)

# 正規分布によって出力結果をランダムにする
def normal(value, mu, sigma):
    noise = np.random.normal(mu, sigma)
    return max(value + noise, 0)  # 負の値を防ぐため0以下は0にする

# 伝送レートのシミュレーション
def simulate_transmission_rate(distance, max_steps_per_episode, mu, sigma_ratio, base_band):
    transmission_rates = []
    
    for step in range(max_steps_per_episode):
        # 受信電力を計算
        received_power = friss_law(distance)
        
        # SNRを計算
        snr = received_power / noise_power
        
        # 伝送レートを計算
        rate = shannon_capacity(snr, base_band)

        # 正規分布のノイズを追加
        rate_with_noise = normal(rate, mu, rate*sigma_ratio)
        if rate_with_noise < 200:
            rate_with_noise = 200
        transmission_rates.append(rate_with_noise)
    
    return transmission_rates
