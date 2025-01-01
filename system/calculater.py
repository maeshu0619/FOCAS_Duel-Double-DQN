import numpy as np
import math
import random

# 平均伝送レートの計算関数
def compute_average(bandwidth_legacy, step, duration):
    """
    start_timeからend_timeの間でbandwidth_legacy配列を積分し、平均値を算出する。
    """
    # スライスの開始と終了インデックスを計算
    start_index = max(0, step - duration)  # 負の値を防ぐ
    end_index = step  # 現在のステップまで

    # bandwidth_legacy の指定範囲を抽出
    selected_bandwidth_legacy = bandwidth_legacy[start_index:end_index]

    # 時間範囲を計算（抽出されたデータのインデックスに基づく）
    time_range = np.arange(start_index, end_index)

    # 積分を計算（NumPyのtrapzを使用）
    if len(selected_bandwidth_legacy) > 1:
        integral_value = np.trapz(selected_bandwidth_legacy, time_range)
        average_value = integral_value / duration
    else:
        average_value = 0  # デフォルト値（データが不足している場合）

    return average_value


#　QoEの計算関数
def qoe_cal(mode, steps_per_episode, bitrate_legacy, resolution_legacy, 
            bitrate_list, resolution_list, quality_vc_legacy, 
            bandwidth_legacy, resblock_info, gaze_coordinates, 
            size_legacy, depth_legacy, latency_constraint):
    alpha = 10
    beta = 2
    gamma = 2

    if steps_per_episode > 0:
        now_rate = bitrate_legacy[steps_per_episode]
        pre_rate = bitrate_legacy[steps_per_episode-1]
    else:
        now_rate = bitrate_legacy[steps_per_episode]
    now_bandwidth = bandwidth_legacy[steps_per_episode]

    episode_fin = False

    if mode == 0: # ABR
        quality = utility(now_rate)
        if now_bandwidth < now_rate:
            '''
            QoE-Driven Cache Management for HTTP Adaptive Bit Rate Streaming Over Wireless Networks
            引用数164 出版年数2013
            '''
            quality -= 50

        # 時間ジッタの計算
        if steps_per_episode == 0:
            jitter_t = 0
        else:
            jitter_t = abs(utility(now_rate) - utility(pre_rate))

        # 空間ジッタの計算
        jitter_s = 0 # 空間ジッタは0

    elif mode == 1: # FOCAS
        pass
    elif mode == 2: # Adaptive FOCAS
        resolution = resolution_legacy[steps_per_episode]
        resblock_time = resblock_info[0]
        resblock_quality = resblock_info[1]
        gaze_xy  = gaze_coordinates[steps_per_episode]
        size_fovea = size_legacy[steps_per_episode][0]
        size_blend = size_legacy[steps_per_episode][1]
        depth_fovea = depth_legacy[steps_per_episode][0]
        depth_blend = depth_legacy[steps_per_episode][1]
        depth_peri = depth_legacy[steps_per_episode][2]

        fovea_area = size_via_resolution(gaze_xy, resolution, size_fovea, depth_fovea, resblock_time)
        fovea_time = fovea_area * (depth_fovea - depth_blend) * resblock_time
        blend_area = size_via_resolution(gaze_xy, resolution, size_blend, depth_blend, resblock_time)
        blend_time = blend_area * (depth_blend - depth_peri) * resblock_time
        peri_time = resolution[0] * resolution[1] * depth_peri * resblock_time
        all_cal_time = fovea_time + blend_time + peri_time

        if all_cal_time > latency_constraint:
            episode_fin = True
        
        # 各領域の動画サイズを計算
        resolution_fovea  = [resolution[0]*resblock_quality**depth_fovea,
                          resolution[1]*resblock_quality**depth_fovea]
        resolution_blend  = [resolution[0]*resblock_quality**depth_blend,
                          resolution[1]*resblock_quality**depth_blend]
        resolution_peri  = [resolution[0]*resblock_quality**depth_peri,
                          resolution[1]*resblock_quality**depth_peri]
        
        # 各領域のビットレート（動画品質）を動画サイズから予測計算
        quality_fovea = resolution_to_quality(bitrate_list, resolution_list, resolution_fovea)
        quality_blend = resolution_to_quality(bitrate_list, resolution_list, resolution_blend)
        quality_peri = resolution_to_quality(bitrate_list, resolution_list, resolution_peri)

        ratio_fovea = area_percentage(resolution, size_fovea)
        ratio_blend = area_percentage(resolution, size_blend)
        quality = (utility(quality_fovea)*ratio_fovea) + (utility(quality_blend)*(ratio_blend-ratio_fovea)) + (utility(quality_peri)*(1-ratio_blend))
        
        # 時間ジッタの計算
        if steps_per_episode == 0:
            jitter_t = 0
        else:
            jitter_t = abs(quality - quality_vc_legacy[steps_per_episode-1])

        # 空間ジッタの計算
        jitter_s = ((utility(quality_fovea) - quality)**2 + (utility(quality_blend) - quality)**2 + (utility(quality_peri) - quality)**2) / 3
        
        print(f'quality_fovea: {quality_fovea}, quality_blend: {quality_blend}, quality_peri: {quality_peri}')

    print(f'quality: {quality}, jitter_t: {jitter_t}, jitter_s: {jitter_s}')

    reward = alpha * quality - beta * jitter_t - gamma * jitter_s
    return quality, jitter_t, jitter_s, reward, episode_fin

# 各領域のサイズ（ピクセル数）
def size_via_resolution(gaze_xy, resolution, size, depth, resblock_time):
    video_width = resolution[1]
    video_height = resolution[0]
    x = int(gaze_xy[0] * video_width / 1080)
    y = int(gaze_xy[1] * video_height / 1920)
    r = size
    # 動画外に視線座標が出ないように矯正
    if x < 0:
        x = 0
    elif y < 0:
        y = 0
    elif x > video_width:
        x = video_width
    elif y > video_height:
        y = video_width

    #print(f'x: {x}, y: {y}, r: {r}')
    if x - r < 0:
        if y - r < 0:
            S1 = x * math.sqrt(r**2 - x**2) / 2
            S2 = y * math.sqrt(r**2 - y**2) / 2
            S3 = x * y
            radian = 360 - (90 + math.degrees(math.acos(x/r)) + math.degrees(math.acos(y/r)))
            S4 = r**2 * math.pi * radian / 360
            area = S1 + S2 + S3 + S4
        elif y - r > 0 and y + r < video_height:
            S1 = x * math.sqrt(r**2 - x**2)
            radian = 360 - (math.degrees(math.acos(x/r))*2)
            S2 = r**2 * math.pi * radian /360
            area = S1 + S2
        elif y + r < video_height:
            S1 = x * math.sqrt(r**2 - x**2) / 2
            S2 = (video_height - y) * math.sqrt(r**2 - (video_height - y)**2) / 2
            S3 = x * (video_height - y)
            radian = 360 - (90 + math.degrees(math.acos(x/r)) + math.degrees(math.acos((video_height - y)/r)))
            S4 = r**2 * math.pi * radian / 360
            area = S1 + S2 + S3 + S4
    elif x - r > 0 and x + r < video_width:
        if y - r < 0:
            S1 = y * math.sqrt(r**2 - y**2)
            radian = 360 - (math.degrees(math.acos(y/r))*2)
            S2 = r**2 * math.pi * radian /360
            area = S1 + S2
        elif y - r > 0 and y + r < video_height:
            area = r**2 * math.pi
        elif y + r > video_height:
            S1 = (video_height - y) * math.sqrt(r**2 - (video_height - y)**2)
            radian = 360 - (math.degrees(math.acos((video_height - y)/r))*2)
            S2 = r**2 * math.pi * radian /360
            area = S1 + S2
    elif x + r > video_width:
        if y - r < 0:
            S1 = (video_width - x) * math.sqrt(r**2 - (video_width - x)**2) / 2
            S2 = y * math.sqrt(r**2 - y**2) / 2
            S3 = (video_width - x) * y
            radian = 360 - (90 + math.degrees(math.acos((video_width - x)/r)) + math.degrees(math.acos(y/r)))
            S4 = r**2 * math.pi * radian / 360
            area = S1 + S2 + S3 + S4
        elif y - r > 0 and y + r < video_height:
            S1 = (video_width - x) * math.sqrt(r**2 - (video_width - x)**2)
            radian = 360 - (math.degrees(math.acos((video_width - x)/r))*2)
            S2 = r**2 * math.pi * radian /360
            area = S1 + S2
        elif y + r > video_height:
            S1 = (video_width - x) * math.sqrt(r**2 - (video_width - x)**2) / 2
            S2 = (video_height - y) * math.sqrt(r**2 - (video_height - y)**2) / 2
            S3 = (video_width - x) * (video_height - y)
            radian = 360 - (90 + math.degrees(math.acos((video_width - x)/r)) + math.degrees(math.acos((video_height - y)/r)))
            S4 = r**2 * math.pi * radian / 360
            area = S1 + S2 + S3 + S4

    return area

# 動画サイズ間の比率から解像度を予測して計算
def resolution_to_quality(bitrate_list, resolution_list, resolution):
    rate_index = -1
    over_quality = 0
    ratio = 0  # ratio を初期化しておく
    quality = 0  # デフォルトの品質値を設定しておく
    
    for i in range(len(resolution_list)):
        if resolution[0] < resolution_list[i][0]:
            rate_index = i
            break
        elif rate_index == len(resolution_list)-1 and resolution[0] > resolution_list[i][0]:
            over_quality = 1
            break
    if over_quality == 0:
        if rate_index == 0:
            ratio = resolution[0] / resolution_list[rate_index][0]
            quality = bitrate_list[rate_index] * (ratio)
        elif rate_index == -1:
            print('rate_index Error in resolution_to_quality of calculater.py')
        else:
            ratio = (resolution[0] - resolution_list[rate_index-1][0]) / (resolution_list[rate_index][0] - resolution_list[rate_index-1][0])
            quality = bitrate_list[rate_index-1] * (1 + ratio)
    elif over_quality == 1:
        ratio = resolution[0] / resolution_list[len(resolution_list)-1][0]
        quality = bitrate_list[len(resolution_list)-1] * ratio
    
    print(f'ratio: {ratio}, quality: {quality}')
    return quality

# 各領域が動画サイズ内の何％を占めるかを計算
def area_percentage(resolution, size):
    ratio = size**2 * math.pi / (resolution[0] * resolution[1])
    return ratio


# eを底に持った対数計算における効用関数
def utility(bitrate):
    log = np.log(1 + bitrate)
    return log

# パラメータ
P_t = 0.1  # 送信電力 (W)
G_t = 2  # 送信アンテナゲイン
G_r = 2  # 受信アンテナゲイン
frequency = 2.4e9  # 周波数 (Hz, 2.4 GHz)
c = 3e8  # 光速 (m/s)
wavelength = c / frequency  # 波長 (m)
noise_power = 1e-6  # ノイズ電力 (W)
capacity = 1


# 伝送レートの計算関数
def rate_cal(time_in_video):
    
    # 距離を正規分布で生成 (平均500m、標準偏差150m)
    #np.random.seed(42)  # 再現性のため乱数シードを固定
    distances = np.random.normal(500, 150)
    distances = np.clip(distances, 1, 1000)  # 1～1000mにクリッピング

    # フリスの公式による受信信号強度 (Pr)
    Pr = P_t * G_t * G_r * (wavelength / (4 * np.pi * distances)) ** 2

    # SNRの計算
    snr = Pr / noise_power

    # シャノンの公式
    bandwidth = capacity / np.log2(1 + snr)
    
    #print(f'band: {bandwidth}, distances: {distances}')

    return bandwidth, distances
