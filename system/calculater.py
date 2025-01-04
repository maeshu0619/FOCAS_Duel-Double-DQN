import numpy as np
import math
import random

# https://github.com/godka/ABR-DQN.git
# Stick: A Harmonious Fusion of Buffer-based and Learning-based Approach for Adaptive Streaming
# リバッファリングペナルティの計算
def calculate_rebuffering_penalty(now_bandwidth, now_rate, segment_length, buffer_size):
    """
    リバッファリングペナルティを計算する関数。

    Args:
        now_bandwidth (float): 現在の帯域幅 (kbps)。
        now_rate (float): 現在選択されたビットレート (kbps)。
        segment_length (float): 動画セグメントの長さ (秒)。デフォルトは1秒。
        buffer_size (float): 現在のバッファサイズ (秒)。デフォルトは0秒。
        beta (float): リバッファリングペナルティの重み。デフォルトは4.3。

    Returns:
        float: リバッファリングペナルティの値。
    """
    # 帯域幅とビットレートをバイト単位で計算
    segment_size = now_rate * segment_length / 8  # ビットからバイトへ変換
    download_time = segment_size / (now_bandwidth / 8)  # ダウンロード時間 (秒)

    # リバッファリング時間を計算
    rebuffering_time = max(download_time - buffer_size, 0.0)

    return rebuffering_time

#　QoEの計算関数
def qoe_cal(mode, steps_per_episode, time_in_training, bitrate_legacy, resolution_legacy, 
            bitrate_list, resolution_list, quality_vc_legacy, 
            bandwidth_legacy, resblock_info, gaze_coordinates, 
            size_legacy, depth_legacy, latency_constraint):
    alpha = 10
    beta = 2
    gamma = 2

    segment_length = 1.0 # セグメントの長さ (秒)
    buffer_size = 0.5    # 現在のバッファサイズ (秒)、一般的な適応ビットレート（ABR）アルゴリズムで短いバッファサイズを持つことを想定しているため。
    sigma = 4.3 * 2         # リバッファリングペナルティの重み。参考にしたリポジトリに基づく。
    rebuffer = 0 # リバッファリングペナルティ

    if steps_per_episode > 0:
        now_rate = bitrate_legacy[time_in_training]
        pre_rate = bitrate_legacy[time_in_training-1]
    else:
        now_rate = bitrate_legacy[time_in_training]
    now_bandwidth = bandwidth_legacy[time_in_training]

    episode_fin = False

    if mode == 0: # ABR
        # 動画品質QoEの計算
        quality = utility(now_rate)
        if now_bandwidth < now_rate: # リバッファリングペナルティ
            rebuffer = calculate_rebuffering_penalty(now_bandwidth, now_rate, segment_length, buffer_size)
            #quality -= sigma * rebuffer

        # 時間ジッタの計算
        if steps_per_episode == 0:
            jitter_t = 0
        else:
            jitter_t = abs(utility(now_rate) - utility(pre_rate))

        # 空間ジッタの計算
        jitter_s = 0 # 空間ジッタは0

    elif mode == 1: # FOCAS
        resolution = resolution_legacy[time_in_training]
        resblock_time = resblock_info[0]
        resblock_quality = resblock_info[1]
        gaze_xy  = gaze_coordinates[time_in_training]
        size_fovea = size_legacy[time_in_training][0]
        size_blend = size_legacy[time_in_training][1]
        depth_fovea = depth_legacy[time_in_training][0]
        depth_blend = depth_legacy[time_in_training][1]
        depth_peri = depth_legacy[time_in_training][2]

        fovea_area = size_via_resolution(gaze_xy, resolution, size_fovea, depth_fovea, resblock_time)
        fovea_time = fovea_area * (depth_fovea - depth_blend) * resblock_time
        blend_area = size_via_resolution(gaze_xy, resolution, size_blend, depth_blend, resblock_time)
        blend_time = blend_area * (depth_blend - depth_peri) * resblock_time
        peri_time = resolution[0] * resolution[1] * depth_peri * resblock_time
        all_cal_time = fovea_time + blend_time + peri_time

        if all_cal_time > latency_constraint: # レイテンシ制約超過をした場合、この行動をもう選択しないようにする
            episode_fin = True
            print(f'latency constraint exceedance')
        print(f'calculation time is {all_cal_time}')

        # 各領域の動画サイズを計算
        resolution_fovea  = [resolution[0]*resblock_quality**depth_fovea,
                          resolution[1]*resblock_quality**depth_fovea]
        resolution_blend  = [resolution[0]*resblock_quality**depth_blend,
                          resolution[1]*resblock_quality**depth_blend]
        resolution_peri  = [resolution[0]*resblock_quality**depth_peri,
                          resolution[1]*resblock_quality**depth_peri]
        
        # 各領域のビットレート（動画品質）を動画サイズから予測計算
        print(f'resolution_fovea: {resolution_fovea}, resolution_blend: {resolution_blend}, resolution_peri: {resolution_peri}')
        
        quality_fovea = resolution_to_quality(bitrate_list, resolution_list, resolution_fovea)
        quality_blend = resolution_to_quality(bitrate_list, resolution_list, resolution_blend)
        quality_peri = resolution_to_quality(bitrate_list, resolution_list, resolution_peri)

        ratio_fovea = area_percentage(resolution, size_fovea)
        ratio_blend = area_percentage(resolution, size_blend)
        quality = (utility(quality_fovea)*ratio_fovea) + (utility(quality_blend)*(ratio_blend-ratio_fovea)) + (utility(quality_peri)*(1-ratio_blend))
        
        # リバッファリングペナルティ
        if now_bandwidth < now_rate: 
            rebuffer = calculate_rebuffering_penalty(now_bandwidth, now_rate, segment_length, buffer_size)
        
        # 時間ジッタの計算
        if steps_per_episode == 0:
            jitter_t = 0
        else:
            jitter_t = abs(quality - quality_vc_legacy[time_in_training-1])

        # 空間ジッタの計算
        jitter_s = ((utility(quality_fovea) - quality)**2 + (utility(quality_blend) - quality)**2 + (utility(quality_peri) - quality)**2) / 3
        
        print(f'quality_fovea: {quality_fovea}, quality_blend: {quality_blend}, quality_peri: {quality_peri}')
    elif mode == 2: # Adaptive FOCAS
        resolution = resolution_legacy[time_in_training]
        resblock_time = resblock_info[0]
        resblock_quality = resblock_info[1]
        gaze_xy  = gaze_coordinates[time_in_training]
        size_fovea = size_legacy[time_in_training][0]
        size_blend = size_legacy[time_in_training][1]
        depth_fovea = depth_legacy[time_in_training][0]
        depth_blend = depth_legacy[time_in_training][1]
        depth_peri = depth_legacy[time_in_training][2]

        fovea_area = size_via_resolution(gaze_xy, resolution, size_fovea, depth_fovea, resblock_time)
        fovea_time = fovea_area * (depth_fovea - depth_blend) * resblock_time
        blend_area = size_via_resolution(gaze_xy, resolution, size_blend, depth_blend, resblock_time)
        blend_time = blend_area * (depth_blend - depth_peri) * resblock_time
        peri_time = resolution[0] * resolution[1] * depth_peri * resblock_time
        all_cal_time = fovea_time + blend_time + peri_time

        if all_cal_time > latency_constraint: # レイテンシ制約超過をした場合、この行動をもう選択しないようにする
            episode_fin = True
            print(f'latency constraint exceedance')
        print(f'calculation time is {all_cal_time}')
        
        # 各領域の動画サイズを計算
        resolution_fovea  = [resolution[0]*resblock_quality**depth_fovea,
                          resolution[1]*resblock_quality**depth_fovea]
        resolution_blend  = [resolution[0]*resblock_quality**depth_blend,
                          resolution[1]*resblock_quality**depth_blend]
        resolution_peri  = [resolution[0]*resblock_quality**depth_peri,
                          resolution[1]*resblock_quality**depth_peri]
        
        # 各領域のビットレート（動画品質）を動画サイズから予測計算
        print(f'resolution_fovea: {resolution_fovea}, resolution_blend: {resolution_blend}, resolution_peri: {resolution_peri}')

        quality_fovea = resolution_to_quality(bitrate_list, resolution_list, resolution_fovea)
        quality_blend = resolution_to_quality(bitrate_list, resolution_list, resolution_blend)
        quality_peri = resolution_to_quality(bitrate_list, resolution_list, resolution_peri)


        ratio_fovea = area_percentage(resolution, size_fovea)
        ratio_blend = area_percentage(resolution, size_blend)
        quality = (utility(quality_fovea)*ratio_fovea) + (utility(quality_blend)*(ratio_blend-ratio_fovea)) + (utility(quality_peri)*(1-ratio_blend))
        
        # リバッファリングペナルティ
        if now_bandwidth < now_rate: 
            rebuffer = calculate_rebuffering_penalty(now_bandwidth, now_rate, segment_length, buffer_size)
        
        # 時間ジッタの計算
        if steps_per_episode == 0:
            jitter_t = 0
        else:
            jitter_t = abs(quality - quality_vc_legacy[time_in_training-1])

        # 空間ジッタの計算
        jitter_s = ((utility(quality_fovea) - quality)**2 + (utility(quality_blend) - quality)**2 + (utility(quality_peri) - quality)**2) / 3
        
        print(f'quality_fovea: {quality_fovea}, quality_blend: {quality_blend}, quality_peri: {quality_peri}')

    print(f'quality: {quality}, jitter_t: {jitter_t}, jitter_s: {jitter_s}, rebuffer: {rebuffer}')

    reward = alpha * quality - beta * jitter_t - gamma * jitter_s - sigma * rebuffer
    return quality, jitter_t, jitter_s, rebuffer, reward, episode_fin

# 各領域のサイズ（ピクセル数）
def size_via_resolution(gaze_xy, resolution, size, depth, resblock_time):
    video_width = resolution[0]
    video_height = resolution[1]
    y = int(gaze_xy[0] * video_width / 1080)
    x = int(gaze_xy[1] * video_height / 1920)
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

    print(f'x: {x}, y: {y}, r: {r}, video_width: {video_width}, video_height: {video_height}')
    if r != 0:
        if x - r <= 0:
            if y - r <= 0:
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
            elif y + r >= video_height:
                S1 = x * math.sqrt(r**2 - x**2) / 2
                S2 = (video_height - y) * math.sqrt(r**2 - (video_height - y)**2) / 2
                S3 = x * (video_height - y)
                radian = 360 - (90 + math.degrees(math.acos(x/r)) + math.degrees(math.acos((video_height - y)/r)))
                S4 = r**2 * math.pi * radian / 360
                area = S1 + S2 + S3 + S4
        elif x - r > 0 and x + r < video_width:
            if y - r <= 0:
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
        elif x + r >= video_width:
            if y - r <= 0:
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
            elif y + r >= video_height:
                S1 = (video_width - x) * math.sqrt(r**2 - (video_width - x)**2) / 2
                S2 = (video_height - y) * math.sqrt(r**2 - (video_height - y)**2) / 2
                S3 = (video_width - x) * (video_height - y)
                radian = 360 - (90 + math.degrees(math.acos((video_width - x)/r)) + math.degrees(math.acos((video_height - y)/r)))
                S4 = r**2 * math.pi * radian / 360
                area = S1 + S2 + S3 + S4
    else:
        area = 0

    return area

# 動画サイズ間の比率から解像度を予測して計算
def resolution_to_quality(bitrate_list, resolution_list, resolution):
    rate_index = 0
    over_quality = 0
    ratio = 0  # ratio を初期化しておく
    quality = 0  # デフォルトの品質値を設定しておく
    
    for i in range(len(resolution_list)):
        if resolution[0] < resolution_list[i][0]:
            rate_index = i
            break
        rate_index += 1

    if rate_index != len(resolution_list):
        if rate_index == 0:
            ratio = resolution[0] / resolution_list[rate_index][0]
            quality = bitrate_list[rate_index] * (ratio)
        else:
            ratio = (resolution[0] - resolution_list[rate_index-1][0]) / (resolution_list[rate_index][0] - resolution_list[rate_index-1][0])
            quality = bitrate_list[rate_index-1] * (1 + ratio)
    else:
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
