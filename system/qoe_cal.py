import numpy as np
import math
import random
from system.gaussian_weight import calculate_video_quality, calculate_weights, calculate_weights_peripheral


#　QoEの計算関数
def qoe_cal(mode, steps_per_episode, time_in_training, bitrate_legacy, resolution_legacy, 
            bitrate_list, resolution_list, quality_vc_legacy, 
            bandwidth_legacy, resblock_info, gaze_coordinates, 
            size_legacy, depth_legacy, latency_constraint, debug_log):
    alpha = 10 # 動画品質の重み
    beta = 2 # 時間ジッタの重み
    gamma = 2 # 空間ジッタの重み
    sigma = 4.3 * 15 # リバッファリングペナルティの重み。

    segment_length = 1.0 # セグメントの長さ (秒)
    rebuffer = 0 # リバッファリングペナルティ

    sigma_h, sigma_w = 64, 64 # ガウス分布による重み係数の計算における平均、分散。FOCASに基づく。

    if steps_per_episode > 0: # 初めのステップでなければ一つ前の解像度も取得
        now_rate = bitrate_legacy[steps_per_episode]
        pre_rate = bitrate_legacy[steps_per_episode-1]
    else:
        now_rate = bitrate_legacy[steps_per_episode]

    now_bandwidth = bandwidth_legacy[time_in_training] # 現在の帯域幅

    action_invalid_judge = False

    if mode == 0: # ABR
        resolution = resolution_legacy[time_in_training] # 現在の動画サイズ
        gaze_yx  = gaze_coordinates[steps_per_episode] # 視線情報取得

        # 重みづけされた各領域の品質を計算
        quality = calculate_video_quality(gaze_yx, resolution, 0, 0, 
                                          0, 0, utility(now_rate), 
                                          sigma_h, sigma_w)
        debug_log.write(f'utility quality: {utility(now_rate)}\n')
        
        # リバッファリングペナルティ
        if now_bandwidth < now_rate:
            rebuffer = calculate_rebuffering_penalty(now_bandwidth, now_rate, segment_length)

        # 時間ジッタの計算
        if steps_per_episode == 0:
            jitter_t = 0
        else:
            jitter_t = abs(utility(now_rate) - utility(pre_rate))

        # 空間ジッタの計算
        jitter_s = 0 # 空間ジッタは0

    elif mode == 1: # FOCAS
        resolution = resolution_legacy[time_in_training] # 現在の動画サイズ（一定）
        resblock_time = resblock_info[0] # 1ピクセルがResBlock一層通過する時間
        resblock_quality = resblock_info[1] # 1ピクセルがResBlock一層通過して向上する品質の倍率
        gaze_yx  = gaze_coordinates[steps_per_episode] # 視線情報取得
        size_fovea = size_legacy[time_in_training][0] # フォビア領域サイズ
        size_blend = size_legacy[time_in_training][1] # ブレンド領域サイズ
        depth_fovea = depth_legacy[time_in_training][0] # フォビア領域深度
        depth_blend = depth_legacy[time_in_training][1] # ブレンド領域深度
        depth_peri = depth_legacy[time_in_training][2] # 周辺領域深度

        fovea_area = size_via_resolution(gaze_yx, resolution, size_fovea, depth_fovea, resblock_time, debug_log) # フォビア領域のピクセル数
        fovea_time = fovea_area * (depth_fovea - depth_blend) * resblock_time # フォビア領域の計算時間
        blend_area = size_via_resolution(gaze_yx, resolution, size_blend, depth_blend, resblock_time, debug_log) # ブレンド領域のピクセル数
        blend_time = blend_area * (depth_blend - depth_peri) * resblock_time # ブレンド領域の計算時間
        peri_time = resolution[0] * resolution[1] * depth_peri * resblock_time # 周辺領域の計算時間
        all_cal_time = fovea_time + blend_time + peri_time # 全計算時間

        if all_cal_time > latency_constraint: # レイテンシ制約超過をした場合、この行動をもう選択しないようにする
            action_invalid_judge = True
            debug_log.write(f'///// latency constraint exceedance /////\n')
        #debug_log.write(f'calculation time is {all_cal_time}\n')

        # 各領域の動画品質サイズを予測計算
        resolution_fovea  = [resolution[0]*resblock_quality**depth_fovea,
                          resolution[1]*resblock_quality**depth_fovea]
        resolution_blend  = [resolution[0]*resblock_quality**depth_blend,
                          resolution[1]*resblock_quality**depth_blend]
        resolution_peri  = [resolution[0]*resblock_quality**depth_peri,
                          resolution[1]*resblock_quality**depth_peri]
        debug_log.write(f'resolution_fovea: {resolution_fovea}, resolution_blend: {resolution_blend}, resolution_peri: {resolution_peri}\n')
        
        # 各領域の動画品質サイズから解像度を計算
        quality_fovea = resolution_to_quality(bitrate_list, resolution_list, resolution_fovea)
        quality_blend = resolution_to_quality(bitrate_list, resolution_list, resolution_blend)
        quality_peri = resolution_to_quality(bitrate_list, resolution_list, resolution_peri)

        # 各領域の動画全体の何％を占めるか計算
        ratio_fovea = fovea_area / (resolution[0]*resolution[1])
        ratio_blend = (blend_area-fovea_area) / (resolution[0]*resolution[1])

        quality_ave = quality_fovea*ratio_fovea + quality_blend*(ratio_blend - ratio_fovea) + quality_peri*(1-ratio_blend)
        debug_log.write(f'quality fovea: {quality_fovea}, quality blend: {quality_blend}, quality peri: {quality_peri}\n')
        debug_log.write(f'ratio_fovea: {ratio_fovea}, ratio_blend: {ratio_blend}\n')        
        
        # 重みづけされた各領域の品質を計算
        quality_fovea_weighted = calculate_weights(resolution, gaze_yx, 0, size_fovea, sigma_h, sigma_w, debug_log)*utility(quality_fovea)
        quality_blend_weighted = calculate_weights(resolution, gaze_yx, size_fovea, size_blend, sigma_h, sigma_w, debug_log)*utility(quality_blend)
        quality_peri_weighted = calculate_weights_peripheral(resolution, gaze_yx, size_blend, sigma_h, sigma_w, debug_log)*utility(quality_peri)

        quality = calculate_video_quality(gaze_yx, resolution, size_fovea, size_blend, 
                                          utility(quality_fovea), utility(quality_blend), utility(quality_peri), 
                                          sigma_h, sigma_w)
        debug_log.write(f'utility quality: {utility(quality_fovea)}, {utility(quality_blend)}, {utility(quality_peri)}\n')
        
        # リバッファリングペナルティ
        if now_bandwidth < now_rate: 
            rebuffer = calculate_rebuffering_penalty(now_bandwidth, now_rate, segment_length)
        
        # 時間ジッタ
        if steps_per_episode == 0:
            jitter_t = 0
        else:
            jitter_t = abs(quality - quality_vc_legacy[time_in_training-1])

        # 空間ジッタ
        jitter_s = (utility(quality_fovea) - utility(quality_ave))**2*ratio_fovea + (utility(quality_blend) - utility(quality_ave))**2*(ratio_blend-ratio_fovea) + (utility(quality_peri) - utility(quality_ave))**2*(1-ratio_blend)
        
        debug_log.write(f'quality_fovea: {quality_fovea}, quality_blend: {quality_blend}, quality_peri: {quality_peri}\n')

    elif mode == 2: # Adaptive FOCAS
        resolution = resolution_legacy[time_in_training] # 現在の動画品質サイズ
        resblock_time = resblock_info[0] # 1ピクセルがResBlock一層通過するのにかかる時間
        resblock_quality = resblock_info[1] # 1ピクセルがResBlock一層通過することによって向上する解像度の倍率
        gaze_yx  = gaze_coordinates[steps_per_episode] # 現在の視線座標
        size_fovea = size_legacy[time_in_training][0] # フォビア領域サイズ
        size_blend = size_legacy[time_in_training][1] # ブレンド領域サイズ
        depth_fovea = depth_legacy[time_in_training][0] # フォビア領域深度
        depth_blend = depth_legacy[time_in_training][1] # ブレンド領域深度
        depth_peri = depth_legacy[time_in_training][2] # 周辺領域深度

        fovea_area = size_via_resolution(gaze_yx, resolution, size_fovea, depth_fovea, resblock_time, debug_log) # フォビア領域のピクセル数
        fovea_time = fovea_area * (depth_fovea - depth_blend) * resblock_time # フォビア領域の計算時間
        blend_area = size_via_resolution(gaze_yx, resolution, size_blend, depth_blend, resblock_time, debug_log) # ブレンド領域のピクセル数
        blend_time = blend_area * (depth_blend - depth_peri) * resblock_time # ブレンド領域の計算時間
        peri_time = resolution[0] * resolution[1] * depth_peri * resblock_time # 周辺領域の計算時間
        all_cal_time = fovea_time + blend_time + peri_time # 全計算時間

        if all_cal_time > latency_constraint: # レイテンシ制約超過をした場合、この行動をもう選択しないようにする
            action_invalid_judge = True
            debug_log.write(f'latency constraint exceedance\n')
        debug_log.write(f'calculation time is {all_cal_time}\n')

        # 各領域の動画品質サイズを予測計算
        resolution_fovea  = [resolution[0]*resblock_quality**depth_fovea,
                          resolution[1]*resblock_quality**depth_fovea]
        resolution_blend  = [resolution[0]*resblock_quality**depth_blend,
                          resolution[1]*resblock_quality**depth_blend]
        resolution_peri  = [resolution[0]*resblock_quality**depth_peri,
                          resolution[1]*resblock_quality**depth_peri]
        debug_log.write(f'resolution_fovea: {resolution_fovea}, resolution_blend: {resolution_blend}, resolution_peri: {resolution_peri}\n')

        # 各領域の動画品質サイズから解像度を計算
        quality_fovea = resolution_to_quality(bitrate_list, resolution_list, resolution_fovea)
        quality_blend = resolution_to_quality(bitrate_list, resolution_list, resolution_blend)
        quality_peri = resolution_to_quality(bitrate_list, resolution_list, resolution_peri)

        # 各領域の動画全体の何％を占めるか計算
        ratio_fovea = fovea_area / (resolution[0]*resolution[1])
        ratio_blend = (blend_area-fovea_area) / (resolution[0]*resolution[1])

        quality_ave = quality_fovea*ratio_fovea + quality_blend*(ratio_blend - ratio_fovea) + quality_peri*(1-ratio_blend)
        debug_log.write(f'quality fovea: {quality_fovea}, quality blend: {quality_blend}, quality peri: {quality_peri}\n')
        debug_log.write(f'ratio_fovea: {ratio_fovea}, ratio_blend: {ratio_blend}\n')
        debug_log.write(f'quality average: {quality_ave}\n')
        
        # 重みづけされた各領域の品質を計算
        quality_fovea_weighted = calculate_weights(resolution, gaze_yx, 0, size_fovea, sigma_h, sigma_w, debug_log)*utility(quality_fovea)
        quality_blend_weighted = calculate_weights(resolution, gaze_yx, size_fovea, size_blend, sigma_h, sigma_w, debug_log)*utility(quality_blend)
        quality_peri_weighted = calculate_weights_peripheral(resolution, gaze_yx, size_blend, sigma_h, sigma_w, debug_log)*utility(quality_peri)

        quality = calculate_video_quality(gaze_yx, resolution, size_fovea, size_blend, 
                                          utility(quality_fovea), utility(quality_blend), utility(quality_peri), 
                                          sigma_h, sigma_w)
        debug_log.write(f'utility quality: {utility(quality_fovea)}, {utility(quality_blend)}, {utility(quality_peri)}\n')
        
        # リバッファリングペナルティ
        if now_bandwidth < now_rate: 
            rebuffer = calculate_rebuffering_penalty(now_bandwidth, now_rate, segment_length)
        
        # 時間ジッタ
        if steps_per_episode == 0:
            jitter_t = 0
        else:
            jitter_t = abs(quality - quality_vc_legacy[time_in_training-1])

        # 空間ジッタ
        jitter_s = (utility(quality_fovea) - utility(quality_ave))**2*ratio_fovea + (utility(quality_blend) - utility(quality_ave))**2*(ratio_blend-ratio_fovea) + (utility(quality_peri) - utility(quality_ave))**2*(1-ratio_blend)
        
        debug_log.write(f'quality_fovea: {quality_fovea}, quality_blend: {quality_blend}, quality_peri: {quality_peri}\n')

    # 報酬の計算
    reward = alpha * quality - beta * jitter_t - gamma * jitter_s - sigma * rebuffer

    debug_log.write(f'reward: {reward}, quality: {quality}, jitter_t: {jitter_t}, jitter_s: {jitter_s}, rebuffer: {rebuffer}\n')

    return quality, jitter_t, jitter_s, rebuffer, reward, action_invalid_judge

# https://github.com/godka/ABR-DQN.git
### Stick: A Harmonious Fusion of Buffer-based and Learning-based Approach for Adaptive Streaming
# D-DASH: A Deep Q-Learning Framework for DASH Video Streaming
# リバッファリングペナルティの計算
def calculate_rebuffering_penalty(now_bandwidth, now_rate, segment_length):
    # 帯域幅とビットレートをバイト単位で計算
    segment_size = now_rate * segment_length  # ビットからバイトへ変換
    download_time = segment_size / (now_bandwidth)  # ダウンロード時間 (秒)
    return download_time

# 各領域のサイズ（ピクセル数）
def size_via_resolution(gaze_yx, resolution, size, depth, resblock_time, debug_log):
    video_height = resolution[0]
    video_width = resolution[1]
    y = int(gaze_yx[0] * video_height / 1080)
    x = int(gaze_yx[1] * video_width / 1920)
    r = size
    # 動画外に視線座標が出ないように矯正
    x = max(0, min(video_width - 1, x))
    y = max(0, min(video_height - 1, y))

    debug_log.write(f'x: {x}, y: {y}, r: {r}, video_width: {video_width}, video_height: {video_height}\n')
    if r != 0:
        if x - r <= 0:
            if y - r <= 0:
                area = (x+r)*(y+r)
            elif y - r > 0 and y + r < video_height:
                area = r*(x+r)
            elif y + r >= video_height:
                area = (video_height+r-y)*(x+r)
        elif x - r > 0 and x + r < video_width:
            if y - r <= 0:
                area = r*(y+r)                
            elif y - r > 0 and y + r < video_height:
                area = r**2
            elif y + r >= video_height:
                area = (video_height+r-y)*r
        elif x + r >= video_width:
            if y - r <= 0:
                area = (video_width+r-x)*(y+r)
            elif y - r > 0 and y + r < video_height:
                area = r*(video_width+r-x)
            elif y + r >= video_height:
                area = (video_height+r-y)*(video_width+r-x)
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
    
    return quality

# 各領域が動画サイズ内の何％を占めるかを計算
def area_percentage(resolution, size):
    ratio = size**2 / (resolution[0] * resolution[1])
    return ratio

# eを底に持った対数計算における効用関数
def utility(bitrate):
    log = np.log(1 + bitrate)
    return log

# リストの平均値を出力
def ave_cal(legacy, max_steps_per_episode):
    sum = 0
    length = len(legacy)
    for i in range(length-max_steps_per_episode, length):
        sum += legacy[i]
    ave = sum / max_steps_per_episode
    return ave