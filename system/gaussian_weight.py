import numpy as np

def calculate_video_quality(gaze_yx, resolution, size_fovea, size_blend,
                            quality_fovea, quality_blend, quality_peri, sigma_h, sigma_w):
    ratio = 270 / resolution[1] # サイズを270*490に合わせる

    video_height = resolution[1] * ratio
    video_width = resolution[0] * ratio

    gaze_y = gaze_yx[0]
    gaze_x = gaze_yx[1]

    # 視線座標を調整（動画範囲内に制限）
    gaze_y = max(0, min(video_height - 1, gaze_y))
    gaze_x = max(0, min(video_width - 1, gaze_x))
    
    size_fovea = size_fovea * ratio
    size_blend = size_blend * ratio
    
    # ガウス分布の重みを計算
    sigma_h = sigma_h*ratio
    sigma_w = sigma_w*ratio
    
    # 各領域のマスク作成
    y_coords, x_coords = np.meshgrid(np.arange(video_height), np.arange(video_width), indexing='ij')

    # フォビア領域
    fovea_mask = ((gaze_y - size_fovea) <= y_coords) & (y_coords <= (gaze_y + size_fovea)) & \
                 ((gaze_x - size_fovea) <= x_coords) & (x_coords <= (gaze_x + size_fovea))

    # 中間領域
    blend_mask = ((gaze_y - size_blend) <= y_coords) & (y_coords <= (gaze_y + size_blend)) & \
                 ((gaze_x - size_blend) <= x_coords) & (x_coords <= (gaze_x + size_blend)) & \
                 ~fovea_mask

    # 周辺領域
    peripheral_mask = ~fovea_mask & ~blend_mask
    
    # ガウス分布による重みの計算
    gaussian_weights = (1 / (2 * np.pi * sigma_h * sigma_w)) * np.exp(
        -(((x_coords - gaze_x) ** 2) / (2 * sigma_h ** 2) + ((y_coords - gaze_y) ** 2) / (2 * sigma_w ** 2))
    )

    # フォビア領域、中間領域、周辺領域の重みを取得
    fovea_weight = np.sum(gaussian_weights[fovea_mask]) * quality_fovea
    blend_weight = np.sum(gaussian_weights[blend_mask]) * quality_blend
    peripheral_weight = np.sum(gaussian_weights[peripheral_mask]) * quality_peri
    
    # 全体の重みを計算
    total_weight = fovea_weight + blend_weight + peripheral_weight

    # 動画全体の品質を計算
    total_quality = total_weight / np.sum(gaussian_weights)

    return total_quality

# 指定されたマスク範囲内のみでガウス分布の重みを計算。
def calculate_gaussian_weights_optimized(video_width, video_height, y, x, sigma_h, sigma_w, mask):
    y_coords, x_coords = np.where(mask)

    # ガウス分布の重みをマスク範囲内で計算
    weights = (1 / (2 * np.pi * sigma_h * sigma_w)) * np.exp(
        -(((x_coords - x) ** 2) / (2 * sigma_h ** 2) + ((y_coords - y) ** 2) / (2 * sigma_w ** 2))
    )

    return weights, y_coords, x_coords

def calculate_weights(resolution, gaze_yx, inner_radius, outer_radius, sigma_h, sigma_w, debug_log):
    video_height, video_width = resolution
    ratio_small = 270 / video_height
    ratio = 1 / 4
    video_height, video_width = video_height*ratio_small, video_width*ratio_small

    y, x = int(gaze_yx[0]*ratio), int(gaze_yx[1]*ratio) 
    inner_radius = inner_radius * ratio_small
    outer_radius = outer_radius * ratio_small
    
    # 動画外に視線座標が出ないように矯正
    x = max(0, min(video_width, x))
    y = max(0, min(video_height, y))

    # 各ピクセルの座標を生成
    y_coords, x_coords = np.meshgrid(np.arange(video_height), np.arange(video_width), indexing='ij')

    # 各ピクセルから視線位置までの距離を計算
    distances = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)

    # ドーナツ型領域のマスクを作成
    mask = (distances >= inner_radius) & (distances <= outer_radius)

    # ガウス分布の重みを最適化して計算（マスク範囲内のみ）
    weights, mask_y, mask_x = calculate_gaussian_weights_optimized(video_width, video_height, y, x, sigma_h, sigma_w, mask)

    # 領域内のガウス分布重みの総和を計算
    weighted_sum = np.sum(weights)

    # デバッグログを記録
    debug_log.write(
        f"Weghts: {weighted_sum}, Resolution: {resolution}, Gaze: {gaze_yx}, Inner Radius: {inner_radius}, Outer Radius: {outer_radius}\n"
    )


    return weighted_sum

def calculate_weights_peripheral(resolution, gaze_yx, inner_radius, sigma_h, sigma_w, debug_log):
    video_height, video_width = resolution
    ratio_small = 270 / video_height
    ratio = 1 / 4
    video_height, video_width = video_height*ratio_small, video_width*ratio_small

    y, x = int(gaze_yx[0]*ratio), int(gaze_yx[1]*ratio) 
    inner_radius = inner_radius * ratio_small
    
    # 動画外に視線座標が出ないように矯正
    x = max(0, min(video_width, x))
    y = max(0, min(video_height, y))

    # 各ピクセルの座標を生成
    y_coords, x_coords = np.meshgrid(np.arange(video_height), np.arange(video_width), indexing='ij')

    # 各ピクセルから視線位置までの距離を計算
    distances = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)

    # 内円より外側の領域をマスク
    mask = distances >= inner_radius

    # ガウス分布の重みを計算
    weights = (1 / (2 * np.pi * sigma_h * sigma_w)) * np.exp(
        -(((x_coords - x) ** 2) / (2 * sigma_h ** 2) + ((y_coords - y) ** 2) / (2 * sigma_w ** 2))
    )

    # 内円より外側のガウス分布重みの総和を計算
    weighted_sum = np.sum(weights[mask])

    # デバッグログを記録
    debug_log.write(
        f"Weghts: {weighted_sum}, Resolution: {resolution}, Gaze: {gaze_yx}, radius: {inner_radius}\n"
    )

    # 重みを返す
    return weighted_sum



"""

def aaa(resolution, gaze_yx, inner_radius, sigma_h, sigma_w):
    video_height, video_width = resolution
    ratio_small = 270 / video_height
    ratio = 1 / 4
    video_height, video_width = video_height*ratio_small, video_width*ratio_small

    y, x = int(gaze_yx[0]*ratio), int(gaze_yx[1]*ratio) 
    inner_radius = inner_radius * ratio_small

    # 動画外に視線座標が出ないように矯正
    x = max(0, min(video_width, x))
    y = max(0, min(video_height, y))
    print(f'{video_height}, {video_width}, {y}, {x}, {inner_radius}')


    # 各ピクセルの座標を生成
    y_coords, x_coords = np.meshgrid(np.arange(video_height), np.arange(video_width), indexing='ij')

    # 各ピクセルから視線位置までの距離を計算
    distances = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)

    # 内円より外側の領域をマスク
    mask = distances >= inner_radius

    # ガウス分布の重みを計算
    weights = (1 / (2 * np.pi * sigma_h * sigma_w)) * np.exp(
        -(((x_coords - x) ** 2) / (2 * sigma_h ** 2) + ((y_coords - y) ** 2) / (2 * sigma_w ** 2))
    )

    # 内円より外側のガウス分布重みの総和を計算
    weighted_sum = np.sum(weights[mask])

    # 重みを返す
    return weighted_sum

import time
r1 = [1080, 1920]
r2 = [540, 960]
r3 = [270, 480]
r4 = [108, 192]
g1 = [2000, 2000]
g2 = [100, 100]
g3 = [50, 20]
inner_radius = 0
sigma_h = 64
sigma_w = 64

a1 = aaa(r1, g1, inner_radius, sigma_h, sigma_w)
a2 = aaa(r2, g1, inner_radius, sigma_h, sigma_w)
a3 = aaa(r3, g1, inner_radius, sigma_h, sigma_w)
a4 = aaa(r4, g1, inner_radius, sigma_h, sigma_w)
b1 = aaa(r1, g2, inner_radius, sigma_h, sigma_w)
b2 = aaa(r2, g2, inner_radius, sigma_h, sigma_w)
b3 = aaa(r3, g2, inner_radius, sigma_h, sigma_w)
b4 = aaa(r4, g2, inner_radius, sigma_h, sigma_w)
c1 = aaa(r1, g3, inner_radius, sigma_h, sigma_w)
c2 = aaa(r2, g3, inner_radius, sigma_h, sigma_w)
c3 = aaa(r3, g3, inner_radius, sigma_h, sigma_w)
c4 = aaa(r4, g3, inner_radius, sigma_h, sigma_w)

def t():
    #st = time.time()
    print(f'{a1}, {a1/a1}')
    print(f'{a2}, {a2/a1}')
    print(f'{a3}, {a3/a1}')
    print(f'{a4}, {a4/a1}\n')
    print(f'{b1}, {b1/b1}')
    print(f'{b2}, {b2/b1}')
    print(f'{b3}, {b3/b1}')
    print(f'{b4}, {b4/b1}\n')
    print(f'{c1}, {c1/c1}')
    print(f'{c2}, {c2/c1}')
    print(f'{c3}, {c3/c1}')
    print(f'{c4}, {c4/c1}')
    #et = time.time()
    #print(et-st)

t()

"""