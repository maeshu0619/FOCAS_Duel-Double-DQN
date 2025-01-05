import numpy as np

def calculate_gaussian_weights(video_width, video_height, y, x, sigma_h, sigma_w):
    # 各ピクセルの座標を生成
    y_coords, x_coords = np.meshgrid(np.arange(video_height), np.arange(video_width), indexing='ij')

    # ガウス分布の重みを計算
    weights = (1 / (2 * np.pi * sigma_h * sigma_w)) * np.exp(
        -(((x_coords - x) ** 2) / (2 * sigma_h ** 2) + ((y_coords - y) ** 2) / (2 * sigma_w ** 2))
    )

    return weights

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
    y, x = int(gaze_yx[0]), int(gaze_yx[1])
    inner_radius = inner_radius * 1920 / video_width
    outer_radius = outer_radius * 1920 / video_width
    
    # 動画外に視線座標が出ないように矯正
    x = max(0, min(video_width - 1, x))
    y = max(0, min(video_height - 1, y))

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
        f"Resolution: {resolution}, Gaze: {gaze_yx}, Inner Radius: {inner_radius}, "
        f"Outer Radius: {outer_radius}, Weighted Sum: {weighted_sum}\n"
    )


    return weighted_sum

def calculate_weights_peripheral(resolution, gaze_yx, inner_radius, sigma_h, sigma_w, debug_log):
    video_height, video_width = resolution
    y, x = int(gaze_yx[0]), int(gaze_yx[1])
    inner_radius = inner_radius * 1920 / video_width

    # 動画外に視線座標が出ないように矯正
    x = max(0, min(video_width - 1, x))
    y = max(0, min(video_height - 1, y))

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
    debug_log.write(f'video height: {video_height}, video width: {video_width}, gaze(y, x): ({y}, {x}), weights: {weights}, weighted_sum: {weighted_sum}')

    # 重み付けされた品質を返す
    return weighted_sum
