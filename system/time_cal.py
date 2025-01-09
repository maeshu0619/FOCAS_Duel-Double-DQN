def debug_time(seconds):
    hours = int(seconds // 3600)  # 秒を3600で割って時間を計算
    minutes = int((seconds % 3600) // 60)  # 残り秒数から分を計算
    secs = seconds % 60  # 残り秒数を計算

    # フォーマットされた文字列を返す
    return f"{hours}h {minutes}m {secs:.2f}s"
