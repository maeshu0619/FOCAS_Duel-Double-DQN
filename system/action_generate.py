bitrate_list = [0,1,2,3]
depth_fovea_list = [6,7,8,9,10] # フォビア深度リスト
depth_blend_list = [3,4,5,6,7] # ブレンド深度リスト
depth_peri_list = [1,2,3,4,5] # 周辺深度リスト
size_list = [x / 12 for x in [0,1,2,3,4,5]] # サイズリスト


def focas_combination():
    all_combi = []
    combi = []
    for size_fovea in range(5):
        combi.append(size_fovea)
        for size_blend in range(size_fovea, 5):
            combi.append(size_blend)

            for depth_fovea in range(5):
                for depth_blend in range(5):
                    for depth_peri in range(5):
                        if depth_fovea_list[depth_fovea] > depth_blend_list[depth_blend] > depth_peri_list[depth_peri]:
                            combi.append(depth_fovea)
                            combi.append(depth_blend)
                            combi.append(depth_peri)

                            all_combi.append(combi)
                            combi = combi[:-3]

            combi = combi[:-1]
        combi = combi[:-1]

    return all_combi

def a_focas_combination():
    all_combi = []
    combi = []
    for bitrate in range(4):
        combi.append(bitrate)
        for size_fovea in range(5):
            combi.append(size_fovea)
            for size_blend in range(size_fovea, 5):
                combi.append(size_blend)

                for depth_fovea in range(5):
                    for depth_blend in range(5):
                        for depth_peri in range(5):
                            if depth_fovea_list[depth_fovea] > depth_blend_list[depth_blend] > depth_peri_list[depth_peri]:
                                combi.append(depth_fovea)
                                combi.append(depth_blend)
                                combi.append(depth_peri)

                                all_combi.append(combi)
                                combi = combi[:-3]

                combi = combi[:-1]
            combi = combi[:-1]
        combi = combi[:-1]

    return all_combi

