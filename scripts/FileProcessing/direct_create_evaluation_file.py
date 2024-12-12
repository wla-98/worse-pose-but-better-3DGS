import numpy as np

def convert_images_to_keyframe(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        lines = infile.readlines()

        # 跳过以 '#' 开头的注释行，找到第一个非注释行的索引
        start_index = 0
        for i, line in enumerate(lines):
            if not line.strip().startswith('#'):
                start_index = i
                break

        # 从第一个非注释行开始处理
        for i in range(start_index, len(lines), 2):
            metadata = lines[i].strip().split()

            # 检查元数据行是否有足够的字段
            if len(metadata) < 10:
                print(f"Skipping line {i+1}: insufficient data -> {metadata}")
                continue

            # 解包元数据
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = metadata

            # 从 NAME 中提取 timestamp
            if not name.endswith('.png'):
                print(f"Skipping line {i+1}: invalid NAME format -> {name}")
                continue

            timestamp = name.replace('.png', '')

            # 转换 T_CW 到 t_cw
            q_cw = [float(qw), float(qx), float(qy), float(qz)]
            t_cw = [float(tx), float(ty), float(tz)]

            # 输出新的格式行
            outfile.write(f"{timestamp} {t_cw[0]:.8f} {t_cw[1]:.8f} {t_cw[2]:.8f} {q_cw[1]:.8f} {q_cw[2]:.8f} {q_cw[3]:.8f} {q_cw[0]:.8f}\n")

# 使用示例
input_file = '/home/wang/catkin_ws/src/3dgs-dataset/tum-fg2-desk-orb/sparse/0/images.txt'
output_file = '/home/wang/catkin_ws/src/3dgs-dataset/tum-fg2-desk-orb/keyframe_test.txt'
convert_images_to_keyframe(input_file, output_file)
print(f"Conversion completed! Keyframe data saved to {output_file}")
