import numpy as np

def quaternion_inverse(q):
    """计算四元数的逆（假设四元数已归一化）"""
    qw, qx, qy, qz = q
    return [qw, -qx, -qy, -qz]

def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    
def transform_tcw_to_twc(q, t):
    """
    从 T_CW 转换到 T_WC
    :param q: 四元数 [qw, qx, qy, qz]
    :param t: 平移向量 [tx, ty, tz]
    :return: 转换后的 [qw, qx, qy, qz], [tx, ty, tz]
    """
    q_inv = quaternion_inverse(q)
    t_wc = -np.dot(quaternion_to_rotation_matrix(q_inv), np.array(t))
    return q_inv, t_wc.tolist()

def convert_images_to_keyframe(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        lines = infile.readlines()

        # 跳过以 '#' 开头的注释行，找到第一个非注释行的索引
        start_index = 0
        for i, line in enumerate(lines):
            if not line.strip().startswith('#'):
                start_index = i
                break

        # 处理数据行并提取时间戳
        metadata_lines = []
        for i in range(start_index, len(lines), 2):
            metadata = lines[i].strip().split()
            if len(metadata) < 10:
                print(f"Skipping line {i+1}: insufficient data -> {metadata}")
                continue

            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = metadata

            if not name.endswith('.png'):
                print(f"Skipping line {i+1}: invalid NAME format -> {name}")
                continue

            # 提取时间戳并存储元数据
            timestamp = name.replace('.png', '')
            metadata_lines.append((timestamp, metadata))

        # 按时间戳排序
        metadata_lines.sort(key=lambda x: float(x[0]))  # 假设时间戳是数字字符串

        # 转换 T_CW 到 T_WC 并写入输出文件
        for timestamp, metadata in metadata_lines:
            qw, qx, qy, qz, tx, ty, tz = map(float, metadata[1:8])  # 提取四元数和平移
            q_cw = [qw, qx, qy, qz]
            t_cw = [tx, ty, tz]
            q_wc, t_wc = transform_tcw_to_twc(q_cw, t_cw)

            # 输出新的格式行
            outfile.write(f"{timestamp} {t_wc[0]} {t_wc[1]} {t_wc[2]} {q_wc[1]} {q_wc[2]} {q_wc[3]} {q_wc[0]}\n")

# 使用示例
input_file = '/home/wang/catkin_ws/src/3dgs-dataset/kitti/sparse-colmap/0/images.txt'
output_file = '/home/wang/catkin_ws/src/3dgs-dataset/kitti/colmap.txt'
convert_images_to_keyframe(input_file, output_file)
print(f"Conversion completed! Keyframe data saved to {output_file}")
