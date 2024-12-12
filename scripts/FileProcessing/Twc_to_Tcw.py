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

        # 从第一个非注释行开始处理
        for i in range(0, len(lines)):
            metadata = lines[i].strip().split()

            # 解包元数据
            timestamp, tx, ty, tz, qx, qy, qz, qw= metadata

            # 转换 T_CW 到 T_WC
            q_cw = [float(qw), float(qx), float(qy), float(qz)]
            t_cw = [float(tx), float(ty), float(tz)]
            q_wc, t_wc = transform_tcw_to_twc(q_cw, t_cw)

            # 输出新的格式行
            outfile.write(f"{timestamp} {t_wc[0]:.8f} {t_wc[1]:.8f} {t_wc[2]:.8f} {q_wc[1]:.8f} {q_wc[2]:.8f} {q_wc[3]:.8f} {q_wc[0]:.8f}\n")

# 使用示例
# input_file = '/home/wang/project/gaussian-splatting-pose/camera_poses_init.txt'
# output_file = '/home/wang/project/gaussian-splatting-pose/camera_poses_init_evo.txt'

input_file = '/home/wang/catkin_ws/src/3dgs-dataset/tank-family/camera_poses_final.txt'
output_file = '/home/wang/catkin_ws/src/3dgs-dataset/tank-family/camera_poses_final_evo.txt'
convert_images_to_keyframe(input_file, output_file)
print(f"Conversion completed! Keyframe data saved to {output_file}")
