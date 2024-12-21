import os
import shutil
from pathlib import Path
import numpy as np

def convert_TWC_to_TCW(tx, ty, tz, qx, qy, qz, qw):
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # Translation vector and rotation matrix for TWC
    TWC_translation = np.array([tx, ty, tz])
    TWC_rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # Invert TWC to get TCW
    TCW_rotation = TWC_rotation.T
    TCW_translation = -np.dot(TCW_rotation, TWC_translation)

    # Convert back to quaternion
    TCW_quaternion = R.from_matrix(TCW_rotation).as_quat()
    return TCW_translation.tolist() + TCW_quaternion.tolist()

def match_images(images_txt_path, images_folder, output_folder, output_txt_path):
    # 读取images.txt中的时间戳和其他数据
    with open(images_txt_path, 'r') as f:
        timestamps_data = [line.strip().split() for line in f if line.strip()]

    # 提取时间戳列表
    timestamps = [float(data[0]) for data in timestamps_data]

    # 获取images文件夹下所有图片文件
    image_files = [file for file in os.listdir(images_folder) if file.endswith(".png")]

    # 提取文件名中的时间戳并与四位小数对齐
    image_timestamps = {
        float(file[:-4]): file for file in image_files
    }

    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 准备保存到新的 images.txt 文件的内容
    output_txt_lines = []

    for idx, ts in enumerate(timestamps, start=1):
        # 找到与当前时间戳最接近的图片时间戳
        closest_ts = min(image_timestamps.keys(), key=lambda x: abs(x - ts))
        
        # 获取对应的图片文件名
        image_file = image_timestamps[closest_ts]

        # 源文件和目标文件路径
        src_path = os.path.join(images_folder, image_file)
        dest_path = os.path.join(output_folder, image_file)

        # 复制文件
        shutil.copy2(src_path, dest_path)
        print(f"Matched {ts} to {closest_ts} -> {image_file}")

        # 第一行信息
        tx, ty, tz, qx, qy, qz, qw = map(float, timestamps_data[idx - 1][1:])
        twc_tx, twc_ty, twc_tz, twc_qx, twc_qy, twc_qz, twc_qw = convert_TWC_to_TCW(tx, ty, tz, qx, qy, qz, qw)
        output_txt_lines.append(f"{idx} {twc_qw} {twc_qx} {twc_qy} {twc_qz} {twc_tx} {twc_ty} {twc_tz} 1 {image_file}")

        # 第二行信息
        output_txt_lines.append("0 0 0")

    # 保存到新的 images.txt 文件
    with open(output_txt_path, 'w') as f:
        f.write("# IMAGE_ID QW  QX  QY  QZ  TX  TY  TZ  CAMERA_ID  NAME\n")
        f.write("# POINTS2D[] as (X  Y  POINT3D_ID)\n")
        f.write("\n".join(output_txt_lines))

# 示例使用
images_txt_path = '/home/wang/catkin_ws/src/sampled_ground_truth.txt'  
images_folder = '/home/wang/catkin_ws/src/3dgs-dataset/tum-fg2-desk-orb/images'  # 替换为你的 images 文件夹路径
output_folder = '/home/wang/catkin_ws/src/images'  # 替换为保存匹配图片的文件夹路径
output_txt_path = 'images.txt'  # 替换为新的 images.txt 文件路径

match_images(images_txt_path, images_folder, output_folder, output_txt_path)
