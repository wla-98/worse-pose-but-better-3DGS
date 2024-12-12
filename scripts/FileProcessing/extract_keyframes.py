import os
import shutil

# 文件路径和文件夹
images_txt_path = "/home/wang/catkin_ws/src/3dgs-dataset/kitti/sparse/0/images.txt"  # 替换为 images.txt 文件路径
source_folder = "/home/wang/catkin_ws/src/3dgs-dataset/kitti/images"
destination_folder = "/home/wang/catkin_ws/src/3dgs-dataset/kitti/images-keyframes"

# 创建目标文件夹（如果不存在）
os.makedirs(destination_folder, exist_ok=True)

# 提取图像名称
image_names = []
with open(images_txt_path, 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines), 2):  # 每两行为一组，取第一行
        parts = lines[i].strip().split()
        if parts:  # 确保非空
            image_name = parts[-1]  # 取最后一个部分
            image_names.append(image_name)

# 复制图像到目标文件夹
for image_name in image_names:
    source_path = os.path.join(source_folder, image_name)
    destination_path = os.path.join(destination_folder, image_name)
    if os.path.exists(source_path):  # 确保源文件存在
        shutil.copy(source_path, destination_path)
    else:
        print(f"Image {source_path} not found.")

print("All keyframes have been processed.")