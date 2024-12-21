import random
import numpy as np


def sample_lines(input_file, output_file, num_lines_to_sample):
    """
    从输入文件中均匀随机选择指定行数的内容，并写入到输出文件中。

    参数:
    input_file (str): 输入文件的路径
    output_file (str): 输出文件的路径
    num_lines_to_sample (int): 要随机选择的行数
    """
    all_lines = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                all_lines.append(line)

    total_lines = len(all_lines)
    if num_lines_to_sample >= total_lines:
        print("要采样的行数不能大于或等于文件总行数")
        return

    # 生成等间隔的索引，确保均匀采样
    step = total_lines // num_lines_to_sample
    selected_indices = np.arange(0, total_lines, step)[:num_lines_to_sample]
    selected_lines = [all_lines[i] for i in selected_indices]

    with open(output_file, 'w') as f:
        f.writelines(selected_lines)


# 使用示例
input_file_path = "/home/wang/catkin_ws/src/3dgs-dataset/tum-fg2-desk-orb/groundtruth.txt"
output_file_path = "sampled_ground_truth.txt"
num_lines_to_sample = 261
sample_lines(input_file_path, output_file_path, num_lines_to_sample)