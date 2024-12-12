# 输入文件路径和输出文件路径
input_file = "/home/wang/catkin_ws/src/3dgs-dataset/tum-fg2-desk-orb/keyframe.txt"
output_file = "/home/wang/catkin_ws/src/3dgs-dataset/tum-fg2-desk-orb/keyframe_true.txt"

# 读取并处理文件
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # 按空格分割行
        parts = line.split()
        if parts:
            # 处理第一项到小数点后4位
            parts[0] = f"{float(parts[0]):.4f}"
            # 将处理后的行写入输出文件
            outfile.write(" ".join(parts) + "\n")

print("文件处理完成，已保存到:", output_file)
