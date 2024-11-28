def add_id_and_append_to_txt(input_file_path, output_file_path):
    new_lines = []
    id_num = 1
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            if line.startswith('#'):
                new_lines.append(line)
            else:
                new_line = str(id_num) + " " + line.strip() + " 0 0 0\n"
                new_lines.append(new_line)
                id_num += 1

    with open(output_file_path, 'w') as outfile:
        outfile.writelines(new_lines)

# 示例用法
input_file_path = '/home/wang/catkin_ws/src/3dgs-dataset/tum-fg1-xyz/sparse-map/0/points3D.txt'
output_file_path = '/home/wang/catkin_ws/src/3dgs-dataset/tum-fg1-xyz/sparse-map/0/points3D.txt'
add_id_and_append_to_txt(input_file_path, output_file_path)