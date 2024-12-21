import random


def select_and_renumber(input_file_path, output_file_path):
    selected_lines = []
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()
        header = lines[0:2]
        data_lines = lines[2:]
        random.seed()
        selected_indices = random.sample(range(len(data_lines)), 50)
        for index in sorted(selected_indices):
            selected_lines.append(data_lines[index])

    with open(output_file_path, 'w') as outfile:
        outfile.writelines(header)
        for new_id, line in enumerate(selected_lines, 1):
            parts = line.split(" ")
            parts[0] = str(new_id)
            new_line = " ".join(parts)
            outfile.write(new_line)


# 示例用法
input_file_path = '/home/wang/catkin_ws/src/3dgs-dataset/tum-fg2-desk-orb/sparse/0/points3D.txt'
output_file_path = '/home/wang/catkin_ws/src/3dgs-dataset/tum-fg2-desk-orb/sparse/0/points3D.txt'
select_and_renumber(input_file_path, output_file_path)