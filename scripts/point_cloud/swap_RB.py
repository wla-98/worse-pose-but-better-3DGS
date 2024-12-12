def swap_r_b(input_file_path, output_file_path):
    id=1
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            if line.startswith('#'):
                # 直接写入注释行
                outfile.write(line)
            else:
                # 按空格分割每行数据
                parts = line.strip().split(" ")
                # 交换R和B的值
                r_index = 4
                b_index = 6
                parts[r_index], parts[b_index] = parts[b_index], parts[r_index]
                parts[0]=str(id)
                id+=1
                # 重新拼接成新行
                new_line = " ".join(parts)
                outfile.write(new_line + '\n')

# 示例用法
input_file_path = '/home/wang/catkin_ws/src/wla_orb/orb-output/20241126_201214/points3D-map.txt'
output_file_path = '/home/wang/catkin_ws/src/wla_orb/orb-output/20241126_201214/points3D.txt'
swap_r_b(input_file_path, output_file_path)