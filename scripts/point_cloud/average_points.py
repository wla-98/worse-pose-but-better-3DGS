points_dict = {}
# 打开文件进行读取
with open('/home/wang/catkin_ws/src/3dgs-dataset/tum-fg3-long-office-household/sparse/0/points3D.txt', 'r') as file:
    for line in file.readlines():
        if line.startswith('#'):
            continue
        parts = line.strip().split(' ')
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        point_key = (x, y, z)
        r = int(parts[4])
        g = int(parts[5])
        b = int(parts[6])
        error = float(parts[7])
        track_info = list(map(int, parts[8:]))
        if point_key in points_dict:
            # 对于重复的点，合并信息并求平均颜色值
            existing_data = points_dict[point_key]
            existing_data[0] = (existing_data[0] + r) / 2
            existing_data[1] = (existing_data[1] + g) / 2
            existing_data[2] = (existing_data[2] + b) / 2
            existing_data[3] += error
            existing_data[4].extend(track_info)
        else:
            points_dict[point_key] = [r, g, b, error, track_info]

# 整理数据并保存到新文件（可以根据需求修改保存的文件名）
with open('/home/wang/catkin_ws/src/3dgs-dataset/tum-fg3-long-office-household/sparse/0/points3D.txt', 'w') as new_file:
    id = 1
    for point_key, data in points_dict.items():
        x, y, z = point_key
        r = int(round(data[0]))
        g = int(round(data[1]))
        b = int(round(data[2]))
        error = data[3]
        track_info = " ".join(map(str, data[4]))
        new_file.write(f"{id} {point_key[0]} {point_key[1]} {point_key[2]} {r} {g} {b} {error} {track_info}\n")
        id += 1