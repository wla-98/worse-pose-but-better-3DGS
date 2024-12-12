import open3d as o3d
import numpy as np
import struct

# 读取自定义点云格式
def read_custom_ply(file_path):
    with open(file_path, 'rb') as f:
        lines = []
        while True:
            line = f.readline().decode('utf-8').strip()
            if line == "end_header":
                break
            lines.append(line)

        header = "\n".join(lines)
        properties = [line for line in lines if line.startswith("property")]
        dtype = []
        for prop in properties:
            parts = prop.split()
            if len(parts) >= 3:
                dtype.append((parts[2], np.float32))  # assuming float properties

        vertex_count = int([line for line in lines if line.startswith("element vertex")][0].split()[-1])
        data = np.frombuffer(f.read(), dtype=dtype, count=vertex_count)

    return header, data

# 写入自定义点云格式
def write_custom_ply(file_path, header, data):
    # 更新 header 中的点云个数
    updated_header = []
    for line in header.split("\n"):
        if line.startswith("element vertex"):
            updated_header.append(f"element vertex {len(data)}")
        else:
            updated_header.append(line)
    
    updated_header = "\n".join(updated_header)

    with open(file_path, 'wb') as f:
        f.write((updated_header + "\nend_header\n").encode('utf-8'))
        f.write(data.tobytes())


def remove_outliers(data, nb_neighbors=30, std_ratio=2, radius=0.3, min_points=10):
    # 构建 Open3D 点云对象
    xyz = np.vstack((data['x'], data['y'], data['z'])).T
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)

    # 1. 统计滤波移除离群点
    _, ind_stat = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # 2. 半径滤波移除离群点
    _, ind_radius = point_cloud.remove_radius_outlier(nb_points=min_points, radius=radius)

    # 3. 组合滤波结果：取两个滤波结果的交集
    ind_combined = set(ind_stat).intersection(ind_radius)

    # 4. 构造掩码数组保留有效点
    mask = np.zeros(len(data), dtype=bool)
    mask[list(ind_combined)] = True

    # 5. 返回过滤后的数据
    filtered_data = data[mask]
    return filtered_data

# 主函数
def main():
    input_ply = "/home/wang/catkin_ws/src/3dgs-dataset/lab-small-room/3dgs-colmap/point_cloud/iteration_30000/point_cloud1.ply"
    output_ply = "/home/wang/catkin_ws/src/3dgs-dataset/lab-small-room/3dgs-colmap/point_cloud/iteration_30000/point_cloud.ply"

    # 读取点云
    header, data = read_custom_ply(input_ply)

    # 移除离群点
    filtered_data = remove_outliers(data, nb_neighbors=20, std_ratio=2.0)

    # 保存过滤后的点云
    write_custom_ply(output_ply, header, filtered_data)
    print(f"Filtered point cloud saved to {output_ply}")

if __name__ == "__main__":
    main()
