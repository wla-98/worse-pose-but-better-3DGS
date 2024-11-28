import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import struct
import open3d as o3d

# 存储点云数据
points_data = []

def callback(msg):
    global points_data

    # 原始数据长度
    point_step = msg.point_step
    new_points = []

    for i in range(msg.width):
        # 计算偏移量
        offset = i * point_step

        # 提取 x, y, z
        x = struct.unpack_from('f', msg.data, offset)[0]
        y = struct.unpack_from('f', msg.data, offset + 4)[0]
        z = struct.unpack_from('f', msg.data, offset + 8)[0]

        # 提取 rgb，作为 32 位整数
        rgb_int = struct.unpack_from('I', msg.data, offset + 16)[0]

        # 提取 RGB 分量
        r = (rgb_int >> 16) & 0xFF
        g = (rgb_int >> 8) & 0xFF
        b = rgb_int & 0xFF
    
        # 检查 RGB 值是否在合理范围内
        if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
            new_points.append((x, y, z, r / 255.0, g / 255.0, b / 255.0))  # 归一化 RGB 值到 [0, 1]

    points_data.extend(new_points)

    # 每次更新时保存到文件
    save_to_ply()

def save_to_ply():
    if len(points_data) == 0:
        return

    # 创建 Open3D 点云对象
    cloud = o3d.geometry.PointCloud()

    # 提取点和颜色
    points = np.array([[p[0], p[1], p[2]] for p in points_data])
    colors = np.array([[p[3], p[4], p[5]] for p in points_data])

    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    # 保存为 PLY 文件
    o3d.io.write_point_cloud("point_cloud_data.ply", cloud)

def listener():
    rospy.init_node('point_cloud_listener', anonymous=True)
    rospy.Subscriber('/slam/keyframe_point3d', PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
