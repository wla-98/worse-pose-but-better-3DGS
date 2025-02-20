import open3d as o3d
import numpy as np

def read_pointcloud_from_file(filename):
    points = []
    colors = []

    with open(filename, 'r') as file:
        for line in file:
            # Skip comments and header lines
            if line.startswith('#') or line.strip() == "":
                continue
            
            # Split the line into tokens
            tokens = line.strip().split()

            
            # Extract X, Y, Z, R, G, B
            x, y, z = map(float, tokens[1:4])
            r, g, b = map(int, tokens[4:7])

            points.append([x, y, z])
            colors.append([b/ 255.0, g / 255.0, r / 255.0])  # Normalize colors to [0, 1]

    return np.array(points), np.array(colors)

def visualize_pointcloud(points, colors):
    # Create Open3D point cloud object
    pointcloud = o3d.geometry.PointCloud()
    
    # Set points and colors
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pointcloud])

if __name__ == "__main__":
    # Path to the point cloud file
    filename = "/home/wang/catkin_ws/src/3dgs-dataset/kitti/sparse-keyframe/0/points3D.txt"  # Replace with your file path
    
    # Read the point cloud data
    points, colors = read_pointcloud_from_file(filename)
    
    # Visualize the point cloud
    visualize_pointcloud(points, colors)
