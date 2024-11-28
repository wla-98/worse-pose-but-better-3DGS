import os
import rosbag
from cv_bridge import CvBridge
import cv2
import numpy as np

import rospy


def generate_rosbag(image_folder, times_file, output_bag_file):
    bridge = CvBridge()

    with rosbag.Bag(output_bag_file, 'w') as bag:
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
        with open(times_file, 'r') as f:
            time_stamps = f.readlines()

        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                continue

            time_stamp = float(time_stamps[i].strip())

            img_msg = bridge.cv2_to_imgmsg(image, encoding='bgr8')
            img_msg.header.stamp = rospy.Time.from_sec(time_stamp)
            img_msg.header.frame_id = str(i)

            bag.write('/image', img_msg, img_msg.header.stamp)

    print(f"已成功生成rosbag文件: {output_bag_file}")


if __name__ == "__main__":
    image_folder = '/home/wang/catkin_ws/src/wla_orb/dataset/kitti/00/image_2'
    times_file = '/home/wang/catkin_ws/src/wla_orb/dataset/kitti/00/times.txt'
    output_bag_file = '/home/wang/catkin_ws/src/wla_orb/dataset/kitti-00.bag'

    generate_rosbag(image_folder, times_file, output_bag_file)