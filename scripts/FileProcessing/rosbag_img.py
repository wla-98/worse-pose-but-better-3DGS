import rosbag
import os
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def extract_images_from_rosbag(bag_file, output_dir):
    bridge = CvBridge()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['camera/image_raw']):
            # 获取精确到小数点后6位的时间戳
            timestamp_str = "{:.6f}".format(t.to_sec())

            image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # 生成包含时间戳的文件名
            image_name = f"{timestamp_str}.png"
            image_path = os.path.join(output_dir, image_name)

            cv2.imwrite(image_path, image)

if __name__ == "__main__":
    bag_file = "/home/wang/catkin_ws/src/wla_orb/dataset/Caterpillar.bag"
    output_dir = "/home/wang/catkin_ws/src/3dgs-dataset/tank-Caterpillar/images"

    extract_images_from_rosbag(bag_file, output_dir)