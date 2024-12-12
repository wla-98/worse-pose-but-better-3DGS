import time
import sys
import os
import rosbag
import roslib
import rospy
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image

TOPIC = 'camera/image_raw'

def my_resize(my_img, x, y):
    resized = cv2.resize(my_img, (int(my_img.shape[1] * x), int(my_img.shape[0] * y)))
    return resized


def CreateVideoBag(videopath, bagname):
    """Creates a bag file with a video file"""
    print(videopath)
    print(bagname)
    bag = rosbag.Bag(bagname, 'w')
    cap = cv2.VideoCapture(videopath)
    cb = CvBridge()
    prop_fps = cap.get(cv2.CAP_PROP_FPS)  # 帧速率

    if prop_fps!= prop_fps or prop_fps <= 1e-2:
        print("Warning: can't get FPS. Assuming 24.")
        prop_fps = 29.97

    print(prop_fps)

    target_fps = 15  # 目标帧率设置为15帧
    frame_id = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 降低分辨率为原来的四分之一
        frame = my_resize(frame, 1 / 4, 1 / 4)

        if frame_count % (int(prop_fps) // target_fps) == 0:
            stamp = rospy.rostime.Time.from_sec(float(frame_id) / prop_fps)
            frame_id += 1
            image = cb.cv2_to_imgmsg(frame, encoding='bgr8')
            image.header.stamp = stamp
            image.header.frame_id = "camera"
            bag.write(TOPIC, image, stamp)

        frame_count += 1

    cap.release()
    bag.close()


if __name__ == "__main__":
    CreateVideoBag('/home/wang/catkin_ws/src/wla_orb/dataset/Caterpillar.mp4',
                   '/home/wang/catkin_ws/src/wla_orb/dataset/Caterpillar.bag')