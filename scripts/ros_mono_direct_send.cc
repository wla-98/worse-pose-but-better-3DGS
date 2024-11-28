/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <sys/stat.h> // For mkdir function
#include <sys/types.h>
#include <errno.h>    // For errno
#include <string.h>   // For strerror
#include <geometry_msgs/PoseStamped.h>  // 用于发布相机位姿
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <thread>  // To create separate threads
#include <atomic>  // To use atomic flags for safe thread synchronization
#include "../include/System.h"
#include "../include/ROSMassageCreate.h"

using namespace std;

class ImageGrabber {
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ros::Publisher& pose_pub, ros::Publisher& rgb_pub, ros::Publisher& points_pub)
        : mpSLAM(pSLAM), posePublisher(pose_pub), rgbPublisher(rgb_pub), pointPublisher(points_pub) {
        prevKeyFrameId = -1;
        finishedGlobalBA = false;
        insertKeyframe = false;
    }
    
    void GrabImage(const sensor_msgs::ImageConstPtr& msg);
    
    void ProcessKeyFrame(int frameId, const std::tuple<sensor_msgs::Image, geometry_msgs::PoseStamped, sensor_msgs::PointCloud2>& data);
    // Background thread function to check the flags and publish messages
    void BackgroundThread();
    std::thread backgroundThread;  // Thread to run the background check
    // 使用 unordered_set 记录已经发布的关键帧
    std::unordered_set<int> publishedFrames;
private:
    ORB_SLAM3::System* mpSLAM;
    ros::Publisher posePublisher;
    ros::Publisher rgbPublisher;
    ros::Publisher pointPublisher;
    sensor_msgs::ImagePtr CurRGBImage;
    int prevKeyFrameId;
    bool finishedGlobalBA;
    bool insertKeyframe;

    std::mutex mtx;  // Mutex for synchronizing access to flags
};

// Define the method to create PoseStamped messages
void ImageGrabber::ProcessKeyFrame(int frameId, const std::tuple<sensor_msgs::Image, geometry_msgs::PoseStamped, sensor_msgs::PointCloud2>& data) {
    // 从 mKeyFrameData 中解包数据
    const sensor_msgs::Image& image = std::get<0>(data);              // RGB 图像
    const geometry_msgs::PoseStamped& poseMsg = std::get<1>(data);    // 位姿消息
    const sensor_msgs::PointCloud2& cloudMsg = std::get<2>(data);     // 三维点云
    // 发布 RGB 图像
    rgbPublisher.publish(image);
    // 发布位姿
    posePublisher.publish(poseMsg);
    // 发布三维点云
    pointPublisher.publish(cloudMsg);

}

// Background thread to check if new keyframes are available and process them
void ImageGrabber::BackgroundThread() {
    ros::Rate rate(25);  // 每秒检查25次
    while (ros::ok()) {
        std::lock_guard<std::mutex> lock(mtx);
        bool isLoopClosure = mpSLAM->mpLoopCloser->finishedGlobalBA;
        if (!isLoopClosure) {
            // 遍历 mKeyFrameData，处理尚未发布的关键帧
            for (const auto& [frameId, data] : mpSLAM->mpLocalMapper->mKeyFrameData) {
                // 检查是否已经发布过该关键帧
                if (publishedFrames.find(frameId) == publishedFrames.end()) {
                    // 未发布的关键帧，处理并标记为已发布
                    ProcessKeyFrame(frameId, data);
                    publishedFrames.insert(frameId);  // 将该帧标记为已发布
                }
            }
        }
        else{
            // 遍历所有关键帧并发布它们的位姿
            std::vector<ORB_SLAM3::KeyFrame*> allKeyFrames = mpSLAM->GetAtlas()->GetAllKeyFrames();
            sort(allKeyFrames.begin(),allKeyFrames.end(),ORB_SLAM3::KeyFrame::lId);

            // 检查关键帧数量是否足够
            size_t numKeyFrames = allKeyFrames.size();
            if (numKeyFrames >= 10) {
                // 提取最后 10 个关键帧
                auto last10KeyFrames = allKeyFrames.end() - 10;

                for (ORB_SLAM3::KeyFrame* pKF : std::vector<ORB_SLAM3::KeyFrame*>(last10KeyFrames, allKeyFrames.end())) {
                    if (!pKF || pKF->isBad()) continue;

                    // 生成位姿信息
                    geometry_msgs::PoseStamped poseMsg = CreatePoseMessage(pKF);
                    posePublisher.publish(poseMsg);

                    // 提取 RGB 信息
                    int keyFrameId = pKF->mnId;
                    auto it = mpSLAM->mpTracker->mnIdToBGRMap.find(keyFrameId);
                    if (it != mpSLAM->mpTracker->mnIdToBGRMap.end()) {
                        cv::Mat rgbImage = it->second;
                        // 创建并处理 RGB 消息 (假设有一个函数 CreateRosImage)
                        sensor_msgs::Image imageMsg = CreateRosImage(rgbImage, keyFrameId);

                        // 这里可以根据需求发布或存储 RGB 消息
                        rgbPublisher.publish(imageMsg);
                        mpSLAM->mpTracker->mnIdToBGRMap.erase(keyFrameId);  // 删除已处理的 keyframe ID
                    }
                }
            }

            for (ORB_SLAM3::KeyFrame* pKF : allKeyFrames) {
                geometry_msgs::PoseStamped poseMsg = CreatePoseMessage(pKF);
                posePublisher.publish(poseMsg);
            }
            mpSLAM->mpLoopCloser->finishedGlobalBA = false;
        }
        rate.sleep();
    }
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg) {

    // Convert the ROS image message to cv::Mat
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    // Track the image using the SLAM system
    mpSLAM->TrackMonocular(cv_ptr->image, cv_ptr->header.stamp.toSec());

}

int main(int argc, char** argv) {
    ros::init(argc, argv, "Mono");
    ros::start();

    if (argc != 3) {
        cerr << endl << "Usage: rosrun ORB_SLAM3 Mono path_to_vocabulary path_to_settings" << endl;
        ros::shutdown();
        return 1;
    }

    // Initialize SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);

    ros::NodeHandle nodeHandler;

    // Publisher for keyframe pose
    ros::Publisher pose_pub = nodeHandler.advertise<geometry_msgs::PoseStamped>("/slam/keyframe_pose", 100);

    // Publisher for RGB image
    ros::Publisher image_pub = nodeHandler.advertise<sensor_msgs::Image>("/slam/keyframe_image", 100);
    ros::Publisher point3d_pub = nodeHandler.advertise<sensor_msgs::PointCloud2>("/slam/keyframe_point3d", 100);

    // Create ImageGrabber and subscribe to image topic
    ImageGrabber igb(&SLAM, pose_pub, image_pub, point3d_pub);
    igb.backgroundThread = std::thread(&ImageGrabber::BackgroundThread, &igb);

    ros::Subscriber sub = nodeHandler.subscribe("/camera/image_raw", 100, &ImageGrabber::GrabImage, &igb);

    ros::spin();
    // Stop all threads
    SLAM.Shutdown();

    // Wait for background thread to finish
    igb.backgroundThread.join();

    ros::shutdown();

    return 0;
}