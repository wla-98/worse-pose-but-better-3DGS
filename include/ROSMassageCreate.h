#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

// 必要的头文件包含
#include <string>
#include <vector>
#include "KeyFrame.h"
#include "Atlas.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"
#include "Settings.h"
#include <opencv2/core.hpp>      // 用于基础数据结构 (如 cv::Mat)
#include <geometry_msgs/PoseStamped.h>  // 用于发布相机位姿
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <mutex>

// 函数声明
sensor_msgs::Image CreateRosImage(const cv::Mat& matImage, int mnid);
geometry_msgs::PoseStamped CreatePoseMessage(ORB_SLAM3::KeyFrame* pKF);
sensor_msgs::PointCloud2 createMapPoints(ORB_SLAM3::KeyFrame* pKF, cv::Mat &image);

#endif // COMMON_FUNCTIONS_H
