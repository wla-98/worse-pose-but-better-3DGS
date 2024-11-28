#include "ROSMassageCreate.h"
#include <System.h>

sensor_msgs::Image CreateRosImage(const cv::Mat& matImage, int mnid) {
    sensor_msgs::Image imgMsg;

    try {
        // 使用 cv_bridge 将 cv::Mat 转换为 ROS Image 消息
        cv_bridge::CvImage cvImage;
        cvImage.header.stamp = ros::Time::now();  // 设置时间戳
        cvImage.header.frame_id = std::to_string(mnid);  // 设置帧 ID
        cvImage.encoding = sensor_msgs::image_encodings::BGR8;  // 设置图像编码格式
        cvImage.image = matImage;  // 将 cv::Mat 图像赋值给 cvImage

        // 返回转换后的 ROS 图像消息
        imgMsg = *cvImage.toImageMsg();
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    return imgMsg;
}

geometry_msgs::PoseStamped CreatePoseMessage(ORB_SLAM3::KeyFrame* pKF) {
    Sophus::SE3f Tcw = pKF->GetPose();   
    
    Eigen::Quaternionf q = Tcw.unit_quaternion(); 
    Eigen::Vector3f t = Tcw.translation(); 

    geometry_msgs::PoseStamped poseMsg;
    poseMsg.header.stamp = ros::Time(pKF->mTimeStamp);  
    poseMsg.header.frame_id = std::to_string(pKF->mnId) + "_" + "0";

    poseMsg.pose.position.x = t(0);
    poseMsg.pose.position.y = t(1);
    poseMsg.pose.position.z = t(2);
    poseMsg.pose.orientation.w = q.w();
    poseMsg.pose.orientation.x = q.x();
    poseMsg.pose.orientation.y = q.y();
    poseMsg.pose.orientation.z = q.z();

    return poseMsg;
}

// create MapPoints function 3dpoints with rgb
sensor_msgs::PointCloud2 createMapPoints(ORB_SLAM3::KeyFrame* pKF, cv::Mat &image) {
    const std::vector<ORB_SLAM3::MapPoint*>& vpMapPoints = pKF->mvpMapPoints;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->header.frame_id = std::to_string(pKF->mnId);
    cloud->header.stamp = pcl_conversions::toPCL(ros::Time::now());

    for (size_t i = 0; i < vpMapPoints.size(); i++) {
        ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
        if (pMP && !pMP->isBad()) {
            size_t pointID = pMP->mnId;

            // Get the 3D coordinates of the point
            Eigen::Vector3f pos = pMP->GetWorldPos();
            pcl::PointXYZRGB point;
            point.x = pos.x();
            point.y = pos.y();
            point.z = pos.z();

            // Get RGB from the current image
            int u = pKF->mvKeysUn[i].pt.x;
            int v = pKF->mvKeysUn[i].pt.y;

            if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
                cv::Vec3b rgb = image.at<cv::Vec3b>(v, u);
                point.r = rgb[2];
                point.g = rgb[1];
                point.b = rgb[0];
            }
            
            cloud->points.push_back(point);
        }
    }
    
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    sensor_msgs::PointCloud2 pointcloudMsg;
    pcl::toROSMsg(*cloud, pointcloudMsg);

    return pointcloudMsg;
}