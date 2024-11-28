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

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include "../include/System.h"

using namespace std;

// Function to check if a file exists
bool fileExists(const string& filename) {
    ifstream file(filename.c_str());
    return file.good();
}

// Function to create an empty file
bool createFile(const string& filename) {
    ofstream file(filename.c_str());
    if (file.is_open()) {
        file.close();
        return true;
    }
    return false;
}

// Function to create a directory
bool createDirectory(const string& dirname) {
    if (mkdir(dirname.c_str(), 0777) == 0 || errno == EEXIST) {
        return true;
    }
    return false;
}

// Function to create directory with error handling
bool createDirectoryWithParents(const string& dirname) {
    size_t pos = 0;
    do {
        pos = dirname.find('/', pos + 1);
        string subdir = dirname.substr(0, pos);
        if (mkdir(subdir.c_str(), 0777) && errno!= EEXIST) {
            cerr << "Error: Failed to create directory " << subdir << " - " << strerror(errno) << endl;
            return false;
        }
    } while (pos!= string::npos);
    return true;
}

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM):mpSLAM(pSLAM){}

    void GrabImage(const sensor_msgs::ImageConstPtr& msg);

    ORB_SLAM3::System* mpSLAM;
};

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    mpSLAM->TrackMonocular(cv_ptr->image,cv_ptr->header.stamp.toSec());
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "Mono");
    ros::start();

    if (argc!= 3) {
        cerr << endl << "Usage: rosrun ORB_SLAM3 Mono path_to_vocabulary path_to_settings" << endl;
        ros::shutdown();
        return 1;
    }

    // 创建 SLAM 系统，初始化所有线程，准备处理帧
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nodeHandler;
    // 订阅相机图像话题
    ros::Subscriber sub = nodeHandler.subscribe("/camera/image_raw", 1, &ImageGrabber::GrabImage,&igb);

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Get current time
    auto now = chrono::system_clock::now();
    auto now_c = chrono::system_clock::to_time_t(now);
    stringstream ss;
    ss << put_time(localtime(&now_c), "%Y%m%d_%H%M%S");
    string currentTime = ss.str();

    // Save camera trajectory
    string directoryName = "/home/wang/catkin_ws/src/wla_orb/orb-output/" + currentTime;

    // Ensure parent directory exists
    if (!createDirectoryWithParents("./dataset")) {
        ros::shutdown();
        return -1;
    }

    // Create the necessary directory
    if (!createDirectory(directoryName)) {
        cerr << "Error: Failed to create directory " << directoryName << " - " << strerror(errno) << endl;
        ros::shutdown();
        return -1;
    }

    // Check and create necessary files
    vector<string> filenames = {
        directoryName + "/KeyFrameTrajectory.txt",
        directoryName + "/images.txt",
        directoryName + "/points3D-keyframe.txt",
        directoryName + "/points3D-map.txt"

    };

    for (const string& filename : filenames) {
        if (!fileExists(filename) &&!createFile(filename)) {
            cerr << "Error: Failed to create file " << filename << endl;
            ros::shutdown();
            return -1;
        }
    }

    SLAM.SaveKeyFrameTrajectoryTUM(directoryName + "/KeyFrameTrajectory.txt");
    SLAM.SaveKeyPointsAndMapPoints(directoryName + "/images.txt");

    SLAM.SavePointcloudFromKeyframes(directoryName + "/points3D-keyframe.txt");
    SLAM.SavePointcloud(directoryName + "/points3D-map.txt");

    ros::shutdown();

    return 0;
}