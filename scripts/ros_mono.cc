#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <string>
#include <sys/stat.h> // For mkdir function
#include <errno.h>    // For errno
#include <string.h>   // For strerror
#include <geometry_msgs/PoseStamped.h>  // 用于发布相机位姿
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp> // 用于递归创建目录

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

// Function to create directory with Boost
bool createDirectoryWithBoost(const string& path) {
    try {
        boost::filesystem::create_directories(path);
        return true;
    } catch (const boost::filesystem::filesystem_error& e) {
        cerr << "Error creating directory: " << e.what() << endl;
        return false;
    }
}

// 保存摄像头参数到文件，使用cv::FileStorage替换yaml-cpp库相关操作
bool SaveCameraParametersToFile(const string& yamlFile, const string& outputFile) {
    // 使用cv::FileStorage读取文件
    cv::FileStorage fs(yamlFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Error: Failed to open input YAML file " << yamlFile << endl;
        return false;
    }

    // 提取参数
    double fx = (double)fs["Camera.fx"];
    double fy = (double)fs["Camera.fy"];
    double cx = (double)fs["Camera.cx"];
    double cy = (double)fs["Camera.cy"];
    int width = (int)fs["Camera.width"];
    int height = (int)fs["Camera.height"];

    // 生成输出内容
    string cameraInfo = "1 PINHOLE " + to_string(width) + " " + to_string(height) + " " +
                        to_string(fx) + " " + to_string(fy) + " " +
                        to_string(cx) + " " + to_string(cy);

    // 写入到文件
    ofstream outFile(outputFile);
    if (!outFile) {
        cerr << "Error: Failed to open output file " << outputFile << endl;
        return false;
    }

    outFile << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << endl;
    outFile << cameraInfo << endl;

    cout << "Camera parameters successfully saved to " << outputFile << endl;
    return true;
}

class ImageGrabber {
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM) : mpSLAM(pSLAM) {}

    void GrabImage(const sensor_msgs::ImageConstPtr& msg);

    ORB_SLAM3::System* mpSLAM;
};

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg) {
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

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

    // 创建 SLAM 系统，初始化所有线程，准备处理帧
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nodeHandler;
    // 订阅相机图像话题
    ros::Subscriber sub = nodeHandler.subscribe("/camera/image_raw", 1, &ImageGrabber::GrabImage, &igb);

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
    string directoryName = "./src/wla_orb/orb-output/" + currentTime + "/sparse/0";
    string imagesDirectory = "./src/wla_orb/orb-output/" + currentTime + "/images";

    // Create the necessary directory
    if (!createDirectoryWithBoost(directoryName)) {
        cerr << "Error: Failed to create directory " << directoryName << endl;
        ros::shutdown();
        return -1;
    }
    // Create the images directory
    if (!createDirectoryWithBoost(imagesDirectory)) {
        cerr << "Error: Failed to create directory " << imagesDirectory << endl;
        ros::shutdown();
        return -1;
    }

    // Check and create necessary files
    vector<string> filenames = {
        directoryName + "/KeyFrameTrajectory.txt",
        directoryName + "/cameras.txt",
        directoryName + "/images.txt",
        directoryName + "/points3D.txt"
    };

    for (const string& filename : filenames) {
        if (!fileExists(filename) && !createFile(filename)) {
            cerr << "Error: Failed to create file " << filename << endl;
            ros::shutdown();
            return -1;
        }
    }

    SLAM.SaveKeyFrameTrajectoryTUM(directoryName + "/KeyFrameTrajectory.txt");
    SLAM.SaveKeyPointsAndMapPoints(directoryName + "/images.txt");
    if (!SaveCameraParametersToFile(argv[2], directoryName + "/cameras.txt")) {
        cerr << "Error: Failed to save camera parameters" << endl;
        return -1;
    }
    SLAM.SavePointcloudFromKeyframes(directoryName + "/points3D.txt", imagesDirectory);

    ros::shutdown();

    return 0;
}
