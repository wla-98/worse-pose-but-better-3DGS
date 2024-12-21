# Worse-Pose-but-Better-3DGS

## Project Overview
This project, titled "Worse-Pose-but-Better-3DGS", is designed to [Describe in detail the core objective or main function here. For instance, it focuses on leveraging specific algorithms or techniques to reconstruct a more accurate and detailed 3D geometric structure even when dealing with input data having relatively poor poses. The methodology might involve advanced image processing, pose estimation refinement, or other innovative approaches.]

## Installation Instructions
To set up the environment for this project, you need to install several components. Here are the links and brief descriptions for each of them:

### Windows 11 with WSL2
- **Link**: [https://learn.microsoft.com/en-us/windows/wsl/install](https://learn.microsoft.com/en-us/windows/wsl/install)
- **Description**: Windows Subsystem for Linux 2 (WSL2) allows you to run a Linux environment natively on your Windows 11 system. Follow the official Microsoft documentation provided in the link to install and configure WSL2 properly. This step is crucial as our project relies on a Linux-based environment for many of its operations.

### Anaconda
- **Link**: [https://docs.anaconda.com/anaconda/install/](https://docs.anaconda.com/anaconda/install/)
- **Description**: Anaconda is a popular platform for managing Python environments and packages. It is recommended to follow the installation guide on the provided link to install Anaconda on your system. Additionally, for this project, it is advisable to create an Anaconda environment based on the `environment.yaml` file in the MONOGS project (https://github.com/muskie82/MonoGS) to manage the necessary Python libraries efficiently and ensure compatibility with other components of our project.

### ROS-noetic
- **Link**: [https://wiki.ros.org/noetic#Installation](https://wiki.ros.org/noetic#Installation)
- **Description**: The Robot Operating System (ROS) in its noetic version is an essential part of our project for handling robotic-related tasks, such as sensor data processing, robot control, and more. Refer to the ROS wiki link for detailed installation instructions to get ROS-noetic up and running on your system. Note that ROS-noetic comes with libraries like OpenCV 4.2.0, PCL, Boost, etc., which are utilized within our project's framework.

### ORBSLAM3
- **Link**: [https://github.com/UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- **Description**: ORB_SLAM3 is a state-of-the-art Simultaneous Localization and Mapping (SLAM) library. It plays a significant role in our project for tasks like mapping the environment and localizing within it. Visit the GitHub page to learn about its installation process, which may involve cloning the repository, installing dependencies, and compiling the code.

### CUDA 11.8
- **Link**: [https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
- **Description**: CUDA 11.8 is required for leveraging the GPU computing capabilities in our project. Follow the instructions on the provided NVIDIA developer website to download and install CUDA 11.8 on your WSL2 (Ubuntu 20.04) environment. Make sure to select the appropriate options based on your system configuration (targeting Linux, x86_64 architecture, WSL-Ubuntu distribution, version 2.0, and deb_local as the target type).

### MONOGS
- **Link**: [https://github.com/muskie82/MonoGS](https://github.com/muskie82/MonoGS)
- **Description**: The MONOGS project implements a dense SLAM system based on 3D Gaussian Splatting, which won the highlight and best demo award at CVPR 2024. It for the first time demonstrates a monocular SLAM method solely relying on 3D Gaussian Splatting, mainly utilizing the gradient propagation with camera poses.

**Recommendation**: 
The project has been developed and tested in a specific environment setup. We are using **Windows 11 with WSL2 running Ubuntu 20.04** as the base Linux environment. **ROS-noetic** is an integral part of our system, which conveniently provides useful libraries such as **OpenCV 4.2.0**, **PCL 1.10**, and Boost out-of-the-box. **Eigen 3.1.0** is another key library utilized in our project. For efficient management of Python libraries and to ensure compatibility with other components, we recommend creating an Anaconda environment following the **`environment.yaml`** file in the MONOGS project. Also, **CUDA-TOOLKIT 11.8** is essential for taking advantage of GPU acceleration in relevant parts of our project. Sticking to these specific environment and library versions can help minimize potential compatibility issues and ensure the smooth running of the project. However, if you do decide to use different versions, be aware that you may encounter difficulties and might need to perform additional configuration and debugging to make the project work as expected.


## Usage
### ORB-SLAM INITIALIZATION
1. **Create a ROS Workspace**:
    - Open your terminal in the WSL2 (Ubuntu 20.04) environment.
    - Create a new directory for your ROS workspace (you can choose any name you prefer, here we'll use `my_ros_ws` as an example). Run the following command:
      ```bash
      mkdir -p "my_ros_ws"/src
      cd "my_ros_ws"
      catkin init
