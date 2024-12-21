# Worse-Pose-but-Better-3DGS

[Project](https://wla-98.github.io/worse-pose-but-better-3DGS/)

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
2. **Clone the Project**:
    - Navigate into the `src` folder within the newly created ROS workspace:
        ```bash
        cd “my_ros_ws”/src
    - Then, clone the `Worse-Pose-but-Better-3DGS` project repository using the following `git` command:
        ```bash
        git clone https://github.com/wla-98/worse-pose-but-better-3DGS.git
    - After cloning the project, move back to the root directory of the ROSWorkspace:
        ```bash
        cd ..
    - Initial install the 3rdparty libraries and run the following command to build the ROS packages in the workspace:
        ```bash
        sh src/wla_orb/build_3rdparty.sh    
        catkin_make
    - Once the build process is completed successfully, you need to source theWorkspace to make the ROS packages and executables available in your current terminal session. Run the following command:
        ```bash
        source devel/setup.bash
3. **Example of Running with TUM Dataset:**
    - open a new terminal and run the following command to start the ROS master:

        New Terminal
        ```bash
        roscore
    - To run the project using the `TUM` dataset, specifically the desk sequence, you can use the following `rosrun` command. Replace the paths with the actual paths on your system if they are different:
        ```bash
        rosrun wla_orb ros_mono src/wla_orb/Vocabulary/ORBvoc.txt src/wla_orb/config/TUM3.yaml
    - This will start the ORB-SLAM initialization process with the specified vocabulary file and configuration file for the desk sequence of the TUM dataset. This will pop up two GUIs. One subscribes to the image information in the ROSBAG, and the other will display the key frames and the sparse point cloud map in real time.
    - Open the rosbag file in another terminal:

        New Terminal
        - Download the rosbag file from the TUM dataset website (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) and place it in the `dataset` folder of the project.
            ```bash
            wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.bag -P src/wla_orb/dataset
        - Run the following command to play the rosbag file:
            ```bash
            rosbag play src/wla_orb/dataset/rgbd_dataset_freiburg1_desk.bag /camera/rgb/image_color:=/camera/image_raw
    - The ORB-SLAM system will start processing the data from the rosbag file and display the key frames and the sparse point cloud map in real time.

### Gaussian Splatting Optimization


=======

[Project](https://wla-98.github.io/worse-pose-but-better-3DGS/)

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
      

2. **Clone the Project**:
    - Navigate into the `src` folder within the newly created ROS workspace:
      ```bash
      cd “my_ros_ws”/src
    - Then, clone the `Worse-Pose-but-Better-3DGS` project repository using the following `git` command:
      ```bash
      git clone https://github.com/wla-98/worse-pose-but-better-3DGS.git
    - After cloning the project, move back to the root directory of the ROSWorkspace:
      ```bash
      cd..
      catkin_make
    - Once the build process is completed successfully, you need to source theWorkspace to make the ROS packages and executables available in your current terminal session. Run the following command:
        ```bash
        source devel/setup.bash

3. **Example of Running with TUM Dataset (desk):**
    - To run the project using the `TUM` dataset, specifically the desk sequence, you can use the following `rosrun` command. Replace the paths with the actual paths on your system if they are different:
        ```bash
        rosrun wla_orb ros_mono src/wla_orb/Vocabulary/ORBvoc.txt src/wla_orb/config/TUM1.yaml
    - This will start the ORB-SLAM initialization process with the specified vocabulary file and configuration file for the desk sequence of the TUM dataset. The system will then attempt to reconstruct the 3D geometric structure based on the input data from the dataset, leveraging the algorithms and techniques implemented in our Worse-Pose-but-Better-3DGS project. You can observe the progress and results in the terminal output, and depending on the implementation, there may also be visualizations or saved files generated for further analysis.
>>>>>>> 5dddb95400d2543712c28ce7c238dbd6c314cdb5
