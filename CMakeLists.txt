cmake_minimum_required(VERSION 3.0.2)
project(wla_orb)

# 设置构建类型为 Release
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
set(CMAKE_PREFIX_PATH "/opt/ros/noetic;/usr/lib/x86_64-linux-gnu/cmake/opencv4;/usr/lib/x86_64-linux-gnu/cmake/pcl")

# 编译标志
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(PYTHON_EXECUTABLE /usr/bin/python3)

set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
set(PCL_DIR "/usr/lib/x86_64-linux-gnu/cmake/pcl")
set(FLANN_INCLUDE_DIR "/usr/include/flann")
set(FLANN_LIBRARY "/usr/lib/x86_64-linux-gnu/libflann.so")

# OpenCV 依赖
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV > 4.2 not found.")
endif()

# 其他依赖
include_directories("/usr/include/eigen3")
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

# # 查找 yaml-cpp 库（假设使用的是动态库，若是静态库可相应调整查找方式）
# find_package(YAML-CPP REQUIRED)

# # 如果找到 yaml-cpp 库，将其包含路径添加到项目中（用于头文件查找）
# if(YAML-CPP_FOUND)
#     include_directories(${YAML-CPP_INCLUDE_DIRS})
# endif(
  
find_package(Boost 1.71 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# 包含头文件路径
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/CameraModels
  ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# 设置共享库输出路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# 生成共享库 libORB_SLAM3.so
add_library(ORB_SLAM3 SHARED
  src/System.cc
  src/Tracking.cc
  src/LocalMapping.cc
  src/LoopClosing.cc
  src/ORBextractor.cc
  src/ORBmatcher.cc
  src/FrameDrawer.cc
  src/Converter.cc
  src/MapPoint.cc
  src/KeyFrame.cc
  src/Atlas.cc
  src/Map.cc
  src/MapDrawer.cc
  src/Optimizer.cc
  src/Frame.cc
  src/KeyFrameDatabase.cc
  src/Sim3Solver.cc
  src/Viewer.cc
  src/ImuTypes.cc
  src/G2oTypes.cc
  src/CameraModels/Pinhole.cpp
  src/CameraModels/KannalaBrandt8.cpp
  src/OptimizableTypes.cpp
  src/MLPnPsolver.cpp
  src/GeometricTools.cc
  src/TwoViewReconstruction.cc
  src/Config.cc
  src/Settings.cc
  src/ROSMassageCreate.cc
  include/System.h
  include/Tracking.h
  include/LocalMapping.h
  include/LoopClosing.h
  include/ORBextractor.h
  include/ORBmatcher.h
  include/FrameDrawer.h
  include/Converter.h
  include/MapPoint.h
  include/KeyFrame.h
  include/Atlas.h
  include/Map.h
  include/MapDrawer.h
  include/Optimizer.h
  include/Frame.h
  include/KeyFrameDatabase.h
  include/Sim3Solver.h
  include/Viewer.h
  include/ImuTypes.h
  include/G2oTypes.h
  include/CameraModels/GeometricCamera.h
  include/CameraModels/Pinhole.h
  include/CameraModels/KannalaBrandt8.h
  include/OptimizableTypes.h
  include/MLPnPsolver.h
  include/GeometricTools.h
  include/TwoViewReconstruction.h
  include/SerializationUtils.h
  include/Config.h
  include/Settings.h
  include/ROSMassageCreate.h
)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(ORB_SLAM3
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${Pangolin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
  -lboost_serialization
  -lcrypto
)

# 设置可执行文件输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/execute)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,/usr/lib/x86_64-linux-gnu")

# 生成项目
catkin_package()
# 添加 ros_mono 可执行文件
add_executable(ros_mono scripts/ros_mono.cc)
add_dependencies(ros_mono ORB_SLAM3)
target_link_libraries(ros_mono ${YAML-CPP_LIBRARIES})
target_link_libraries(ros_mono
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${Pangolin_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/lib/libORB_SLAM3.so
  -lboost_system
)

# 添加 camera_pose_visualizer 可执行文件
add_executable(camera_pose_visualizer scripts/camera_pose_visualizer.cc)
add_dependencies(camera_pose_visualizer ORB_SLAM3)

target_link_libraries(camera_pose_visualizer
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${Pangolin_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/lib/libORB_SLAM3.so
  -lboost_system
  ${PCL_LIBRARIES}
)

add_executable(ros_mono_direct_send scripts/ros_mono_direct_send.cc)
add_dependencies(ros_mono_direct_send ORB_SLAM3)

target_link_libraries(ros_mono_direct_send
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${Pangolin_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/lib/libORB_SLAM3.so
  -lboost_system
  ${PCL_LIBRARIES}
)
# add_executable(points3d_visualizer scripts/points3d_visualizer.cc)
# target_link_libraries(points3d_visualizer
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBS}
#   ${EIGEN3_LIBS}
#   ${Pangolin_LIBRARIES}
#   ${PROJECT_SOURCE_DIR}/lib/libORB_SLAM3.so
#   -lboost_system
#   ${PCL_LIBRARIES}
# )


