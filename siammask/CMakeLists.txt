cmake_minimum_required(VERSION 2.6)

project(alexnet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(OpenCV_DIR   "/usr/local")
find_package(OpenCV)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)
include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/media/honsen/MyProj/myProject/tensorRT_Pro-main/lean/TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-11.4.cudnn8.2/TensorRT-8.2.1.8/include)
link_directories(/media/honsen/MyProj/myProject/tensorRT_Pro-main/lean/TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-11.4.cudnn8.2/TensorRT-8.2.1.8/lib)

# add_executable(alexnet ${PROJECT_SOURCE_DIR}/alex.cpp)
add_executable(myNet ${PROJECT_SOURCE_DIR}/myNet.cpp)
# target_link_libraries(alexnet nvinfer)
# target_link_libraries(alexnet cudart)
target_link_libraries(myNet nvinfer)
target_link_libraries(myNet cudart)
target_link_libraries(myNet nvonnxparser)
target_link_libraries(myNet ${OpenCV_LIBS})
add_definitions(-O2 -pthread)

