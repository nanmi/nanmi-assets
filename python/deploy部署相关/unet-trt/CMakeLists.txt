cmake_minimum_required(VERSION 3.10)
project(unet LANGUAGES CXX)
set(TARGET_NAME "unet")
# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD  11)
#SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_BUILD_TYPE "Release")

# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# tensorrt
set(TENSORRT_INCLUDE_DIR /usr/local/TensorRT-8.0.1.6/include/)
set(TENSORRT_LIBRARY_DIR /usr/local/TensorRT-8.0.1.6/lib/)
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# execute
# add_library(${TARGET_NAME} SHARED unet.cpp)
add_executable(${TARGET_NAME} unet.cpp)

target_link_libraries(${TARGET_NAME} nvinfer ${CUDA_LIBRARIES} ${OpenCV_LIBS})

