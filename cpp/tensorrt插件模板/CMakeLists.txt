cmake_minimum_required(VERSION 2.6)

project(test1)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/plugins)
include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugins/AddPlugin.cu)
target_link_libraries(myplugins nvinfer cudart)

# find_package(OpenCV)
# include_directories(OpenCV_INCLUDE_DIRS)

file(GLOB DIR_SRCS ./*.cpp)
add_executable(test1 ${DIR_SRCS})
target_link_libraries(test1 nvinfer)
target_link_libraries(test1 cudart)
target_link_libraries(test1 myplugins)
# target_link_libraries(test1 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

