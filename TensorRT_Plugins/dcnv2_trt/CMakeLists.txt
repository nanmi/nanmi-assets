cmake_minimum_required(VERSION 3.10)

project(dcnv2)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86)

include_directories(${PROJECT_SOURCE_DIR})

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# tensorrt
set(TRT_OSS_ROOT /home/zhangbo/libraries/TensorRT-8.2.4.2)
include_directories(${TRT_OSS_ROOT}/include)
link_directories(${TRT_OSS_ROOT}/lib)


# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations")#弃用警告过滤

cuda_add_library(dcnv2 SHARED ${PROJECT_SOURCE_DIR}/DCNv2Plugin.cpp
                                ${PROJECT_SOURCE_DIR}/DCNv2Plugin.cu
                                                    )
target_link_libraries(dcnv2 nvinfer nvparsers cudart cublas nvinfer_plugin)

# add_definitions(-O2 -pthread)

