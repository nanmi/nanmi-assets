cmake_minimum_required(VERSION 3.10)

project(plugin_build_example)
add_definitions(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(CUDA REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
set(TENSORRT_INCLUDE_DIR /usr/local/TensorRT-8.0.1.6/include)
set(TENSORRT_LIBRARY_DIR /usr/local/TensorRT-8.0.1.6/lib)
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})

# deepstream
set(DEEPSTREAM_ROOT /opt/nvidia/deepstream/deepstream/sources)
include_directories(${DEEPSTREAM_ROOT}/includes)
link_directories(${DEEPSTREAM_ROOT}/libs)

cuda_add_library(nvdsinfer_custom_lib_impl SHARED ${CMAKE_SOURCE_DIR}/yololayer.cu
                                ${CMAKE_SOURCE_DIR}/nvdsparser_custom_impl.cpp
                                                    )
target_link_libraries(nvdsinfer_custom_lib_impl nvinfer_plugin nvinfer nvparsers
                                                cudart cublas stdc++fs)

add_definitions(-O2 -pthread)

