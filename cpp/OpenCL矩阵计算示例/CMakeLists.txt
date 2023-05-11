cmake_minimum_required(VERSION 3.12)
project(YourProjectName)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(cl_gemm opencl_gemm.cpp)
target_link_libraries(cl_gemm ${OpenCL_LIBRARIES})
