cmake_minimum_required(VERSION 3.0)
project(cxx_use_python VERSION 7.2.2)
set(CMAKE_CXX_FLAGS "-std=c++11")


# Python
find_package(PythonLibs REQUIRED)
# find_package(PythonLibs PATHS /home/nvidia/data/zhangbo/library/opencv-4.5.1 NO_DEFAULT_PATH)
include_directories(${PYTHON_INCLUDE_DIRS})

file(GLOB test_source
    host.cxx
    )
add_executable(cxx_use_python ${test_source})
target_compile_options(cxx_use_python PUBLIC -std=c++11 -Wall -Wfloat-conversion)
target_link_libraries(cxx_use_python ${PYTHON_LIBRARIES})
# target_link_libraries(cxx_use_python ${CMAKE_CURRENT_LIST_DIR}/lib/script_function.cpython-36m-x86_64-linux-gnu.so)



