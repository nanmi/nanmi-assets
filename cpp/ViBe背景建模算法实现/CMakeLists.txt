cmake_minimum_required(VERSION 3.10)
project(vide_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD  11)

# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_PATH})

# execute
add_executable(vide_test main.cpp vibe.cpp)

target_link_libraries(vide_test ${OpenCV_LIBS})

