cmake_minimum_required(VERSION 3.10)
project(test)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

# Add executable
add_executable(test ransac.cpp)

# Add OpenCV library
target_link_libraries(test PRIVATE ${OpenCV_LIBS})

