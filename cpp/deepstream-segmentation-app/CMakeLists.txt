cmake_minimum_required(VERSION 3.10)
project(segmentation-app)
set(PROJECT_NAME segmentation-app)

set(${CMAKE_BUILD_TYPE} "Debug")

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")


# link third libraries
find_package(PkgConfig REQUIRED)
set(libs glib-2.0 gstreamer-1.0 json-glib-1.0 gstreamer-video-1.0 x11)
foreach(lib ${libs})
    pkg_search_module(${lib} REQUIRED ${lib})
    include_directories(${${lib}_INCLUDE_DIRS})
    list(APPEND EXLIBS ${${lib}_LIBRARIES})
endforeach()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Deepstream libs
set(DEEPSTREAM_ROOT /opt/nvidia/deepstream/deepstream)
include_directories(${DEEPSTREAM_ROOT}/sources/includes)
link_directories(${DEEPSTREAM_ROOT}/lib)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Threads
find_package(Threads REQUIRED)


# append src/head
file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

file(GLOB_RECURSE COMMON_INCLUDE_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/*.h
	)

# add library
# add_library(${PROJECT_NAME} ${TARGET} ${SRC_FILES})
add_executable(${PROJECT_NAME} deepstream_segmentation_app.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${EXLIBS})
target_link_libraries(${PROJECT_NAME} Threads::Threads)
target_link_libraries(${PROJECT_NAME} nvdsgst_meta
                                      nvds_meta
                                      nvdsgst_helper
                                      nvdsgst_smartrecord
                                      nvds_utils
                                      nvds_msgbroker
                                      m dl
                                      gstrtspserver-1.0
                                      nvbufsurface
                                      jpeg)

