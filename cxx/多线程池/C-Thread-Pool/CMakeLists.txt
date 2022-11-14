cmake_minimum_required(VERSION 3.10)
project(thpool_library)

# Set include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(thpool_h ${CMAKE_CURRENT_SOURCE_DIR}/thpool.h)

# Set output for library
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib)

# Add library
add_library(thpool STATIC thpool.c)

# Add pthreads
find_package(Threads)
target_link_libraries(thpool ${CMAKE_THREAD_LIBS_INIT})

# Set C++11 support
set_property(TARGET thpool PROPERTY CXX_STANDARD 11)
set_property(TARGET thpool PROPERTY CXX_STANDARD_REQUIRED ON)


# Set -03 optimization
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-O3" COMPILER_SUPPORTS_O3)
if (COMPILER_SUPPORTS_O3)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Disable debugging
add_definitions(-DNDEBUG)

# Install library
install(TARGETS thpool DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
install(FILES ${thpool_h} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/include)
