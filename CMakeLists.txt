cmake_minimum_required(VERSION 2.8)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release") 
set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(/usr/local/include/eigen3)
add_executable(pose_graph pose_optimization.cpp tic_toc.h)
