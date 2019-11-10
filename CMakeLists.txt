cmake_minimum_required(VERSION 2.8)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release") 
set(CMAKE_CXX_FLAGS "-std=c++11")
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS}
 ${OpenCV_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)


include_directories(/usr/local/include/eigen3)
set( G2O_LIBS  g2o_core g2o_types_sba g2o_solver_csparse g2o_solver_dense g2o_stuff g2o_csparse_extension)




add_executable(pose_graph pose_optimization.cpp tic_toc.h)

target_link_libraries(pose_graph ${G2O_LIBS} ${OpenCV_LIBS})
