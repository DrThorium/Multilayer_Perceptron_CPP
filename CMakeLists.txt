cmake_minimum_required(VERSION 3.24)
project(Multilayer_Perceptron)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message("${CMAKE_MODULE_PATH}")
message("${EIGEN3_INCLUDE_DIR}")

set(CMAKE_CXX_STANDARD 17)

add_executable(Multilayer_Perceptron main.cpp MultilayerPerceptron.cpp MultilayerPerceptron.h)
