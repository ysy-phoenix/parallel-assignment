cmake_minimum_required(VERSION 3.20)
project(tools)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES graphTools.cpp ../common/graph.cpp)
add_executable(graphTools ${SOURCE_FILES})

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")