cmake_minimum_required(VERSION 3.20)
project(pagerank)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp page_rank.cpp ../common/graph.cpp)
add_executable(pagerank ${SOURCE_FILES})

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")