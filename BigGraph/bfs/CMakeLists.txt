cmake_minimum_required(VERSION 3.20)
project(pagerank)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp bfs.cpp ../common/graph.cpp)
add_executable(bfs ${SOURCE_FILES})

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")