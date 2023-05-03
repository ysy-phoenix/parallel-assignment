cmake_minimum_required(VERSION 3.20)

project(BigGraph)

# Add subdirectories
add_subdirectory(tools)
add_subdirectory(pagerank)
add_subdirectory(bfs)

find_package(OpenMP REQUIRED)
