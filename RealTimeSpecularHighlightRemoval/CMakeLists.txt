cmake_minimum_required(VERSION 3.25)
project(removal CUDA)

set(CMAKE_CUDA_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# COMPILER FLAGS AND DEFINITIONS
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=sm_75)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)
set(THRUST_LIBRARIES thrust)

add_executable(removal
        main.cu
        Clustering.h
        ColorProcessing.h
        SpecularHighlightRemoval.h
        useCUDA.h
        Clustering.cu
        ColorProcessing.cu
        SpecularHighlightRemoval.cu
        )

set(LIBS ${LIBS} ${OpenCV_LIBS} ${CUDA_LIBRARIES})

target_link_libraries(removal ${LIBS})

# Set the CUDA flags for the executable
set_target_properties(
        removal PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75"
)
