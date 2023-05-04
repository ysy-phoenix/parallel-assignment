#ifndef GPUCLUSTER_H
#define GPUCLUSTER_H

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <thrust\device_ptr.h>
#include <thrust\copy.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\count.h>

#define THREADS 32

void GPULoadThrustImages(int size);

void GPUCluster(
        float *minChromaticImage, float *maxChromaticImage, int *clusterImage,
        float *minCenters, float *maxCenters, int minClusterIndex,
        int maxClusterIndex, int maxMinClusterIndex,
        size_t minPitch, size_t maxPitch, size_t clusterPitch,
        int rows, int cols
);

#endif