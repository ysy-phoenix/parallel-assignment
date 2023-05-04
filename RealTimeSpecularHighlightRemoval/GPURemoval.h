#ifndef GPUREMOVAL_H
#define GPUREMOVAL_H

#include <cuda_runtime.h>
#include <cstdio>
#include <thrust\device_ptr.h>
#include <thrust\copy.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\count.h>
#include <thrust\sort.h>
#include <thrust\remove.h>

#define NUM_THREAD 32

void deviceLoadRatio(int size);

void deviceCalculateMinMaxRange(
        unsigned char *image, unsigned char *minImage, unsigned char *maxImage, unsigned char *rangeImage,
        size_t pitch, size_t minPitch, size_t maxPitch, size_t rangePitch,
        int rows, int cols
);

void deviceCalculatePseudoChromaticity(
        unsigned char *image, unsigned char *minImage, float *minChromaticImage, float *maxChromaticImage,
        unsigned char *maskImage,
        size_t pitch, size_t minPitch, size_t minChromaticPitch, size_t maxChromaticPitch, size_t maskPitch,
        float minMean, int rows, int cols
);

void
deviceCalculateIntensityRatio(
        int *clusterImage, unsigned char *rangeImage, unsigned char *maxImage, float *ratioImage,
        size_t clusterPitch, size_t rangePitch, size_t maxPitch, size_t ratioPitch,
        float minMean, float midPercent, int k, int useSort, int alpha, float beta, float gamma,
        int rows, int cols
);

void deviceSeparateComponents(
        unsigned char *image, unsigned char *specularImage, unsigned char *diffuseImage, unsigned char *maxImage,
        unsigned char *rangeImage, unsigned char *maskImage, float *ratioImage,
        size_t pitch, size_t specularPitch, size_t diffusePitch, size_t maxPitch,
        size_t rangePitch, size_t maskPitch, size_t ratioPitch,
        int rows, int cols
);

#endif