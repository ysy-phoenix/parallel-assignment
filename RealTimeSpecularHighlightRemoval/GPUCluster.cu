#include "GPUCluster.h"

thrust::device_vector<float> thrustMinImage;
thrust::device_vector<float> thrustMaxImage;
__constant__ float constantMinCenters[3];
__constant__ float constantMaxCenters[3];

struct isNotZero {
    __host__ __device__
    bool operator()(const float x) {
        return (x != 0);
    }
};

__device__ float computeDistance(float x1, float y1, float x2, float y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

__global__ void loadClusters(
        float *minImage, float *maxImage, float *minCenters, float *maxCenters,
        int minClusterIndex, int maxClusterIndex, int maxMinClusterIndex,
        size_t minPitch, size_t maxPitch,
        int cols
) {

    int xMinIndex = minClusterIndex % cols;
    int yMinIndex = minClusterIndex / cols;
    int xMaxIndex = maxClusterIndex % cols;
    int yMaxIndex = maxClusterIndex / cols;
    int xMaxMinIndex = maxMinClusterIndex % cols;
    int yMaxMinIndex = maxMinClusterIndex / cols;

    auto *minImgCluster1 = (float *) ((char *) minImage + yMinIndex * minPitch);
    auto *maxImgCluster1 = (float *) ((char *) maxImage + yMinIndex * maxPitch);
    auto *minImgCluster2 = (float *) ((char *) minImage + yMaxIndex * minPitch);
    auto *maxImgCluster2 = (float *) ((char *) maxImage + yMaxIndex * maxPitch);
    auto *minImgCluster3 = (float *) ((char *) minImage + yMaxMinIndex * minPitch);
    auto *maxImgCluster3 = (float *) ((char *) maxImage + yMaxMinIndex * maxPitch);

    minCenters[0] = minImgCluster1[xMinIndex];
    minCenters[1] = minImgCluster2[xMaxIndex];
    minCenters[2] = minImgCluster3[xMaxMinIndex];

    maxCenters[0] = maxImgCluster1[xMinIndex];
    maxCenters[1] = maxImgCluster2[xMaxIndex];
    maxCenters[2] = maxImgCluster3[xMaxMinIndex];

}

__global__ void assignClusters(
        float *minImage, float *maxImage, int *clusterImage,
        size_t minPitch, size_t maxPitch, size_t clusterPitch,
        int rows, int cols
) {

    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    auto *minImg = (float *) ((char *) minImage + y * minPitch);
    auto *maxImg = (float *) ((char *) maxImage + y * maxPitch);
    int *clusterImg = (int *) ((char *) clusterImage + y * clusterPitch);

    float minVal = minImg[x];
    float maxVal = maxImg[x];
    if (minVal > 0 && maxVal > 0) {
        auto minDist = (float) (rows * cols);
        for (int cluster = 0; cluster < 3; ++cluster) {
            float dist = computeDistance(minVal, maxVal, constantMinCenters[cluster], constantMaxCenters[cluster]);
            if (dist < minDist) {
                minDist = dist;
                clusterImg[x] = cluster + 1;
            }
        }
    } else {
        clusterImg[x] = 0;
    }

}

__global__ void copyToThrust(
        float *image, float *thrustImage, int *maskImage, int clusterIndex,
        size_t clusterPitch, int rows, int cols
) {

    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    int *maskImg = (int *) ((char *) maskImage + y * clusterPitch);
    auto *img = (float *) ((char *) image + y * clusterPitch);

    if (maskImg[x] == clusterIndex) {
        thrustImage[y * cols + x] = img[x];
    } else {
        thrustImage[y * cols + x] = 0;
    }
}

void GPUCheckError(const char *functionName) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("%s: %s\n", functionName, cudaGetErrorString(error));
    }
}

unsigned int GPURoundUp(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

void GPULoadThrustImages(int size) {
    thrustMinImage.resize(size);
    thrustMaxImage.resize(size);
}

void GPUCluster(
        float *minChromaticImage, float *maxChromaticImage, int *clusterImage,
        float *minCenters, float *maxCenters,
        int minClusterIndex, int maxClusterIndex, int maxMinClusterIndex,
        size_t minPitch, size_t maxPitch, size_t clusterPitch,
        int rows, int cols
) {

    float hostMinCenters[3], hostMaxCenters[3];
    dim3 threads(THREADS, THREADS);
    dim3 grid(GPURoundUp(cols, threads.x), GPURoundUp(rows, threads.y));

    // first K-mean
    loadClusters<<<1, 1>>>(
            minChromaticImage, maxChromaticImage,
            minCenters, maxCenters,
            minClusterIndex, maxClusterIndex, maxMinClusterIndex,
            minPitch, maxPitch, cols
    );
    cudaMemcpyToSymbol(constantMinCenters, minCenters, 3 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(constantMaxCenters, maxCenters, 3 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    assignClusters<<<grid, threads>>>(
            minChromaticImage, maxChromaticImage, clusterImage,
            minPitch, maxPitch, clusterPitch,
            rows, cols
    );

    // second K-mean
    for (int cluster = 1; cluster <= 3; cluster++) {
        copyToThrust<<<grid, threads>>>(
                minChromaticImage, thrust::raw_pointer_cast(thrustMinImage.data()),
                clusterImage, cluster, clusterPitch, rows, cols
        );
        copyToThrust<<<grid, threads>>>(
                maxChromaticImage, thrust::raw_pointer_cast(thrustMaxImage.data()),
                clusterImage, cluster, clusterPitch, rows, cols
        );
        float size = (float) thrust::count_if(thrustMinImage.begin(), thrustMinImage.end(), isNotZero());
        float minSum = thrust::reduce(thrustMinImage.begin(), thrustMinImage.end());
        float maxSum = thrust::reduce(thrustMaxImage.begin(), thrustMaxImage.end());
        hostMinCenters[cluster - 1] = minSum / size;
        hostMaxCenters[cluster - 1] = maxSum / size;
    }
    cudaMemcpyToSymbol(constantMinCenters, hostMinCenters, 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constantMaxCenters, hostMaxCenters, 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
    assignClusters<<<grid, threads>>>(
            minChromaticImage, maxChromaticImage, clusterImage,
            minPitch, maxPitch, clusterPitch,
            rows, cols
    );
    GPUCheckError("GPUCluster");
}