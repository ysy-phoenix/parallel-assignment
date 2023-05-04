#include "GPURemoval.h"

thrust::device_vector<float> thrustRatio;
thrust::device_vector<float> thrustCompressedRatio;

__device__ inline int deviceRound(float x) {
    return ((x > 0) ? (int) (x + 0.5f) : (int) (x - 0.5f));
}

template<typename T>
__device__ inline T deviceThreeMin(T a, T b, T c) {
    return min(a, min(b, c));
}

template<typename T>
__device__ inline T deviceThreeMax(T a, T b, T c) {
    return max(a, max(b, c));
}

__global__ void calculateMinMaxRange(
        unsigned char *image,
        unsigned char *minImage,
        unsigned char *maxImage,
        unsigned char *rangeImage,
        size_t pitch,
        size_t minPitch,
        size_t maxPitch,
        size_t rangePitch,
        int rows,
        int cols
) {
    // get the row and column for the current thread
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    unsigned char *originalImg = image + y * pitch;
    unsigned char *minImg = minImage + y * minPitch;
    unsigned char *maxImg = maxImage + y * maxPitch;
    unsigned char *rangeImg = rangeImage + y * rangePitch;

    // separate the three channels, note that OpenCV stores them in BGR order
    int blue = originalImg[x * 3 + 0];
    int red = originalImg[x * 3 + 2];
    int green = originalImg[x * 3 + 1];

    int minVal = deviceThreeMin(blue, red, green);
    int maxVal = deviceThreeMax(blue, red, green);
    minImg[x] = minVal;
    maxImg[x] = maxVal;
    rangeImg[x] = maxVal - minVal;
}

__global__ void calculatePseudoChromaticity(
        unsigned char *image,
        unsigned char *minImage,
        float *minChromaticImage,
        float *maxChromaticImage,
        unsigned char *maskImage,
        size_t pitch,
        size_t minPitch,
        size_t minChromaticPitch,
        size_t maxChromaticPitch,
        size_t maskPitch,
        float meanMin,
        int rows,
        int cols
) {
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    unsigned char *img = image + y * pitch;
    unsigned char *minImg = minImage + y * minPitch;
    auto *minChromaticImg = (float *) ((char *) minChromaticImage + y * minChromaticPitch);
    auto *maxChromaticImg = (float *) ((char *) maxChromaticImage + y * maxChromaticPitch);
    unsigned char *maskImg = maskImage + y * maskPitch;

    int minVal = minImg[x];
    bool mask = ((float) minVal > meanMin);
    if (mask) {
        float redChromatic = (float) img[x * 3 + 2] - (float) minVal + (float) meanMin;
        float greenChromatic = (float) img[x * 3 + 1] - (float) minVal + (float) meanMin;
        float blueChromatic = (float) img[x * 3 + 0] - (float) minVal + (float) meanMin;
        float sum = redChromatic + greenChromatic + blueChromatic;
        redChromatic /= sum;
        greenChromatic /= sum;
        blueChromatic /= sum;
        minChromaticImg[x] = deviceThreeMin(redChromatic, greenChromatic, blueChromatic);
        maxChromaticImg[x] = deviceThreeMax(redChromatic, greenChromatic, blueChromatic);
    } else {
        minChromaticImg[x] = 0;
        maxChromaticImg[x] = 0;
    }
    maskImg[x] = mask;
}

__global__ void calculateIntensityRatio(
        int *clusterImage,
        unsigned char *rangeImage,
        unsigned char *maxImage,
        float *ratio,
        size_t clusterPitch,
        size_t rangePitch,
        size_t maxPitch,
        float minMean,
        int cluster,
        int rows,
        int cols
) {

    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    int *clusterImg = (int *) ((char *) clusterImage + y * clusterPitch);
    unsigned char *rangeImg = rangeImage + y * rangePitch;
    unsigned char *maxImg = maxImage + y * maxPitch;

    int clusterIndex = clusterImg[x];
    int range = rangeImg[x];
    int maxVal = maxImg[x];

    if (clusterIndex == cluster && (float) range > minMean) {
        ratio[y * cols + x] = (float) maxVal / ((float) range + 1e-10f);
    } else {
        ratio[y * cols + x] = 0;
    }

}

__global__ void assignCompressedRatio(
        int *clusterImage,
        float *ratioImage,
        const float *compressedRatio,
        size_t clusterPitch,
        size_t ratioPitch,
        int index,
        int cluster,
        int rows,
        int cols
) {

    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    int *clusterImg = (int *) ((char *) clusterImage + y * clusterPitch);
    auto *ratioImg = (float *) ((char *) ratioImage + y * ratioPitch);

    if (clusterImg[x] == cluster) {
        ratioImg[x] = compressedRatio[index];
    }

}

__global__ void assignIntensityRatio(
        int *clusterImage,
        float *ratioImage,
        float intensityRatio,
        size_t clusterPitch,
        size_t ratioPitch,
        int cluster,
        int rows,
        int cols
) {

    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    int *clusterImg = (int *) ((char *) clusterImage + y * clusterPitch);
    auto *ratioImg = (float *) ((char *) ratioImage + y * ratioPitch);

    if (clusterImg[x] == cluster) {
        ratioImg[x] = intensityRatio;
    }

}

__global__ void separateComponents(
        unsigned char *image,
        unsigned char *specularImage,
        unsigned char *diffuseImage,
        unsigned char *maxImage,
        unsigned char *rangeImage,
        unsigned char *maskImage,
        float *ratioImage,
        size_t originalPitch,
        size_t specularPitch,
        size_t diffusePitch,
        size_t maximumPitch,
        size_t rangePitch,
        size_t maskPitch,
        size_t ratioPitch,
        int rows,
        int cols
) {

    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    unsigned char *img = image + y * originalPitch;
    unsigned char *specularImg = specularImage + y * specularPitch;
    unsigned char *diffuseImg = diffuseImage + y * diffusePitch;
    unsigned char *maskImg = maskImage + y * maskPitch;

    if (maskImg[x] & 1) {
        unsigned char *maxImg = maxImage + y * maximumPitch;
        unsigned char *rangeImg = rangeImage + y * rangePitch;
        auto *ratioImg = (float *) ((char *) ratioImage + y * ratioPitch);
        int value = deviceRound((float) maxImg[x] - ratioImg[x] * (float) rangeImg[x]);
        int specular = max(value, 0);
        specularImg[x * 3 + 0] = specular;
        specularImg[x * 3 + 1] = specular;
        specularImg[x * 3 + 2] = specular;
        diffuseImg[x * 3 + 0] = min(max(img[x * 3 + 0] - specular, 0), 255);
        diffuseImg[x * 3 + 1] = min(max(img[x * 3 + 1] - specular, 0), 255);
        diffuseImg[x * 3 + 2] = min(max(img[x * 3 + 2] - specular, 0), 255);
    } else {
        specularImg[x * 3 + 0] = 0;
        specularImg[x * 3 + 1] = 0;
        specularImg[x * 3 + 2] = 0;
        diffuseImg[x * 3 + 0] = img[x * 3 + 0];
        diffuseImg[x * 3 + 1] = img[x * 3 + 1];
        diffuseImg[x * 3 + 2] = img[x * 3 + 2];
    }

}

struct zeroCompare {
    __host__ __device__
    bool operator()(const float x) {
        return (x != 0);
    }
};

struct greater {
    float value;

    explicit greater(float val) {
        value = val;
    }

    __host__ __device__
    bool operator()(const float x) const {
        return (x > value);
    }
};

inline int roundValue(float x) {
    return ((x > 0) ? lround(x) : (int) (x - 0.5));
}

inline unsigned int roundUp(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

void checkError(const char *functionName) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("%s: %s\n", functionName, cudaGetErrorString(error));
    }
}

void deviceLoadRatio(int size) {
    thrustRatio.resize(size);
}

void deviceCalculateMinMaxRange(
        unsigned char *image,
        unsigned char *minImage,
        unsigned char *maxImage,
        unsigned char *rangeImage,
        size_t pitch,
        size_t minPitch,
        size_t maxPitch,
        size_t rangePitch,
        int rows,
        int cols
) {

    dim3 threads(NUM_THREAD, NUM_THREAD);
    dim3 grid(roundUp(cols, threads.x), roundUp(rows, threads.y));
    calculateMinMaxRange<<<grid, threads>>>(image, minImage, maxImage, rangeImage, pitch, minPitch,
                                            maxPitch, rangePitch, rows, cols);
    checkError("deviceCalculateMinMaxRange");

}

void deviceCalculatePseudoChromaticity(
        unsigned char *image,
        unsigned char *minImage,
        float *minChromaticImage,
        float *maxChromaticImage,
        unsigned char *maskImage,
        size_t pitch,
        size_t minPitch,
        size_t minChromaticPitch,
        size_t maxChromaticPitch,
        size_t maskPitch,
        float minMean,
        int rows,
        int cols
) {

    dim3 threads(NUM_THREAD, NUM_THREAD);
    dim3 grid(roundUp(cols, threads.x), roundUp(rows, threads.y));
    calculatePseudoChromaticity<<<grid, threads>>>(
            image, minImage, minChromaticImage, maxChromaticImage, maskImage,
            pitch, minPitch, minChromaticPitch, maxChromaticPitch, maskPitch,
            minMean, rows, cols
    );
    checkError("deviceCalculatePseudoChromaticity");

}

void deviceCalculateIntensityRatio(
        int *clusterImage,
        unsigned char *rangeImage,
        unsigned char *maxImage,
        float *ratioImage,
        size_t clusterPitch,
        size_t rangePitch,
        size_t maxPitch,
        size_t ratioPitch,
        float minMean,
        float midPercent,
        int k, int useSort,
        int alpha, float beta, float gamma,
        int rows, int cols
) {

    dim3 threads(NUM_THREAD, NUM_THREAD);
    dim3 grid(roundUp(cols, threads.x), roundUp(rows, threads.y));

    for (int cluster = 1; cluster <= k; ++cluster) {
        calculateIntensityRatio<<<grid, threads>>>(
                clusterImage, rangeImage, maxImage, thrust::raw_pointer_cast(thrustRatio.data()),
                clusterPitch, rangePitch, maxPitch, minMean, cluster, rows, cols
        );
        int compressedSize = (rows * cols) - (int) thrust::count(thrustRatio.begin(), thrustRatio.end(), 0);
        if (useSort) {
            thrustCompressedRatio.resize(compressedSize);
            // copy non-zero values to thrustCompressedRatio
            thrust::copy_if(thrustRatio.begin(), thrustRatio.end(), thrustCompressedRatio.begin(), zeroCompare());
            thrust::sort(thrustCompressedRatio.begin(), thrustCompressedRatio.end());
            assignCompressedRatio<<<grid, threads>>>(
                    clusterImage, ratioImage, thrust::raw_pointer_cast(thrustCompressedRatio.data()),
                    clusterPitch, ratioPitch, roundValue((float) compressedSize * midPercent), cluster, rows, cols
            );
        } else {
            float sum = thrust::reduce(thrustRatio.begin(), thrustRatio.end());
            float estimatedRatio = (sum / (float) compressedSize);
            for (int iteration = 0; iteration < alpha; ++iteration) {
                int greaterCount = (int) thrust::count_if(thrustRatio.begin(), thrustRatio.end(),
                                                          greater(estimatedRatio));
                int lessCount = compressedSize - greaterCount;
                if ((float) lessCount / (float) compressedSize > beta) {
                    estimatedRatio -= (estimatedRatio * gamma);
                } else if ((float) greaterCount / (float) compressedSize > beta) {
                    estimatedRatio += (estimatedRatio * gamma);
                } else {
                    break;
                }
            }
            assignIntensityRatio<<<grid, threads>>>(
                    clusterImage, ratioImage, estimatedRatio,
                    clusterPitch, ratioPitch, cluster, rows, cols
            );
        }
    }
    checkError("deviceCalculateIntensityRatio");

}

void deviceSeparateComponents(
        unsigned char *image,
        unsigned char *specularImage,
        unsigned char *diffuseImage,
        unsigned char *maxImage,
        unsigned char *rangeImage,
        unsigned char *maskImage,
        float *ratioImage,
        size_t pitch,
        size_t specularPitch,
        size_t diffusePitch,
        size_t maxPitch,
        size_t rangePitch,
        size_t maskPitch,
        size_t ratioPitch,
        int rows,
        int cols
) {

    dim3 threads(NUM_THREAD, NUM_THREAD);
    dim3 grid(roundUp(cols, threads.x), roundUp(rows, threads.y));
    separateComponents<<<grid, threads>>>(
            image, specularImage, diffuseImage, maxImage, rangeImage, maskImage, ratioImage,
            pitch, specularPitch, diffusePitch, maxPitch, rangePitch, maskPitch, ratioPitch, rows, cols
    );
    checkError("deviceSeparateComponents");

}