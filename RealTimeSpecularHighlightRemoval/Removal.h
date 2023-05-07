#ifndef REMOVAL_H
#define REMOVAL_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

#define USE_CUDA


#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "GPURemoval.h"
#include "GPUCluster.h"
#endif

class Removal {
public:
    Removal();

    ~Removal();

    void initialize(int imageRows, int imageCols);

    cv::Mat run(cv::Mat& image);

    // getters
    inline int getNumberOfIterations() const { return this->alpha; }

    inline float getThreshold() const { return this->beta; }

    inline float getStepValue() const { return this->gamma; }

    inline bool isSortEnabled() const { return this->useSort; }

    // setters
    void setNumberOfIterations(int alphaVal) { this->alpha = alphaVal; }

    void setThreshold(float betaVal) { this->beta = betaVal; }

    void setStepValue(float gammaVal) { this->gamma = gammaVal; }

    void enableSort() { this->useSort = true; }

    void disableSort() { this->useSort = false; }

private:
    // static functions
    static inline float calculateDistance(float x1, float y1, float x2, float y2);

    static inline int round(float x);

    // below corresponds to the variables in the paper
    // I^\min I^\max I^\range I^\mask
#ifdef USE_CUDA
    cv::cuda::GpuMat deviceImage, deviceDiffuseImage, deviceSpecularImage;
    cv::cuda::GpuMat deviceMinImage, deviceMaxImage, deviceRangeImage, deviceMaskImage;
    cv::cuda::GpuMat deviceMinChromaticImage, deviceMaxChromaticImage, deviceClusterImage, deviceRatioImage;
    float *deviceMinCenters{}, *deviceMaxCenters{};
#else
    cv::Mat minImage, maxImage, rangeImage, maskImage;
    cv::Mat minChromaticImage, maxChromaticImage, clusterImage, ratioImage;
    float *ratio{}, minCenters[3]{}, maxCenters[3]{};

    unsigned char* img{}, *minImg{}, *maxImg{}, *rangeImg{}, *maskImg{};
    unsigned char* specularImg{}, *diffuseImg{};
    float *minChromaticImg{}, *maxChromaticImg{}, *ratioImg{};
    int *clusterImg{};

    // number of pixels
    int cols{}, numPixel{};

    // CPU member functions
    void computeMinMaxRange();

    void computePseudoChromaticity();

    void CPUCluster();

    void estimateRatio();
#endif

    // seperated diffuse and specular images
    cv::Mat diffuseImage, specularImage;

    // \overline{I^\min}
    cv::Scalar minMean;

    // three cluster seeds: lowest minimum, highest maximum, and highest minimum chromaticity values.
    double minValue{}, maxValue{}, maxMinValue{};
    cv::Point minLocation, maxLocation, maxMinLocation;

    float midPercent{};
    int alpha{};        // maximum number of iterations
    float beta{};       // a threshold value
    float gamma{};      // a step value
    float epsilon{};    // a small value
    bool useSort{};     // whether to use sort or not
};

#endif