#include "Removal.h"

Removal::Removal() {
    this->midPercent = 0.5;
    this->alpha = 3;
    this->beta = 0.51;
    this->gamma = 0.025;
    this->useSort = true;
}

Removal::~Removal() {
#ifdef USE_CUDA
    cudaFree(&deviceMinCenters);
    cudaFree(&deviceMaxCenters);
#else
    delete[] ratio;
#endif
}

void Removal::initialize(int imageRows, int imageCols) {

#ifdef USE_CUDA
    deviceDiffuseImage = cv::cuda::GpuMat(imageRows, imageCols, CV_8UC3);
    deviceSpecularImage = cv::cuda::GpuMat(imageRows, imageCols, CV_8UC3);
    deviceMinImage = cv::cuda::GpuMat(imageRows, imageCols, CV_8UC1);
    deviceMaxImage = cv::cuda::GpuMat(imageRows, imageCols, CV_8UC1);
    deviceRangeImage = cv::cuda::GpuMat(imageRows, imageCols, CV_8UC1);
    deviceMaskImage = cv::cuda::GpuMat(imageRows, imageCols, CV_8UC1);
    deviceMinChromaticImage = cv::cuda::GpuMat(imageRows, imageCols, CV_32FC1);
    deviceMaxChromaticImage = cv::cuda::GpuMat(imageRows, imageCols, CV_32FC1);
    deviceClusterImage = cv::cuda::GpuMat(imageRows, imageCols, CV_32S);
    deviceRatioImage = cv::cuda::GpuMat(imageRows, imageCols, CV_32FC1);
    deviceLoadRatio(imageRows * imageCols);
    GPULoadThrustImages(imageRows * imageCols);
    cudaMalloc(&deviceMinCenters, 3 * sizeof(float));
    cudaMalloc(&deviceMaxCenters, 3 * sizeof(float));
#else
    minImage = cv::Mat(imageRows, imageCols, CV_8UC1);
    minChromaticImage = cv::Mat::zeros(imageRows, imageCols, CV_32FC1);
    maxChromaticImage = cv::Mat::zeros(imageRows, imageCols, CV_32FC1);
    clusterImage = cv::Mat(imageRows, imageCols, CV_32S);       // note 3 channels
    ratio = (float *) malloc(imageRows * imageCols * sizeof(float));
    maxImage = cv::Mat(imageRows, imageCols, CV_8UC1);
    rangeImage = cv::Mat(imageRows, imageCols, CV_8UC1);
    maskImage = cv::Mat(imageRows, imageCols, CV_8UC1);
    ratioImage = cv::Mat(imageRows, imageCols, CV_32FC1);

    // define the pointers to the image data
    minImg = minImage.ptr<unsigned char>();
    maxImg = maxImage.ptr<unsigned char>();
    rangeImg = rangeImage.ptr<unsigned char>();
    maskImg = maskImage.ptr<unsigned char>();
    minChromaticImg = minChromaticImage.ptr<float>();
    maxChromaticImg = maxChromaticImage.ptr<float>();
    clusterImg = clusterImage.ptr<int>();
    ratioImg = ratioImage.ptr<float>();

    this->epsilon = 1e-10f;
#endif
}

/*
Pipeline to remove specular highlights in real time
1. Compute minimum, maximum and range values for each pixel 
2. Compute the mean of minimum values
3. Estimate the pseudo-chromaticity values
4. Cluster regions in the minimum-maximum pseudo-chromaticity space
5. Estimate the single intensity ratio per cluster
6. Separate specular from diffuse components
*/

#ifndef USE_CUDA
#include "CPURemoval.h"
#endif


cv::Mat Removal::run(cv::Mat &image) {
    // first clone the image
    diffuseImage = image.clone();
    specularImage = image.clone();

#ifdef USE_CUDA
    deviceImage = cv::cuda::GpuMat(image);
    auto deviceImgPtr = deviceImage.ptr();
    auto deviceMinImgPtr = deviceMinImage.ptr();
    auto deviceMaxImgPtr = deviceMaxImage.ptr();
    auto deviceRangeImgPtr = deviceRangeImage.ptr();
    auto deviceMinChromaticImgPtr = deviceMinChromaticImage.ptr<float>();
    auto deviceMaxChromaticImgPtr = deviceMaxChromaticImage.ptr<float>();
    auto deviceMaskImgPtr = deviceMaskImage.ptr();
    auto deviceClusterImgPtr = deviceClusterImage.ptr<int>();
    auto deviceRatioImgPtr = deviceRatioImage.ptr<float>();
    auto deviceSpecularImgPtr = deviceSpecularImage.ptr();
    auto deviceDiffuseImgPtr = deviceDiffuseImage.ptr();

    auto pitch = deviceImage.step;
    auto minPitch = deviceMinImage.step;
    auto maxPitch = deviceMaxImage.step;
    auto rangePitch = deviceRangeImage.step;
    auto maskPitch = deviceMaskImage.step;
    auto clusterPitch = deviceClusterImage.step;
    auto minChromaticPitch = deviceMinChromaticImage.step;
    auto maxChromaticPitch = deviceMaxChromaticImage.step;
    auto ratioPitch = deviceRatioImage.step;
    auto specularPitch = deviceSpecularImage.step;
    auto diffusePitch = deviceDiffuseImage.step;
    auto rows = deviceImage.rows;
    auto cols = deviceImage.cols;

    // calculate the minimum, maximum and range images
    deviceCalculateMinMaxRange(
            deviceImgPtr, deviceMinImgPtr, deviceMaxImgPtr, deviceRangeImgPtr,
            pitch, minPitch, maxPitch, rangePitch, rows, cols
    );

    cv::Mat hostMinImage;
    deviceMinImage.download(hostMinImage);
    cv::meanStdDev(hostMinImage, minMean, stdDevMean);
    deviceCalculatePseudoChromaticity(
            deviceImgPtr, deviceMinImgPtr,
            deviceMinChromaticImgPtr, deviceMaxChromaticImgPtr, deviceMaskImgPtr,
            pitch, minPitch, minChromaticPitch, maxChromaticPitch, maskPitch, (float) minMean(0),
            rows, cols
    );

    cv::Mat hostMaskImage;
    deviceMaskImage.download(hostMaskImage);
    deviceMinChromaticImage.download(hostMinImage);

    cv::minMaxLoc(hostMinImage, &minValue, &maxMinValue, &minLocation, &maxMinLocation, hostMaskImage);
    cv::minMaxLoc(hostMinImage, nullptr, &maxValue, nullptr, &maxLocation, hostMaskImage);

    auto minClusterIndex = minLocation.y * deviceImage.cols + minLocation.x;
    auto maxClusterIndex = maxLocation.y * deviceImage.cols + maxLocation.x;
    auto maxMinClusterIndex = maxMinLocation.y * deviceImage.cols + maxMinLocation.x;
    GPUCluster(
            deviceMinChromaticImgPtr, deviceMaxChromaticImgPtr, deviceClusterImgPtr,
            deviceMinCenters, deviceMaxCenters,
            minClusterIndex, maxClusterIndex, maxMinClusterIndex, minChromaticPitch, maxChromaticPitch, clusterPitch,
            rows, cols
    );

    deviceCalculateIntensityRatio(
            deviceClusterImgPtr, deviceRangeImgPtr, deviceMaxImgPtr, deviceRatioImgPtr, clusterPitch,
            rangePitch, maxPitch, ratioPitch, (float) minMean(0),
            midPercent, 3, useSort, alpha, beta, gamma,
            rows, cols
    );

    deviceSeparateComponents(
            deviceImgPtr, deviceSpecularImgPtr, deviceDiffuseImgPtr,
            deviceMaxImgPtr, deviceRangeImgPtr, deviceMaskImgPtr, deviceRatioImgPtr,
            pitch, specularPitch, diffusePitch, maxPitch, rangePitch, maskPitch, ratioPitch,
            rows, cols
    );

    deviceSpecularImage.download(specularImage);
    deviceDiffuseImage.download(diffuseImage);
    return diffuseImage;
#else

    cols = image.cols;
    numPixel = image.rows * image.cols;

    img = image.ptr<unsigned char>();
    specularImg = specularImage.ptr<unsigned char>();
    diffuseImg = diffuseImage.ptr<unsigned char>();

    // compute the minimum, maximum and range images
    computeMinMaxRange();

    // use OpenCV to compute the mean of the minimum image
    minMean = cv::mean(minImage);

    // compute the pseudo-chromaticity images
    computePseudoChromaticity();

    // find the minimum and maximum chromaticity values of minimum chromaticity images
    cv::minMaxLoc(minChromaticImage, &minValue, &maxMinValue, &minLocation,
                  &maxMinLocation, maskImage);
    // find the maximum chromaticity values of maximum chromaticity images
    cv::minMaxLoc(maxChromaticImage, nullptr, &maxValue, nullptr,
                  &maxLocation, maskImage);

    // Cluster regions in the minimum-maximum pseudo-chromaticity space
    // Actually use K-means twice
    CPUCluster();

    // Estimate the single intensity ratio per cluster
    estimateRatio();

    // separate the specular and diffuse components
    specularImage.setTo(0);
    for (int pixel = 0; pixel < numPixel; ++pixel) {
        if (maskImg[pixel] == 1) {
            int specular = round((float) maxImg[pixel] - ratioImg[pixel] * (float) rangeImg[pixel]);
            specular = std::max(specular, 0);
            for (int channel = 0; channel < 3; ++channel) {
                int index = pixel * 3 + channel;
                specularImg[index] = specular;
                diffuseImg[index] = std::min(std::max(img[index] - specularImg[index], 0), 255);
            }
        }
    }
    
    return diffuseImage;
#endif

}

inline float Removal::calculateDistance(float x1, float y1, float x2, float y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

inline int Removal::round(float x) {
    return (((x) > 0) ? lround(x) : (int) ((x) - 0.5));
}
