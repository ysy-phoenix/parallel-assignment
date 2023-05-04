#ifndef CPUREMOVAL_H
#define CPUREMOVAL_H

#include "Removal.h"

#ifndef USE_CUDA

int compare(const void *a, const void *b) {
    return (*(float *) a > *(float *) b) ? 1 : ((*(float *) a == *(float *) b) ? 0 : -1);
}

template<typename T>
inline T threeMin(T a, T b, T c) {
    return std::min(std::min(a, b), c);
}

template<typename T>
inline T threeMax(T a, T b, T c) {
    return std::max(std::max(a, b), c);
}

template<typename T>
inline int getCluster(T dist1, T dist2, T dist3) {
    return (dist1 <= dist2) ? ((dist1 <= dist3) ? 1 : 3) : ((dist2 < dist3) ? 2 : 3);
}

void Removal::computeMinMaxRange() {
    // Compute the minimum, maximum and range images
    for (int pixel = 0; pixel < numPixel; ++pixel) {
        // separate the three channels, note that OpenCV stores them in BGR order
        int blue = img[pixel * 3 + 0];
        int green = img[pixel * 3 + 1];
        int red = img[pixel * 3 + 2];

        // set I^min and I^max
        minImg[pixel] = threeMin(red, green, blue);
        maxImg[pixel] = threeMax(red, green, blue);
        rangeImg[pixel] = maxImg[pixel] - minImg[pixel];
    }
}

void Removal::computePseudoChromaticity() {
    // Estimate the pseudo-chromaticity image
    for (int pixel = 0; pixel < numPixel; ++pixel) {
        // mask determines whether a pixel is in the specular highlight region
        maskImg[pixel] = (minImg[pixel] > minMean(0)) ? 1 : 0;
        if (maskImg[pixel] & 1) {
            // calculate I^{psf} = I - I^min
            int blue = img[pixel * 3 + 0];
            int green = img[pixel * 3 + 1];
            int red = img[pixel * 3 + 2];
            double redChromatic = red - minImg[pixel] + minMean(0);
            double greenChromatic = green - minImg[pixel] + minMean(0);
            double blueChromatic = blue - minImg[pixel] + minMean(0);

            // calculate \Lambda^{psf}
            double sum = redChromatic + greenChromatic + blueChromatic;
            redChromatic /= sum;
            greenChromatic /= sum;
            blueChromatic /= sum;
            // \Lambda^{psf}_\min
            minChromaticImg[pixel] = (float) threeMin(redChromatic, greenChromatic, blueChromatic);
            // \Lambda^{psf}_\max
            maxChromaticImg[pixel] = (float) threeMax(redChromatic, greenChromatic, blueChromatic);
        }
    }
}

void Removal::CPUCluster() {
    int minIndex = minLocation.y * cols + minLocation.x;
    int maxIndex = maxLocation.y * cols + maxLocation.x;
    int minMaxIndex = maxMinLocation.y * cols + maxMinLocation.x;
    for (int pixel = 0; pixel < numPixel; ++pixel) {
        if (maskImg[pixel] & 1) {
            float minChromatic = minChromaticImg[pixel];
            float maxChromatic = maxChromaticImg[pixel];
            float dist1 = calculateDistance(minChromatic, maxChromatic,
                                            minChromaticImg[minIndex], maxChromaticImg[minIndex]);
            float dist2 = calculateDistance(minChromatic, maxChromatic,
                                            minChromaticImg[maxIndex], maxChromaticImg[maxIndex]);
            float dist3 = calculateDistance(minChromatic, maxChromatic,
                                            minChromaticImg[minMaxIndex], maxChromaticImg[minMaxIndex]);

            // assign the pixel to the cluster with the minimum distance
            clusterImg[pixel] = getCluster(dist1, dist2, dist3);
        }
    }

    for (int cluster = 1; cluster <= 3; ++cluster) {
        int count = 0;
        minCenters[cluster - 1] = 0;
        maxCenters[cluster - 1] = 0;
        for (int pixel = 0; pixel < numPixel; ++pixel) {
            if (clusterImg[pixel] == cluster) {
                minCenters[cluster - 1] += minChromaticImg[pixel];
                maxCenters[cluster - 1] += maxChromaticImg[pixel];
                ++count;
            }
        }
        minCenters[cluster - 1] /= (float) count;
        maxCenters[cluster - 1] /= (float) count;
    }

    // K-means twice
    for (int pixel = 0; pixel < numPixel; ++pixel) {
        if (maskImage.ptr<unsigned char>()[pixel] == 1) {
            float minChromatic = minChromaticImg[pixel];
            float maxChromatic = maxChromaticImg[pixel];
            float dist1 = calculateDistance(minChromatic, maxChromatic,
                                            minCenters[0], maxCenters[0]);
            float dist2 = calculateDistance(minChromatic, maxChromatic,
                                            minCenters[1], maxCenters[1]);
            float dist3 = calculateDistance(minChromatic, maxChromatic,
                                            minCenters[2], maxCenters[2]);
            clusterImg[pixel] = getCluster(dist1, dist2, dist3);
        }
    }
}

void Removal::estimateRatio() {
    for (int cluster = 1; cluster <= 3; ++cluster) {

        float estimatedRatio;
        int count = 0;

        if (useSort) {
            for (int pixel = 0; pixel < numPixel; ++pixel) {
                if (clusterImg[pixel] == cluster && rangeImg[pixel] > minMean(0)) {
                    ratio[count] = (float) maxImg[pixel] / ((float) rangeImg[pixel] + epsilon);
                    ++count;
                }
            }
            qsort(ratio, count, sizeof(float), compare);
            estimatedRatio = ratio[round((float) count * midPercent)];

        } else {

            float sumValue = 0;
            for (int pixel = 0; pixel < numPixel; ++pixel) {
                if (clusterImage.ptr<int>()[pixel] == cluster && rangeImg[pixel] > minMean(0)) {
                    ratio[count] = (float) maxImg[pixel] / ((float) rangeImg[pixel] + epsilon);
                    sumValue += ratio[count];
                    ++count;
                }
            }
            // calculate the estimated ratio
            estimatedRatio = sumValue / (float) count;

            // count both the number of pixels with ratio greater than the estimated ratio
            // and the number of pixels with ratio less than the estimated ratio
            for (int iteration = 0; iteration < alpha; ++iteration) {

                int lessCount = 0;
                int greaterCount = 0;

                for (int index = 0; index < count; ++index) {
                    if (ratio[index] > estimatedRatio) {
                        ++greaterCount;
                    } else {
                        ++lessCount;
                    }
                }

                // update the estimated ratio
                if ((float) lessCount / (float) count > beta) {
                    estimatedRatio -= (estimatedRatio * gamma);
                } else if ((float) greaterCount / (float) count > beta) {
                    estimatedRatio += (estimatedRatio * gamma);
                } else {
                    break;
                }
            }

        }

        // set the ratio of pixels in the cluster to the estimated ratio
        for (int pixel = 0; pixel < numPixel; ++pixel) {
            if (clusterImg[pixel] == cluster) {
                ratioImg[pixel] = estimatedRatio;
            }
        }
    }
}

#endif

#endif