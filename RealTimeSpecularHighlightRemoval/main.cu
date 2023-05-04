#include <opencv2\opencv.hpp>
#include "Removal.h"
#include <chrono>

int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " img\n";
        exit(1);
    }

    cv::Mat inputImage = cv::imread(argv[1]);
    cv::Mat outputImage;

    Removal removal;
    removal.initialize(inputImage.rows, inputImage.cols);

    auto begin = std::chrono::high_resolution_clock::now();
    outputImage = removal.run(inputImage);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "ns"
              << std::endl;

    while (cv::waitKey(33) != 13) {
        cv::imshow("Input Image", inputImage);
        cv::imshow("Output Image", outputImage);
    }

    return 0;

}
