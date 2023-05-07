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

    auto begin = std::chrono::high_resolution_clock::now();
    Removal removal;
    removal.initialize(inputImage.rows, inputImage.cols);


    outputImage = removal.run(inputImage);
    auto end = std::chrono::high_resolution_clock::now();
    #ifndef USE_CUDA
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us"
              << std::endl;
    #endif

    // save output image
    cv::imwrite("fix.png", outputImage);
    while (cv::waitKey(33) != 13) {
        cv::imshow("Input Image", inputImage);
        cv::imshow("Output Image", outputImage);
    }

    return 0;

}
