#pragma once

#include <cuda_runtime.h>
#include <npp.h>
#include "utils.h"

// Declaration of CUDA kernel function (implemented in edge_detection.cu)
extern "C" void launchEdgeDetectionKernel(
    const unsigned char* dx,
    const unsigned char* dy,
    unsigned char* output,
    int width,
    int height,
    int stride
);

class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();
    
    // Image processing functions
    Image convertToGrayscale(const Image& inputImage);
    Image applyGaussianBlur(const Image& inputImage, float sigma = 2.0f);
    Image detectEdges(const Image& inputImage);
    Image resizeImage(const Image& inputImage, int newWidth, int newHeight);
    
private:
    // CUDA helper functions
    void checkCudaError(cudaError_t error, const char* message);
    void checkNppError(NppStatus status, const char* message);
}; 