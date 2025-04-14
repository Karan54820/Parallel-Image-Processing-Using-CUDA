#include "../include/image_processor.h"
#include <stdexcept>
#include <iostream>

// Constructor
ImageProcessor::ImageProcessor() {
    // Initialize NPP library
    const NppLibraryVersion* libVer = nppGetLibVersion();
    if (libVer == nullptr) {
        throw std::runtime_error("Failed to initialize NPP library");
    }
}

// Destructor
ImageProcessor::~ImageProcessor() {
    // Nothing to clean up for now
}

// Convert image to grayscale
Image ImageProcessor::convertToGrayscale(const Image& inputImage) {
    // Only process RGB or RGBA images
    if (inputImage.channels != 3 && inputImage.channels != 4) {
        throw std::runtime_error("Grayscale conversion requires RGB or RGBA input image");
    }
    
    // Create output image (single channel)
    Image outputImage;
    outputImage.width = inputImage.width;
    outputImage.height = inputImage.height;
    outputImage.channels = 1;
    outputImage.step = inputImage.width; // One byte per pixel for grayscale
    outputImage.isDeviceMemory = true;
    
    // Allocate device memory for output
    const size_t outputSize = outputImage.height * outputImage.step;
    outputImage.data = allocDeviceMemory(outputSize);
    
    NppStatus status;
    
    if (inputImage.channels == 3) {
        // RGB to Grayscale conversion
        status = nppiRGBToGray_8u_C3C1R(
            inputImage.data, inputImage.step,
            outputImage.data, outputImage.step,
            {inputImage.width, inputImage.height}
        );
    } else { // 4 channels (RGBA)
        // RGBA to Grayscale conversion
        status = nppiRGBToGray_8u_AC4C1R(
            inputImage.data, inputImage.step,
            outputImage.data, outputImage.step,
            {inputImage.width, inputImage.height}
        );
    }
    
    checkNppError(status, "Error converting image to grayscale");
    return outputImage;
}

// Apply Gaussian blur
Image ImageProcessor::applyGaussianBlur(const Image& inputImage, float sigma) {
    // Create output image with same format as input
    Image outputImage;
    outputImage.width = inputImage.width;
    outputImage.height = inputImage.height;
    outputImage.channels = inputImage.channels;
    outputImage.step = inputImage.step;
    outputImage.isDeviceMemory = true;
    
    // Allocate device memory for output
    const size_t outputSize = outputImage.height * outputImage.step;
    outputImage.data = allocDeviceMemory(outputSize);
    
    // Calculate kernel size based on sigma (typically 3*sigma on each side)
    int kernelSize = static_cast<int>(sigma * 6 + 1);
    if (kernelSize % 2 == 0) kernelSize++; // Ensure odd kernel size
    
    NppStatus status;
    NppiSize oSizeROI = {inputImage.width, inputImage.height};
    
    // Create a mask based on kernel size (3x3, 5x5, etc.)
    NppiMaskSize maskSize;
    if (kernelSize <= 3) 
        maskSize = NPP_MASK_SIZE_3_X_3;
    else if (kernelSize <= 5) 
        maskSize = NPP_MASK_SIZE_5_X_5;
    else 
        maskSize = NPP_MASK_SIZE_7_X_7;
    
    // Different functions for different channels
    if (inputImage.channels == 1) {
        // Apply Gaussian filter for grayscale image
        status = nppiFilterGauss_8u_C1R(
            inputImage.data, inputImage.step,
            outputImage.data, outputImage.step,
            oSizeROI, maskSize
        );
    } else if (inputImage.channels == 3) {
        // Apply Gaussian filter for RGB image
        status = nppiFilterGauss_8u_C3R(
            inputImage.data, inputImage.step,
            outputImage.data, outputImage.step,
            oSizeROI, maskSize
        );
    } else { // 4 channels
        // Apply Gaussian filter for RGBA image
        status = nppiFilterGauss_8u_C4R(
            inputImage.data, inputImage.step,
            outputImage.data, outputImage.step,
            oSizeROI, maskSize
        );
    }
    
    checkNppError(status, "Error applying Gaussian blur");
    return outputImage;
}

// Edge detection using Sobel operator
Image ImageProcessor::detectEdges(const Image& inputImage) {
    // First, convert to grayscale if needed
    Image grayImage;
    if (inputImage.channels == 1) {
        // Already grayscale, just make a copy
        grayImage.width = inputImage.width;
        grayImage.height = inputImage.height;
        grayImage.channels = 1;
        grayImage.step = inputImage.width;
        grayImage.isDeviceMemory = true;
        
        size_t size = grayImage.height * grayImage.step;
        grayImage.data = allocDeviceMemory(size);
        
        copyHostToDevice(grayImage.data, inputImage.data, size);
    } else {
        // Convert to grayscale
        grayImage = convertToGrayscale(inputImage);
    }
    
    // Create output image (grayscale)
    Image outputImage;
    outputImage.width = grayImage.width;
    outputImage.height = grayImage.height;
    outputImage.channels = 1;
    outputImage.step = grayImage.width;
    outputImage.isDeviceMemory = true;
    
    // Allocate device memory for output and temporary gradients
    const size_t outputSize = outputImage.height * outputImage.step;
    outputImage.data = allocDeviceMemory(outputSize);
    
    // Allocate device memory for sobel derivatives
    unsigned char* dx = allocDeviceMemory(outputSize);
    unsigned char* dy = allocDeviceMemory(outputSize);
    
    // Apply Sobel filter
    NppiSize oSizeROI = {grayImage.width, grayImage.height};
    NppStatus status;
    
    // Calculate dx (horizontal gradient)
    status = nppiFilterSobelHoriz_8u_C1R(
        grayImage.data, grayImage.step,
        dx, grayImage.step,
        oSizeROI
    );
    checkNppError(status, "Error calculating horizontal gradient");
    
    // Calculate dy (vertical gradient)
    status = nppiFilterSobelVert_8u_C1R(
        grayImage.data, grayImage.step,
        dy, grayImage.step,
        oSizeROI
    );
    checkNppError(status, "Error calculating vertical gradient");
    
    // Use our custom CUDA kernel to combine gradients
    try {
        launchEdgeDetectionKernel(
            dx, dy, outputImage.data,
            grayImage.width, grayImage.height, grayImage.step
        );
    } catch (const std::exception& e) {
        // Clean up first
        freeDeviceMemory(dx);
        freeDeviceMemory(dy);
        if (inputImage.channels != 1) {
            freeImage(grayImage);
        }
        throw; // Re-throw the exception
    }
    
    // Clean up
    freeDeviceMemory(dx);
    freeDeviceMemory(dy);
    if (inputImage.channels != 1) {
        freeImage(grayImage);
    }
    
    return outputImage;
}

// Resize image
Image ImageProcessor::resizeImage(const Image& inputImage, int newWidth, int newHeight) {
    // Create output image
    Image outputImage;
    outputImage.width = newWidth;
    outputImage.height = newHeight;
    outputImage.channels = inputImage.channels;
    outputImage.step = newWidth * inputImage.channels;
    outputImage.isDeviceMemory = true;
    
    // Allocate device memory for output
    const size_t outputSize = outputImage.height * outputImage.step;
    outputImage.data = allocDeviceMemory(outputSize);
    
    NppStatus status;
    NppiSize srcSize = {inputImage.width, inputImage.height};
    NppiRect srcROI = {0, 0, inputImage.width, inputImage.height};
    NppiSize dstSize = {newWidth, newHeight};
    NppiRect dstROI = {0, 0, newWidth, newHeight};
    
    // Resize based on channel count
    if (inputImage.channels == 1) {
        status = nppiResize_8u_C1R(
            inputImage.data, inputImage.step, srcSize, srcROI,
            outputImage.data, outputImage.step, dstSize, dstROI,
            NPPI_INTER_LINEAR
        );
    } else if (inputImage.channels == 3) {
        status = nppiResize_8u_C3R(
            inputImage.data, inputImage.step, srcSize, srcROI,
            outputImage.data, outputImage.step, dstSize, dstROI,
            NPPI_INTER_LINEAR
        );
    } else { // 4 channels
        status = nppiResize_8u_C4R(
            inputImage.data, inputImage.step, srcSize, srcROI,
            outputImage.data, outputImage.step, dstSize, dstROI,
            NPPI_INTER_LINEAR
        );
    }
    
    checkNppError(status, "Error resizing image");
    return outputImage;
}

// Error checking helper functions
void ImageProcessor::checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::string errorMsg = message;
        errorMsg += ": ";
        errorMsg += cudaGetErrorString(error);
        throw std::runtime_error(errorMsg);
    }
}

void ImageProcessor::checkNppError(NppStatus status, const char* message) {
    if (status != NPP_SUCCESS) {
        std::string errorMsg = message;
        errorMsg += ": NPP error code ";
        errorMsg += std::to_string(status);
        throw std::runtime_error(errorMsg);
    }
} 