#pragma once

#include <string>
#include <cuda_runtime.h>

// Simple image structure
struct Image {
    unsigned char* data;    // Image data (RGBA format)
    int width;              // Image width
    int height;             // Image height
    int channels;           // Number of channels (typically 1, 3 or 4)
    int step;               // Row stride in bytes
    bool isDeviceMemory;    // Whether the data is stored in device memory
};

// Image loading and saving functions
Image loadImage(const std::string& filePath);
void saveImage(const std::string& filePath, const Image& image);
void freeImage(Image& image);

// CUDA memory management helpers
unsigned char* allocDeviceMemory(size_t size);
void freeDeviceMemory(unsigned char* ptr);
void copyHostToDevice(unsigned char* dst, const unsigned char* src, size_t size);
void copyDeviceToHost(unsigned char* dst, const unsigned char* src, size_t size);

// Utility functions
int divUp(int a, int b); 