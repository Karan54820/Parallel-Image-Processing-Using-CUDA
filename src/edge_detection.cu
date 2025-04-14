#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdexcept>
#include <string>
#include "../include/utils.h"

/**
 * CUDA Kernel for combining horizontal and vertical gradients 
 * to create an edge detection image
 */
__global__ void combineEdgeGradientsKernel(
    const unsigned char* dx, 
    const unsigned char* dy,
    unsigned char* output,
    int width, 
    int height, 
    int stride
) {
    // Calculate global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within image bounds
    if (x < width && y < height) {
        // Calculate pixel position in the arrays
        int pos = y * stride + x;
        
        // Read gradient values
        float dx_val = static_cast<float>(dx[pos]);
        float dy_val = static_cast<float>(dy[pos]);
        
        // Combine gradients using sqrt(dx^2 + dy^2)
        float edge_val = sqrtf(dx_val * dx_val + dy_val * dy_val);
        
        // Clamp to valid range [0, 255]
        edge_val = edge_val > 255.0f ? 255.0f : edge_val;
        
        // Write result to output
        output[pos] = static_cast<unsigned char>(edge_val);
    }
}

/**
 * Host function to launch the edge detection kernel
 */
extern "C" void launchEdgeDetectionKernel(
    const unsigned char* dx,
    const unsigned char* dy,
    unsigned char* output,
    int width,
    int height,
    int stride
) {
    // Define thread block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    // Launch the kernel
    combineEdgeGradientsKernel<<<gridSize, blockSize>>>(
        dx, dy, output, width, height, stride
    );
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to launch edge detection kernel: " + 
                               std::string(cudaGetErrorString(error)));
    }
    
    // Wait for kernel to finish
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        throw std::runtime_error("Edge detection kernel execution failed: " + 
                               std::string(cudaGetErrorString(error)));
    }
} 