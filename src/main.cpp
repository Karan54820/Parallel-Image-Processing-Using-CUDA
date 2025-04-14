/**
 * CUDA Image Processing
 * 
 * This program demonstrates batch image processing using CUDA.
 * Operations include: grayscale conversion, blurring, and edge detection.
 */

 #include <iostream>
 #include <string>
 #include <vector>
 #include <chrono>
 #include <filesystem>
 #include <fstream>
 #include <algorithm>
 #include <cuda_runtime.h>
 #include <npp.h>
 #include "../include/image_processor.h"
 #include "../include/utils.h"
 
 namespace fs = std::filesystem;
 
 // Set this to true for debug output
 const bool DEBUG_MODE = false;
 
 int main(int argc, char** argv) {
     try {
         std::cout << "CUDA Batch Image Processing" << std::endl;
         
         // Initialize CUDA and NPP
         cudaError_t cudaStatus = cudaSetDevice(0);
         if (cudaStatus != cudaSuccess) {
             throw std::runtime_error("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
         }
         
         // Print device info
         int deviceCount = 0;
         cudaGetDeviceCount(&deviceCount);
         if (deviceCount == 0) {
             throw std::runtime_error("No CUDA capable devices found");
         }
         
         cudaDeviceProp deviceProp;
         cudaGetDeviceProperties(&deviceProp, 0);
         std::cout << "Using GPU: " << deviceProp.name << std::endl;
         
         // Parse command line arguments
         std::string inputDir = "./data";
         std::string outputDir = "./results";
         
         for (int i = 1; i < argc; i++) {
             std::string arg = argv[i];
             if (arg == "--input" && i + 1 < argc) {
                 inputDir = argv[++i];
             } else if (arg == "--output" && i + 1 < argc) {
                 outputDir = argv[++i];
             }
         }
         
         // Create output directory if it doesn't exist
         if (!fs::exists(outputDir)) {
             fs::create_directory(outputDir);
         }
         
         // Create a log file
         std::ofstream logFile(outputDir + "/processing_log.txt");
         if (!logFile) {
             throw std::runtime_error("Failed to create log file");
         }
         
         logFile << "CUDA Batch Image Processing Log" << std::endl;
         logFile << "================================" << std::endl;
         logFile << "Device: " << deviceProp.name << std::endl;
         logFile << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
         logFile << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
         logFile << "================================" << std::endl << std::endl;
         
         // Get all image files from the input directory
         std::vector<std::string> imageFiles;
         if (fs::exists(inputDir)) {
             for (const auto& entry : fs::directory_iterator(inputDir)) {
                 if (entry.is_regular_file()) {
                     std::string extension = entry.path().extension().string();
                     // Convert to lowercase
                     std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                     
                     if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp") {
                         imageFiles.push_back(entry.path().string());
                     }
                 }
             }
         } else {
             throw std::runtime_error("Input directory does not exist: " + inputDir);
         }
         
         if (imageFiles.empty()) {
             throw std::runtime_error("No image files found in input directory: " + inputDir);
         }
         
         std::cout << "Found " << imageFiles.size() << " image files to process" << std::endl;
         logFile << "Found " << imageFiles.size() << " image files to process" << std::endl;
         
         // Process each image
         ImageProcessor processor;
         
         auto totalStartTime = std::chrono::high_resolution_clock::now();
         
         // Count successful processing
         int successCount = 0;
         
         for (const auto& imagePath : imageFiles) {
             std::string filename = fs::path(imagePath).filename().string();
             std::string filenameNoExt = fs::path(imagePath).stem().string();
             
             logFile << "Processing: " << filename << std::endl;
             std::cout << "Processing: " << filename << "..." << std::flush;
             
             try {
                 auto startTime = std::chrono::high_resolution_clock::now();
                 
                 // Load image (this creates a synthetic 640x480 image for our demo)
                 Image img = loadImage(imagePath);
                 
                 // Check if image loaded correctly
                 if (!img.data) {
                     throw std::runtime_error("Failed to load image data");
                 }
                 
                 // Create output paths
                 std::string grayPath = outputDir + "/" + filenameNoExt + "_gray.png";
                 std::string blurPath = outputDir + "/" + filenameNoExt + "_blur.png";
                 std::string edgePath = outputDir + "/" + filenameNoExt + "_edge.png";
                 
                 // Upload image to GPU
                 Image deviceImg;
                 deviceImg.width = img.width;
                 deviceImg.height = img.height;
                 deviceImg.channels = img.channels;
                 deviceImg.step = img.step;
                 deviceImg.isDeviceMemory = true;
                 
                 size_t imageSize = deviceImg.height * deviceImg.step;
                 deviceImg.data = allocDeviceMemory(imageSize);
                 
                 // Copy image data to GPU
                 copyHostToDevice(deviceImg.data, img.data, imageSize);
                 
                 if (DEBUG_MODE) {
                     std::cout << "Uploaded image to GPU: " << deviceImg.width << "x" << deviceImg.height 
                               << ", " << deviceImg.channels << " channels, " << imageSize << " bytes" << std::endl;
                 }
                 
                 // Process image
                 Image grayImg = processor.convertToGrayscale(deviceImg);
                 Image blurImg = processor.applyGaussianBlur(deviceImg);
                 Image edgeImg = processor.detectEdges(deviceImg);
                 
                 // Save processed images
                 saveImage(grayPath, grayImg);
                 saveImage(blurPath, blurImg);
                 saveImage(edgePath, edgeImg);
                 
                 // Free memory
                 freeImage(deviceImg);
                 freeImage(grayImg);
                 freeImage(blurImg);
                 freeImage(edgeImg);
                 freeImage(img);
                 
                 auto endTime = std::chrono::high_resolution_clock::now();
                 auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
                 
                 logFile << "  Completed in " << duration << " ms" << std::endl;
                 std::cout << " Completed in " << duration << " ms" << std::endl;
                 
                 successCount++;
             } catch (const std::exception& e) {
                 logFile << "  Error processing " << filename << ": " << e.what() << std::endl;
                 std::cout << std::endl << "  Error processing " << filename << ": " << e.what() << std::endl;
             }
         }
         
         auto totalEndTime = std::chrono::high_resolution_clock::now();
         auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totalEndTime - totalStartTime).count();
         
         logFile << std::endl << "================================" << std::endl;
         logFile << "Total processing time: " << totalDuration << " ms" << std::endl;
         logFile << "Average time per image: " << totalDuration / imageFiles.size() << " ms" << std::endl;
         logFile << "Successfully processed " << successCount << " of " << imageFiles.size() << " images" << std::endl;
         
         std::cout << std::endl << "Processing complete!" << std::endl;
         std::cout << "Total processing time: " << totalDuration << " ms" << std::endl;
         std::cout << "Average time per image: " << totalDuration / imageFiles.size() << " ms" << std::endl;
         std::cout << "Successfully processed " << successCount << " of " << imageFiles.size() << " images" << std::endl;
         
         // Cleanup
         logFile.close();
         cudaDeviceReset();
         
         return 0;
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
 } 