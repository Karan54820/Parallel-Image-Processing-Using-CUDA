# CUDA Batch Image Processing

This project demonstrates parallel image processing using NVIDIA CUDA and NPP (NVIDIA Performance Primitives) to process a large number of images efficiently.

## Features

- Process multiple images in parallel using CUDA
- Apply various image processing operations:
  - Grayscale conversion
  - Gaussian blur
  - Edge detection
  - Image resizing
- Performance benchmarking and logging

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+
- NPP (NVIDIA Performance Primitives)
- C++17 compatible compiler

## Project Structure

```
.
├── bin/              # Executable files
├── data/             # Input image data
├── include/          # Header files
├── results/          # Processed output images
├── scripts/          # Utility scripts
├── src/              # Source code
├── Makefile          # Build configuration
└── README.md         # This file
```

## Building the Project

To build the project, simply run:

```bash
make
```

This will create the executable in the `bin/` directory.

## Downloading Sample Data

The project includes a Python script to download sample images from the USC-SIPI Image Database:

```bash
cd scripts
python download_sample_images.py
```

This will download a set of sample images to the `data/` directory.

## Running the Project

To run the project with default settings:

```bash
make run
```

This will process all images in the `data/` directory and save the results to the `results/` directory.

You can also run the executable directly with custom parameters:

```bash
./bin/image_processor --input path/to/input --output path/to/output
```

## Performance

The application measures and logs the processing time for each image, providing insights into the performance gains from GPU acceleration. These metrics are saved in `results/processing_log.txt`.

## Implementation Details

This project uses the NVIDIA NPP library for core image processing operations:

- `nppiRGBToGray_8u_C3C1R` - Grayscale conversion
- `nppiFilterGaussAdvanced_8u_C3R` - Gaussian blur
- `nppiFilterSobelHorizBorder_8u_C1R` and `nppiFilterSobelVertBorder_8u_C1R` - Edge detection
- `nppiResize_8u_C3R` - Image resizing

## Extending the Project

You can extend this project by:

1. Adding new image processing operations
2. Implementing custom CUDA kernels for specific effects
3. Adding batch processing for video frames
4. Integrating with other libraries like OpenCV

## License

This project is provided as-is under the MIT License. 