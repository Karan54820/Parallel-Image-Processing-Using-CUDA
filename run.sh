#!/bin/bash
# End-to-end script to run the CUDA Batch Image Processing project

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print colored messages
print_message() {
    echo -e "\e[1;34m[INFO]\e[0m $1"
}

print_success() {
    echo -e "\e[1;32m[SUCCESS]\e[0m $1"
}

print_error() {
    echo -e "\e[1;31m[ERROR]\e[0m $1"
}

# Check for required dependencies
print_message "Checking dependencies..."

# Check for CUDA
if ! command_exists nvcc; then
    print_error "CUDA toolkit (nvcc) not found. Please install CUDA toolkit."
    exit 1
fi

# Check for Python (for generating test data)
if ! command_exists python3; then
    print_error "Python 3 not found. It's needed for generating test data."
    exit 1
fi

# Check for Python packages
python3 -c "import numpy, PIL" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    print_message "Installing required Python packages..."
    pip install numpy pillow
fi

# Create necessary directories
print_message "Creating project directories..."
mkdir -p bin data results

# Generate synthetic test data
print_message "Generating synthetic test images..."
cd scripts || exit 1
python3 generate_synthetic_images.py --count 25 --output ../data
cd ..

# Build the project
print_message "Building the project..."
make clean
make

# Run the CUDA image processor
if [ -f bin/image_processor ]; then
    print_message "Running CUDA batch image processor..."
    time bin/image_processor --input data --output results
    
    # Check results
    image_count=$(ls -1 data | wc -l)
    result_count=$(ls -1 results | grep -v "log" | wc -l)
    
    print_message "Processed $image_count input images into $result_count output files"
    
    # Display processing log
    if [ -f results/processing_log.txt ]; then
        print_message "Processing log summary:"
        grep "Total processing time:" results/processing_log.txt
        grep "Average time per image:" results/processing_log.txt
    fi
    
    print_success "CUDA image processing completed successfully!"
else
    print_error "Failed to build the project. Check build errors above."
    exit 1
fi 