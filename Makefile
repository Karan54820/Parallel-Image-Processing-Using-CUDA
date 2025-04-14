################################################################################
# CUDA Batch Image Processing Makefile
################################################################################

# Define the compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++17 -I/usr/local/cuda/include -Iinclude
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc

# Define directories
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data
RESULTS_DIR = results

# Define source files and target executable
CPP_SRC = $(wildcard $(SRC_DIR)/*.cpp)
CU_SRC = $(wildcard $(SRC_DIR)/*.cu)
CPP_OBJ = $(CPP_SRC:.cpp=.o)
CU_OBJ = $(CU_SRC:.cu=.o)
TARGET = $(BIN_DIR)/image_processor

# Define CUDA architecture flags (can be adjusted based on your GPU)
CUDA_ARCH = -gencode arch=compute_60,code=sm_60 \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80

# Define the default rule
all: directories $(TARGET)

# Create necessary directories
directories:
	mkdir -p $(BIN_DIR) $(RESULTS_DIR)

# Rule for building the target executable
$(TARGET): $(CPP_OBJ) $(CU_OBJ)
	$(NVCC) $(CUDA_ARCH) $^ -o $@ $(LDFLAGS)

# Rule for building C++ object files
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(CUDA_ARCH) $(CXXFLAGS) -c $< -o $@

# Rule for building CUDA object files
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CUDA_ARCH) $(CXXFLAGS) -c $< -o $@

# Rule for running the application
run: $(TARGET)
	$(TARGET) --input $(DATA_DIR) --output $(RESULTS_DIR)

# Rule for generating sample data (if no real data is available)
sample_data:
	mkdir -p $(DATA_DIR)
	cd scripts && python generate_synthetic_images.py
	
# Clean up
clean:
	rm -rf $(BIN_DIR)/* $(SRC_DIR)/*.o $(RESULTS_DIR)/*

# Installation rule
install:
	@echo "No installation required."

# Help command
help:
	@echo "Available make commands:"
	@echo "  make                - Build the project."
	@echo "  make run            - Run the project on all images in the data directory."
	@echo "  make sample_data    - Generate sample data (for testing)."
	@echo "  make clean          - Clean up the build files and results."
	@echo "  make help           - Display this help message."

.PHONY: all directories run clean install help sample_data