#!/usr/bin/env python3
"""
Generate synthetic test images for CUDA image processing project
This script creates a set of synthetic test images of various sizes
"""

import os
import sys
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import random

def generate_gradient_image(width, height, name, output_dir):
    """Generate a gradient test image"""
    # Create gradient arrays
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Create RGB channels
    r = np.uint8(X * 255)
    g = np.uint8(Y * 255)
    b = np.uint8(255 - X * 255)
    
    # Combine into RGB image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    
    # Create PIL image and save
    pil_img = Image.fromarray(img)
    output_path = output_dir / f"{name}_gradient_{width}x{height}.png"
    pil_img.save(output_path)
    print(f"Generated gradient image: {output_path}")
    return output_path

def generate_noise_image(width, height, name, output_dir):
    """Generate a noise test image"""
    # Create random noise
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Create PIL image and save
    pil_img = Image.fromarray(noise)
    output_path = output_dir / f"{name}_noise_{width}x{height}.png"
    pil_img.save(output_path)
    print(f"Generated noise image: {output_path}")
    return output_path

def generate_pattern_image(width, height, name, output_dir):
    """Generate a pattern test image (checkerboard)"""
    # Create a blank image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Define checkerboard parameters
    square_size = min(width, height) // 8
    
    # Generate checkerboard pattern
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            # Determine if this square should be colored
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                # Random color for this square
                color = np.random.randint(100, 256, 3, dtype=np.uint8)
                
                # Fill the square
                y_end = min(y + square_size, height)
                x_end = min(x + square_size, width)
                img[y:y_end, x:x_end] = color
    
    # Create PIL image and save
    pil_img = Image.fromarray(img)
    output_path = output_dir / f"{name}_pattern_{width}x{height}.png"
    pil_img.save(output_path)
    print(f"Generated pattern image: {output_path}")
    return output_path

def generate_circle_image(width, height, name, output_dir):
    """Generate an image with circles"""
    # Create a blank image (white background)
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Generate random circles
    num_circles = random.randint(5, 20)
    
    for _ in range(num_circles):
        # Random circle parameters
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        radius = random.randint(20, min(width, height) // 4)
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
        
        # Draw circle
        y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
        mask = x*x + y*y <= radius*radius
        
        # Apply color to circle area
        img[mask] = color
    
    # Create PIL image and save
    pil_img = Image.fromarray(img)
    output_path = output_dir / f"{name}_circles_{width}x{height}.png"
    pil_img.save(output_path)
    print(f"Generated circles image: {output_path}")
    return output_path

def main():
    """Main function to generate synthetic test images"""
    parser = argparse.ArgumentParser(description="Generate synthetic test images for CUDA processing")
    parser.add_argument("--output", default="../data", help="Output directory for generated images")
    parser.add_argument("--count", type=int, default=10, help="Number of images to generate of each type")
    parser.add_argument("--min-size", type=int, default=512, help="Minimum image dimension")
    parser.add_argument("--max-size", type=int, default=2048, help="Maximum image dimension")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Image generation functions
    generators = [
        generate_gradient_image,
        generate_noise_image,
        generate_pattern_image,
        generate_circle_image
    ]
    
    # Generate images
    for i in range(args.count):
        # Random size between min and max
        width = random.randint(args.min_size, args.max_size)
        height = random.randint(args.min_size, args.max_size)
        
        # Generate images of each type
        for generator in generators:
            generator(width, height, f"test{i:03d}", output_dir)
    
    print(f"Generated {args.count * len(generators)} images in {output_dir}")

if __name__ == "__main__":
    main() 