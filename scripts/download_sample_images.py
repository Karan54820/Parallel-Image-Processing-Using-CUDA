#!/usr/bin/env python3
"""
Download sample images for CUDA image processing project
This script will download images from the USC-SIPI image database
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import argparse

# Base URL for the USC-SIPI image database
BASE_URL = "https://sipi.usc.edu/database/"

# Sample images to download
IMAGE_SETS = {
    "misc": "misc.zip",  # Miscellaneous images
    "aerials": "aerials.zip",  # Aerial photographs
    "textures": "textures.zip"  # Texture patterns
}

def download_file(url, destination):
    """Download a file from a URL to a destination path"""
    print(f"Downloading {url} to {destination}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"Successfully downloaded {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file to a directory"""
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Successfully extracted {zip_path}")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def main():
    """Main function to download sample images"""
    parser = argparse.ArgumentParser(description="Download sample images for CUDA image processing")
    parser.add_argument("--output", default="../data", help="Output directory for downloaded images")
    parser.add_argument("--set", choices=["all"] + list(IMAGE_SETS.keys()), default="misc", 
                        help="Image set to download (default: misc)")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which sets to download
    sets_to_download = list(IMAGE_SETS.keys()) if args.set == "all" else [args.set]
    
    # Download and extract each set
    for image_set in sets_to_download:
        if image_set not in IMAGE_SETS:
            print(f"Unknown image set: {image_set}")
            continue
            
        zip_file = IMAGE_SETS[image_set]
        url = f"{BASE_URL}{zip_file}"
        download_path = output_dir / zip_file
        
        # Download the zip file
        if download_file(url, download_path):
            # Extract it
            extract_zip(download_path, output_dir)
            
            # Remove the zip file
            download_path.unlink()

if __name__ == "__main__":
    main() 