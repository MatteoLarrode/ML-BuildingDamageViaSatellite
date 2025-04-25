#!/usr/bin/env python
"""
Script to download and preprocess a single Sentinel-1 zip file.
Downloads to data/raw/sentinel, processes, then cleans up the raw file.

Usage:
    python download_process_single_sentinel_file.py <sentinel_url> <aoi_path> <output_dir> [--snap_gpt_path PATH]

Example:
    python download_process_single_sentinel_file.py https://datapool.asf.alaska.edu/GRD_HD/SA/S1A_IW_GRDH_1SDV_20230427T154057_20230427T154122_048284_05CE71_87D9.zip ../utils/AOI_bboxes/aoi_shifa.geojson ../data/preprocessed/sentinel
"""
import os
import sys
import subprocess
import argparse
import shutil
from datetime import datetime

def download_sentinel_data(url, output_dir):
    """Download Sentinel-1 data from URL to output directory"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename from URL
    filename = os.path.basename(url)
    output_path = os.path.join(output_dir, filename)
    
    # Check if file already exists
    if os.path.exists(output_path):
        print(f"File {filename} already exists at {output_path}")
        return output_path
    
    print(f"Downloading {filename} from {url}")
    print(f"Output location: {output_path}")
    
    # Run download script
    start_time = datetime.now()
    
    # Download using subprocess
    cmd = ["python", "../scripts/download_process_single_sentinel_file.py", url, 
           "--output_dir", output_dir]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Download failed with return code {result.returncode}")
        return None
    
    # Verify download
    if not os.path.exists(output_path):
        print(f"ERROR: Expected file not found at {output_path}")
        return None
    
    download_time = datetime.now() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024*1024)
    
    print(f"Download completed in {download_time}")
    print(f"File saved to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    
    return output_path

def preprocess_sentinel_data(zip_path, aoi_path, output_dir, snap_gpt_path="/Applications/esa-snap/bin/gpt"):
    """Preprocess Sentinel-1 data using SNAP"""
    
    if not os.path.exists(zip_path):
        print(f"ERROR: Zip file not found at {zip_path}")
        return False
    
    if not os.path.exists(aoi_path):
        print(f"ERROR: AOI file not found at {aoi_path}")
        return False
    
    print(f"Preprocessing {os.path.basename(zip_path)}")
    print(f"Using AOI: {aoi_path}")
    print(f"Output directory: {output_dir}")
    
    # Run preprocessing script
    start_time = datetime.now()
    
    cmd = ["python", "../scripts/preprocess_single_sentinel_zip.py", 
           zip_path, aoi_path, output_dir, "--snap_gpt_path", snap_gpt_path]
    
    print(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Preprocessing failed with return code {result.returncode}")
        return False
    
    preprocess_time = datetime.now() - start_time
    print(f"Preprocessing completed in {preprocess_time}")
    
    # Check output directory
    product_id = os.path.basename(zip_path).split('.')[0]
    date_str = product_id.split('_')[5][:8]  # Extract date YYYYMMDD
    date_output_dir = os.path.join(output_dir, date_str)
    
    if os.path.exists(date_output_dir):
        files = os.listdir(date_output_dir)
        print(f"Found {len(files)} files in output directory {date_output_dir}:")
        for file in files:
            print(f" - {file}")
        return True
    else:
        print(f"WARNING: Expected output directory {date_output_dir} not found!")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download and preprocess Sentinel-1 data')
    parser.add_argument('url', help='URL of Sentinel-1 zip file')
    parser.add_argument('aoi_path', help='Path to AOI GeoJSON file')
    parser.add_argument('output_dir', help='Directory for preprocessed outputs')
    parser.add_argument('--raw_dir', default='../data/raw/sentinel', 
                        help='Directory for raw data (default: ../data/raw/sentinel)')
    parser.add_argument('--snap_gpt_path', default='gpt', 
                        help='Path to SNAP GPT executable (default: "gpt")')
    parser.add_argument('--keep_raw', action='store_true', 
                        help='Keep raw data after processing')
    
    args = parser.parse_args()
    
    # Start timing
    overall_start_time = datetime.now()
    
    # Step 1: Download data
    zip_path = download_sentinel_data(args.url, args.raw_dir)
    if not zip_path:
        print("Download failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Preprocess data
    success = preprocess_sentinel_data(
        zip_path, args.aoi_path, args.output_dir, args.snap_gpt_path
    )
    
    # Step 3: Clean up raw data
    if success and not args.keep_raw:
        print(f"Removing raw data file: {zip_path}")
        os.remove(zip_path)
        print("Raw data removed successfully")
    
    # Final timing
    total_time = datetime.now() - overall_start_time
    print(f"\nTotal processing time: {total_time}")
    
    if success:
        print("Process completed successfully!")
        return 0
    else:
        print("Process failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
