#!/usr/bin/env python
"""
Script to preprocess a single Sentinel-1 zip file using the process_single_file function.

Usage:
    python preprocess_single_sentinel_zip.py path/to/zipfile.zip path/to/aoi.geojson output_directory [path/to/snap/gpt]

Example:
    python preprocess_single_sentinel_zip.py ../data/raw/sentinel/ref/S1A_IW_GRDH_1SDV_20230427T154057_20230427T154122_048284_05CE71_87D9.zip ../utils/AOI_bboxes/aoi_shifa.geojson ../data/preprocessed/sentinel /Applications/esa-snap/bin/gpt
"""

import argparse

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocessing_sentinel_helpers import process_single_file

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess a single Sentinel-1 zip file.')
    parser.add_argument('zip_path', help='Path to the Sentinel-1 zip file')
    parser.add_argument('aoi_path', help='Path to the AOI GeoJSON file')
    parser.add_argument('output_dir', help='Directory to save preprocessed outputs')
    parser.add_argument('--snap_gpt_path', default='/Applications/esa-snap/bin/gpt', 
                        help='Path to SNAP GPT executable (default: "gpt", assumes it\'s in PATH)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.zip_path):
        print(f"Error: Zip file not found: {args.zip_path}")
        sys.exit(1)
    
    if not os.path.exists(args.aoi_path):
        print(f"Error: AOI file not found: {args.aoi_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print processing information
    print(f"Processing Sentinel-1 file: {os.path.basename(args.zip_path)}")
    print(f"Using AOI: {os.path.basename(args.aoi_path)}")
    print(f"Output directory: {args.output_dir}")
    print(f"SNAP GPT path: {args.snap_gpt_path}")
    
    # Process the file
    try:
        result = process_single_file(
            zip_path=args.zip_path,
            aoi_path=args.aoi_path,
            output_dir=args.output_dir,
            snap_gpt_path=args.snap_gpt_path
        )
        
        # Check result
        if result:
            print("\nProcessing completed successfully!")
            print(f"Output files: {result}")
        else:
            print("\nProcessing failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
