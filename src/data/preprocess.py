import os
import pandas as pd
import geopandas as gpd
from pathlib import Path

def get_project_root():
    """Get the project root directory."""
    # This assumes the script is in src/data/
    return Path(__file__).parent.parent.parent

def load_building_data(region_name):
    """
    Load building data for a specific region.
    
    Args:
        region_name: Name of the region (e.g., 'Gaza')
        
    Returns:
        dict with footprints and grid DataFrames
    """
    project_root = get_project_root()
    
    # Set paths for building data
    footprints_dir = os.path.join(project_root, "data", "raw", "building_data", "footprints")
    grid_dir = os.path.join(project_root, "data", "raw", "building_data", "grid")
    
    # Find files for the specified region
    footprint_files = [f for f in os.listdir(footprints_dir) 
                      if f.endswith('.csv') and f.startswith(f"{region_name}_")]
    grid_files = [f for f in os.listdir(grid_dir) 
                 if f.endswith('.csv') and f.startswith(f"{region_name}_")]
    
    result = {}
    
    # Load footprint data if available
    if footprint_files:
        footprint_file = footprint_files[0]
        footprint_path = os.path.join(footprints_dir, footprint_file)
        result['footprints'] = pd.read_csv(footprint_path)
        print(f"Loaded footprint data from: {footprint_file}")
    else:
        result['footprints'] = None
        print(f"No footprint data found for region: {region_name}")
    
    # Load grid data if available
    if grid_files:
        grid_file = grid_files[0]
        grid_path = os.path.join(grid_dir, grid_file)
        result['grid'] = pd.read_csv(grid_path)
        print(f"Loaded grid data from: {grid_file}")
    else:
        result['grid'] = None
        print(f"No grid data found for region: {region_name}")
    
    return result