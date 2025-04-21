import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import fiona

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

def find_unosat_gdb():
    """Find UNOSAT GDB files in the data directory."""
    project_root = get_project_root()
    labels_dir = os.path.join(project_root, "data", "raw", "labels")
    
    gdb_paths = []
    for root, dirs, files in os.walk(labels_dir):
        for dir_name in dirs:
            if dir_name.endswith('.gdb'):
                gdb_path = os.path.join(root, dir_name)
                gdb_paths.append(gdb_path)
    
    return gdb_paths

def list_gdb_layers(gdb_path):
    """List all layers in a GDB file."""
    if not os.path.exists(gdb_path):
        print(f"GDB file not found: {gdb_path}")
        return []
    
    try:
        layers = fiona.listlayers(gdb_path)
        return layers
    except Exception as e:
        print(f"Error listing layers in GDB: {str(e)}")
        return []

def load_unosat_layer(gdb_path, layer_name):
    """
    Load a specific layer from a UNOSAT GDB file.
    
    Args:
        gdb_path: Path to the GDB file
        layer_name: Name of the layer to load
        
    Returns:
        GeoDataFrame with the layer data
    """
    if not os.path.exists(gdb_path):
        print(f"GDB file not found: {gdb_path}")
        return None
    
    try:
        # Check if the layer exists
        layers = fiona.listlayers(gdb_path)
        if layer_name not in layers:
            print(f"Layer '{layer_name}' not found in GDB. Available layers: {', '.join(layers)}")
            return None
        
        # Read the layer
        gdf = gpd.read_file(gdb_path, layer=layer_name)
        print(f"Loaded layer '{layer_name}' with {len(gdf)} features")
        return gdf
    except Exception as e:
        print(f"Error loading layer from GDB: {str(e)}")
        return None

def load_all_unosat_layers(gdb_path=None):
    """
    Load all layers from a UNOSAT GDB file.
    
    Args:
        gdb_path: Path to the GDB file. If None, will find GDB files automatically.
        
    Returns:
        Dictionary with layer names as keys and GeoDataFrames as values
    """
    if gdb_path is None:
        # Find GDB files
        gdb_paths = find_unosat_gdb()
        if not gdb_paths:
            print("No UNOSAT GDB files found.")
            return {}
        # Use the first GDB file found
        gdb_path = gdb_paths[0]
    
    if not os.path.exists(gdb_path):
        print(f"GDB file not found: {gdb_path}")
        return {}
    
    try:
        # List all layers in the GDB
        layers = fiona.listlayers(gdb_path)
        print(f"Found {len(layers)} layers in {gdb_path}:")
        for layer in layers:
            print(f" - {layer}")
        
        # Load each layer
        layers_dict = {}
        for layer in layers:
            try:
                gdf = gpd.read_file(gdb_path, layer=layer)
                layers_dict[layer] = gdf
                print(f"Loaded layer '{layer}' with {len(gdf)} features")
            except Exception as e:
                print(f"Error loading layer '{layer}': {str(e)}")
        
        return layers_dict
    except Exception as e:
        print(f"Error loading layers from GDB: {str(e)}")
        return {}