from pathlib import Path
import json
import fiona
import geopandas as gpd
import os
import pandas as pd


# ---- Setup ----
def get_project_root():
    """Get the project root directory."""
    # This assumes the script is in src/data/
    return Path(__file__).parent.parent.parent


# --- UNOSAT GDB Functions ----
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

def load_all_unosat_damage_sites():
    """
    Load damage sites layers from all available UNOSAT GDBs.
    
    Returns:
        Dictionary with date keys and GeoDataFrames as values, containing damage data
    """
    # Find all UNOSAT GDB directories
    project_root = get_project_root()
    labels_dir = os.path.join(project_root, "data", "raw", "labels", "unosat")
    
    # Check if the directory exists
    if not os.path.exists(labels_dir):
        print(f"UNOSAT data directory not found: {labels_dir}")
        return {}
    
    # Find all date subdirectories
    date_gdbs = {}
    for date_dir in os.listdir(labels_dir):
        full_path = os.path.join(labels_dir, date_dir)
        if os.path.isdir(full_path):
            # Find GDB files in this directory
            for root, dirs, files in os.walk(full_path):
                for dir_name in dirs:
                    if dir_name.endswith('.gdb'):
                        gdb_path = os.path.join(root, dir_name)
                        date_gdbs[date_dir] = gdb_path
                        break
    
    if not date_gdbs:
        print("No UNOSAT GDB files found.")
        return {}
    
    print(f"Found {len(date_gdbs)} UNOSAT GDB datasets:")
    for date, gdb_path in sorted(date_gdbs.items()):
        print(f" - {date}: {gdb_path}")
    
    # Load damage sites from each GDB
    damage_data = {}
    for date, gdb_path in sorted(date_gdbs.items()):
        # List layers in the GDB
        layers = list_gdb_layers(gdb_path)
        
        # Find the damage sites layer
        damage_layer = None
        for layer in layers:
            if 'damage' in layer.lower() and ('sites' in layer.lower() or 'building' in layer.lower()):
                damage_layer = layer
                break
        
        if damage_layer:
            try:
                # Load the layer
                gdf = load_unosat_layer(gdb_path, damage_layer)
                if gdf is not None and len(gdf) > 0:
                    damage_data[date] = {
                        'gdf': gdf,
                        'layer_name': damage_layer
                    }
                    print(f"Loaded {len(gdf)} damage points from {date} ({damage_layer})")
            except Exception as e:
                print(f"Error loading damage layer from {date}: {str(e)}")
    
    return damage_data

def inspect_layer_properties(gdb_path, layer_name):
    """
    Inspect the properties of a specific layer in a GDB file.
    
    Args:
        gdb_path: Path to the GDB file
        layer_name: Name of the layer to inspect
    """
    print(f"Inspecting layer: {layer_name} in {gdb_path}")
    
    # Open the layer and load as GeoDataFrame
    gdf = gpd.read_file(gdb_path, layer=layer_name)
    
    # Basic information
    print(f"\nBasic Information:")
    print(f"Number of features: {len(gdf)}")
    print(f"Geometry type: {gdf.geom_type.iloc[0] if len(gdf) > 0 else 'Unknown'}")
    print(f"CRS: {gdf.crs}")
    
    # Column information
    print(f"\nColumns ({len(gdf.columns)} total):")
    for col in gdf.columns:
        # Get column dtype
        dtype = gdf[col].dtype
        # Count non-null values
        non_null = gdf[col].count()
        # Show some sample values (first 3)
        sample_values = gdf[col].dropna().head(3).tolist()
        
        print(f" - {col}: {dtype}, {non_null}/{len(gdf)} non-null values")
        print(f"   Sample values: {sample_values}")
    
    # Look for potential damage columns
    print(f"\nPotential damage columns:")
    damage_cols = [col for col in gdf.columns if 'damage' in col.lower()]
    for col in damage_cols:
        value_counts = gdf[col].value_counts()
        print(f" - {col}:")
        print(f"   Unique values: {value_counts.index.tolist()}")
        print(f"   Value counts: {value_counts.tolist()}")
    
    # Look for ID columns
    print(f"\nPotential ID columns:")
    id_cols = [col for col in gdf.columns if 'id' in col.lower()]
    for col in id_cols:
        # Check if values are unique
        unique_count = gdf[col].nunique()
        print(f" - {col}: {unique_count} unique values")
        
    # Look for date columns
    print(f"\nPotential date columns:")
    date_cols = [col for col in gdf.columns if any(term in col.lower() for term in ['date', 'time', 'sensor'])]
    for col in date_cols:
        # Show unique values
        unique_values = gdf[col].dropna().unique()
        print(f" - {col}: {len(unique_values)} unique values")
        if len(unique_values) > 0:
            print(f"   First few values: {unique_values[:3]}")
    
    return gdf

def convert_layer_to_long_format(gdb_path, layer_name):
    """
    Converts a GDB layer from wide format to long format, where each row represents
    a unique point observation at a specific epoch.
    
    Parameters:
    -----------
    gdb_path : str
        Path to the GDB file
    layer_name : str
        Name of the layer within the GDB
        
    Returns:
    --------
    pd.DataFrame
        Long format dataframe with each row representing a point at a specific epoch
    """
    # Read the GDB layer
    print(f"Reading layer: {layer_name} from {gdb_path}")
    gdf = gpd.read_file(gdb_path, layer=layer_name)
    
    # Generate a unique ID for each point since there's no explicit ID
    # Using the geometry as part of the ID since it's unique for each point
    gdf['point_id'] = [f"p{i}" for i in range(len(gdf))]
    
    # Keep track of the original geometries and location info
    location_cols = ['Governorate', 'Municipality', 'Neighborhood', 'geometry']
    locations = gdf[['point_id'] + location_cols].copy()
    
    # Create an empty list to store dataframes for each epoch
    epoch_dfs = []
    
    # Process each epoch (1-11)
    # Note: First epoch seems to have different column naming pattern
    # For epoch 1, the columns don't have a suffix
    epoch1_df = gdf[['point_id', 'SensorDate', 'SensorID', 'ConfidenceID', 'Main_Damage_Site_Class']].copy()
    # Only keep rows where SensorDate is not null
    epoch1_df = epoch1_df[~pd.isna(epoch1_df['SensorDate'])].copy()
    if not epoch1_df.empty:
        epoch1_df['epoch'] = 1
        epoch1_df.rename(columns={
            'SensorDate': 'date',
            'SensorID': 'sensor_id',
            'ConfidenceID': 'confidence_id',
            'Main_Damage_Site_Class': 'damage_class'
        }, inplace=True)
        epoch_dfs.append(epoch1_df)
    
    # For epochs 2-11, the columns have numeric suffixes
    for i in range(2, 12):
        date_col = f'SensorDate_{i}'
        sensor_id_col = f'SensorID_{i}'
        confidence_id_col = f'ConfidenceID_{i}'
        damage_class_col = f'Main_Damage_Site_Class_{i}'
        damage_status_col = f'Damage_Status_{i}'
        
        # Only select rows where the date is not null for this epoch
        epoch_df = gdf[['point_id', date_col, sensor_id_col, confidence_id_col, 
                         damage_class_col, damage_status_col]].copy()
        epoch_df = epoch_df[~pd.isna(epoch_df[date_col])].copy()
        
        if not epoch_df.empty:
            epoch_df['epoch'] = i
            epoch_df.rename(columns={
                date_col: 'date',
                sensor_id_col: 'sensor_id',
                confidence_id_col: 'confidence_id',
                damage_class_col: 'damage_class',
                damage_status_col: 'damage_status'
            }, inplace=True)
            epoch_dfs.append(epoch_df)
    
    # Combine all epoch dataframes
    if epoch_dfs:
        long_df = pd.concat(epoch_dfs, ignore_index=True)
        
        # Merge with the location data
        long_df = pd.merge(long_df, locations, on='point_id', how='left')
        
        # Sort by point_id and date
        long_df.sort_values(['point_id', 'date'], inplace=True)
        
        return long_df
    else:
        print("No valid epoch data found.")
        return None