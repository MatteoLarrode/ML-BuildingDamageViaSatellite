import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import fiona

# ---- Setup ----
def get_project_root():
    """Get the project root directory."""
    # This assumes the script is in src/data/
    return Path(__file__).parent.parent.parent

# --- Building Loading Functions ----
# From PWTT: https://github.com/oballinger/PWTT/tree/main?tab=readme-ov-file
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

def gdb_to_geojson(gdb_path, layer_name, output_path):
    """
    Transforms a UNOSAT GDB layer into a GeoJSON format similar to the https://github.com/prs-eth/ukraine-damage-mapping-tool/blob/main/data/unosat_labels.geojson.
    
    Parameters:
    -----------
    gdb_path : str
        Path to the GDB file
    layer_name : str
        Name of the layer in the GDB
    output_path : str
        Path to save the output GeoJSON file
    """
    # Read the GDB layer
    gdf = gpd.read_file(gdb_path, layer=layer_name)
    
    # Initialize empty list to store features
    features = []
    
    # Loop through each point in the GDB
    for idx, row in gdf.iterrows():
        # Get the base geometry
        geom = row['geometry']
        
        # Prepare the basic properties
        base_properties = {
            "unosat_id": row['SiteID'],
            "city": row['Municipality'],
            "country": "Palestine",
            "gdb": row['EventCode'],
            "layer": f"Gaza_CDA_{row['SensorDate_11'].strftime('%Y%m%d')}" if pd.notna(row['SensorDate_11']) else None,
        }
        
        # Get additional geographic properties
        if pd.notna(row['Governorate']):
            base_properties["governorate"] = row['Governorate']
        if pd.notna(row['Municipality']):
            base_properties["municipality"] = row['Municipality']
        if pd.notna(row['Neighborhood']):
            base_properties["neighborhood"] = row['Neighborhood']
        
        # Process each epoch (time period) where data exists
        for ep in range(1, 12):  # 11 epochs in the dataset
            sensor_date_col = f'SensorDate_{ep}' if ep > 1 else 'SensorDate'
            damage_class_col = f'Main_Damage_Site_Class_{ep}' if ep > 1 else 'Main_Damage_Site_Class'
            
            # Skip if no data for this epoch
            if pd.isna(row.get(sensor_date_col)):
                continue
                
            # Create properties for this epoch
            props = base_properties.copy()
            props["ep"] = ep
            
            # Add date
            date_val = row.get(sensor_date_col)
            if pd.notna(date_val):
                props["date"] = date_val.isoformat()
            
            # Add damage value
            damage_val = row.get(damage_class_col)
            if pd.notna(damage_val):
                props["damage"] = int(damage_val) if isinstance(damage_val, (int, float)) else damage_val
            
            # Add previous damage and date if not the first epoch
            if ep > 1:
                prev_ep = ep - 1
                prev_date_col = f'SensorDate_{prev_ep}' if prev_ep > 1 else 'SensorDate'
                prev_damage_col = f'Main_Damage_Site_Class_{prev_ep}' if prev_ep > 1 else 'Main_Damage_Site_Class'
                
                prev_date = row.get(prev_date_col)
                prev_damage = row.get(prev_damage_col)
                
                props["prev_date"] = prev_date.isoformat() if pd.notna(prev_date) else None
                props["prev_damage"] = int(prev_damage) if pd.notna(prev_damage) and isinstance(prev_damage, (int, float)) else -99
            else:
                props["prev_date"] = None
                props["prev_damage"] = -99
            
            # Create the GeoJSON feature
            feature = {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Point",
                    "coordinates": [geom.x, geom.y]  # longitude, latitude
                }
            }
            
            features.append(feature)
    
    # Create the GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "name": "unosat_labels",
        "crs": {
            "type": "name", 
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f)
    
    print(f"Successfully converted to GeoJSON with {len(features)} features")
    return geojson