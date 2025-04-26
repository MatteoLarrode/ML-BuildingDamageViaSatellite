from pathlib import Path
import json
import fiona
import geopandas as gpd
import os
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap, MarkerCluster
from datetime import datetime

# ---- Setup ----
def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.absolute()

# ---- Exploration funtions ----
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

def load_unosat_damage_sites():
    """
    Load damage sites layer from the UNOSAT GDB.
    
    Returns:
        GeoDataFrame containing damage data
    """
    # Path to the UNOSAT GDB
    project_root = get_project_root()
    gdb_path = os.path.join(project_root, "data", "raw", "labels", "UNOSAT_GazaStrip_CDA_25February2025.gdb")
    
    print(f"Loading UNOSAT data from: {gdb_path}")
    
    # Check if the file exists
    if not os.path.exists(gdb_path):
        print(f"UNOSAT data file not found: {gdb_path}")
        return None
    
    # List the layers in the GDB
    layers = list_gdb_layers(gdb_path)
    
    # Find the damage sites layer - it should be the first layer as per your example
    damage_layer = None
    for layer in layers:
        if 'Damage' in layer or 'damage' in layer:
            damage_layer = layer
            break
    
    if not damage_layer and len(layers) > 0:
        damage_layer = layers[0]  # Use the first layer if no "damage" layer found
        
    if damage_layer:
        try:
            # Load the layer
            print(f"Loading layer: {damage_layer}")
            gdf = load_unosat_layer(gdb_path, damage_layer)
            
            if gdf is not None and len(gdf) > 0:
                print(f"Successfully loaded {len(gdf)} damage points")
                
                # Print column information to help with debugging
                print("\nColumns in the damage data:")
                for col in gdf.columns:
                    print(f" - {col}")
                
                # Check for damage classification columns
                damage_cols = [col for col in gdf.columns if 'Damage' in col or 'damage' in col]
                print(f"\nFound {len(damage_cols)} damage-related columns")
                
                return gdf
            else:
                print("Failed to load damage data")
                return None
        except Exception as e:
            print(f"Error loading damage layer: {str(e)}")
            return None
    else:
        print(f"No damage layer found in the GDB. Available layers: {', '.join(layers)}")
        return None

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

# ----- Preprocessing functions -----
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap, MarkerCluster
from datetime import datetime

def process_unosat_damage_labels(gdf, date_index=7, aoi_path=None, output_dir="../data/processed/labels", 
                                positive_classes=[1, 2], include_moderate=False):
    """
    Process UNOSAT damage labels focusing on a specific date index.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        The raw UNOSAT damage labels GeoDataFrame
    date_index : int, default=7
        Which date index to use (e.g., 7 for May 3, 2024)
    aoi_path : str, optional
        Path to AOI GeoJSON file to filter labels spatially
    output_dir : str, default="../data/processed/labels"
        Directory to save processed labels
    positive_classes : list, default=[1, 2]
        Damage classes to be considered as positive (damaged) examples
        In UNOSAT: 1=Destroyed, 2=Severe Damage, 3=Moderate Damage, 
        4=Possible Damage, 6=No Visible Damage, 11=Possible Damage from adjacent impact
    include_moderate : bool, default=False
        Whether to include moderate damage in the dataset (will be labeled as 0 for binary classification)
        
    Returns:
    --------
    GeoDataFrame
        Processed and cleaned damage labels
    """
    print(f"Processing UNOSAT damage labels for date index {date_index}...")
    
    if gdf is None:
        print("Error: No damage data provided")
        return None
    
    # Create a copy to avoid modifying the original
    damage_labels = gdf.copy()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get column names for the specific date index
    date_col = f"SensorDate_{date_index}"
    damage_class_col = f"Main_Damage_Site_Class_{date_index}"
    damage_status_col = f"Damage_Status_{date_index}"
    
    # Check if the columns exist
    if damage_class_col not in damage_labels.columns:
        print(f"Error: Column '{damage_class_col}' not found in the data")
        print(f"Available columns: {damage_labels.columns.tolist()}")
        return None
    
    # Filter to rows that have data for this date index
    valid_mask = ~damage_labels[damage_class_col].isna()
    valid_labels = damage_labels[valid_mask].copy()
    
    print(f"Found {len(valid_labels)} valid damage labels for date index {date_index}")
    
    # Get the actual date from the data
    if date_col in valid_labels.columns:
        unique_dates = valid_labels[date_col].dropna().unique()
        date_str = pd.to_datetime(unique_dates[0]).strftime('%Y%m%d') if len(unique_dates) > 0 else f"date_{date_index}"
        print(f"Date: {unique_dates[0] if len(unique_dates) > 0 else 'Unknown'}")
    else:
        print(f"Warning: Date column '{date_col}' not found")
        date_str = f"date_{date_index}"
    
    # Create a simplified and cleaned version with available columns
    columns_to_keep = ['SiteID', 'geometry']
    
    # Add date column if it exists
    if date_col in valid_labels.columns:
        columns_to_keep.append(date_col)
    
    # Add damage class column
    columns_to_keep.append(damage_class_col)
    
    # Add damage status column if it exists
    if damage_status_col in valid_labels.columns:
        columns_to_keep.append(damage_status_col)
    
    # Add location columns if they exist
    for col in ['Governorate', 'Municipality', 'Neighborhood']:
        if col in valid_labels.columns:
            columns_to_keep.append(col)
    
    # Create cleaned dataset with available columns
    cleaned_labels = valid_labels[columns_to_keep].copy()
    
    # Rename columns for clarity
    column_mapping = {}
    if date_col in cleaned_labels.columns:
        column_mapping[date_col] = 'date'
    column_mapping[damage_class_col] = 'damage_class'
    if damage_status_col in cleaned_labels.columns:
        column_mapping[damage_status_col] = 'damage_status'
    
    cleaned_labels = cleaned_labels.rename(columns=column_mapping)
    
    # Convert damage class codes to descriptive categories
    # Based on UNOSAT classification from the provided image
    damage_class_map = {
        1: 'Destroyed',
        2: 'Severe Damage',
        3: 'Moderate Damage',
        4: 'Possible Damage',
        6: 'No Visible Damage',
        11: 'Possible Damage from adjacent impact/debris'
    }
    
    # Apply mapping
    cleaned_labels['damage_class_desc'] = cleaned_labels['damage_class'].map(damage_class_map)
    
    # Filter to keep only desired classes
    if not include_moderate:
        # Only keep destroyed, severe damage, and no damage
        class_filter = cleaned_labels['damage_class'].isin(positive_classes + [6])  # 6 = No Visible Damage
        filtered_labels = cleaned_labels[class_filter].copy()
        print(f"Filtered from {len(cleaned_labels)} to {len(filtered_labels)} points " +
              f"(keeping only Destroyed, Severe Damage, and No Visible Damage)")
        cleaned_labels = filtered_labels
    else:
        # Keep all classes but still mark the binary label properly
        print("Keeping all damage classes in the dataset")
    
    # Create binary damage indicator (1 = damaged, 0 = not damaged)
    # Only Destroyed (1) and Severe Damage (2) are considered as damaged
    cleaned_labels['is_damaged'] = cleaned_labels['damage_class'].apply(
        lambda x: 1 if x in positive_classes else 0
    )
    
    # Create categorical damage level (0 = none, 1 = moderate, 2 = severe, 3 = destroyed)
    damage_level_map = {
        1: 3,  # Destroyed -> 3
        2: 2,  # Severe Damage -> 2
        3: 1,  # Moderate Damage -> 1
        4: 0,  # Possible Damage -> 0
        6: 0,  # No Visible Damage -> 0
        11: 0  # Possible Damage from adjacent impact -> 0
    }
    
    cleaned_labels['damage_level'] = cleaned_labels['damage_class'].map(damage_level_map)
    
    # Extract coordinates for easier access
    cleaned_labels['lon'] = cleaned_labels.geometry.x
    cleaned_labels['lat'] = cleaned_labels.geometry.y
    
    # Filter by AOI if provided
    if aoi_path and os.path.exists(aoi_path):
        print(f"Filtering by AOI: {aoi_path}")
        aoi = gpd.read_file(aoi_path)
        
        # Make sure CRS matches
        if aoi.crs != cleaned_labels.crs:
            aoi = aoi.to_crs(cleaned_labels.crs)
            
        # Spatial join to filter points in the AOI
        before_count = len(cleaned_labels)
        cleaned_labels = gpd.sjoin(cleaned_labels, aoi, predicate='within', how='inner')
        after_count = len(cleaned_labels)
        print(f"Filtered from {before_count} to {after_count} points within the AOI")
    
    # Save to file
    output_file = os.path.join(output_dir, f"damage_labels_{date_str}.gpkg")
    cleaned_labels.to_file(output_file, driver="GPKG")
    print(f"Saved processed labels to {output_file}")
    
    # Create a CSV version for easier use with other tools
    csv_file = os.path.join(output_dir, f"damage_labels_{date_str}.csv")
    # Drop geometry column for CSV
    cleaned_labels_csv = cleaned_labels.drop(columns=['geometry']).copy()
    cleaned_labels_csv.to_csv(csv_file, index=False)
    print(f"Saved CSV version to {csv_file}")
    
    # Print summary statistics
    print("\nDamage class distribution:")
    class_counts = cleaned_labels['damage_class_desc'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count/len(cleaned_labels)*100:.1f}%)")
    
    damaged_count = cleaned_labels['is_damaged'].sum()
    print(f"\nTotal damaged structures: {damaged_count} ({damaged_count/len(cleaned_labels)*100:.1f}%)")
    print(f"Total undamaged structures: {len(cleaned_labels) - damaged_count} ({(len(cleaned_labels) - damaged_count)/len(cleaned_labels)*100:.1f}%)")
    
    return cleaned_labels

def create_control_points(damage_labels, aoi_path, min_distance=100, num_points=None):
    """
    Generate control points away from damaged areas to serve as negative examples
    
    Parameters:
    -----------
    damage_labels : GeoDataFrame
        GeoDataFrame with damage labels
    aoi_path : str
        Path to AOI GeoJSON file
    min_distance : float, default=100
        Minimum distance (meters) from any damage point
    num_points : int, optional
        Number of control points to generate (default: same as damage points)
        
    Returns:
    --------
    GeoDataFrame
        Control points with the same structure as damage_labels
    """
    import geopandas as gpd
    from shapely.geometry import Point
    import random
    
    # Set number of points to match damage points if not specified
    if num_points is None:
        num_points = len(damage_labels)
    
    print(f"Generating {num_points} control points at least {min_distance}m from damage...")
    
    # Load AOI
    aoi = gpd.read_file(aoi_path)
    
    # Make sure CRS matches
    if aoi.crs != damage_labels.crs:
        aoi = aoi.to_crs(damage_labels.crs)
    
    # Get AOI boundary as a single geometry
    aoi_boundary = aoi.unary_union
    
    # Create a buffer around all damage points
    print("Creating damage buffer zone...")
    # Convert to UTM for accurate distance measurement
    if damage_labels.crs.is_geographic:
        damage_utm = damage_labels.to_crs("EPSG:32636")  # UTM zone 36N for Gaza
        damage_buffer = gpd.GeoDataFrame(geometry=damage_utm.geometry.buffer(min_distance))
    else:
        damage_buffer = gpd.GeoDataFrame(geometry=damage_labels.geometry.buffer(min_distance))
    
    damage_buffer = damage_buffer.unary_union
    
    # Convert back to original CRS if needed
    if damage_labels.crs.is_geographic:
        # Convert the unary_union back to a GeoDataFrame with the original CRS
        damage_buffer_gdf = gpd.GeoDataFrame(geometry=[damage_buffer], crs="EPSG:32636")
        damage_buffer = damage_buffer_gdf.to_crs(damage_labels.crs).geometry[0]
    
    # Generate random points within the AOI but outside damage zones
    control_points = []
    num_attempts = 0
    
    print("Generating random points...")
    while len(control_points) < num_points and num_attempts < 50000:
        num_attempts += 1
        
        if num_attempts % 5000 == 0:
            print(f"  Made {num_attempts} attempts, found {len(control_points)} valid points...")
        
        # Get AOI bounds
        minx, miny, maxx, maxy = aoi_boundary.bounds
        
        # Generate random point within bounds
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        point = Point(x, y)
        
        # Check if point is within AOI boundary but outside damage buffer
        if point.within(aoi_boundary) and not point.within(damage_buffer):
            control_points.append(point)
    
    print(f"Found {len(control_points)} valid control points after {num_attempts} attempts")
    
    # Create GeoDataFrame of control points
    control_gdf = gpd.GeoDataFrame(
        {'geometry': control_points},
        crs=damage_labels.crs
    )
    
    # Add columns for consistency
    control_gdf['is_damaged'] = 0
    control_gdf['damage_class'] = 6  # "No Visible Damage"
    control_gdf['damage_class_desc'] = 'No Visible Damage'
    control_gdf['lon'] = control_gdf.geometry.x
    control_gdf['lat'] = control_gdf.geometry.y
    
    return control_gdf

def visualize_damage_labels(damage_labels, output_dir="../data/processed/labels"):
    """
    Create visualizations of the damage labels.
    
    Parameters:
    -----------
    damage_labels : GeoDataFrame
        Processed damage labels
    output_dir : str, default="../data/processed/labels"
        Directory to save visualizations
        
    Returns:
    --------
    None
    """
    print("Creating visualizations of damage labels...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get date for naming
    date_str = pd.to_datetime(damage_labels['date'].iloc[0]).strftime('%Y%m%d') \
               if not pd.isna(damage_labels['date'].iloc[0]) else "unknown_date"
    
    # 1. Static plot of damage distribution
    plt.figure(figsize=(10, 6))
    damage_counts = damage_labels['damage_class_desc'].value_counts()
    colors = ['green', 'yellow', 'orange', 'red', 'grey', 'blue']
    damage_counts.plot(kind='bar', color=colors[:len(damage_counts)])
    plt.title(f'Distribution of Damage Classes ({date_str})')
    plt.xlabel('Damage Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"damage_distribution_{date_str}.png"), dpi=300)
    plt.close()
    
    # 2. Damage by governorate
    plt.figure(figsize=(12, 8))
    gov_damage = damage_labels.groupby(['Governorate', 'damage_class_desc']).size().unstack().fillna(0)
    gov_damage.plot(kind='bar', stacked=True, colormap='RdYlGn_r')
    plt.title(f'Damage Distribution by Governorate ({date_str})')
    plt.xlabel('Governorate')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Damage Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"damage_by_governorate_{date_str}.png"), dpi=300)
    plt.close()
    
    # 3. Create an interactive map
    # Get center coordinates
    center_lat = damage_labels['lat'].mean()
    center_lon = damage_labels['lon'].mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, 
                  tiles='CartoDB positron')
    
    # Add clustered markers
    marker_cluster = MarkerCluster().add_to(m)
    
    # Define color map for damage classes
    color_map = {
        'No Damage': 'green',
        'Minor/Moderate Damage': 'yellow',
        'Severe Damage': 'orange',
        'Destroyed': 'red',
        'Possible Damage': 'gray',
        'Under Construction': 'blue'
    }
    
    # Sample points if there are too many
    sample_size = min(10000, len(damage_labels))
    if len(damage_labels) > sample_size:
        sample_labels = damage_labels.sample(sample_size)
        print(f"Sampling {sample_size} points for the interactive map")
    else:
        sample_labels = damage_labels
    
    # Add markers
    for idx, row in sample_labels.iterrows():
        color = color_map.get(row['damage_class_desc'], 'gray')
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"ID: {row['SiteID']}<br>Class: {row['damage_class_desc']}<br>Location: {row['Neighborhood']}, {row['Municipality']}"
        ).add_to(marker_cluster)
    
    # Add heatmap layer
    heat_data = [[row['lat'], row['lon'], row['damage_level'] if not np.isnan(row['damage_level']) else 0] 
                for idx, row in sample_labels.iterrows() if 'damage_level' in row]
    
    HeatMap(heat_data, radius=15, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}, 
           min_opacity=0.5, blur=10).add_to(m)
    
    # Save map
    m.save(os.path.join(output_dir, f"damage_map_{date_str}.html"))
    print(f"Saved interactive map to {os.path.join(output_dir, f'damage_map_{date_str}.html')}")
    
    return

def create_damage_buffer_gdf(damage_labels, buffer_distance=10, only_damaged=True):
    """
    Create a GeoDataFrame with buffer areas around damaged structures.
    Useful for spatial joins with backscatter data.
    
    Parameters:
    -----------
    damage_labels : GeoDataFrame
        Processed damage labels
    buffer_distance : float, default=10
        Buffer distance in meters
    only_damaged : bool, default=True
        If True, only create buffers around damaged structures
        
    Returns:
    --------
    GeoDataFrame
        Buffered geometries around damage points
    """
    # Create a copy to avoid modifying the original
    gdf = damage_labels.copy()
    
    # Filter to only damaged structures if requested
    if only_damaged:
        gdf = gdf[gdf['is_damaged'] == 1].copy()
        print(f"Creating buffers around {len(gdf)} damaged structures")
    else:
        print(f"Creating buffers around {len(gdf)} structures")
    
    # Make sure the CRS is in meters for accurate buffering
    if gdf.crs.is_geographic:
        # Convert to a projected CRS (UTM zone appropriate for Gaza)
        gdf = gdf.to_crs("EPSG:32636")  # UTM zone 36N
        
    # Create buffer geometries
    gdf['geometry'] = gdf.geometry.buffer(buffer_distance)
    
    # Convert back to original CRS if it was geographic
    if damage_labels.crs.is_geographic:
        gdf = gdf.to_crs(damage_labels.crs)
    
    return gdf

def create_damage_raster(damage_labels, resolution=10, output_path=None, aoi_path=None):
    """
    Create a raster representation of damage levels compatible with Sentinel-1 data.
    
    Parameters:
    -----------
    damage_labels : GeoDataFrame
        Processed damage labels
    resolution : float, default=10
        Resolution in meters (should match Sentinel-1 resolution)
    output_path : str, optional
        Path to save the output raster file
    aoi_path : str, optional
        Path to AOI GeoJSON to define the raster extent
        
    Returns:
    --------
    tuple
        (raster_data, transform, crs)
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    
    # Determine the bounds for the raster
    if aoi_path and os.path.exists(aoi_path):
        aoi = gpd.read_file(aoi_path)
        if aoi.crs != damage_labels.crs:
            aoi = aoi.to_crs(damage_labels.crs)
        bounds = aoi.total_bounds
    else:
        bounds = damage_labels.total_bounds
    
    # Calculate raster dimensions
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    
    # Create transform
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    # Convert damage points to shapes with values
    shapes = []
    buffer_distance = resolution / 2  # Buffer half the resolution
    
    # Convert to UTM for accurate buffering
    if damage_labels.crs.is_geographic:
        gdf_utm = damage_labels.to_crs("EPSG:32636")  # UTM zone 36N
    else:
        gdf_utm = damage_labels.copy()
    
    # Create buffered shapes
    for idx, row in gdf_utm.iterrows():
        if 'damage_level' in row and not pd.isna(row['damage_level']):
            # Buffer the point to create a small circle
            buffered = row.geometry.buffer(buffer_distance)
            
            # Convert back to original CRS
            if damage_labels.crs.is_geographic:
                buffered = gpd.GeoSeries([buffered], crs="EPSG:32636").to_crs(damage_labels.crs)[0]
                
            # Add to shapes list with its damage level
            shapes.append((buffered, row['damage_level']))
    
    # Rasterize the shapes
    raster_data = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        default_value=0,
        dtype=np.float32
    )
    
    # Save to file if output path provided
    if output_path:
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=rasterio.float32,
            crs=damage_labels.crs,
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(raster_data, 1)
        print(f"Saved damage raster to {output_path}")
    
    return raster_data, transform, damage_labels.crs


def integrate_unosat_with_backscatter(damage_labels, backscatter_gdf, buffer_distance=10, aoi_path=None):
    """
    Integrate UNOSAT damage labels with the backscatter data.
    
    Parameters:
    -----------
    damage_labels : GeoDataFrame
        Processed damage labels
    backscatter_gdf : GeoDataFrame
        Extracted backscatter values as a GeoDataFrame
    buffer_distance : float, default=10
        Buffer distance in meters to account for positional uncertainty
    aoi_path : str, optional
        Path to AOI GeoJSON to filter the data
        
    Returns:
    --------
    GeoDataFrame
        Backscatter data with damage labels
    """
    print("Integrating UNOSAT damage labels with backscatter data...")
    
    # Create buffers around damage points
    damage_buffers = create_damage_buffer_gdf(damage_labels, buffer_distance=buffer_distance)
    
    # Make sure CRS is the same for both datasets
    if backscatter_gdf.crs != damage_buffers.crs:
        backscatter_gdf = backscatter_gdf.to_crs(damage_buffers.crs)
    
    # Perform spatial join
    print(f"Performing spatial join with {len(damage_buffers)} damage buffers...")
    joined_data = gpd.sjoin(backscatter_gdf, damage_buffers, how='left', predicate='within')
    
    # Fill missing values for damage columns
    damage_cols = ['damage_class', 'damage_class_desc', 'is_damaged', 'damage_level']
    for col in damage_cols:
        if col in joined_data.columns:
            joined_data[col] = joined_data[col].fillna(0 if col != 'damage_class_desc' else 'No Damage')
    
    # Count how many points got damage labels
    damage_count = joined_data['is_damaged'].sum() if 'is_damaged' in joined_data.columns else 0
    print(f"Added damage labels to {joined_data['is_damaged'].notnull().sum()} backscatter points")
    print(f"Found {damage_count} damaged pixels ({damage_count/len(joined_data)*100:.2f}% of total)")
    
    return joined_data