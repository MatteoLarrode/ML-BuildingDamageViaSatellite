import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, mapping
from datetime import datetime
import re
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

def extract_backscatter_from_tiff(tiff_path, polarization=None, aoi_path=None):
    """
    Extract backscatter values from a single Sentinel-1 GeoTIFF file.
    
    Parameters:
    -----------
    tiff_path : str
        Path to the GeoTIFF file
    polarization : str, optional
        Polarization ('VV' or 'VH'), detected from filename if not provided
    aoi_path : str, optional
        Path to AOI GeoJSON file to mask the data (if None, all pixels are used)
    
    Returns:
    --------
    gdf : GeoDataFrame
        GeoDataFrame with backscatter values and coordinates
    """
    # Extract metadata from filename
    filename = os.path.basename(tiff_path)
    
    # Determine polarization from filename if not provided
    if polarization is None:
        if 'vv' in filename.lower():
            polarization = 'VV'
        elif 'vh' in filename.lower():
            polarization = 'VH'
        else:
            raise ValueError(f"Could not determine polarization from filename: {filename}")
    
    # Extract date from filename or directory
    date_match = re.search(r'(\d{8})(?:_|T)', filename)
    if date_match:
        date_str = date_match.group(1)
    else:
        # Try to get date from directory name
        parent_dir = os.path.basename(os.path.dirname(tiff_path))
        if re.match(r'^\d{8}$', parent_dir):
            date_str = parent_dir
        else:
            raise ValueError(f"Could not extract date from filename or directory: {tiff_path}")
    
    # Determine orbit direction based on acquisition time
    if 'T15' in filename:
        orbit = 'ASC'  # Ascending (afternoon)
    elif 'T03' in filename:
        orbit = 'DESC'  # Descending (morning)
    else:
        orbit = 'UNK'  # Unknown
    
    # Load AOI if provided
    aoi = None
    if aoi_path and os.path.exists(aoi_path):
        aoi = gpd.read_file(aoi_path)
    
    # Open the tiff file
    with rasterio.open(tiff_path) as src:
        # Get the image transform
        transform = src.transform
        
        # Mask with AOI if provided
        if aoi is not None:
            masked_data, masked_transform = mask(src, aoi.geometry, crop=True, all_touched=True, nodata=np.nan)
            data = masked_data[0]  # First (and only) band
            transform = masked_transform
        else:
            data = src.read(1)  # Read the first (and only) band
        
        # Get coordinates for all valid pixels
        height, width = data.shape
        rows, cols = np.where(~np.isnan(data))
        
        # Create a list to store data
        pixel_data = []
        
        # Process each valid pixel
        for row, col in zip(rows, cols):
            # Get the value
            value = data[row, col]
            
            # Convert pixel coordinates to geographic coordinates
            x, y = rasterio.transform.xy(transform, row, col, offset='center')
            
            # Add to pixel data
            pixel_data.append({
                'row': row,
                'col': col,
                'x': x,
                'y': y,
                'lon': x,
                'lat': y, 
                'backscatter': value,
                'geometry': Point(x, y),
                'date': date_str,
                'polarization': polarization,
                'orbit': orbit,
                'file': filename
            })
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(pixel_data, geometry='geometry')
    
    # Set CRS from source file
    with rasterio.open(tiff_path) as src:
        gdf.crs = src.crs
    
    return gdf

def extract_ml_features(points_gdf, backscatter_gdf, buffer_distance=10):
    """
    Extract machine learning features from backscatter data for given points
    
    Parameters:
    -----------
    points_gdf : GeoDataFrame
        Points to extract features for (both damaged and control)
    backscatter_gdf : GeoDataFrame
        Backscatter values
    buffer_distance : float, default=10
        Buffer distance (meters) around points
        
    Returns:
    --------
    DataFrame
        Features for machine learning
    """
    # Create buffers around points
    if points_gdf.crs.is_geographic:
        points_utm = points_gdf.to_crs("EPSG:32636")  # UTM zone 36N
        buffered = points_utm.copy()
        buffered['geometry'] = points_utm.geometry.buffer(buffer_distance)
        buffered = buffered.to_crs(points_gdf.crs)
    else:
        buffered = points_gdf.copy()
        buffered['geometry'] = buffered.geometry.buffer(buffer_distance)
    
    # Extract features for each point
    all_features = []
    
    for idx, row in buffered.iterrows():
        point_buffer = row.geometry
        
        # Find backscatter points within this buffer
        matching_points = backscatter_gdf[backscatter_gdf.geometry.within(point_buffer)]
        
        if len(matching_points) == 0:
            continue
        
        # Split into reference and post periods
        ref_data = matching_points[matching_points['period'] == 'reference']
        post_data = matching_points[matching_points['period'] == 'post']
        
        if len(ref_data) == 0 or len(post_data) == 0:
            continue
        
        # Process each polarization separately
        for pol in ['VV', 'VH']:
            ref_pol = ref_data[ref_data['polarization'] == pol]
            post_pol = post_data[post_data['polarization'] == pol]
            
            if len(ref_pol) == 0 or len(post_pol) == 0:
                continue
            
            # Extract features
            features = {
                'point_id': idx,
                'lon': row['lon'],
                'lat': row['lat'],
                'is_damaged': row['is_damaged'],
                'polarization': pol,
                
                # Reference period statistics
                'ref_mean': ref_pol['backscatter'].mean(),
                'ref_std': ref_pol['backscatter'].std() if len(ref_pol) > 1 else 0,
                'ref_min': ref_pol['backscatter'].min(),
                'ref_max': ref_pol['backscatter'].max(),
                'ref_range': ref_pol['backscatter'].max() - ref_pol['backscatter'].min(),
                'ref_count': len(ref_pol),
                
                # Post period statistics
                'post_mean': post_pol['backscatter'].mean(),
                'post_std': post_pol['backscatter'].std() if len(post_pol) > 1 else 0,
                'post_min': post_pol['backscatter'].min(),
                'post_max': post_pol['backscatter'].max(),
                'post_range': post_pol['backscatter'].max() - post_pol['backscatter'].min(),
                'post_count': len(post_pol),
                
                # Change metrics
                'change_mean': post_pol['backscatter'].mean() - ref_pol['backscatter'].mean(),
                'change_std': post_pol['backscatter'].std() - ref_pol['backscatter'].std() if len(post_pol) > 1 and len(ref_pol) > 1 else 0,
                'change_magnitude': abs(post_pol['backscatter'].mean() - ref_pol['backscatter'].mean()),
                
                # Statistical change metrics (similar to PWTT)
                't_statistic': (post_pol['backscatter'].mean() - ref_pol['backscatter'].mean()) / 
                              (((ref_pol['backscatter'].std()**2 / len(ref_pol)) + 
                                (post_pol['backscatter'].std()**2 / len(post_pol)))**0.5) 
                              if len(ref_pol) > 1 and len(post_pol) > 1 else 0
            }
            
            # Add percent change (handling division by zero)
            if ref_pol['backscatter'].mean() != 0:
                features['percent_change'] = ((post_pol['backscatter'].mean() - ref_pol['backscatter'].mean()) / 
                                             abs(ref_pol['backscatter'].mean())) * 100
            else:
                features['percent_change'] = 0
                
            # Add information about orbit if available
            if 'orbit' in ref_pol.columns:
                features['orbit'] = ref_pol['orbit'].iloc[0]
                
            # Add damage class information if available
            if 'damage_class' in row and row['is_damaged'] == 1:
                features['damage_class'] = row['damage_class'] 
                features['damage_class_desc'] = row['damage_class_desc']
                
            all_features.append(features)
    
    # Create DataFrame
    features_df = pd.DataFrame(all_features)
    
    return features_df

def collect_all_backscatter_values_optimized(data_dir, aoi_path=None, polarizations=None, sample_rate=1.0):
    """
    Optimized function to collect backscatter values from all GeoTIFFs.
    This version focuses on performance with large datasets.
    
    Parameters:
    -----------
    data_dir : str
        Base directory containing preprocessed Sentinel-1 data
    aoi_path : str, optional
        Path to AOI GeoJSON file
    polarizations : list, optional
        List of polarizations to include (default: ['VV', 'VH'])
    sample_rate : float, optional
        Fraction of pixels to sample (1.0 = all pixels, 0.1 = 10% of pixels)
    
    Returns:
    --------
    gdf : GeoDataFrame
        Combined GeoDataFrame with backscatter values from all files
    """
    if polarizations is None:
        polarizations = ['VV', 'VH']
    
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask
    from shapely.geometry import Point
    import os
    import glob
    import re
    from datetime import datetime
    from tqdm.notebook import tqdm
    
    # Load AOI if provided
    aoi = None
    if aoi_path and os.path.exists(aoi_path):
        aoi = gpd.read_file(aoi_path)
    
    # Find all date subdirectories
    date_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and re.match(r'^\d{8}$', d)]
    date_dirs.sort()  # Sort chronologically
    
    # Separate into reference (pre-conflict) and post-conflict
    reference_dates = [d for d in date_dirs if d < '20231007']  # Dates before conflict (Oct 7, 2023)
    post_dates = [d for d in date_dirs if d >= '20231007']  # Dates on or after conflict
    
    print(f"Found {len(reference_dates)} reference dates and {len(post_dates)} post-conflict dates")
    
    # Create spatial sampling mask if needed
    sampled_pixels_mask = None
    if sample_rate < 1.0:
        print("Creating a spatial sample...")
        
        # Find a representative file
        sample_file = None
        for date_dir in date_dirs:
            tiff_files = glob.glob(os.path.join(data_dir, date_dir, f"*_{polarizations[0].lower()}_db.tif"))
            if tiff_files:
                sample_file = tiff_files[0]
                break
        
        if not sample_file:
            raise ValueError("No suitable files found for sampling")
        
        # Create sampling mask directly from raster data
        with rasterio.open(sample_file) as src:
            # Apply AOI mask if provided
            if aoi is not None:
                data, transform = mask(src, aoi.geometry, crop=True, all_touched=True, nodata=np.nan)
                data = data[0]  # First band
            else:
                data = src.read(1)
                transform = src.transform
            
            # Generate random sample mask (True where we keep pixels)
            height, width = data.shape
            random_mask = np.random.random(size=(height, width)) < sample_rate
            
            # Only sample where we have valid data
            valid_data = ~np.isnan(data)
            sampled_pixels_mask = np.logical_and(random_mask, valid_data)
            
            print(f"Created sample mask with {np.sum(sampled_pixels_mask)} pixels " +
                  f"({np.sum(sampled_pixels_mask) / np.sum(valid_data) * 100:.1f}% of valid pixels)")
    
    # Function to process a single file with optimizations
    def process_file_optimized(tiff_path, period, polarization=None):
        # Extract metadata from filename
        filename = os.path.basename(tiff_path)
        
        # Determine polarization from filename if not provided
        if polarization is None:
            if 'vv' in filename.lower():
                polarization = 'VV'
            elif 'vh' in filename.lower():
                polarization = 'VH'
            else:
                raise ValueError(f"Could not determine polarization from filename: {filename}")
        
        # Extract date from filename or directory
        date_match = re.search(r'(\d{8})(?:_|T)', filename)
        if date_match:
            date_str = date_match.group(1)
        else:
            # Try to get date from directory name
            parent_dir = os.path.basename(os.path.dirname(tiff_path))
            if re.match(r'^\d{8}$', parent_dir):
                date_str = parent_dir
            else:
                raise ValueError(f"Could not extract date from filename or directory: {tiff_path}")
        
        # Determine orbit direction based on acquisition time
        if 'T15' in filename:
            orbit = 'ASC'  # Ascending (afternoon)
        elif 'T03' in filename:
            orbit = 'DESC'  # Descending (morning)
        else:
            orbit = 'UNK'  # Unknown
        
        # Open the tiff file
        with rasterio.open(tiff_path) as src:
            # Apply AOI and sampling masks
            if aoi is not None:
                data, transform = mask(src, aoi.geometry, crop=True, all_touched=True, nodata=np.nan)
                data = data[0]  # First band
            else:
                data = src.read(1)
                transform = src.transform
            
            # Apply sampling mask if provided
            if sampled_pixels_mask is not None:
                # Ensure masks have same dimensions, resize if needed
                if sampled_pixels_mask.shape != data.shape:
                    print(f"Warning: Sample mask shape {sampled_pixels_mask.shape} doesn't match data shape {data.shape}")
                    # If dimensions don't match, we need a different approach
                    # For simplicity, just sample randomly from valid pixels
                    valid_data = ~np.isnan(data)
                    random_mask = np.random.random(size=data.shape) < sample_rate
                    mask_to_apply = np.logical_and(random_mask, valid_data)
                else:
                    mask_to_apply = sampled_pixels_mask
                
                # Apply the mask
                rows, cols = np.where(np.logical_and(mask_to_apply, ~np.isnan(data)))
            else:
                # Just get valid pixels
                rows, cols = np.where(~np.isnan(data))
            
            # Create arrays for efficiency
            n_pixels = len(rows)
            backscatter_values = data[rows, cols]
            
            # Vectorized coordinate transformation
            xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
            
            # Create pixel IDs
            grid_resolution = 0.0001
            x_grid = (np.array(xs) / grid_resolution).round().astype(int).astype(str)
            y_grid = (np.array(ys) / grid_resolution).round().astype(int).astype(str)
            pixel_ids = np.array([f"{x}_{y}" for x, y in zip(x_grid, y_grid)])
            
            # Create data frame efficiently
            df = pd.DataFrame({
                'row': rows,
                'col': cols,
                'lon': xs,
                'lat': ys,
                'backscatter': backscatter_values,
                'date': date_str,
                'polarization': polarization,
                'orbit': orbit,
                'period': period,
                'pixel_id': pixel_ids
            })
            
            # Add geometries
            geometries = [Point(x, y) for x, y in zip(xs, ys)]
            
            # Return GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=src.crs)
            return gdf
    
    # Process all files and collect data
    all_data = []
    
    # Process reference dates
    print("Processing reference dates...")
    for date_dir in tqdm(reference_dates):
        for pol in polarizations:
            tiff_files = glob.glob(os.path.join(data_dir, date_dir, f"*_{pol.lower()}_db.tif"))
            for tiff_file in tiff_files:
                try:
                    gdf = process_file_optimized(tiff_file, 'reference', pol)
                    all_data.append(gdf)
                except Exception as e:
                    print(f"Error processing {tiff_file}: {str(e)}")
    
    # Process post-conflict dates
    print("Processing post-conflict dates...")
    for date_dir in tqdm(post_dates):
        for pol in polarizations:
            tiff_files = glob.glob(os.path.join(data_dir, date_dir, f"*_{pol.lower()}_db.tif"))
            for tiff_file in tiff_files:
                try:
                    gdf = process_file_optimized(tiff_file, 'post', pol)
                    all_data.append(gdf)
                except Exception as e:
                    print(f"Error processing {tiff_file}: {str(e)}")
    
    # Combine all data
    print("Combining all data...")
    all_backscatter = pd.concat(all_data, ignore_index=True)
    
    print(f"Total records: {len(all_backscatter)}")
    print(f"Unique pixels: {all_backscatter['pixel_id'].nunique()}")
    
    return all_backscatter

def extract_time_series_features(backscatter_gdf, pixel_id_col='pixel_id'):
    """
    Extract time series features for each unique pixel.
    
    Parameters:
    -----------
    backscatter_gdf : GeoDataFrame
        Combined GeoDataFrame with backscatter values
    pixel_id_col : str, optional
        Name of column containing pixel identifiers
    
    Returns:
    --------
    feature_gdf : GeoDataFrame
        GeoDataFrame with time series features for each unique pixel
    """
    # Get unique pixel IDs
    pixel_ids = backscatter_gdf[pixel_id_col].unique()
    
    # Prepare a list to store feature rows
    feature_data = []
    
    print(f"Extracting features for {len(pixel_ids)} unique pixels...")
    
    # Process each unique pixel
    for pixel_id in tqdm(pixel_ids):
        # Get data for this pixel
        pixel_data = backscatter_gdf[backscatter_gdf[pixel_id_col] == pixel_id]
        
        # Skip if pixel doesn't have both reference and post data
        if not (pixel_data['period'] == 'reference').any() or not (pixel_data['period'] == 'post').any():
            continue
        
        # Get reference period data for VV and VH
        ref_vv = pixel_data[(pixel_data['period'] == 'reference') & (pixel_data['polarization'] == 'VV')]
        ref_vh = pixel_data[(pixel_data['period'] == 'reference') & (pixel_data['polarization'] == 'VH')]
        
        # Get post period data for VV and VH
        post_vv = pixel_data[(pixel_data['period'] == 'post') & (pixel_data['polarization'] == 'VV')]
        post_vh = pixel_data[(pixel_data['period'] == 'post') & (pixel_data['polarization'] == 'VH')]
        
        # Skip if any polarization is missing data
        if len(ref_vv) == 0 or len(ref_vh) == 0 or len(post_vv) == 0 or len(post_vh) == 0:
            continue
        
        # Get representative location (use the mean coordinates)
        lon = pixel_data['lon'].mean()
        lat = pixel_data['lat'].mean()
        
        # Extract time series features
        features = {
            'pixel_id': pixel_id,
            'lon': lon,
            'lat': lat,
            'geometry': Point(lon, lat),
            
            # Reference period statistics - VV
            'ref_vv_mean': ref_vv['backscatter'].mean(),
            'ref_vv_std': ref_vv['backscatter'].std(),
            'ref_vv_min': ref_vv['backscatter'].min(),
            'ref_vv_max': ref_vv['backscatter'].max(),
            'ref_vv_range': ref_vv['backscatter'].max() - ref_vv['backscatter'].min(),
            'ref_vv_count': len(ref_vv),
            
            # Reference period statistics - VH
            'ref_vh_mean': ref_vh['backscatter'].mean(),
            'ref_vh_std': ref_vh['backscatter'].std(),
            'ref_vh_min': ref_vh['backscatter'].min(),
            'ref_vh_max': ref_vh['backscatter'].max(), 
            'ref_vh_range': ref_vh['backscatter'].max() - ref_vh['backscatter'].min(),
            'ref_vh_count': len(ref_vh),
            
            # Reference period - cross-polarization ratio (VV/VH) in dB (subtraction)
            'ref_ratio_mean': ref_vv['backscatter'].mean() - ref_vh['backscatter'].mean(),
            
            # Post period statistics - VV
            'post_vv_mean': post_vv['backscatter'].mean(),
            'post_vv_std': post_vv['backscatter'].std(),
            'post_vv_min': post_vv['backscatter'].min(),
            'post_vv_max': post_vv['backscatter'].max(),
            'post_vv_range': post_vv['backscatter'].max() - post_vv['backscatter'].min(),
            'post_vv_count': len(post_vv),
            
            # Post period statistics - VH
            'post_vh_mean': post_vh['backscatter'].mean(),
            'post_vh_std': post_vh['backscatter'].std(),
            'post_vh_min': post_vh['backscatter'].min(),
            'post_vh_max': post_vh['backscatter'].max(),
            'post_vh_range': post_vh['backscatter'].max() - post_vh['backscatter'].min(),
            'post_vh_count': len(post_vh),
            
            # Post period - cross-polarization ratio (VV/VH) in dB (subtraction)
            'post_ratio_mean': post_vv['backscatter'].mean() - post_vh['backscatter'].mean(),
            
            # Change features (post - reference)
            'vv_change_mean': post_vv['backscatter'].mean() - ref_vv['backscatter'].mean(),
            'vh_change_mean': post_vh['backscatter'].mean() - ref_vh['backscatter'].mean(),
            'ratio_change_mean': (post_vv['backscatter'].mean() - post_vh['backscatter'].mean()) - 
                               (ref_vv['backscatter'].mean() - ref_vh['backscatter'].mean()),
            
            # Change magnitude (absolute values)
            'vv_change_magnitude': abs(post_vv['backscatter'].mean() - ref_vv['backscatter'].mean()),
            'vh_change_magnitude': abs(post_vh['backscatter'].mean() - ref_vh['backscatter'].mean()),
            
            # Variability changes
            'vv_std_change': post_vv['backscatter'].std() - ref_vv['backscatter'].std(),
            'vh_std_change': post_vh['backscatter'].std() - ref_vh['backscatter'].std(),
            
            # Normalized changes (as percentages of reference values)
            'vv_percent_change': ((post_vv['backscatter'].mean() - ref_vv['backscatter'].mean()) / 
                                abs(ref_vv['backscatter'].mean())) * 100 if ref_vv['backscatter'].mean() != 0 else np.nan,
            'vh_percent_change': ((post_vh['backscatter'].mean() - ref_vh['backscatter'].mean()) / 
                                abs(ref_vh['backscatter'].mean())) * 100 if ref_vh['backscatter'].mean() != 0 else np.nan,
        }
        
        feature_data.append(features)
    
    # Create GeoDataFrame
    feature_gdf = gpd.GeoDataFrame(feature_data, geometry='geometry')
    
    # Set CRS (assuming the input GeoDataFrame has a valid CRS)
    feature_gdf.crs = backscatter_gdf.crs
    
    print(f"Extracted features for {len(feature_gdf)} pixels")
    
    return feature_gdf

def explore_backscatter_evolution(backscatter_gdf, pixel_id, figsize=(14, 8)):
    """
    Explore the temporal evolution of backscatter for a specific pixel.
    
    Parameters:
    -----------
    backscatter_gdf : GeoDataFrame
        Combined GeoDataFrame with backscatter values
    pixel_id : str
        ID of the pixel to analyze
    figsize : tuple, optional
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Get data for the specified pixel
    pixel_data = backscatter_gdf[backscatter_gdf['pixel_id'] == pixel_id].copy()
    
    if len(pixel_data) == 0:
        print(f"No data found for pixel ID: {pixel_id}")
        return None
    
    # Convert dates to datetime objects for plotting
    pixel_data['datetime'] = pd.to_datetime(pixel_data['date'], format='%Y%m%d')
    
    # Split by polarization
    vv_data = pixel_data[pixel_data['polarization'] == 'VV']
    vh_data = pixel_data[pixel_data['polarization'] == 'VH']
    
    # Get the conflict start date
    conflict_date = pd.to_datetime('20231007', format='%Y%m%d')
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot VV backscatter
    vv_data.sort_values('datetime').plot.scatter(
        x='datetime', y='backscatter', 
        c=vv_data['orbit'].map({'ASC': 'blue', 'DESC': 'orange'}),
        ax=axes[0], s=60, alpha=0.7
    )
    axes[0].set_title(f'VV Backscatter Evolution for Pixel {pixel_id}')
    axes[0].set_ylabel('Backscatter (dB)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot VH backscatter
    vh_data.sort_values('datetime').plot.scatter(
        x='datetime', y='backscatter', 
        c=vh_data['orbit'].map({'ASC': 'blue', 'DESC': 'orange'}),
        ax=axes[1], s=60, alpha=0.7
    )
    axes[1].set_title('VH Backscatter Evolution')
    axes[1].set_ylabel('Backscatter (dB)')
    axes[1].grid(True, alpha=0.3)
    
    # Calculate and plot VV/VH ratio
    dates = []
    ratios = []
    orbit_types = []
    periods = []
    
    # Group by date and orbit
    for (date, orbit), group in pixel_data.groupby(['date', 'orbit']):
        vv_val = group[group['polarization'] == 'VV']['backscatter'].values
        vh_val = group[group['polarization'] == 'VH']['backscatter'].values
        
        # Only add if we have both polarizations
        if len(vv_val) > 0 and len(vh_val) > 0:
            dates.append(pd.to_datetime(date, format='%Y%m%d'))
            ratios.append(vv_val[0] - vh_val[0])  # Ratio in dB is a subtraction
            orbit_types.append(orbit)
            periods.append('reference' if pd.to_datetime(date, format='%Y%m%d') < conflict_date else 'post')
    
    # Create a DataFrame for the ratio
    ratio_df = pd.DataFrame({
        'datetime': dates,
        'ratio': ratios,
        'orbit': orbit_types,
        'period': periods
    })
    
    # Plot the ratio
    ratio_df.sort_values('datetime').plot.scatter(
        x='datetime', y='ratio', 
        c=ratio_df['orbit'].map({'ASC': 'blue', 'DESC': 'orange'}),
        ax=axes[2], s=60, alpha=0.7
    )
    axes[2].set_title('VV/VH Ratio Evolution (dB)')
    axes[2].set_ylabel('VV/VH Ratio (dB)')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    
    # Add a vertical line for the conflict start date
    for ax in axes:
        ax.axvline(x=conflict_date, color='red', linestyle='--', alpha=0.7)
        ax.text(conflict_date, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                'Conflict Start', rotation=90, color='red', ha='right')
        
        # Add a legend
        if ax == axes[0]:
            ax.legend(['', 'Conflict Start', 'Ascending Orbit', 'Descending Orbit'])
    
    plt.tight_layout()
    
    # Calculate and print statistics
    print(f"Pixel ID: {pixel_id}")
    print(f"Location: Lon {pixel_data['lon'].mean():.6f}, Lat {pixel_data['lat'].mean():.6f}")
    print("\nBackscatter Statistics:")
    
    ref_vv = vv_data[vv_data['period'] == 'reference']['backscatter']
    ref_vh = vh_data[vh_data['period'] == 'reference']['backscatter']
    post_vv = vv_data[vv_data['period'] == 'post']['backscatter']
    post_vh = vh_data[vh_data['period'] == 'post']['backscatter']
    
    print(f"Reference VV: Mean = {ref_vv.mean():.2f} dB, Std = {ref_vv.std():.2f} dB, Count = {len(ref_vv)}")
    print(f"Reference VH: Mean = {ref_vh.mean():.2f} dB, Std = {ref_vh.std():.2f} dB, Count = {len(ref_vh)}")
    print(f"Post VV: Mean = {post_vv.mean():.2f} dB, Std = {post_vv.std():.2f} dB, Count = {len(post_vv)}")
    print(f"Post VH: Mean = {post_vh.mean():.2f} dB, Std = {post_vh.std():.2f} dB, Count = {len(post_vh)}")
    
    # Calculate changes
    vv_change = post_vv.mean() - ref_vv.mean()
    vh_change = post_vh.mean() - ref_vh.mean()
    ratio_change = (post_vv.mean() - post_vh.mean()) - (ref_vv.mean() - ref_vh.mean())
    
    print("\nChanges:")
    print(f"VV Change: {vv_change:.2f} dB ({(vv_change/abs(ref_vv.mean()))*100:.1f}%)")
    print(f"VH Change: {vh_change:.2f} dB ({(vh_change/abs(ref_vh.mean()))*100:.1f}%)")
    print(f"VV/VH Ratio Change: {ratio_change:.2f} dB")
    
    return fig

def visualize_backscatter_changes(feature_gdf, figsize=(18, 12), cmap='RdBu_r', center_zero=True):
    """
    Visualize backscatter changes on a map.
    
    Parameters:
    -----------
    feature_gdf : GeoDataFrame
        GeoDataFrame with time series features
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap to use
    center_zero : bool, optional
        Whether to center the colormap at zero
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Define variables to plot
    variables = [
        ('vv_change_mean', 'VV Backscatter Change (dB)'),
        ('vh_change_mean', 'VH Backscatter Change (dB)'),
        ('ratio_change_mean', 'VV/VH Ratio Change (dB)'),
        ('vv_change_magnitude', 'VV Backscatter Change Magnitude (dB)')
    ]
    
    # Plot each variable
    for i, (var, title) in enumerate(variables):
        ax = axes[i]
        
        # Determine vmin and vmax
        if center_zero and var != 'vv_change_magnitude':
            max_abs = max(abs(feature_gdf[var].max()), abs(feature_gdf[var].min()))
            vmin, vmax = -max_abs, max_abs
            cmap_use = cmap
        else:
            vmin, vmax = feature_gdf[var].min(), feature_gdf[var].max()
            cmap_use = 'viridis' if var == 'vv_change_magnitude' else cmap
        
        # Plot
        feature_gdf.plot(
            column=var,
            cmap=cmap_use,
            legend=True,
            ax=ax,
            alpha=0.7,
            vmin=vmin,
            vmax=vmax,
            legend_kwds={'label': var, 'orientation': 'horizontal'}
        )
        
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig