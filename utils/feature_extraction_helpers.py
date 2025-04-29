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
from scipy import stats
from tqdm.notebook import tqdm

def collect_all_backscatter_values_optimized(data_dir, aoi_path=None, polarizations=None, sample_rate=1.0):
    """
    Optimized function to collect backscatter values from all GeoTIFFs.
    This version focuses on performance with large datasets and adds orbit determination.
    
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
    
    if polarizations is None:
        polarizations = ['VV', 'VH']
        
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
    
    # Organize dates by month for orbit determination
    dates_by_month = {}
    for date_str in date_dirs:
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:8]
        
        key = f"{year}-{month}"
        if key not in dates_by_month:
            dates_by_month[key] = []
        
        dates_by_month[key].append(day)
    
    # Sort days within each month
    for key in dates_by_month:
        dates_by_month[key] = sorted(dates_by_month[key])
    
    # Create orbit lookup dictionary
    orbit_lookup = {}
    for key, days in dates_by_month.items():
        for i, day in enumerate(days):
            date_str = f"{key[:4]}{key[5:7]}{day}"
            orbit_lookup[date_str] = "ASC" if i % 2 == 0 else "DESC"
    
    print("Determined orbit directions based on acquisition patterns")
    
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
        
        # Determine orbit direction
        orbit = orbit_lookup.get(date_str, "UNK")
        
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

            # Use the exact resolution from the raster for gridding
            # x_res, y_res = src.res
            # grid_resolution_x = x_res  # Usually around 8.98e-05 degrees
            # grid_resolution_y = y_res  # Usually around 8.98e-05 degrees

            # # Create grid-aligned pixel IDs
            # x_grid = np.round(np.array(xs) / grid_resolution_x).astype(int)
            # y_grid = np.round(np.array(ys) / grid_resolution_y).astype(int) 
            # pixel_ids = np.array([f"{x}_{y}" for x, y in zip(x_grid, y_grid)])
            
            # Create pixel IDs based on the original row and column indices
            pixel_ids = np.array([f"{r}_{c}" for r, c in zip(rows, cols)])
            
            # Create data frame efficiently
            df = pd.DataFrame({
                'row': rows,
                'col': cols,
                'lon': xs,
                'lat': ys,
                'backscatter': backscatter_values,
                'date_str': date_str,
                # Date format to YYYYMMDD
                'backscatter_date': pd.to_datetime(date_str, format='%Y%m%d'),
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
    
    # Summary of orbit distribution
    orbit_counts = all_backscatter['orbit'].value_counts()
    print("\nOrbit distribution:")
    for orbit, count in orbit_counts.items():
        print(f"  {orbit}: {count} observations ({count/len(all_backscatter)*100:.1f}%)")
    
    return all_backscatter

def extract_pixel_timeseries(labeled_points, backscatter_data, buffer_distance=5):
    """
    Extract time series for each labeled point using a buffer to associate with pixels.
    
    Parameters:
    -----------
    labeled_points : GeoDataFrame
        Points with damage labels
    backscatter_data : GeoDataFrame
        Backscatter values from Sentinel-1
    buffer_distance : float, default=5
        Buffer distance in meters (5m recommended based on analysis)
        
    Returns:
    --------
    DataFrame with time series associated with each labeled point
    """
    print(f"Extracting time series for {len(labeled_points)} points with {buffer_distance}m buffer...")
    
    # Ensure CRS match
    if labeled_points.crs != backscatter_data.crs:
        print(f"Converting points to match backscatter CRS: {backscatter_data.crs}")
        labeled_points = labeled_points.to_crs(backscatter_data.crs)
    
    # Create buffers for spatial search
    if labeled_points.crs.is_geographic:
        # Use UTM for accurate buffering
        points_utm = labeled_points.to_crs("EPSG:32636")  # UTM zone 36N for Gaza
        buffered_points = points_utm.copy()
        buffered_points['geometry'] = points_utm.geometry.buffer(buffer_distance)
        buffered_points = buffered_points.to_crs(labeled_points.crs)
    else:
        buffered_points = labeled_points.copy()
        buffered_points['geometry'] = buffered_points.geometry.buffer(buffer_distance)
    
    # Ensure we have a spatial index
    if not hasattr(backscatter_data, 'sindex') or backscatter_data.sindex is None:
        print("Creating spatial index for faster processing...")
        backscatter_data = backscatter_data.copy()
        backscatter_data.sindex  # This creates the index
    
    # List to store results
    timeseries_records = []
    
    # Process each labeled point
    for idx, point in tqdm(buffered_points.iterrows(), total=len(buffered_points), 
                           desc="Extracting time series"):
        # Find backscatter values within buffer
        possible_matches_idx = list(backscatter_data.sindex.intersection(point.geometry.bounds))
        if not possible_matches_idx:
            continue
            
        possible_matches = backscatter_data.iloc[possible_matches_idx]
        matches = possible_matches[possible_matches.intersects(point.geometry)]
        
        if len(matches) == 0:
            continue
        
        # Get the label information
        label_info = {col: point[col] for col in labeled_points.columns 
                     if col != 'geometry' and not pd.isna(point[col])}
        
        # Extract unique pixel IDs in the buffer
        pixel_ids = matches['pixel_id'].unique()
        
        # For a 5m buffer, there should mostly be one pixel per point (based on analysis)
        # But if we have multiple, let's handle each pixel separately
        for pixel_id in pixel_ids:
            pixel_data = matches[matches['pixel_id'] == pixel_id]
            
            # Add each time series point
            for _, row in pixel_data.iterrows():
                record = {
                    'point_id': idx,
                    'pixel_id': pixel_id,
                    'backscatter_date': row['backscatter_date'],
                    'polarization': row['polarization'],
                    'orbit': row['orbit'],
                    'period': row['period'],
                    'backscatter': row['backscatter'],
                    'point_lon': point.geometry.x if hasattr(point.geometry, 'x') else point.geometry.centroid.x,
                    'point_lat': point.geometry.y if hasattr(point.geometry, 'y') else point.geometry.centroid.y,
                    'pixel_lon': row['lon'],
                    'pixel_lat': row['lat'],
                    **label_info
                }
                timeseries_records.append(record)
    
    # Convert to DataFrame
    timeseries_df = pd.DataFrame(timeseries_records)
    
    print(f"Extracted {len(timeseries_df)} time series points for {timeseries_df['point_id'].nunique()} labeled points")
    print(f"Covering {timeseries_df['pixel_id'].nunique()} unique pixels")
    
    # Basic stats by polarization and orbit
    if 'polarization' in timeseries_df.columns and 'orbit' in timeseries_df.columns:
        print("\nTime series distribution by polarization and orbit:")
        counts = timeseries_df.groupby(['polarization', 'orbit']).size().unstack(fill_value=0)
        print(counts)
    
    return timeseries_df

def create_stable_point_timeseries(timeseries):
    """
    Create a more stable time series for each point by averaging values across
    all pixels ever associated with that point, for each date.
    
    Optimized for performance with large datasets.
    """
    # Create a mapping of point_id to unique pixel_ids (faster than groupby.unique)
    point_pixel_map = timeseries.groupby('point_id')['pixel_id'].apply(set).to_dict()
    
    # Create a DataFrame with one row per point containing essential point information
    point_info = timeseries[['point_id', 'is_damaged', 'point_lon', 'point_lat']].drop_duplicates('point_id')
    
    # Prepare a list to store results for each point
    results = []
    
    # Process each point
    for _, point_row in point_info.iterrows():
        point_id = point_row['point_id']
        
        # Get associated pixels for this point
        pixel_ids = point_pixel_map[point_id]
        
        # Filter data for these pixels (faster than multiple .isin() calls)
        mask = timeseries['pixel_id'].isin(pixel_ids)
        pixels_data = timeseries[mask]
        
        # Group and aggregate in one operation
        avg_data = (pixels_data
            .groupby(['backscatter_date', 'polarization', 'orbit', 'period'], observed=True)
            .agg(
                backscatter=('backscatter', 'mean'),
                pixel_count=('pixel_id', 'nunique')
            )
            .reset_index()
        )
        
        # Add point information (faster than column-by-column assignment)
        for col in point_row.index:
            avg_data[col] = point_row[col]
        
        results.append(avg_data)
    
    # Combine all results
    stable_timeseries = pd.concat(results, ignore_index=True)
    return stable_timeseries


def create_timeseries_features(timeseries_df):
    """
    Create feature vectors from time series data as specified in the paper.
    Each pixel+orbit combination will have one row with features from both polarizations.
    The final feature vector is φ(xi) ∈ R^{4N} where N=7 (summary statistics).
    
    Parameters:
    -----------
    timeseries_df : DataFrame
        
    Returns:
    --------
    DataFrame with feature vectors for each pixel/orbit combination
    """
    print("Creating time series features...")
    
    # Make a copy to avoid modifying the original
    df = timeseries_df.copy()
    
    # Split data into reference and assessment periods
    ref_data = df[df['period'] == 'reference']
    post_data = df[df['period'] == 'post']
    
    print(f"Reference period: {len(ref_data)} observations")
    print(f"Assessment period: {len(post_data)} observations")
    
    # Define the summary statistics to compute
    stats_functions = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'median': np.median,
        'std': lambda x: np.std(x) if len(x) > 1 else 0,
        'kurtosis': lambda x: stats.kurtosis(x) if len(x) > 3 else 0,
        'skew': lambda x: stats.skew(x) if len(x) > 2 else 0
    }
    
    # Group by pixel and orbit (combining polarizations into one feature vector)
    group_cols = ['pixel_id', 'orbit']
    
    # Store the results
    features = []
    
    # Process each pixel+orbit group
    for (pixel_id, orbit), group in tqdm(df.groupby(group_cols), 
                                       desc="Extracting features", 
                                       total=df.groupby(group_cols).ngroups):
        
        # Create feature record
        feature_record = {
            'pixel_id': pixel_id,
            'orbit': orbit
        }
        
        # Add point information (use the first point associated with this pixel)
        for col in ['point_id', 'point_lon', 'point_lat', 'is_damaged', 'damage_class', 'damage_class_desc']:
            if col in group.columns:
                feature_record[col] = group[col].iloc[0]
        
        # Process each polarization separately, then combine into one feature vector
        for pol in ['VV', 'VH']:
            pol_group = group[group['polarization'] == pol]
            
            # Skip if we don't have this polarization
            if len(pol_group) == 0:
                continue
                
            # Get reference and assessment data for this polarization
            pol_ref = pol_group[pol_group['period'] == 'reference']
            pol_post = pol_group[pol_group['period'] == 'post']
            
            # Skip if we don't have data for both periods
            if len(pol_ref) == 0 or len(pol_post) == 0:
                continue
            
            # Calculate summary statistics for reference period
            for stat_name, stat_func in stats_functions.items():
                feature_record[f'ref_{pol}_{stat_name}'] = stat_func(pol_ref['backscatter'])
            
            # Calculate summary statistics for assessment period
            for stat_name, stat_func in stats_functions.items():
                feature_record[f'post_{pol}_{stat_name}'] = stat_func(pol_post['backscatter'])
            
            # Add observation counts
            feature_record[f'ref_{pol}_count'] = len(pol_ref)
            feature_record[f'post_{pol}_count'] = len(pol_post)
        
        # Make sure we have both polarizations
        if (any(f'ref_VV_{stat}' in feature_record for stat in stats_functions) and 
            any(f'ref_VH_{stat}' in feature_record for stat in stats_functions)):
            features.append(feature_record)
    
    # Create DataFrame
    features_df = pd.DataFrame(features)
    
    # Calculate change metrics across polarizations (feature engineering)
    for pol in ['VV', 'VH']:
        # Skip if we don't have this polarization
        if not any(f'ref_{pol}_mean' in col for col in features_df.columns):
            continue
            
        features_df[f'change_{pol}_mean'] = features_df[f'post_{pol}_mean'] - features_df[f'ref_{pol}_mean']
        features_df[f'change_{pol}_magnitude'] = abs(features_df[f'post_{pol}_mean'] - features_df[f'ref_{pol}_mean'])
        features_df[f'change_{pol}_relative'] = (
            features_df[f'change_{pol}_magnitude'] / abs(features_df[f'ref_{pol}_mean'].replace(0, np.nan)).fillna(1)
        )
        
        # Cross-polarization metrics (only for VV, calculating VV/VH ratio)
        if pol == 'VV' and 'ref_VH_mean' in features_df.columns:
            # In dB, ratio is subtraction
            features_df['ref_ratio'] = features_df['ref_VV_mean'] - features_df['ref_VH_mean']
            features_df['post_ratio'] = features_df['post_VV_mean'] - features_df['post_VH_mean']
            features_df['change_ratio'] = features_df['post_ratio'] - features_df['ref_ratio']
    
    print(f"Created {len(features_df)} feature vectors")
    
    if 'is_damaged' in features_df.columns:
        damaged_count = features_df['is_damaged'].sum()
        print(f"Class distribution: {damaged_count} damaged, {len(features_df) - damaged_count} undamaged")
    
    return features_df

def extract_ml_features_stable(stable_timeseries):
    """
    Extract machine learning features from the stabilized time series data.
    
    Features include:
    - Statistical metrics for reference and post-conflict periods
    - Change metrics between periods
    - Statistical tests comparing distributions
    
    Returns a DataFrame with one row per point and features as columns.
    """
    # Ensure we have datetime objects
    if isinstance(stable_timeseries['backscatter_date'].iloc[0], str):
        stable_timeseries['backscatter_date'] = pd.to_datetime(stable_timeseries['backscatter_date'])
    
    # Initialize the results list
    feature_records = []
    
    # Process each point
    for point_id, point_data in stable_timeseries.groupby('point_id'):
        # Initialize the feature record with point_id and damage status
        is_damaged = point_data['is_damaged'].iloc[0]
        
        feature_record = {
            'point_id': point_id,
            'is_damaged': is_damaged,
            'point_lon': point_data['point_lon'].iloc[0],
            'point_lat': point_data['point_lat'].iloc[0]
        }
        
        # Process each combination of polarization and orbit
        for (pol, orbit), combo_data in point_data.groupby(['polarization', 'orbit']):
            # Separate reference and post-conflict data
            ref_data = combo_data[combo_data['period'] == 'reference']['backscatter']
            post_data = combo_data[combo_data['period'] == 'post']['backscatter']
            
            # Skip if we don't have enough data
            if len(ref_data) < 5 or len(post_data) < 5:
                continue
                
            # Basic statistics for reference period
            feature_record[f'{pol}_{orbit}_ref_mean'] = ref_data.mean()
            feature_record[f'{pol}_{orbit}_ref_median'] = ref_data.median()
            feature_record[f'{pol}_{orbit}_ref_std'] = ref_data.std()
            feature_record[f'{pol}_{orbit}_ref_min'] = ref_data.min()
            feature_record[f'{pol}_{orbit}_ref_max'] = ref_data.max()
            feature_record[f'{pol}_{orbit}_ref_range'] = ref_data.max() - ref_data.min()
            feature_record[f'{pol}_{orbit}_ref_iqr'] = np.percentile(ref_data, 75) - np.percentile(ref_data, 25)
            feature_record[f'{pol}_{orbit}_ref_skew'] = stats.skew(ref_data)
            feature_record[f'{pol}_{orbit}_ref_kurtosis'] = stats.kurtosis(ref_data)
            
            # Basic statistics for post-conflict period
            feature_record[f'{pol}_{orbit}_post_mean'] = post_data.mean()
            feature_record[f'{pol}_{orbit}_post_median'] = post_data.median()
            feature_record[f'{pol}_{orbit}_post_std'] = post_data.std()
            feature_record[f'{pol}_{orbit}_post_min'] = post_data.min()
            feature_record[f'{pol}_{orbit}_post_max'] = post_data.max()
            feature_record[f'{pol}_{orbit}_post_range'] = post_data.max() - post_data.min()
            feature_record[f'{pol}_{orbit}_post_iqr'] = np.percentile(post_data, 75) - np.percentile(post_data, 25)
            feature_record[f'{pol}_{orbit}_post_skew'] = stats.skew(post_data)
            feature_record[f'{pol}_{orbit}_post_kurtosis'] = stats.kurtosis(post_data)
            
            # Change metrics
            feature_record[f'{pol}_{orbit}_mean_diff'] = post_data.mean() - ref_data.mean()
            feature_record[f'{pol}_{orbit}_median_diff'] = post_data.median() - ref_data.median()
            feature_record[f'{pol}_{orbit}_std_diff'] = post_data.std() - ref_data.std()
            feature_record[f'{pol}_{orbit}_range_diff'] = feature_record[f'{pol}_{orbit}_post_range'] - feature_record[f'{pol}_{orbit}_ref_range']
            
            # Ratio metrics (with safeguard against division by zero)
            if ref_data.mean() != 0:
                feature_record[f'{pol}_{orbit}_mean_ratio'] = post_data.mean() / ref_data.mean()
            else:
                feature_record[f'{pol}_{orbit}_mean_ratio'] = np.nan
                
            if ref_data.std() != 0:
                feature_record[f'{pol}_{orbit}_std_ratio'] = post_data.std() / ref_data.std()
            else:
                feature_record[f'{pol}_{orbit}_std_ratio'] = np.nan
        
        feature_records.append(feature_record)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(feature_records)
    
    # Calculate number of features
    num_features = len(features_df.columns) - 4  # Subtract the ID and coordinate columns
    print(f"Extracted {num_features} features for {len(features_df)} points")
    
    return features_df