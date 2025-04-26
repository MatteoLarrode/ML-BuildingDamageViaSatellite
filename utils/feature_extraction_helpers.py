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
                'date_str': date_str,
                # Date format to YYYYMMDD
                'date': pd.to_datetime(date_str, format='%Y%m%d'),
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
        
        # For a 5m buffer, we should mostly have one pixel per point (based on analysis)
        # But if we have multiple, let's handle each pixel separately
        for pixel_id in pixel_ids:
            pixel_data = matches[matches['pixel_id'] == pixel_id]
            
            # Add each time series point
            for _, row in pixel_data.iterrows():
                record = {
                    'point_id': idx,
                    'pixel_id': pixel_id,
                    'date': row['date'],
                    'polarization': row['polarization'],
                    'orbit': row['orbit'],
                    'period': row['period'],
                    'backscatter': row['backscatter'],
                    'point_lon': point.geometry.x,
                    'point_lat': point.geometry.y,
                    'pixel_lon': row.geometry.x,
                    'pixel_lat': row.geometry.y,
                    **label_info
                }
                timeseries_records.append(record)
    
    # Convert to DataFrame
    timeseries_df = pd.DataFrame(timeseries_records)
    
    # Convert date to datetime
    if 'date' in timeseries_df.columns:
        timeseries_df['date'] = pd.to_datetime(timeseries_df['date'], format='%Y%m%d')
    
    print(f"Extracted {len(timeseries_df)} time series points for {timeseries_df['point_id'].nunique()} labeled points")
    print(f"Covering {timeseries_df['pixel_id'].nunique()} unique pixels")
    
    # Basic stats by polarization and orbit
    if 'polarization' in timeseries_df.columns and 'orbit' in timeseries_df.columns:
        print("\nTime series distribution by polarization and orbit:")
        counts = timeseries_df.groupby(['polarization', 'orbit']).size().unstack(fill_value=0)
        print(counts)
    
    return timeseries_df

def create_timeseries_features(timeseries_df, reference_dates=None, assessment_dates=None):
    """
    Create feature vectors from time series data as specified in the paper.
    
    Parameters:
    -----------
    timeseries_df : DataFrame
        Time series data from extract_pixel_timeseries function
    reference_dates : list, optional
        List of dates to use as reference period (pre-conflict)
    assessment_dates : list, optional
        List of dates to use as assessment period (post-conflict)
        
    Returns:
    --------
    DataFrame with feature vectors for each pixel/polarization/orbit combination
    """
    print("Creating time series features...")
    
    # Make a copy to avoid modifying the original
    df = timeseries_df.copy()
    
    # If dates not provided, infer them
    if reference_dates is None or assessment_dates is None:
        all_dates = pd.to_datetime(df['date'].unique()).sort_values()
        conflict_start = pd.to_datetime('2023-10-07')
        
        if reference_dates is None:
            reference_dates = [d for d in all_dates if d < conflict_start]
            print(f"Inferred reference dates: {reference_dates}")
        
        if assessment_dates is None:
            assessment_dates = [d for d in all_dates if d >= conflict_start]
            print(f"Inferred assessment dates: {assessment_dates}")
    
    # Convert reference_dates and assessment_dates to datetime if they're not already
    if reference_dates is not None and not isinstance(reference_dates[0], pd.Timestamp):
        reference_dates = pd.to_datetime(reference_dates)
    
    if assessment_dates is not None and not isinstance(assessment_dates[0], pd.Timestamp):
        assessment_dates = pd.to_datetime(assessment_dates)
    
    # Split data into reference and assessment periods
    ref_data = df[df['date'].isin(reference_dates)] if reference_dates is not None else df[df['period'] == 'reference']
    assess_data = df[df['date'].isin(assessment_dates)] if assessment_dates is not None else df[df['period'] == 'post']
    
    print(f"Reference period: {len(ref_data)} observations")
    print(f"Assessment period: {len(assess_data)} observations")
    
    # Group by pixel, polarization, and orbit
    # Each combination will get its own feature vector
    features = []
    
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
    
    # Group by pixel, polarization, and orbit
    group_cols = ['pixel_id', 'polarization', 'orbit']
    
    # Process each group
    for name, group in tqdm(df.groupby(group_cols), 
                            desc="Extracting features", 
                            total=df.groupby(group_cols).ngroups):
        
        # Get reference and assessment data for this group
        group_ref = group[group['date'].isin(reference_dates)] if reference_dates is not None else group[group['period'] == 'reference']
        group_assess = group[group['date'].isin(assessment_dates)] if assessment_dates is not None else group[group['period'] == 'post']
        
        # Skip if we don't have data for both periods
        if len(group_ref) == 0 or len(group_assess) == 0:
            continue
        
        # Create feature record
        feature_record = {}
        
        # Add identifying information
        pixel_id, polarization, orbit = name
        feature_record['pixel_id'] = pixel_id
        feature_record['polarization'] = polarization
        feature_record['orbit'] = orbit
        
        # Add point information (use the first point associated with this pixel)
        for col in ['point_id', 'point_lon', 'point_lat', 'is_damaged', 'damage_class', 'damage_class_desc']:
            if col in group.columns:
                feature_record[col] = group[col].iloc[0]
        
        # Calculate summary statistics for reference period
        for stat_name, stat_func in stats_functions.items():
            feature_record[f'ref_{stat_name}'] = stat_func(group_ref['backscatter'])
        
        # Calculate summary statistics for assessment period
        for stat_name, stat_func in stats_functions.items():
            feature_record[f'assess_{stat_name}'] = stat_func(group_assess['backscatter'])
        
        # Calculate change metrics
        feature_record['change_mean'] = feature_record['assess_mean'] - feature_record['ref_mean']
        feature_record['change_median'] = feature_record['assess_median'] - feature_record['ref_median']
        feature_record['change_std'] = feature_record['assess_std'] - feature_record['ref_std']
        feature_record['change_magnitude'] = abs(feature_record['assess_mean'] - feature_record['ref_mean'])
        feature_record['change_relative'] = feature_record['change_magnitude'] / abs(feature_record['ref_mean']) if feature_record['ref_mean'] != 0 else 0
        
        # Add observation counts
        feature_record['ref_count'] = len(group_ref)
        feature_record['assess_count'] = len(group_assess)
        
        features.append(feature_record)
    
    # Create DataFrame
    features_df = pd.DataFrame(features)
    
    print(f"Created {len(features_df)} feature vectors")
    
    if 'is_damaged' in features_df.columns:
        damaged_count = features_df['is_damaged'].sum()
        print(f"Class distribution: {damaged_count} damaged, {len(features_df) - damaged_count} undamaged")
    
    return features_df







# def extract_backscatter_for_points(points_gdf, backscatter_gdf, buffer_distance=5):
#     """
#     Extract backscatter values for labeled points from all available dates.
    
#     Parameters:
#     -----------
#     points_gdf : GeoDataFrame
#         GeoDataFrame containing labeled points (damage and control)
#     backscatter_gdf : GeoDataFrame
#         GeoDataFrame containing backscatter values from Sentinel-1
#     buffer_distance : float, default=5
#         Buffer distance in meters to search around each point
        
#     Returns:
#     --------
#     DataFrame
#         DataFrame containing backscatter values for each point and date combination
#     """
#     print(f"Extracting backscatter values for {len(points_gdf)} points with {buffer_distance}m buffer...")
    
#     # Ensure both GeoDataFrames have the same CRS
#     if points_gdf.crs != backscatter_gdf.crs:
#         print(f"Converting points from {points_gdf.crs} to {backscatter_gdf.crs}")
#         points_gdf = points_gdf.to_crs(backscatter_gdf.crs)
    
#     # Create buffers around points for spatial search
#     # If CRS is geographic, convert to projected before buffering
#     if points_gdf.crs.is_geographic:
#         print("Converting to UTM for accurate buffering...")
#         # Use UTM zone 36N for Gaza
#         points_utm = points_gdf.to_crs("EPSG:32636")
#         buffered_points = points_utm.copy()
#         buffered_points['geometry'] = points_utm.geometry.buffer(buffer_distance)
#         buffered_points = buffered_points.to_crs(points_gdf.crs)
#     else:
#         buffered_points = points_gdf.copy()
#         buffered_points['geometry'] = buffered_points.geometry.buffer(buffer_distance)
    
#     # Initialize list to store results
#     results = []
    
#     # Process each point
#     for idx, point in tqdm(buffered_points.iterrows(), total=len(buffered_points), 
#                            desc="Extracting backscatter values"):
#         # Find backscatter data within the buffer
#         if hasattr(backscatter_gdf, 'sindex') and backscatter_gdf.sindex is not None:
#             # Use spatial index if available for faster querying
#             possible_matches_idx = list(backscatter_gdf.sindex.intersection(point.geometry.bounds))
#             if possible_matches_idx:
#                 possible_matches = backscatter_gdf.iloc[possible_matches_idx]
#                 matches = possible_matches[possible_matches.intersects(point.geometry)]
#             else:
#                 matches = backscatter_gdf[backscatter_gdf.intersects(point.geometry)]
#         else:
#             # Fallback to standard spatial query (slower)
#             matches = backscatter_gdf[backscatter_gdf.intersects(point.geometry)]
        
#         # If no backscatter data found for this point, skip
#         if len(matches) == 0:
#             continue
        
#         # Get point attributes to include in results
#         point_attrs = {col: point[col] for col in points_gdf.columns 
#                       if col != 'geometry' and not pd.isna(point[col])}
        
#         # Add each backscatter value to results
#         for _, match in matches.iterrows():
#             result = {
#                 'point_id': idx,
#                 'point_lon': point.geometry.x if hasattr(point.geometry, 'x') else point.geometry.centroid.x,
#                 'point_lat': point.geometry.y if hasattr(point.geometry, 'y') else point.geometry.centroid.y,
#                 'backscatter_lon': match.geometry.x if hasattr(match.geometry, 'x') else match.geometry.centroid.x,
#                 'backscatter_lat': match.geometry.y if hasattr(match.geometry, 'y') else match.geometry.centroid.y,
#                 'distance': point.geometry.distance(match.geometry),
#                 **point_attrs
#             }
            
#             # Add backscatter attributes
#             for col in match.index:
#                 if col != 'geometry' and not pd.isna(match[col]):
#                     if col in result and col not in ['point_id', 'lon', 'lat']:
#                         # Rename to avoid conflicts
#                         result[f'backscatter_{col}'] = match[col]
#                     else:
#                         result[col] = match[col]
            
#             results.append(result)
    
#     # Convert results to DataFrame
#     if not results:
#         print("No backscatter values found for any points!")
#         return pd.DataFrame()
        
#     result_df = pd.DataFrame(results)
    
#     # Print summary
#     print(f"Extracted {len(result_df)} backscatter values for {result_df['point_id'].nunique()} points")
    
#     # Summary by polarization
#     if 'polarization' in result_df.columns:
#         print("\nValues by polarization:")
#         print(result_df['polarization'].value_counts())
    
#     # Summary by orbit
#     if 'orbit' in result_df.columns:
#         print("\nValues by orbit:")
#         print(result_df['orbit'].value_counts())
    
#     # Summary by date
#     if 'date' in result_df.columns:
#         print("\nValues by date:")
#         print(result_df['backscatter_date'].value_counts().sort_index())
    
#     # Summary by damage status
#     if 'is_damaged' in result_df.columns:
#         print("\nValues by damage status:")
#         print(result_df['is_damaged'].value_counts())
    
#     return result_df

# def create_ml_features(backscatter_points_df, reference_dates=None, assessment_dates=None):
#     """
#     Create machine learning features from extracted backscatter values
    
#     Parameters:
#     -----------
#     backscatter_points_df : DataFrame
#         DataFrame from extract_backscatter_for_points function
#     reference_dates : list, optional
#         List of dates to use as reference period (pre-conflict)
#     assessment_dates : list, optional
#         List of dates to use as assessment period (post-conflict)
        
#     Returns:
#     --------
#     DataFrame
#         DataFrame with ML features for each point/polarization/orbit combination
#     """
#     print("Creating machine learning features...")
    
#     # If dates not provided, try to infer them
#     if reference_dates is None or assessment_dates is None:
#         all_dates = backscatter_points_df['date'].unique()
#         all_dates.sort()
        
#         if reference_dates is None:
#             # Use dates before 2023-10-07 (conflict start) as reference
#             reference_dates = [d for d in all_dates if d < '20231007']
#             print(f"Inferred reference dates: {reference_dates}")
        
#         if assessment_dates is None:
#             # Use dates on or after 2023-10-07 as assessment
#             assessment_dates = [d for d in all_dates if d >= '20231007']
#             print(f"Inferred assessment dates: {assessment_dates}")
    
#     # Split data into reference and assessment periods
#     ref_data = backscatter_points_df[backscatter_points_df['date'].isin(reference_dates)]
#     assessment_data = backscatter_points_df[backscatter_points_df['date'].isin(assessment_dates)]
    
#     print(f"Reference period: {len(ref_data)} observations")
#     print(f"Assessment period: {len(assessment_data)} observations")
    
#     # Group by point, polarization, and orbit
#     # Each combination will get its own feature set
#     features = []
    
#     # Identify the group columns
#     group_cols = ['point_id']
#     if 'polarization' in backscatter_points_df.columns:
#         group_cols.append('polarization')
#     if 'orbit' in backscatter_points_df.columns:
#         group_cols.append('orbit')
    
#     # Create a unique group key
#     print(f"Grouping by: {group_cols}")
    
#     # Process each group
#     for name, group in tqdm(backscatter_points_df.groupby(group_cols), 
#                             desc="Extracting features", 
#                             total=backscatter_points_df.groupby(group_cols).ngroups):
        
#         # Get reference and assessment data for this group
#         group_ref = group[group['date'].isin(reference_dates)]
#         group_assessment = group[group['date'].isin(assessment_dates)]
        
#         # Skip if we don't have data for both periods
#         if len(group_ref) == 0 or len(group_assessment) == 0:
#             continue
        
#         # Create feature record
#         feature_record = {}
        
#         # Add identifying information
#         if isinstance(name, tuple):
#             for i, col in enumerate(group_cols):
#                 feature_record[col] = name[i]
#         else:
#             feature_record[group_cols[0]] = name
        
#         # Add point metadata
#         for col in ['is_damaged', 'damage_class', 'damage_class_desc', 'point_lon', 'point_lat']:
#             if col in group.columns:
#                 feature_record[col] = group[col].iloc[0]
        
#         # Calculate reference period statistics
#         feature_record.update({
#             'ref_mean': group_ref['backscatter'].mean(),
#             'ref_std': group_ref['backscatter'].std() if len(group_ref) > 1 else 0,
#             'ref_min': group_ref['backscatter'].min(),
#             'ref_max': group_ref['backscatter'].max(),
#             'ref_range': group_ref['backscatter'].max() - group_ref['backscatter'].min(),
#             'ref_count': len(group_ref)
#         })
        
#         # Calculate assessment period statistics
#         feature_record.update({
#             'assessment_mean': group_assessment['backscatter'].mean(),
#             'assessment_std': group_assessment['backscatter'].std() if len(group_assessment) > 1 else 0,
#             'assessment_min': group_assessment['backscatter'].min(),
#             'assessment_max': group_assessment['backscatter'].max(),
#             'assessment_range': group_assessment['backscatter'].max() - group_assessment['backscatter'].min(),
#             'assessment_count': len(group_assessment)
#         })
        
#         # Calculate change metrics
#         feature_record.update({
#             'change_mean': feature_record['assessment_mean'] - feature_record['ref_mean'],
#             'change_std': feature_record['assessment_std'] - feature_record['ref_std'],
#             'change_range': feature_record['assessment_range'] - feature_record['ref_range'],
#             'change_magnitude': abs(feature_record['assessment_mean'] - feature_record['ref_mean'])
#         })
        
#         # Calculate statistical change metrics (similar to PWTT)
#         # t-statistic = (mean2 - mean1) / sqrt((var1/n1) + (var2/n2))
#         if len(group_ref) > 1 and len(group_assessment) > 1:
#             pooled_std = ((group_ref['backscatter'].var() / len(group_ref)) + 
#                            (group_assessment['backscatter'].var() / len(group_assessment))) ** 0.5
#             if pooled_std > 0:
#                 feature_record['t_statistic'] = (feature_record['assessment_mean'] - 
#                                                feature_record['ref_mean']) / pooled_std
#             else:
#                 feature_record['t_statistic'] = 0
#         else:
#             feature_record['t_statistic'] = 0
        
#         # Calculate percent change (handle division by zero)
#         if feature_record['ref_mean'] != 0:
#             feature_record['percent_change'] = ((feature_record['assessment_mean'] - 
#                                               feature_record['ref_mean']) / 
#                                              abs(feature_record['ref_mean'])) * 100
#         else:
#             feature_record['percent_change'] = 0
        
#         features.append(feature_record)
    
#     # Create DataFrame
#     features_df = pd.DataFrame(features)
    
#     print(f"Created {len(features_df)} feature records")
#     if 'is_damaged' in features_df.columns:
#         print("Class distribution:")
#         print(features_df['is_damaged'].value_counts())
    
#     return features_df




# def extract_backscatter_from_tiff(tiff_path, polarization=None, aoi_path=None):
#     """
#     Extract backscatter values from a single Sentinel-1 GeoTIFF file.
    
#     Parameters:
#     -----------
#     tiff_path : str
#         Path to the GeoTIFF file
#     polarization : str, optional
#         Polarization ('VV' or 'VH'), detected from filename if not provided
#     aoi_path : str, optional
#         Path to AOI GeoJSON file to mask the data (if None, all pixels are used)
    
#     Returns:
#     --------
#     gdf : GeoDataFrame
#         GeoDataFrame with backscatter values and coordinates
#     """
#     # Extract metadata from filename
#     filename = os.path.basename(tiff_path)
    
#     # Determine polarization from filename if not provided
#     if polarization is None:
#         if 'vv' in filename.lower():
#             polarization = 'VV'
#         elif 'vh' in filename.lower():
#             polarization = 'VH'
#         else:
#             raise ValueError(f"Could not determine polarization from filename: {filename}")
    
#     # Extract date from filename or directory
#     date_match = re.search(r'(\d{8})(?:_|T)', filename)
#     if date_match:
#         date_str = date_match.group(1)
#     else:
#         # Try to get date from directory name
#         parent_dir = os.path.basename(os.path.dirname(tiff_path))
#         if re.match(r'^\d{8}$', parent_dir):
#             date_str = parent_dir
#         else:
#             raise ValueError(f"Could not extract date from filename or directory: {tiff_path}")
    
#     # Determine orbit direction based on acquisition time
#     if 'T15' in filename:
#         orbit = 'ASC'  # Ascending (afternoon)
#     elif 'T03' in filename:
#         orbit = 'DESC'  # Descending (morning)
#     else:
#         orbit = 'UNK'  # Unknown
    
#     # Load AOI if provided
#     aoi = None
#     if aoi_path and os.path.exists(aoi_path):
#         aoi = gpd.read_file(aoi_path)
    
#     # Open the tiff file
#     with rasterio.open(tiff_path) as src:
#         # Get the image transform
#         transform = src.transform
        
#         # Mask with AOI if provided
#         if aoi is not None:
#             masked_data, masked_transform = mask(src, aoi.geometry, crop=True, all_touched=True, nodata=np.nan)
#             data = masked_data[0]  # First (and only) band
#             transform = masked_transform
#         else:
#             data = src.read(1)  # Read the first (and only) band
        
#         # Get coordinates for all valid pixels
#         height, width = data.shape
#         rows, cols = np.where(~np.isnan(data))
        
#         # Create a list to store data
#         pixel_data = []
        
#         # Process each valid pixel
#         for row, col in zip(rows, cols):
#             # Get the value
#             value = data[row, col]
            
#             # Convert pixel coordinates to geographic coordinates
#             x, y = rasterio.transform.xy(transform, row, col, offset='center')
            
#             # Add to pixel data
#             pixel_data.append({
#                 'row': row,
#                 'col': col,
#                 'x': x,
#                 'y': y,
#                 'lon': x,
#                 'lat': y, 
#                 'backscatter': value,
#                 'geometry': Point(x, y),
#                 'date': date_str,
#                 'polarization': polarization,
#                 'orbit': orbit,
#                 'file': filename
#             })
    
#     # Create GeoDataFrame
#     gdf = gpd.GeoDataFrame(pixel_data, geometry='geometry')
    
#     # Set CRS from source file
#     with rasterio.open(tiff_path) as src:
#         gdf.crs = src.crs
    
#     return gdf

# def extract_ml_features(points_gdf, backscatter_gdf, buffer_distance=10):
#     """
#     Extract machine learning features from backscatter data for given points
    
#     Parameters:
#     -----------
#     points_gdf : GeoDataFrame
#         Points to extract features for (both damaged and control)
#     backscatter_gdf : GeoDataFrame
#         Backscatter values
#     buffer_distance : float, default=10
#         Buffer distance (meters) around points
        
#     Returns:
#     --------
#     DataFrame
#         Features for machine learning
#     """
#     # Create buffers around points
#     if points_gdf.crs.is_geographic:
#         points_utm = points_gdf.to_crs("EPSG:32636")  # UTM zone 36N
#         buffered = points_utm.copy()
#         buffered['geometry'] = points_utm.geometry.buffer(buffer_distance)
#         buffered = buffered.to_crs(points_gdf.crs)
#     else:
#         buffered = points_gdf.copy()
#         buffered['geometry'] = buffered.geometry.buffer(buffer_distance)
    
#     # Extract features for each point
#     all_features = []
    
#     for idx, row in buffered.iterrows():
#         point_buffer = row.geometry
        
#         # Find backscatter points within this buffer
#         matching_points = backscatter_gdf[backscatter_gdf.geometry.within(point_buffer)]
        
#         if len(matching_points) == 0:
#             continue
        
#         # Split into reference and post periods
#         ref_data = matching_points[matching_points['period'] == 'reference']
#         post_data = matching_points[matching_points['period'] == 'post']
        
#         if len(ref_data) == 0 or len(post_data) == 0:
#             continue
        
#         # Process each polarization separately
#         for pol in ['VV', 'VH']:
#             ref_pol = ref_data[ref_data['polarization'] == pol]
#             post_pol = post_data[post_data['polarization'] == pol]
            
#             if len(ref_pol) == 0 or len(post_pol) == 0:
#                 continue
            
#             # Extract features
#             features = {
#                 'point_id': idx,
#                 'lon': row['lon'],
#                 'lat': row['lat'],
#                 'is_damaged': row['is_damaged'],
#                 'polarization': pol,
                
#                 # Reference period statistics
#                 'ref_mean': ref_pol['backscatter'].mean(),
#                 'ref_std': ref_pol['backscatter'].std() if len(ref_pol) > 1 else 0,
#                 'ref_min': ref_pol['backscatter'].min(),
#                 'ref_max': ref_pol['backscatter'].max(),
#                 'ref_range': ref_pol['backscatter'].max() - ref_pol['backscatter'].min(),
#                 'ref_count': len(ref_pol),
                
#                 # Post period statistics
#                 'post_mean': post_pol['backscatter'].mean(),
#                 'post_std': post_pol['backscatter'].std() if len(post_pol) > 1 else 0,
#                 'post_min': post_pol['backscatter'].min(),
#                 'post_max': post_pol['backscatter'].max(),
#                 'post_range': post_pol['backscatter'].max() - post_pol['backscatter'].min(),
#                 'post_count': len(post_pol),
                
#                 # Change metrics
#                 'change_mean': post_pol['backscatter'].mean() - ref_pol['backscatter'].mean(),
#                 'change_std': post_pol['backscatter'].std() - ref_pol['backscatter'].std() if len(post_pol) > 1 and len(ref_pol) > 1 else 0,
#                 'change_magnitude': abs(post_pol['backscatter'].mean() - ref_pol['backscatter'].mean()),
                
#                 # Statistical change metrics (similar to PWTT)
#                 't_statistic': (post_pol['backscatter'].mean() - ref_pol['backscatter'].mean()) / 
#                               (((ref_pol['backscatter'].std()**2 / len(ref_pol)) + 
#                                 (post_pol['backscatter'].std()**2 / len(post_pol)))**0.5) 
#                               if len(ref_pol) > 1 and len(post_pol) > 1 else 0
#             }
            
#             # Add percent change (handling division by zero)
#             if ref_pol['backscatter'].mean() != 0:
#                 features['percent_change'] = ((post_pol['backscatter'].mean() - ref_pol['backscatter'].mean()) / 
#                                              abs(ref_pol['backscatter'].mean())) * 100
#             else:
#                 features['percent_change'] = 0
                
#             # Add information about orbit if available
#             if 'orbit' in ref_pol.columns:
#                 features['orbit'] = ref_pol['orbit'].iloc[0]
                
#             # Add damage class information if available
#             if 'damage_class' in row and row['is_damaged'] == 1:
#                 features['damage_class'] = row['damage_class'] 
#                 features['damage_class_desc'] = row['damage_class_desc']
                
#             all_features.append(features)
    
#     # Create DataFrame
#     features_df = pd.DataFrame(all_features)
    
#     return features_df

# def extract_time_series_features(backscatter_gdf, pixel_id_col='pixel_id'):
#     """
#     Extract time series features for each unique pixel.
    
#     Parameters:
#     -----------
#     backscatter_gdf : GeoDataFrame
#         Combined GeoDataFrame with backscatter values
#     pixel_id_col : str, optional
#         Name of column containing pixel identifiers
    
#     Returns:
#     --------
#     feature_gdf : GeoDataFrame
#         GeoDataFrame with time series features for each unique pixel
#     """
#     # Get unique pixel IDs
#     pixel_ids = backscatter_gdf[pixel_id_col].unique()
    
#     # Prepare a list to store feature rows
#     feature_data = []
    
#     print(f"Extracting features for {len(pixel_ids)} unique pixels...")
    
#     # Process each unique pixel
#     for pixel_id in tqdm(pixel_ids):
#         # Get data for this pixel
#         pixel_data = backscatter_gdf[backscatter_gdf[pixel_id_col] == pixel_id]
        
#         # Skip if pixel doesn't have both reference and post data
#         if not (pixel_data['period'] == 'reference').any() or not (pixel_data['period'] == 'post').any():
#             continue
        
#         # Get reference period data for VV and VH
#         ref_vv = pixel_data[(pixel_data['period'] == 'reference') & (pixel_data['polarization'] == 'VV')]
#         ref_vh = pixel_data[(pixel_data['period'] == 'reference') & (pixel_data['polarization'] == 'VH')]
        
#         # Get post period data for VV and VH
#         post_vv = pixel_data[(pixel_data['period'] == 'post') & (pixel_data['polarization'] == 'VV')]
#         post_vh = pixel_data[(pixel_data['period'] == 'post') & (pixel_data['polarization'] == 'VH')]
        
#         # Skip if any polarization is missing data
#         if len(ref_vv) == 0 or len(ref_vh) == 0 or len(post_vv) == 0 or len(post_vh) == 0:
#             continue
        
#         # Get representative location (use the mean coordinates)
#         lon = pixel_data['lon'].mean()
#         lat = pixel_data['lat'].mean()
        
#         # Extract time series features
#         features = {
#             'pixel_id': pixel_id,
#             'lon': lon,
#             'lat': lat,
#             'geometry': Point(lon, lat),
            
#             # Reference period statistics - VV
#             'ref_vv_mean': ref_vv['backscatter'].mean(),
#             'ref_vv_std': ref_vv['backscatter'].std(),
#             'ref_vv_min': ref_vv['backscatter'].min(),
#             'ref_vv_max': ref_vv['backscatter'].max(),
#             'ref_vv_range': ref_vv['backscatter'].max() - ref_vv['backscatter'].min(),
#             'ref_vv_count': len(ref_vv),
            
#             # Reference period statistics - VH
#             'ref_vh_mean': ref_vh['backscatter'].mean(),
#             'ref_vh_std': ref_vh['backscatter'].std(),
#             'ref_vh_min': ref_vh['backscatter'].min(),
#             'ref_vh_max': ref_vh['backscatter'].max(), 
#             'ref_vh_range': ref_vh['backscatter'].max() - ref_vh['backscatter'].min(),
#             'ref_vh_count': len(ref_vh),
            
#             # Reference period - cross-polarization ratio (VV/VH) in dB (subtraction)
#             'ref_ratio_mean': ref_vv['backscatter'].mean() - ref_vh['backscatter'].mean(),
            
#             # Post period statistics - VV
#             'post_vv_mean': post_vv['backscatter'].mean(),
#             'post_vv_std': post_vv['backscatter'].std(),
#             'post_vv_min': post_vv['backscatter'].min(),
#             'post_vv_max': post_vv['backscatter'].max(),
#             'post_vv_range': post_vv['backscatter'].max() - post_vv['backscatter'].min(),
#             'post_vv_count': len(post_vv),
            
#             # Post period statistics - VH
#             'post_vh_mean': post_vh['backscatter'].mean(),
#             'post_vh_std': post_vh['backscatter'].std(),
#             'post_vh_min': post_vh['backscatter'].min(),
#             'post_vh_max': post_vh['backscatter'].max(),
#             'post_vh_range': post_vh['backscatter'].max() - post_vh['backscatter'].min(),
#             'post_vh_count': len(post_vh),
            
#             # Post period - cross-polarization ratio (VV/VH) in dB (subtraction)
#             'post_ratio_mean': post_vv['backscatter'].mean() - post_vh['backscatter'].mean(),
            
#             # Change features (post - reference)
#             'vv_change_mean': post_vv['backscatter'].mean() - ref_vv['backscatter'].mean(),
#             'vh_change_mean': post_vh['backscatter'].mean() - ref_vh['backscatter'].mean(),
#             'ratio_change_mean': (post_vv['backscatter'].mean() - post_vh['backscatter'].mean()) - 
#                                (ref_vv['backscatter'].mean() - ref_vh['backscatter'].mean()),
            
#             # Change magnitude (absolute values)
#             'vv_change_magnitude': abs(post_vv['backscatter'].mean() - ref_vv['backscatter'].mean()),
#             'vh_change_magnitude': abs(post_vh['backscatter'].mean() - ref_vh['backscatter'].mean()),
            
#             # Variability changes
#             'vv_std_change': post_vv['backscatter'].std() - ref_vv['backscatter'].std(),
#             'vh_std_change': post_vh['backscatter'].std() - ref_vh['backscatter'].std(),
            
#             # Normalized changes (as percentages of reference values)
#             'vv_percent_change': ((post_vv['backscatter'].mean() - ref_vv['backscatter'].mean()) / 
#                                 abs(ref_vv['backscatter'].mean())) * 100 if ref_vv['backscatter'].mean() != 0 else np.nan,
#             'vh_percent_change': ((post_vh['backscatter'].mean() - ref_vh['backscatter'].mean()) / 
#                                 abs(ref_vh['backscatter'].mean())) * 100 if ref_vh['backscatter'].mean() != 0 else np.nan,
#         }
        
#         feature_data.append(features)
    
#     # Create GeoDataFrame
#     feature_gdf = gpd.GeoDataFrame(feature_data, geometry='geometry')
    
#     # Set CRS (assuming the input GeoDataFrame has a valid CRS)
#     feature_gdf.crs = backscatter_gdf.crs
    
#     print(f"Extracted features for {len(feature_gdf)} pixels")
    
#     return feature_gdf

# def explore_backscatter_evolution(backscatter_gdf, pixel_id, figsize=(14, 8)):
#     """
#     Explore the temporal evolution of backscatter for a specific pixel.
    
#     Parameters:
#     -----------
#     backscatter_gdf : GeoDataFrame
#         Combined GeoDataFrame with backscatter values
#     pixel_id : str
#         ID of the pixel to analyze
#     figsize : tuple, optional
#         Figure size
    
#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         Figure object
#     """
#     # Get data for the specified pixel
#     pixel_data = backscatter_gdf[backscatter_gdf['pixel_id'] == pixel_id].copy()
    
#     if len(pixel_data) == 0:
#         print(f"No data found for pixel ID: {pixel_id}")
#         return None
    
#     # Convert dates to datetime objects for plotting
#     pixel_data['datetime'] = pd.to_datetime(pixel_data['date'], format='%Y%m%d')
    
#     # Split by polarization
#     vv_data = pixel_data[pixel_data['polarization'] == 'VV']
#     vh_data = pixel_data[pixel_data['polarization'] == 'VH']
    
#     # Get the conflict start date
#     conflict_date = pd.to_datetime('20231007', format='%Y%m%d')
    
#     # Create figure
#     fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
#     # Plot VV backscatter
#     vv_data.sort_values('datetime').plot.scatter(
#         x='datetime', y='backscatter', 
#         c=vv_data['orbit'].map({'ASC': 'blue', 'DESC': 'orange'}),
#         ax=axes[0], s=60, alpha=0.7
#     )
#     axes[0].set_title(f'VV Backscatter Evolution for Pixel {pixel_id}')
#     axes[0].set_ylabel('Backscatter (dB)')
#     axes[0].grid(True, alpha=0.3)
    
#     # Plot VH backscatter
#     vh_data.sort_values('datetime').plot.scatter(
#         x='datetime', y='backscatter', 
#         c=vh_data['orbit'].map({'ASC': 'blue', 'DESC': 'orange'}),
#         ax=axes[1], s=60, alpha=0.7
#     )
#     axes[1].set_title('VH Backscatter Evolution')
#     axes[1].set_ylabel('Backscatter (dB)')
#     axes[1].grid(True, alpha=0.3)
    
#     # Calculate and plot VV/VH ratio
#     dates = []
#     ratios = []
#     orbit_types = []
#     periods = []
    
#     # Group by date and orbit
#     for (date, orbit), group in pixel_data.groupby(['date', 'orbit']):
#         vv_val = group[group['polarization'] == 'VV']['backscatter'].values
#         vh_val = group[group['polarization'] == 'VH']['backscatter'].values
        
#         # Only add if we have both polarizations
#         if len(vv_val) > 0 and len(vh_val) > 0:
#             dates.append(pd.to_datetime(date, format='%Y%m%d'))
#             ratios.append(vv_val[0] - vh_val[0])  # Ratio in dB is a subtraction
#             orbit_types.append(orbit)
#             periods.append('reference' if pd.to_datetime(date, format='%Y%m%d') < conflict_date else 'post')
    
#     # Create a DataFrame for the ratio
#     ratio_df = pd.DataFrame({
#         'datetime': dates,
#         'ratio': ratios,
#         'orbit': orbit_types,
#         'period': periods
#     })
    
#     # Plot the ratio
#     ratio_df.sort_values('datetime').plot.scatter(
#         x='datetime', y='ratio', 
#         c=ratio_df['orbit'].map({'ASC': 'blue', 'DESC': 'orange'}),
#         ax=axes[2], s=60, alpha=0.7
#     )
#     axes[2].set_title('VV/VH Ratio Evolution (dB)')
#     axes[2].set_ylabel('VV/VH Ratio (dB)')
#     axes[2].set_xlabel('Date')
#     axes[2].grid(True, alpha=0.3)
    
#     # Add a vertical line for the conflict start date
#     for ax in axes:
#         ax.axvline(x=conflict_date, color='red', linestyle='--', alpha=0.7)
#         ax.text(conflict_date, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
#                 'Conflict Start', rotation=90, color='red', ha='right')
        
#         # Add a legend
#         if ax == axes[0]:
#             ax.legend(['', 'Conflict Start', 'Ascending Orbit', 'Descending Orbit'])
    
#     plt.tight_layout()
    
#     # Calculate and print statistics
#     print(f"Pixel ID: {pixel_id}")
#     print(f"Location: Lon {pixel_data['lon'].mean():.6f}, Lat {pixel_data['lat'].mean():.6f}")
#     print("\nBackscatter Statistics:")
    
#     ref_vv = vv_data[vv_data['period'] == 'reference']['backscatter']
#     ref_vh = vh_data[vh_data['period'] == 'reference']['backscatter']
#     post_vv = vv_data[vv_data['period'] == 'post']['backscatter']
#     post_vh = vh_data[vh_data['period'] == 'post']['backscatter']
    
#     print(f"Reference VV: Mean = {ref_vv.mean():.2f} dB, Std = {ref_vv.std():.2f} dB, Count = {len(ref_vv)}")
#     print(f"Reference VH: Mean = {ref_vh.mean():.2f} dB, Std = {ref_vh.std():.2f} dB, Count = {len(ref_vh)}")
#     print(f"Post VV: Mean = {post_vv.mean():.2f} dB, Std = {post_vv.std():.2f} dB, Count = {len(post_vv)}")
#     print(f"Post VH: Mean = {post_vh.mean():.2f} dB, Std = {post_vh.std():.2f} dB, Count = {len(post_vh)}")
    
#     # Calculate changes
#     vv_change = post_vv.mean() - ref_vv.mean()
#     vh_change = post_vh.mean() - ref_vh.mean()
#     ratio_change = (post_vv.mean() - post_vh.mean()) - (ref_vv.mean() - ref_vh.mean())
    
#     print("\nChanges:")
#     print(f"VV Change: {vv_change:.2f} dB ({(vv_change/abs(ref_vv.mean()))*100:.1f}%)")
#     print(f"VH Change: {vh_change:.2f} dB ({(vh_change/abs(ref_vh.mean()))*100:.1f}%)")
#     print(f"VV/VH Ratio Change: {ratio_change:.2f} dB")
    
#     return fig

# def visualize_backscatter_changes(feature_gdf, figsize=(18, 12), cmap='RdBu_r', center_zero=True):
#     """
#     Visualize backscatter changes on a map.
    
#     Parameters:
#     -----------
#     feature_gdf : GeoDataFrame
#         GeoDataFrame with time series features
#     figsize : tuple, optional
#         Figure size
#     cmap : str, optional
#         Colormap to use
#     center_zero : bool, optional
#         Whether to center the colormap at zero
    
#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         Figure object
#     """
#     # Create a figure with 2x2 subplots
#     fig, axes = plt.subplots(2, 2, figsize=figsize)
#     axes = axes.flatten()
    
#     # Define variables to plot
#     variables = [
#         ('vv_change_mean', 'VV Backscatter Change (dB)'),
#         ('vh_change_mean', 'VH Backscatter Change (dB)'),
#         ('ratio_change_mean', 'VV/VH Ratio Change (dB)'),
#         ('vv_change_magnitude', 'VV Backscatter Change Magnitude (dB)')
#     ]
    
#     # Plot each variable
#     for i, (var, title) in enumerate(variables):
#         ax = axes[i]
        
#         # Determine vmin and vmax
#         if center_zero and var != 'vv_change_magnitude':
#             max_abs = max(abs(feature_gdf[var].max()), abs(feature_gdf[var].min()))
#             vmin, vmax = -max_abs, max_abs
#             cmap_use = cmap
#         else:
#             vmin, vmax = feature_gdf[var].min(), feature_gdf[var].max()
#             cmap_use = 'viridis' if var == 'vv_change_magnitude' else cmap
        
#         # Plot
#         feature_gdf.plot(
#             column=var,
#             cmap=cmap_use,
#             legend=True,
#             ax=ax,
#             alpha=0.7,
#             vmin=vmin,
#             vmax=vmax,
#             legend_kwds={'label': var, 'orientation': 'horizontal'}
#         )
        
#         ax.set_title(title)
#         ax.set_xlabel('Longitude')
#         ax.set_ylabel('Latitude')
#         ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     return fig