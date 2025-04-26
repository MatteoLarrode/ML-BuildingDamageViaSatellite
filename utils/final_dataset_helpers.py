def create_binary_dataset_for_ml(damage_labels, backscatter_gdf, buffer_distance=10,
                                reference_dates=None, assessment_dates=None,
                                orbit_directions=['ASC', 'DESC']):
    """
    Create a dataset for binary classification of damage, following the approach in the study.
    
    Parameters:
    -----------
    damage_labels : GeoDataFrame
        Processed damage labels from UNOSAT
    backscatter_gdf : GeoDataFrame
        Backscatter values from Sentinel-1
    buffer_distance : float, default=10
        Buffer distance in meters around damage points for spatial join
    reference_dates : list, optional
        List of dates for the reference period (pre-conflict)
    assessment_dates : list, optional
        List of dates for the assessment period (post-conflict)
    orbit_directions : list, default=['ASC', 'DESC']
        Orbit directions to process separately
        
    Returns:
    --------
    dict
        Dictionary with datasets for each orbit direction
    """
    import warnings
    warnings.filterwarnings('ignore', 'GeoSeries.isna', FutureWarning)
    
    print("Creating binary dataset for machine learning...")
    
    # Convert damage labels to the same CRS as backscatter data if needed
    if damage_labels.crs != backscatter_gdf.crs:
        damage_labels = damage_labels.to_crs(backscatter_gdf.crs)
    
    # Create buffers around damage points to account for positional uncertainty
    print("Creating buffers around damage points...")
    damage_with_buffers = damage_labels.copy()
    
    # If CRS is not projected, convert to UTM for accurate buffering
    if damage_with_buffers.crs.is_geographic:
        # Convert to a projected CRS (UTM zone appropriate for Gaza)
        damage_with_buffers = damage_with_buffers.to_crs("EPSG:32636")  # UTM zone 36N
        
    # Create buffer geometries
    damage_with_buffers['geometry'] = damage_with_buffers.geometry.buffer(buffer_distance)
    
    # Convert back to original CRS
    if damage_labels.crs.is_geographic:
        damage_with_buffers = damage_with_buffers.to_crs(damage_labels.crs)
    
    # If reference_dates and assessment_dates are not provided, try to infer them
    all_dates = backscatter_gdf['date'].unique()
    all_dates.sort()
    
    if reference_dates is None:
        # Use dates before 2023-10-07 (conflict start) as reference
        reference_dates = [d for d in all_dates if d < '20231007']
        print(f"Inferred reference dates: {reference_dates}")
    
    if assessment_dates is None:
        # Use dates on or after 2023-10-07 as assessment
        assessment_dates = [d for d in all_dates if d >= '20231007']
        print(f"Inferred assessment dates: {assessment_dates}")
    
    # Create datasets for each orbit direction
    orbit_datasets = {}
    
    for orbit in orbit_directions:
        print(f"\nProcessing orbit direction: {orbit}")
        
        # Filter backscatter data by orbit
        orbit_data = backscatter_gdf[backscatter_gdf['orbit'] == orbit].copy()
        
        if len(orbit_data) == 0:
            print(f"No data found for orbit {orbit}")
            continue
        
        # Perform spatial join to add damage labels to backscatter data
        print("Performing spatial join...")
        joined_data = gpd.sjoin(orbit_data, damage_with_buffers, how='inner', predicate='within')
        
        # If no matches found, try with a larger buffer
        if len(joined_data) == 0:
            print(f"No spatial matches found for orbit {orbit}. Try with a larger buffer distance.")
            continue
        
        print(f"Found {len(joined_data)} spatial matches")
        
        # Group by pixel to extract time series
        pixel_groups = joined_data.groupby(['pixel_id', 'polarization'])
        
        print(f"Processing {len(pixel_groups)} unique pixel/polarization combinations...")
        
        # Prepare dataset for this orbit
        ml_data = []
        
        for (pixel_id, pol), group in pixel_groups:
            # Get the damage label for this pixel
            is_damaged = group['is_damaged'].iloc[0]
            damage_class = group['damage_class'].iloc[0]
            damage_desc = group['damage_class_desc'].iloc[0]
            
            # Filter to reference and assessment periods
            ref_data = group[group['date'].isin(reference_dates)]
            assessment_data = group[group['date'].isin(assessment_dates)]
            
            # Skip if we don't have enough data in both periods
            if len(ref_data) == 0 or len(assessment_data) == 0:
                continue
            
            # Calculate statistics for reference period
            ref_stats = {
                'ref_mean': ref_data['backscatter'].mean(),
                'ref_std': ref_data['backscatter'].std(),
                'ref_min': ref_data['backscatter'].min(),
                'ref_max': ref_data['backscatter'].max(),
                'ref_range': ref_data['backscatter'].max() - ref_data['backscatter'].min(),
                'ref_count': len(ref_data)
            }
            
            # Calculate statistics for assessment period
            assessment_stats = {
                'assessment_mean': assessment_data['backscatter'].mean(),
                'assessment_std': assessment_data['backscatter'].std(),
                'assessment_min': assessment_data['backscatter'].min(),
                'assessment_max': assessment_data['backscatter'].max(),
                'assessment_range': assessment_data['backscatter'].max() - assessment_data['backscatter'].min(),
                'assessment_count': len(assessment_data)
            }
            
            # Calculate change features
            change_stats = {
                'change_mean': assessment_stats['assessment_mean'] - ref_stats['ref_mean'],
                'change_std': assessment_stats['assessment_std'] - ref_stats['ref_std'],
                'change_range': assessment_stats['assessment_range'] - ref_stats['ref_range'],
                'change_magnitude': abs(assessment_stats['assessment_mean'] - ref_stats['ref_mean']),
                'percent_change': ((assessment_stats['assessment_mean'] - ref_stats['ref_mean']) / 
                                  abs(ref_stats['ref_mean'])) * 100 if ref_stats['ref_mean'] != 0 else 0
            }
            
            # Combine all features
            ml_record = {
                'pixel_id': pixel_id,
                'polarization': pol,
                'orbit': orbit,
                'lon': group['lon'].iloc[0],
                'lat': group['lat'].iloc[0],
                'is_damaged': is_damaged,
                'damage_class': damage_class,
                'damage_class_desc': damage_desc,
                **ref_stats,
                **assessment_stats,
                **change_stats
            }
            
            ml_data.append(ml_record)
        
        # Create DataFrame for this orbit
        orbit_df = pd.DataFrame(ml_data)
        
        if len(orbit_df) > 0:
            print(f"Created dataset with {len(orbit_df)} samples for orbit {orbit}")
            print(f"Class distribution: {orbit_df['is_damaged'].value_counts().to_dict()}")
            orbit_datasets[orbit] = orbit_df
        else:
            print(f"No valid samples created for orbit {orbit}")
    
    # Combine datasets if requested
    if len(orbit_datasets) > 0:
        all_data = pd.concat([df for df in orbit_datasets.values()])
        print(f"\nCreated combined dataset with {len(all_data)} samples across all orbits")
        print(f"Overall class distribution: {all_data['is_damaged'].value_counts().to_dict()}")
        orbit_datasets['all'] = all_data
    
    return orbit_datasets