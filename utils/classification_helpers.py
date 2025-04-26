def combine_orbit_predictions(pixel_predictions):
    """
    Combine predictions from different orbits as specified in the paper.
    At inference time, compute the overall damage probability map by averaging estimates from different orbits.
    
    Parameters:
    -----------
    pixel_predictions : DataFrame
        DataFrame with predictions for each pixel/polarization/orbit
        Must have columns: 'pixel_id', 'orbit', 'damage_prob'
        
    Returns:
    --------
    DataFrame with combined predictions for each pixel
    """
    # Group by pixel and average predictions
    combined = pixel_predictions.groupby('pixel_id')['damage_prob'].mean().reset_index()
    combined.rename(columns={'damage_prob': 'damage_prob_combined'}, inplace=True)
    
    # Add other useful information from the original predictions
    # Use the most common value for categorical variables
    for col in ['point_id', 'point_lon', 'point_lat', 'is_damaged', 'damage_class']:
        if col in pixel_predictions.columns:
            mode_values = pixel_predictions.groupby('pixel_id')[col].agg(
                lambda x: x.mode()[0] if not x.mode().empty else None
            ).reset_index()
            combined = combined.merge(mode_values, on='pixel_id')
    
    return combined