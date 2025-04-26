import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from shapely.geometry import Point
from tqdm.notebook import tqdm

def evaluate_with_3x3_window(true_labels, predictions, label_coords_cols=['point_lon', 'point_lat'], 
                           pred_coords_cols=['pixel_lon', 'pixel_lat']):
    """
    Evaluate predictions using a 3x3 window around each label as mentioned in the paper.
    
    Parameters:
    -----------
    true_labels : DataFrame
        DataFrame with true damage labels
    predictions : DataFrame
        DataFrame with pixel-wise predictions
    label_coords_cols : list, default=['point_lon', 'point_lat']
        Columns in true_labels containing coordinates
    pred_coords_cols : list, default=['pixel_lon', 'pixel_lat']
        Columns in predictions containing coordinates
        
    Returns:
    --------
    DataFrame with evaluation results
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
    
    # Convert to GeoDataFrames if they're not already
    if not isinstance(true_labels, gpd.GeoDataFrame):
        true_labels = true_labels.copy()
        true_labels['geometry'] = [Point(x, y) for x, y in zip(true_labels[label_coords_cols[0]], 
                                                             true_labels[label_coords_cols[1]])]
        true_labels = gpd.GeoDataFrame(true_labels, geometry='geometry')
    
    if not isinstance(predictions, gpd.GeoDataFrame):
        predictions = predictions.copy()
        predictions['geometry'] = [Point(x, y) for x, y in zip(predictions[pred_coords_cols[0]], 
                                                             predictions[pred_coords_cols[1]])]
        predictions = gpd.GeoDataFrame(predictions, geometry='geometry')
    
    # Ensure CRS match
    if true_labels.crs != predictions.crs and true_labels.crs is not None and predictions.crs is not None:
        predictions = predictions.to_crs(true_labels.crs)
    
    # For each true label, find predictions within a 3x3 pixel window (approximately 30m)
    window_size = 30  # 3 pixels at 10m resolution
    
    results = []
    
    for idx, label in true_labels.iterrows():
        # Create buffer around label point
        if true_labels.crs.is_geographic:
            # Convert to UTM for accurate buffering
            label_utm = gpd.GeoDataFrame([label], geometry=[label.geometry], crs=true_labels.crs)
            label_utm = label_utm.to_crs("EPSG:32636")  # UTM zone 36N
            buffer = label_utm.geometry.buffer(window_size).iloc[0]
            buffer = gpd.GeoDataFrame(geometry=[buffer], crs="EPSG:32636").to_crs(true_labels.crs).geometry.iloc[0]
        else:
            buffer = label.geometry.buffer(window_size)
        
        # Find predictions within the buffer
        nearby_preds = predictions[predictions.geometry.intersects(buffer)]
        
        # If no predictions in buffer, skip
        if len(nearby_preds) == 0:
            continue
        
        # Get the maximum predicted probability in the window
        if 'damage_prob' in nearby_preds.columns:
            max_prob = nearby_preds['damage_prob'].max()
        elif 'damage_prob_combined' in nearby_preds.columns:
            max_prob = nearby_preds['damage_prob_combined'].max()
        else:
            # Try to find any column that might contain probabilities
            prob_cols = [col for col in nearby_preds.columns if 'prob' in col.lower()]
            if prob_cols:
                max_prob = nearby_preds[prob_cols[0]].max()
            else:
                continue
        
        # Record result
        result = {
            'true_label': label['is_damaged'] if 'is_damaged' in label else None,
            'max_prob_in_window': max_prob,
            'num_pixels_in_window': len(nearby_preds)
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics if we have true labels
    if 'true_label' in results_df.columns and not results_df['true_label'].isna().all():
        try:
            # Precision-recall AUC
            average_precision = average_precision_score(results_df['true_label'], results_df['max_prob_in_window'])
            # ROC AUC
            roc_auc = roc_auc_score(results_df['true_label'], results_df['max_prob_in_window'])
            
            print(f"Evaluation metrics using 3x3 window:")
            print(f"  Precision-Recall AUC: {average_precision:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")
            
            # Add to results DataFrame
            results_df = results_df.assign(
                average_precision=average_precision,
                roc_auc=roc_auc
            )
        except:
            print("Could not calculate metrics. Make sure scikit-learn is installed.")
    
    return results_df