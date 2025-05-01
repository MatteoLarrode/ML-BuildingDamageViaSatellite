import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import re
from IPython.display import display, Markdown
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def set_visualization_style():
    plt.style.use('seaborn-v0_8-colorblind')
    font_path = '/Users/matteolarrode/Library/Fonts/cmunss.ttf'
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams.update({
        'text.usetex': False,
        #'font.family': 'serif',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 1.5,
        'lines.markersize': 8,
        'figure.figsize': (10, 6),
        'axes.grid': False, 
        'axes.spines.top': False,  # Remove top spine
        'axes.spines.right': False,  # Remove right spine
        # Add this line to use ASCII hyphen instead of Unicode minus
        'axes.unicode_minus': False
    })

def plot_backscatter_timeseries_by_coordinates(timeseries, lat, lon, radius_meters=20, max_points=2):
    """
    Plot backscatter time series for points closest to the given coordinates
    
    Parameters:
    -----------
    timeseries : DataFrame
        The time series data
    lon : float
        Longitude coordinate
    lat : float
        Latitude coordinate
    radius_meters : float
        Search radius in meters to find nearby points
    max_points : int
        Maximum number of points to plot
    """
    set_visualization_style()

    # Get unique points with their coordinates
    unique_points = timeseries[['point_id', 'point_lon', 'point_lat', 'is_damaged']].drop_duplicates()
    
    # Calculate distances from input coordinates
    # Convert degrees to approximate meters (very rough conversion near equator)
    # 1 degree ≈ 111km at equator
    meter_per_degree = 111000
    dlon = (unique_points['point_lon'] - lon) * meter_per_degree * np.cos(np.radians(lat))
    dlat = (unique_points['point_lat'] - lat) * meter_per_degree
    unique_points['distance_meters'] = np.sqrt(dlon**2 + dlat**2)
    
    # Filter by radius and sort by distance
    nearby_points = unique_points[unique_points['distance_meters'] <= radius_meters].sort_values('distance_meters')
    
    if len(nearby_points) == 0:
        print(f"No points found within {radius_meters}m of coordinates ({lon}, {lat})")
        # Find closest point instead
        closest_point = unique_points.loc[unique_points['distance_meters'].idxmin()]
        print(f"Showing closest point instead, {closest_point['distance_meters']:.1f}m away")
        nearby_points = unique_points.iloc[[unique_points['distance_meters'].idxmin()]]
    
    # Limit to max_points
    nearby_points = nearby_points.head(max_points)
    
    # Print selected points information
    print(f"Selected {len(nearby_points)} points:")
    for _, point in nearby_points.iterrows():
        damage_status = "Damaged" if point['is_damaged'] == 1 else "Undamaged"
        print(f"Point {point['point_id']}: {damage_status}, {point['distance_meters']:.1f}m away")
    
    # Use existing function to plot these points
    point_ids = nearby_points['point_id'].tolist()
    
    # Create figure with subplots - one col per point, two columns for two orbits
    fig, axes = plt.subplots(2, len(point_ids), figsize=(8*len(point_ids), 7))
    if len(point_ids) == 1:
        axes = np.array([axes])  # Ensure axes is 2D
    
    # Colors for polarizations
    vv_color = '#151515'  # Red for VV
    vh_color = '#949494'  # Green for VH
    
    # Process each point
    for i, point_id in enumerate(point_ids):
        # Get data for this point
        point_data = timeseries[timeseries['point_id'] == point_id]
        
        # Get damage status
        is_damaged = point_data['is_damaged'].iloc[0]
        damage_desc = "Damaged" if is_damaged == 1 else "Undamaged"
        
        # Get unique orbits for this point
        orbits = point_data['orbit'].unique()
        
        # Sort orbits to ensure consistent order (ASC first, DESC second if both present)
        orbits = sorted(orbits)
        
        # Plot each orbit in a separate subplot
        for j, orbit in enumerate(orbits[:2]):  # Limit to 2 orbits
            # Skip if we have more orbits than columns
            if j >= 2:
                continue
                
            orbit_data = point_data[point_data['orbit'] == orbit]
            
            # Set title for subplot
            axes[i, j].set_title(f"Orbit {orbit}")
            
            # Plot vertical lines
            conflict_date = pd.to_datetime('2023-10-07')
            # Black dashed line for conflict start
            axes[i, j].axvline(x=conflict_date, color='k', linestyle='--')
            # Red dashed line for damage assessment date (if available)
            if 'date' in point_data.columns and not pd.isna(point_data['date'].iloc[0]):
                assessment_date = pd.to_datetime(point_data['date'].iloc[0])
                axes[i, j].axvline(x=assessment_date, color='r', linestyle='--')
            
            # Plot VV polarization
            vv_data = orbit_data[orbit_data['polarization'] == 'VV']
            vv_data = vv_data.sort_values('backscatter_date')
            if len(vv_data) > 0:
                axes[i, j].plot(vv_data['backscatter_date'], vv_data['backscatter'], 
                               marker='.', linestyle='-', label='VV', color=vv_color)
            
            # Plot VH polarization
            vh_data = orbit_data[orbit_data['polarization'] == 'VH']
            vh_data = vh_data.sort_values('backscatter_date')
            if len(vh_data) > 0:
                axes[i, j].plot(vh_data['backscatter_date'], vh_data['backscatter'], 
                               marker='.', linestyle='-', label='VH', color=vh_color)
            
            # Add grid and legend
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].legend()
            axes[i, j].set_ylabel('Backscatter (dB)')
            
            # Format x-axis
            axes[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axes[i, j].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(axes[i, j].xaxis.get_majorticklabels(), rotation=45)
    
    # Add a suptitle with the coordinates
    plt.suptitle(f"Backscatter Time Series near ({lon:.5f}, {lat:.5f})", fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the suptitle
    plt.show()
    
    return

    
def plot_backscatter_timeseries_by_id(timeseries, point_ids, max_points=2, save_path=None):
    """
    Plot backscatter time series for selected points, separating orbits
    
    Parameters:
    -----------
    timeseries : DataFrame
        The time series data
    point_ids : list
        List of point_ids to plot
    max_points : int
        Maximum number of points to plot
    """
    set_visualization_style()

    # Limit to manageable number of points
    if len(point_ids) > max_points:
        point_ids = point_ids[:max_points]
    
    # Create figure with subplots - one row per point, two columns for two orbits
    fig, axes = plt.subplots(2, len(point_ids), figsize=(8*len(point_ids), 7))
    if len(point_ids) == 1:
        axes = np.array([axes])  # Ensure axes is 2D
    
    # Colors for polarizations
    vv_color = '#151515'  
    vh_color = '#949494'

    # Mapping of orbit numbers to descriptions
    orbit_descriptions = {
        'ASC': "Orbit 43 (ASCENDING)",
        'DESC': "Orbit 94 (DESCENDING)"
        }
    
    # Process each point
    for i, point_id in enumerate(point_ids):
        # Get data for this point
        point_data = timeseries[timeseries['point_id'] == point_id]
        
        # Get damage status
        is_damaged = point_data['is_damaged'].iloc[0]
        damage_desc = "Damaged" if is_damaged == 1 else "Undamaged"
        
        # Get unique orbits for this point
        orbits = point_data['orbit'].unique()
        
        # Sort orbits to ensure consistent order (ASC first, DESC second if both present)
        orbits = sorted(orbits)
        
        # Plot each orbit in a separate subplot
        for j, orbit in enumerate(orbits[:2]):
            orbit_data = point_data[point_data['orbit'] == orbit]
            
            # Set title for subplot
            orbit_title = orbit_descriptions.get(orbit, f"Orbit {orbit}")
            axes[i, j].set_title(orbit_title)
            
            # Plot vertical lines
            conflict_date = pd.to_datetime('2023-10-07')
            prior_date = pd.to_datetime('2023-08-10')
            later_date = pd.to_datetime('2024-01-20')
            # Black dashed line for conflict start
            axes[i, j].axvline(x=conflict_date, color='k', linestyle='--')
            # Dashed lines for prior and later dates
            axes[i, j].axvline(x=prior_date, color='#149954', linestyle='--')
            axes[i, j].axvline(x=later_date, color='#E4312b', linestyle='--')
            
            # Plot VV polarization
            vv_data = orbit_data[orbit_data['polarization'] == 'VV']
            vv_data = vv_data.sort_values('backscatter_date')
            if len(vv_data) > 0:
                axes[i, j].plot(vv_data['backscatter_date'], vv_data['backscatter'], 
                               marker='.', linestyle='-', label='VV', color=vv_color)
            
            # Plot VH polarization
            vh_data = orbit_data[orbit_data['polarization'] == 'VH']
            vh_data = vh_data.sort_values('backscatter_date')
            if len(vh_data) > 0:
                axes[i, j].plot(vh_data['backscatter_date'], vh_data['backscatter'], 
                               marker='.', linestyle='-', label='VH', color=vh_color)
            
            # Add grid and legend
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].legend()
            axes[i, j].set_ylabel('Backscatter (dB)')
            
            # Format x-axis
            axes[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axes[i, j].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(axes[i, j].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()

    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()
    
    return

def create_desc_stats(df):
    # Group features
    polarizations = ['VH', 'VV']
    orbits = ['ASC', 'DESC']
    periods = ['ref', 'post']
    stats = ['mean', 'median', 'min', 'max', 'std']
    
    # Initialize columns for the table
    columns = ['Orbit', 'Polarization', 'Statistic']
    for period in periods:
        columns.append(f'Undamaged {period.capitalize()} (mean ± std)')
        columns.append(f'Damaged {period.capitalize()} (mean ± std)')
    
    # Create empty lists for each column
    data = {col: [] for col in columns}
    
    # Fill the table
    for orbit in orbits:
        for pol in polarizations:
            for stat in stats:
                # Add identifiers
                data['Orbit'].append(orbit)
                data['Polarization'].append(pol)
                data['Statistic'].append(stat)
                
                # Add values for each period and damage status
                for period in periods:
                    feature = f"{pol}_{orbit}_{period}_{stat}"
                    
                    # For undamaged
                    undamaged_mean = df[df['is_damaged'] == 0][feature].mean()
                    undamaged_std = df[df['is_damaged'] == 0][feature].std()
                    data[f'Undamaged {period.capitalize()} (mean ± std)'].append(
                        f"{undamaged_mean:.4f} ± {undamaged_std:.4f}")
                    
                    # For damaged
                    damaged_mean = df[df['is_damaged'] == 1][feature].mean()
                    damaged_std = df[df['is_damaged'] == 1][feature].std()
                    data[f'Damaged {period.capitalize()} (mean ± std)'].append(
                        f"{damaged_mean:.4f} ± {damaged_std:.4f}")
    
    # Create DataFrame
    stats_df = pd.DataFrame(data)
    
    # Add sample size information at the top
    n_undamaged = df[df['is_damaged'] == 0].shape[0]
    n_damaged = df[df['is_damaged'] == 1].shape[0]
    sample_info = pd.DataFrame({
        'Orbit': ['Sample Size'],
        'Polarization': [''],
        'Statistic': ['Count'],
        'Undamaged Ref (mean ± std)': [f"{n_undamaged}"],
        'Damaged Ref (mean ± std)': [f"{n_damaged}"],
        'Undamaged Post (mean ± std)': [f"{n_undamaged}"],
        'Damaged Post (mean ± std)': [f"{n_damaged}"]
    })
    
    # Combine and return
    final_df = pd.concat([sample_info, stats_df], ignore_index=True)
    return final_df

def create_latex_descriptive_table(df):
    # Prepare data
    orbits = ['ASC', 'DESC']
    polarizations = ['VH', 'VV']
    stats = ['mean', 'median', 'min', 'max', 'std']
    periods = ['ref', 'post']
    
    # Start building the LaTeX table
    latex_table = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Descriptive statistics of SAR features by damage status}",
        "\\label{tab:desc-stats}",
        "\\small",
        "\\begin{tabular}{llc|cc|cc}",
        "\\toprule",
        " & & & \\multicolumn{2}{c|}{\\textbf{Undamaged}} & \\multicolumn{2}{c}{\\textbf{Damaged}} \\\\",
        "\\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
        "\\textbf{Orbit} & \\textbf{Polarization} & \\textbf{Statistic} & \\textbf{Reference} & \\textbf{Post} & \\textbf{Reference} & \\textbf{Post} \\\\"
    ]
    
    # Sample size row
    n_undamaged = df[df['is_damaged'] == 0].shape[0]
    n_damaged = df[df['is_damaged'] == 1].shape[0]
    latex_table.append("\\midrule")
    latex_table.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{Sample Size}}}} & {n_undamaged} & {n_undamaged} & {n_damaged} & {n_damaged} \\\\")
    
    # Add data rows
    for orbit in orbits:
        # Add orbit header
        latex_table.append("\\midrule")
        latex_table.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{orbit} orbit}}}} \\\\")
        
        for pol in polarizations:
            # Add polarization header
            latex_table.append("\\midrule")
            latex_table.append(f"\\multicolumn{{1}}{{l}}{{}} & \\multicolumn{{6}}{{l}}{{\\textbf{{{pol} polarization}}}} \\\\")
            
            for stat in stats:
                # Get values for each combination
                ref_undamaged_mean = df[(df['is_damaged'] == 0)][f"{pol}_{orbit}_ref_{stat}"].mean()
                ref_undamaged_std = df[(df['is_damaged'] == 0)][f"{pol}_{orbit}_ref_{stat}"].std()
                post_undamaged_mean = df[(df['is_damaged'] == 0)][f"{pol}_{orbit}_post_{stat}"].mean()
                post_undamaged_std = df[(df['is_damaged'] == 0)][f"{pol}_{orbit}_post_{stat}"].std()
                
                ref_damaged_mean = df[(df['is_damaged'] == 1)][f"{pol}_{orbit}_ref_{stat}"].mean()
                ref_damaged_std = df[(df['is_damaged'] == 1)][f"{pol}_{orbit}_ref_{stat}"].std()
                post_damaged_mean = df[(df['is_damaged'] == 1)][f"{pol}_{orbit}_post_{stat}"].mean()
                post_damaged_std = df[(df['is_damaged'] == 1)][f"{pol}_{orbit}_post_{stat}"].std()
                
                # Format the values
                ref_undamaged = f"{ref_undamaged_mean:.4f} $\\pm$ {ref_undamaged_std:.4f}"
                post_undamaged = f"{post_undamaged_mean:.4f} $\\pm$ {post_undamaged_std:.4f}"
                ref_damaged = f"{ref_damaged_mean:.4f} $\\pm$ {ref_damaged_std:.4f}"
                post_damaged = f"{post_damaged_mean:.4f} $\\pm$ {post_damaged_std:.4f}"
                
                # Add row
                latex_table.append(f" & & {stat} & {ref_undamaged} & {post_undamaged} & {ref_damaged} & {post_damaged} \\\\")
    
    # Close the table
    latex_table.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_table)

def plot_performance_curves(y_true, y_pred_proba, save_path=None):
   """
   Plot ROC curve and Precision-Recall curve side by side
   
   Parameters:
   -----------
   y_true : array-like
       True binary labels
   y_pred_proba : array-like
       Predicted probabilities for the positive class
   save_path : str, optional
       Path to save the figure
   """
   set_visualization_style()

   # Set up figure
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   
   # Plot ROC curve
   fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
   roc_auc = auc(fpr, tpr)
   
   axes[0].plot(fpr, tpr, color='blue', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
   axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
   axes[0].set_xlim([0.0, 1.0])
   axes[0].set_ylim([0.0, 1.05])
   axes[0].set_xlabel('False Positive Rate')
   axes[0].set_ylabel('True Positive Rate')
   axes[0].set_title('ROC Curve')
   axes[0].legend(loc="lower right")
   axes[0].grid(True, alpha=0.3)
   
   # Plot Precision-Recall curve
   precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
   pr_auc = auc(recall, precision)
   
   axes[1].plot(recall, precision, color='green', lw=2,
                label=f'PR curve (AP = {pr_auc:.2f})')
   # Add baseline based on class balance
   baseline = sum(y_true) / len(y_true)
   axes[1].plot([0, 1], [baseline, baseline], color='gray', linestyle='--')
   
   axes[1].set_xlim([0.0, 1.0])
   axes[1].set_ylim([0.0, 1.05])
   axes[1].set_xlabel('Recall')
   axes[1].set_ylabel('Precision')
   axes[1].set_title('Precision-Recall Curve')
   axes[1].legend(loc="lower left")
   axes[1].grid(True, alpha=0.3)
   
   plt.tight_layout()
   
   if save_path:
       plt.savefig(save_path, dpi=300, bbox_inches='tight')
   
   return fig, axes

def plot_metrics_vs_threshold(y_true, y_pred_proba, save_path=None):
   """
   Plot evaluation metrics vs decision threshold
   
   Parameters:
   -----------
   y_true : array-like
       True binary labels
   y_pred_proba : array-like
       Predicted probabilities for the positive class
   save_path : str, optional
       Path to save the figure
   """
   set_visualization_style()
   # Define thresholds to evaluate (from 0.1 to 0.9 as requested)
   thresholds = np.linspace(0.1, 0.9, 101)
   
   # Store metrics
   precisions = []
   recalls = []
   f1s = []
   accuracies = []
   
   # Calculate metrics for each threshold
   for threshold in thresholds:
       y_pred_thresh = (y_pred_proba >= threshold).astype(int)
       precisions.append(precision_score(y_true, y_pred_thresh, zero_division=0))
       recalls.append(recall_score(y_true, y_pred_thresh))
       f1s.append(f1_score(y_true, y_pred_thresh))
       accuracies.append(accuracy_score(y_true, y_pred_thresh))
   
   # Create figure
   fig, ax = plt.subplots(figsize=(10, 5))
   
   # Plot metrics
   ax.plot(thresholds, precisions, label='Precision', linewidth=2)
   ax.plot(thresholds, recalls, label='Recall', linewidth=2)
   ax.plot(thresholds, f1s, label='F1 Score', linewidth=2)
   ax.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
   
   # Add vertical line for default threshold
   ax.axvline(x=0.5, color='grey', linestyle='--', label='Threshold = 0.5')
   
   # Formatting
   ax.set_xlabel('Decision Threshold', fontsize=12)
   ax.set_ylabel('Score', fontsize=12)
   ax.set_xlim([0.1, 0.9])
   ax.set_ylim([0, 1.05])
   ax.legend(loc='best', fontsize=11)
   ax.grid(True, alpha=0.3)
   
   plt.tight_layout()
   
   if save_path:
       plt.savefig(save_path, dpi=300, bbox_inches='tight')
   
   return fig, ax
