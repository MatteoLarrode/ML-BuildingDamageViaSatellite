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
