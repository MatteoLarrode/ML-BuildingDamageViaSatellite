import folium
import geopandas as gpd
import shapely.wkt
import json
import pandas as pd
import numpy as np

def parse_geo_column(df):
    """
    Parse the .geo column from a dataframe and convert it to a GeoDataFrame.
    
    The .geo column can be in various formats:
    - GeoJSON-like string
    - WKT string
    
    This function attempts to identify the format and convert accordingly.
    
    Args:
        df: DataFrame with a .geo column
        
    Returns:
        GeoDataFrame with proper geometry objects
    """
    if '.geo' not in df.columns:
        print("No .geo column found in the dataframe.")
        return None
    
    try:
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check if we're dealing with GeoJSON-like strings
        sample = df_copy['.geo'].iloc[0]
        
        if isinstance(sample, str) and sample.startswith('{'):
            # Handle GeoJSON-like strings
            geometries = []
            for geo_str in df_copy['.geo']:
                try:
                    geo_dict = json.loads(geo_str)
                    if 'geometries' in geo_dict:
                        # MultiGeometry case
                        geom_list = []
                        for g in geo_dict['geometries']:
                            if g['type'] == 'Point':
                                x, y = g['coordinates']
                                geom_list.append(shapely.geometry.Point(x, y))
                        geometry = shapely.geometry.MultiPoint(geom_list)
                    else:
                        # Single geometry case
                        if geo_dict['type'] == 'Point':
                            x, y = geo_dict['coordinates']
                            geometry = shapely.geometry.Point(x, y)
                        elif geo_dict['type'] == 'Polygon':
                            coords = geo_dict['coordinates'][0]  # Outer ring
                            geometry = shapely.geometry.Polygon(coords)
                    geometries.append(geometry)
                except Exception as e:
                    print(f"Error parsing geometry: {e}")
                    geometries.append(None)
            
            # Create GeoDataFrame
            df_copy['geometry'] = geometries
            gdf = gpd.GeoDataFrame(df_copy, geometry='geometry', crs="EPSG:4326")
            return gdf
        
        elif isinstance(sample, str) and 'POINT' in sample.upper():
            # Handle WKT strings
            df_copy['geometry'] = df_copy['.geo'].apply(shapely.wkt.loads)
            gdf = gpd.GeoDataFrame(df_copy, geometry='geometry', crs="EPSG:4326")
            return gdf
        
        else:
            print(f"Unrecognized .geo format: {sample}")
            return None
            
    except Exception as e:
        print(f"Error converting to GeoDataFrame: {e}")
        return None

def create_folium_map(gdf, color_column=None, color_scheme='YlOrRd', title="Building Map", use_class_for_damage=False):
    """
    Create a Folium map to visualize the GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame with geometry column
        color_column: Column to use for coloring features (optional)
        color_scheme: Colormap to use for coloring (optional)
        title: Title for the map (optional)
        use_class_for_damage: If True, use the 'class' column to color buildings
                             (1=damaged, 0=undamaged) instead of color_column
        
    Returns:
        Folium Map object
    """
    # Get centroid of all geometries for map centering
    if gdf is None or len(gdf) == 0:
        # Default to Gaza Strip coordinates if no data
        center = [31.3547, 34.3088]
        zoom = 10
    else:
        # Calculate centroid of all geometries
        try:
            bounds = gdf.geometry.total_bounds
            center = [(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2]
            zoom = 12
        except Exception:
            # Default to Gaza Strip coordinates if calculation fails
            center = [31.3547, 34.3088]
            zoom = 10
    
    # Create base map
    m = folium.Map(location=center, zoom_start=zoom, tiles='OpenStreetMap')
    
    # Add title
    title_html = f'''
        <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    if gdf is not None and len(gdf) > 0:
        if use_class_for_damage and 'class' in gdf.columns:
            # Use the 'class' column for damage visualization (1=damaged, 0=undamaged)
            feature_group = folium.FeatureGroup(name="Buildings (by damage class)")
            
            for idx, row in gdf.iterrows():
                if hasattr(row.geometry, 'geom_type'):
                    # Determine color based on class value
                    is_damaged = row['class'] == 1
                    color = 'red' if is_damaged else 'green'
                    
                    if row.geometry.geom_type == 'Point':
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=5,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.6,
                            tooltip=f"ID: {row['system:index']}, Damaged: {'Yes' if is_damaged else 'No'}"
                        ).add_to(feature_group)
                    elif row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                        folium.GeoJson(
                            row.geometry,
                            style_function=lambda x, color=color: {
                                'fillColor': color,
                                'color': color,
                                'weight': 1,
                                'fillOpacity': 0.6
                            },
                            tooltip=f"ID: {row['system:index']}, Damaged: {'Yes' if is_damaged else 'No'}"
                        ).add_to(feature_group)
            
            feature_group.add_to(m)
            
            # Add a simple legend
            legend_html = """
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; width: 150px; height: 90px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color:white;
                        padding: 10px">
              <span style="color:green;">■</span> Undamaged<br>
              <span style="color:red;">■</span> Damaged
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))
            
        elif color_column and color_column in gdf.columns:
            # Create choropleth map if color column is provided
            folium.Choropleth(
                geo_data=gdf,
                name='choropleth',
                data=gdf,
                columns=['system:index', color_column],
                key_on='feature.properties.system:index',
                fill_color=color_scheme,
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=color_column
            ).add_to(m)
        else:
            # Just add geometries with a single color if no color column
            feature_group = folium.FeatureGroup(name="Buildings")
            
            for idx, row in gdf.iterrows():
                if hasattr(row.geometry, 'geom_type'):
                    if row.geometry.geom_type == 'Point':
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=5,
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.6
                        ).add_to(feature_group)
                    elif row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                        folium.GeoJson(
                            row.geometry,
                            style_function=lambda x: {
                                'fillColor': 'blue',
                                'color': 'blue',
                                'weight': 1,
                                'fillOpacity': 0.6
                            }
                        ).add_to(feature_group)
            
            feature_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m