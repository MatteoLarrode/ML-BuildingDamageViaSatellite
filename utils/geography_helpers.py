import geopandas as gpd
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
import math

def create_rotated_square_aoi(lat, lon, size_km=10.0, angle_deg=0.0, filename="rotated_aoi.geojson"):
    """
    Create a rotated square AOI centered at (lat, lon).
    
    Parameters:
    - lat, lon: center coordinates (float)
    - size_km: side length in kilometers (float)
    - angle_deg: counterclockwise rotation angle in degrees (float)
    - filename: output GeoJSON filename (str)
    """
    size_deg = size_km / 111  # Rough conversion from km to degrees
    half = size_deg / 2

    # Define square corner coordinates around center
    square = Polygon([
        (-half, -half),
        ( half, -half),
        ( half,  half),
        (-half,  half)
    ])
    
    # Rotate square and translate it to the target center
    rotated = rotate(square, angle=angle_deg, origin=(0, 0), use_radians=False)
    positioned = translate(rotated, xoff=lon, yoff=lat)

    # Save to GeoJSON
    gdf = gpd.GeoDataFrame(geometry=[positioned], crs="EPSG:4326")
    gdf.to_file(filename, driver="GeoJSON")
    return filename