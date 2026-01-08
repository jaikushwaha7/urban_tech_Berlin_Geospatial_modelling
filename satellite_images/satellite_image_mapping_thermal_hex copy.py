import json
import numpy as np
import pandas as pd
import cv2
from shapely.geometry import shape
from pyproj import Transformer
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 1. Setup Data and Transformer (UTM 25833 to WGS84)
with open('berlin_hex_district_500m.geojson', 'r') as f:
    hex_data = json.load(f)
transformer = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)

# 10km Altitude Reference Extent
extent = [13.088, 13.761, 52.338, 52.675]

def calculate_averaged_hex_intensity(thermal_img_path):
    # Load thermal image
    img = cv2.imread(thermal_img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    hex_results = []
    
    for feature in hex_data['features']:
        # Get polygon coordinates and transform to pixel space
        geom = shape(feature['geometry'])
        poly_coords = list(geom.exterior.coords)
        
        px_coords = []
        for x_utm, y_utm in poly_coords:
            lon, lat = transformer.transform(x_utm, y_utm)
            # Map Lon/Lat to Image Pixel X/Y
            px = int((lon - extent[0]) / (extent[1] - extent[0]) * w)
            py = int((1 - (lat - extent[2]) / (extent[3] - extent[2])) * h)
            px_coords.append([px, py])
            
        # 2. Create a Mask for this specific Hexagon
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(px_coords)], 1)
        
        # 3. Extract pixels inside the mask and average them
        pixels_in_hex = img[mask == 1]
        
        if len(pixels_in_hex) > 0:
            avg_intensity = np.mean(pixels_in_hex)
            max_intensity = np.max(pixels_in_hex)
        else:
            avg_intensity = 0
            max_intensity = 0

        hex_results.append({
            'avg_lst': round(avg_intensity, 2),
            'max_lst': int(max_intensity),
            'centroid_lon': transformer.transform(geom.centroid.x, geom.centroid.y)[0],
            'centroid_lat': transformer.transform(geom.centroid.x, geom.centroid.y)[1]
        })

    return pd.DataFrame(hex_results)

# Execute
df_averaged = calculate_averaged_hex_intensity('berlin_thermal_delta_plot.png')
df_averaged.to_csv('berlin_hexagon_lst_analysis.csv', index=False)