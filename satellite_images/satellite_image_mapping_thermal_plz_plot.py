import geopandas as gpd
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import shape

# 1. Load Postal Codes and District Boundaries
# Using the URLs you provided
districts_url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
postcodes_url = "https://tsb-opendata.s3.eu-central-1.amazonaws.com/plz/plz.geojson"

def generate_postal_lst_analysis(thermal_img_path):
    # Load GeoJSONs
    gdf_plz = gpd.read_file(postcodes_url).rename(columns={"plz": "postal_code"})
    gdf_plz["postal_code"] = gdf_plz["postal_code"].astype(str)
    
    # Ensure WGS84 for alignment with satellite imagery
    gdf_plz = gdf_plz.to_crs("EPSG:4326")
    
    # Load Thermal Image (Sentinel-3 / LST)
    img = cv2.imread(thermal_img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # Berlin Reference Extent (WGS84)
    extent = [13.088, 13.761, 52.338, 52.675]
    
    plz_results = []
    
    for idx, row in gdf_plz.iterrows():
        geom = row.geometry
        if geom.is_empty: continue
            
        # Create a mask for the Postal Code polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Mapping Polygon coordinates to Image Pixel coordinates
        def get_px_coords(poly):
            coords = []
            for lon, lat in poly.exterior.coords:
                px = int((lon - extent[0]) / (extent[1] - extent[0]) * w)
                py = int((1 - (lat - extent[2]) / (extent[3] - extent[2])) * h)
                coords.append([px, py])
            return np.array(coords)

        if geom.geom_type == 'Polygon':
            cv2.fillPoly(mask, [get_px_coords(geom)], 1)
        else: # MultiPolygon
            for part in geom.geoms:
                cv2.fillPoly(mask, [get_px_coords(part)], 1)
        
        # Calculate Stats from the pixels inside this Postal Code
        pixels = img[mask == 1]
        if len(pixels) > 0:
            plz_results.append({
                'postal_code': row['postal_code'],
                'avg_lst': round(np.mean(pixels), 2),
                'max_lst': np.max(pixels),
                'min_lst': np.min(pixels)
            })

    # Create Results DataFrame
    df_plz_lst = pd.DataFrame(plz_results)
    
    # 2. Plotting the Heatmap
    gdf_plot = gdf_plz.merge(df_plz_lst, on='postal_code')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    gdf_plot.plot(column='avg_lst', cmap='magma', ax=ax, legend=True,
                  legend_kwds={'label': "Average Thermal Intensity (PLZ Level)", 'orientation': "horizontal"})
    
    # Label the top 10 hottest postal codes for clarity
    for idx, row in gdf_plot.nlargest(10, 'avg_lst').iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, row['postal_code'], fontsize=9, 
                ha='center', weight='bold', color='white',
                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))

    plt.title('Berlin Postal Code (PLZ) Thermal Intensity Analysis', fontsize=16)
    plt.axis('off')
    plt.savefig('berlin_plz_lst_map.png', dpi=300, bbox_inches='tight')
    
    # 3. Export Data
    df_plz_lst.to_csv('berlin_postal_lst_data.csv', index=False)
    return df_plz_lst

# Run the analysis
lst_plz_data = generate_postal_lst_analysis('berlin_thermal_delta_plot.png')