import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

# insert documentation here if needed
"""
This script generates a heatmap of vegetation change in Berlin, overlaid with
administrative district boundaries and labels. It utilizes two NDVI (Normalized
Difference Vegetation Index) images from different time points to calculate
pixel-wise changes, which are then visualized.

Key functionalities include:
- Loading Berlin district boundaries from a GeoJSON file.
- `generate_district_change_map`: Loads two NDVI images, computes their
  difference to represent vegetation change, and plots this change as a heatmap.
  District boundaries and names are then overlaid on this heatmap for
  geographical context.

Dependencies:
- `geopandas`
- `pandas`
- `matplotlib`
- `opencv-python` (cv2)
- `numpy`

Usage:
1. Ensure you have the `berlin_bezirke.geojson` file accessible or
   update the `districts_url`.
2. Provide paths to your start and end NDVI image files (e.g., `nvdi_001.png`,
   `nvdi_009.png`). These images should be grayscale representations of NDVI,
   where pixel intensity corresponds to NDVI values.
3. Call `generate_district_change_map` with the image paths and the loaded
   districts GeoDataFrame to produce and save the visualization.
"""


# Note: In a local environment, download the GeoJSON from: 
districts_url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
districts = gpd.read_file(districts_url)

def generate_district_change_map(img_path_start, img_path_end, districts_gdf):
    # Load NDVI Visualizations
    img_start = cv2.imread(img_path_start, cv2.IMREAD_GRAYSCALE)
    img_end = cv2.imread(img_path_end, cv2.IMREAD_GRAYSCALE)
    
    # Calculate Pixel-wise Change (Intensity Delta)
    change = img_end.astype(float) - img_start.astype(float)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Berlin geographic bounds (WGS84)
    # Mapping the 512x512 image to Berlin's bounding box
    extent = [13.08, 13.77, 52.33, 52.68] 
    
    # Plot Change Heatmap
    im = ax.imshow(change, cmap='RdYlGn', extent=extent, alpha=0.8)
    
    # Overlay District Boundaries
    districts_gdf.boundary.plot(ax=ax, color='black', linewidth=1.2)
    
    # Add District Labels
    for idx, row in districts_gdf.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, row['name'], fontsize=9, 
                ha='center', weight='bold', bbox=dict(facecolor='white', alpha=0.6))

    plt.colorbar(im, label='Vegetation Intensity Change (2023 vs 2025)')
    plt.title('Berlin District-Level Vegetation Change Detection')
    plt.axis('off')
    plt.savefig('berlin_district_change_plot.png', bbox_inches='tight')
    plt.show()

# Example Usage:
generate_district_change_map(r'.\NVDI\nvdi_001.png', r'.\NVDI\nvdi_009.png', districts)