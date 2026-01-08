import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Insert documentation about the script
"""
Berlin Boundary Change Visualization
This script loads two satellite images representing vegetation indices
from different years and visualizes the change overlaid with Berlin's boundary.



"""


# 1. Load Berlin Boundary (Administrative level 4/8)
# Using local GeoJSON file from the project
berlin_boundary = gpd.read_file(r'../berlin_heat_data/boundaries/berlin_boundary.geojson')

def plot_berlin_change(img_path_start, img_path_end):
    # Load images
    img_start = cv2.imread(img_path_start, cv2.IMREAD_GRAYSCALE)
    img_end = cv2.imread(img_path_end, cv2.IMREAD_GRAYSCALE)
    
    # Calculate Change (Difference)
    change = img_end.astype(float) - img_start.astype(float)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display the change heatmap
    # Extent logic: Mapping 512x512 pixels to Berlin's coordinate bounds
    bounds = berlin_boundary.total_bounds # [minx, miny, maxx, maxy]
    im = ax.imshow(change, cmap='RdYlGn', extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
    
    # Overlay the Berlin Boundary
    berlin_boundary.boundary.plot(ax=ax, color='black', linewidth=2, label='Berlin Border')
    
    plt.colorbar(im, label='NDVI Change Intensity')
    plt.title('Berlin Vegetation Change Overlay (2023-2025)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.savefig('berlin_boundary_change.png')
    plt.show()

# Run the plot using your first and last file
plot_berlin_change(r'.\NVDI\nvdi_001.png', r'.\NVDI\nvdi_009.png')