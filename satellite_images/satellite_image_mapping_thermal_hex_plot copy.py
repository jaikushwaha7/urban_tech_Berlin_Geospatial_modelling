import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt

# Load GeoJSON manually
with open('berlin_hex_district_500m.geojson', 'r') as f:
    hex_json = json.load(f)

features = hex_json['features']
geoms = [shape(f['geometry']) for f in features]
props = [f['properties'] for f in features]

hex_gdf = gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:25833")

# Load LST CSV
lst_df = pd.read_csv('berlin_hexagon_lst_analysis.csv')
hex_gdf['avg_lst'] = lst_df['avg_lst'].values

# Load Boundary GeoJSON manually
with open('../berlin_heat_data/boundaries/berlin_boundary.geojson', 'r') as f:
    bound_json = json.load(f)


bound_geoms = [shape(f['geometry']) for f in bound_json['features']]
boundary_gdf = gpd.GeoDataFrame(geometry=bound_geoms, crs="EPSG:4326")
boundary_gdf = boundary_gdf.to_crs(hex_gdf.crs)

# Plotting
fig, ax = plt.subplots(figsize=(15, 12))

# Use a dark background to make colors pop
ax.set_facecolor('black')
fig.patch.set_facecolor('white')

# Plot hexagons
hex_gdf.plot(column='avg_lst', cmap='magma', ax=ax, legend=True, 
             legend_kwds={'label': "Mean Thermal Intensity (Averaged across 500m Hexagon)",
                          'orientation': "horizontal", 'pad': 0.05},
             edgecolor='none')

# Plot boundary line
boundary_gdf.boundary.plot(ax=ax, color='#00FFFF', linewidth=2, alpha=0.8)

plt.title('Berlin Hex-Level LST (Zonal Statistics)\nHigh-Resolution Heat Risk Analysis', fontsize=18, pad=20)
plt.axis('off')

plt.savefig('berlin_hex_lst_visual.png', dpi=300, bbox_inches='tight')
print("Generated: berlin_hex_lst_visual.png")

import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from pyproj import Transformer
import cv2
import matplotlib.pyplot as plt

# 1. Load Hexagon GeoJSON
with open('berlin_hex_district_500m.geojson', 'r') as f:
    hex_data = json.load(f)

# 2. Setup Coordinate Transformer (UTM Zone 33N -> WGS84)
transformer = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)

# 3. Create Synthetic Thermal Image with spatial variance to calculate min/max
# Size matches general Berlin aspect ratio
width, height = 1200, 1000
# Define extent used for mapping (WGS84)
extent = [13.088, 13.761, 52.338, 52.675]

# Generate synthetic heatmap
y, x = np.ogrid[:height, :width]
cy, cx = height // 2, width // 2
dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
# Urban heat core + local variations (noise)
base_heat = 150 * np.exp(-dist_from_center / 400) + 50
noise = np.random.normal(0, 20, (height, width))
thermal_data = np.clip(base_heat + noise, 0, 255).astype(np.uint8)

# 4. Zonal Statistics for Min and Max
hex_stats = []
for feature in hex_data['features']:
    geom = shape(feature['geometry'])
    poly_coords = list(geom.exterior.coords)
    
    # Project to pixel space
    px_coords = []
    for x_utm, y_utm in poly_coords:
        lon, lat = transformer.transform(x_utm, y_utm)
        px = int((lon - extent[0]) / (extent[1] - extent[0]) * width)
        py = int((1 - (lat - extent[2]) / (extent[3] - extent[2])) * height)
        px_coords.append([px, py])
    
    # Create mask for hexagon
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(px_coords)], 1)
    pixels = thermal_data[mask == 1]
    
    if len(pixels) > 0:
        h_min = np.min(pixels)
        h_max = np.max(pixels)
        h_diff = h_max - h_min
    else:
        h_min, h_max, h_diff = 0, 0, 0
        
    hex_stats.append({
        'geometry': geom,
        'min_lst': int(h_min),
        'max_lst': int(h_max),
        'diff_lst': int(h_diff)
    })

gdf = gpd.GeoDataFrame(hex_stats, crs="EPSG:25833")

# 5. Plotting Min vs Max side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

gdf.plot(column='min_lst', cmap='magma', ax=ax1, legend=True, 
         legend_kwds={'label': "Min Intensity", 'orientation': "horizontal"})
ax1.set_title("Hexagon-Level MIN LST Intensity", fontsize=16)
ax1.axis('off')

gdf.plot(column='max_lst', cmap='magma', ax=ax2, legend=True, 
         legend_kwds={'label': "Max Intensity", 'orientation': "horizontal"})
ax2.set_title("Hexagon-Level MAX LST Intensity", fontsize=16)
ax2.axis('off')

plt.tight_layout()
plt.savefig('berlin_min_max_lst_comparison.png', dpi=300)

# 6. Plotting the Difference (Range)
fig2, ax3 = plt.subplots(figsize=(12, 10))
gdf.plot(column='diff_lst', cmap='viridis', ax=ax3, legend=True,
         legend_kwds={'label': "Intensity Range (Max - Min)"})
ax3.set_title("LST Intensity Range (Spatial Variance) per Hexagon", fontsize=16)
ax3.axis('off')

plt.savefig('berlin_lst_range_map.png', dpi=300)

# Save the updated stats
gdf.drop(columns='geometry').to_csv('berlin_hexagon_min_max_lst.csv', index=False)

print("Plots and CSV generated.")