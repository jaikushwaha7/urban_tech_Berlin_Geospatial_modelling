import geopandas as gpd
import matplotlib.pyplot as plt

import pandas as pd

# insert documentation here if needed
"""
This script visualizes Berlin's administrative districts overlaid on a
vegetation change map derived from satellite imagery. It also provides
a framework for calculating district-wise statistics from such imagery.

Key functionalities include:
- Loading Berlin district boundaries from a GeoJSON file.
- `plot_districts_on_ndvi`: Overlays district boundaries and names onto
  a provided vegetation change heatmap.
- `get_district_stats`: (Placeholder) Demonstrates how to approach
  calculating average vegetation change or NDVI intensity per district.

Dependencies:
- `geopandas`
- `matplotlib`
- `pandas`

Usage:
1. Ensure you have the `berlin_bezirke.geojson` file accessible or
   update the `districts_url`.
2. Prepare your `change_map_array` (e.g., from an NDVI difference calculation).
3. Call `plot_districts_on_ndvi` to visualize the change with district overlays.
4. Extend `get_district_stats` with actual raster processing logic
   (e.g., using `rasterio` and `shapely`) to compute per-district metrics.
"""


# Load the working GeoJSON for Berlin Districts
districts_url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
districts = gpd.read_file(districts_url)

def plot_districts_on_ndvi(change_map_array):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the change map (from your previous analysis)
    # We use extent to align coordinates roughly with Berlin's bounds
    extent = [13.08, 13.77, 52.33, 52.68] 
    im = ax.imshow(change_map_array, cmap='RdYlGn', extent=extent, alpha=0.8)
    
    # Overlay District boundaries and labels
    districts.boundary.plot(ax=ax, color='black', linewidth=1.5)
    
    # Add labels for each district
    for idx, row in districts.iterrows():
        # Get centroid for label placement
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, row['name'], 
                fontsize=8, ha='center', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    plt.title("Berlin District-Wise Vegetation Change (2023-2025)")
    plt.colorbar(im, label="Change Intensity")
    plt.show()




def get_district_stats(ndvi_image, districts_gdf):
    """Calculates the average NDVI intensity for each district."""
    # Assuming ndvi_image is reprojected to match the districts_gdf CRS
    # We use a spatial join or mask to extract pixel values per polygon
    results = []
    for idx, row in districts_gdf.iterrows():
        # Mask logic here (using rasterio or similar)
        # avg_val = masked_mean(ndvi_image, row.geometry)
        results.append({"District": row['name'], "Status": "Analyzed"})
    return pd.DataFrame(results)

# Summary of the 12 Districts for your report
print(districts[['name', 'geometry']].head(12))