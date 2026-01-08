import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import h3 # Ensure h3-py is installed

# --- PART 1: DATA PREPARATION ---
# Data from the provided image analysis and district stats
districts_data = {
    "District": ["Treptow-Köpenick", "Spandau", "Steglitz-Zehlendorf", "Pankow", "Reinickendorf",
                 "Marzahn-Hellersdorf", "Lichtenberg", "Charlottenburg-Wilmersdorf",
                 "Tempelhof-Schöneberg", "Neukölln", "Friedrichshain-Kreuzberg", "Mitte"],
    "Resilience_Score": [0.92, 0.88, 0.85, 0.74, 0.69, 0.61, 0.58, 0.49, 0.38, 0.22, 0.18, 0.15],
    "Observed_Change": [4.2, 5.1, 1.5, 11.8, 2.3, 6.4, 5.8, 8.2, 12.5, 15.1, 18.2, 20.5]
}
df_districts = pd.DataFrame(districts_data)

# --- PART 2: MODERN CHARTS ---

# 1. Sorted Bar Chart
df_sorted = df_districts.sort_values(by="Resilience_Score", ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(df_sorted["District"], df_sorted["Resilience_Score"], color=plt.cm.viridis(np.linspace(0, 1, 12)))
plt.title("Berlin District Resilience Ranking", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Resilience Index")
plt.tight_layout()
plt.savefig("district_resilience_bar.png")

# 2. Radar Chart
def create_radar(df):
    labels = df["District"].tolist()
    values = df["Resilience_Score"].tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='teal', alpha=0.25)
    ax.plot(angles, values, color='teal', linewidth=2)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title("Ecological Profile Radar", y=1.1)
    plt.savefig("district_radar_chart.png")

create_radar(df_districts)

# --- PART 3: H3 GRID & FAIR RATIO ANALYSIS ---

def analyze_spatial_ratios(geojson_path, target_col='name'):
    # Load Boundaries (Districts or Postal Codes)
    gdf = gpd.read_file(geojson_path)
    
    # H3 Grid Generation (Resolution 9 = ~0.1 km2 units)
    def count_h3_units(geom, res=9):
        # Convert geometry to H3-compatible polyfill
        if geom.geom_type == 'Polygon':
            coords = [geom.exterior.coords]
        else: # MultiPolygon
            coords = [p.exterior.coords for p in geom.geoms]
        
        # Simple count estimation for demonstration
        # For actual H3: hexs = h3.polyfill(polygon_dict, res)
        # return len(hexs)
        return int(geom.area * 100000) # Proxy for demonstration

    gdf['unit_count'] = gdf.geometry.apply(count_h3_units)
    
    # Calculate Ratio: Higher ratio = Higher change intensity per unit area
    # Assuming 'observed_change' is joined to this GDF
    gdf['change_ratio'] = gdf['observed_change'] / gdf['unit_count']
    return gdf

# District URLs for your local environment
# DISTRICTS: https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson
# POSTAL CODES: https://raw.githubusercontent.com/codeforberlin/plz-geojson/master/berlin_plz.geojson

print("Spatial logic defined. Ranking complete.")