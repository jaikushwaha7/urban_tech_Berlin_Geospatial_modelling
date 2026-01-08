import geopandas as gpd
import shapely.geometry as sg
import numpy as np
import folium
from tqdm import tqdm   # nice progress bar (optional)
import os
import unicodedata
import pandas as pd

# insert documentation here if needed
"""
This script generates hexagonal grids for Berlin at both district and postal code levels,
visualizes them using Folium, and integrates external resilience and change intensity data
for thematic mapping.

Key functionalities include:
1.  **Loading Administrative Boundaries**: Fetches Berlin district and postal code
    boundaries from local files or remote URLs.
2.  **Hexagon Grid Generation**: Creates custom flat-topped hexagonal grids over
    the loaded administrative polygons.
3.  **Data Integration**: Merges external resilience and change intensity data
    (from `final_berlin_h3_comparison.csv`) with the generated hexagonal grid
    based on district names.
4.  **Interactive Mapping with Folium**:
    *   Visualizes district-level hexagons, colored by district.
    *   Visualizes postal code-level hexagons, colored by postal code.
    *   Generates thematic maps for 'Resilience Score' and 'Estimated Change Intensity',
        using color gradients to represent values.
    *   Creates a comparison map allowing toggling between different grid layers.
5.  **Output**: Saves generated GeoJSON files for the hexagonal grids and HTML
    files for all Folium maps.

Dependencies:
-   `geopandas`
-   `shapely`
-   `numpy`
-   `folium`
-   `tqdm` (for progress bars)
-   `pandas`
-   `matplotlib` (for colormap generation)
-   `unicodedata` (for robust string matching)

Usage:
-   Ensure `berlin_bezirke.geojson` and `plz.geojson` (or their remote sources)
    are accessible.
-   Place `final_berlin_h3_comparison.csv` in the same directory for resilience
    and change intensity mapping.
-   Run the script to generate GeoJSON and HTML map files in the current directory.
"""


# 1. Load Berlin Administrative Data
# Source: ODIS Berlin / OpenStreetMap
districts_url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
postcodes_url = "https://tsb-opendata.s3.eu-central-1.amazonaws.com/plz/plz.geojson"

import logging

logging.basicConfig(level=logging.INFO)

# Load boundaries from local repository when available, otherwise attempt remote URLs
BOUNDARIES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'berlin_heat_data', 'boundaries'))

def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

district_candidates = [
    os.path.join(BOUNDARIES_DIR, 'berlin_bezirke.geojson'),
    os.path.join(BOUNDARIES_DIR, 'berlin_boundary.geojson'),
]
postcode_candidates = [
    os.path.join(BOUNDARIES_DIR, 'berlin_plz.geojson'),
]

district_fp = _first_existing(district_candidates)
if district_fp:
    districts = gpd.read_file(district_fp)
    logging.info(f'Loaded districts from local file: {district_fp}')
else:
    try:
        districts = gpd.read_file(districts_url)
        logging.info('Loaded districts from remote URL')
    except Exception:
        logging.warning('Failed to load districts remotely; falling back to available local boundary')
        districts = gpd.read_file(os.path.join(BOUNDARIES_DIR, 'berlin_boundary.geojson'))

postcode_fp = _first_existing(postcode_candidates)
if postcode_fp:
    postcodes = gpd.read_file(postcode_fp)
    logging.info(f'Loaded postcodes from local file: {postcode_fp}')
else:
    try:
        postcodes = gpd.read_file(postcodes_url)
        logging.info('Loaded postcodes from remote URL')
    except Exception:
        logging.warning('Failed to load postcodes from original URL; trying alternate source')
        try:
            # Alternate source: Overpass API / OpenStreetMap
            postcodes_alt_url = "https://www.wfsland.de/ogc-services/service?service=WFS&version=2.0.0&request=GetFeature&typeNames=plz&outputFormat=application/json&cql_filter=bundesland='BE'"
            postcodes = gpd.read_file(postcodes_alt_url)
            logging.info('Loaded postcodes from alternate Overpass/WFS source')
        except Exception as e:
            logging.warning(f'Failed to load postcodes from alternate source ({e}); using districts as a proxy')
            postcodes = districts.copy()
            postcodes['plz'] = '00000'

# Ensure GeoDataFrames have JSON-serializable properties for Folium
def stringify_datetime_columns(gdf):
    for col in gdf.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(gdf[col]):
                gdf[col] = gdf[col].astype(str)
        except Exception:
            # ignore columns that raise during type checks
            continue

stringify_datetime_columns(postcodes)
stringify_datetime_columns(districts)

# Ensure active geometry column is set
if 'geometry' in postcodes.columns:
    postcodes = gpd.GeoDataFrame(postcodes, geometry='geometry', crs=postcodes.crs)
if 'geometry' in districts.columns:
    districts = gpd.GeoDataFrame(districts, geometry='geometry', crs=districts.crs)
# ------------------------------------------------------------------
# 0.  Make sure the district layer is in a metre-based CRS
# ------------------------------------------------------------------
if districts.crs.is_geographic:          # WGS84 → ETRS89 / UTM 33N
    districts = districts.to_crs(25833)
else:                                    # already projected
    pass

# ------------------------------------------------------------------
# 1.  Hexagon geometry helpers
# ------------------------------------------------------------------
def hexagon(center, radius):
    """
    Return a POLYGON for a regular hexagon with *flat top*.
    radius = distance centre ⟺ side (also called 'hex size').
    """
    cx, cy = center
    angle = np.linspace(0, 2*np.pi, 7)          # 6 vertices + close
    x = cx + radius * np.cos(angle)
    y = cy + radius * np.sin(angle + np.pi/6)   # rotate 30° for flat top
    return sg.Polygon(np.column_stack([x, y]))

def hex_grid_over_polygon(poly, radius):
    """
    Return list of hexagons whose centroid is inside `poly`.
    radius in the same units as the CRS (here: metres).
    """
    minx, miny, maxx, maxy = poly.bounds
    # horizontal and vertical spacing for flat-topped hexagons
    dx = 3/2 * radius
    dy = np.sqrt(3) * radius

    # generate centres
    xcoords = np.arange(minx - dx, maxx + dx, dx)
    ycoords = np.arange(miny - dy, maxy + dy, dy)
    # offset every second row
    centers = []
    for j, y in enumerate(ycoords):
        offset = 0 if j % 2 == 0 else dx/2
        for x in xcoords:
            centers.append((x + offset, y))

    # keep only those whose centroid is inside the polygon
    hexagons = [hexagon(c, radius) for c in centers if poly.contains(sg.Point(c))]
    return hexagons

# ------------------------------------------------------------------
# 2.  Build hex grids at DISTRICT level
# ------------------------------------------------------------------
print("\n=== CREATING DISTRICT-LEVEL HEXAGON GRID ===")
radius_m = 500          # ≈ 500 m hexagon "size" – change freely
hex_rows_district = []

for _, dist in tqdm(districts.iterrows(), total=len(districts), desc="Processing districts"):
    pol = dist.geometry
    if pol.is_empty or pol is None:
        continue
    hex_polys = hex_grid_over_polygon(pol, radius_m)
    district_name = dist.get('name', dist.get('Gemeinde_name', 'unknown'))
    district_id = dist.get('id', dist.get('BEZ', -1))
    for h in hex_polys:
        hex_rows_district.append({
            'geometry': h,
            'district': district_name,
            'district_id': district_id,
            'level': 'district'
        })

hex_grid_district = gpd.GeoDataFrame(hex_rows_district, crs=districts.crs)
print(f'{len(hex_grid_district)} hexagons created at district level')
hex_grid_district.to_file('berlin_hex_district_500m.geojson', driver='GeoJSON')

# ------------------------------------------------------------------
# 2b. Build hex grids at POSTAL CODE level
# ------------------------------------------------------------------
print("\n=== CREATING POSTAL CODE-LEVEL HEXAGON GRID ===")
if 'postcodes' in locals() and not postcodes.empty and postcodes['plz'].nunique() > 1:
    # Ensure postcodes are in the same CRS as districts
    postcodes_proj = postcodes.to_crs(districts.crs)
    hex_rows_postcode = []
    
    for _, postcode in tqdm(postcodes_proj.iterrows(), total=len(postcodes_proj), desc="Processing postcodes"):
        pol = postcode.geometry
        if pol.is_empty or pol is None:
            continue
        hex_polys = hex_grid_over_polygon(pol, radius_m)
        plz = postcode.get('plz', postcode.get('PLZ', 'unknown'))
        for h in hex_polys:
            hex_rows_postcode.append({
                'geometry': h,
                'plz': plz,
                'level': 'postcode'
            })
    
    hex_grid_postcode = gpd.GeoDataFrame(hex_rows_postcode, crs=districts.crs)
    print(f'{len(hex_grid_postcode)} hexagons created at postal code level')
    hex_grid_postcode.to_file('berlin_hex_postcode_500m.geojson', driver='GeoJSON')
else:
    print("Postal codes not available or insufficient variety - skipping postcode-level grid")
    hex_grid_postcode = None

# Keep district grid as main hex_grid for backward compatibility
hex_grid = hex_grid_district

# optional: spatial join with postcodes if you want plz column
if 'postcodes' in locals() and not postcodes.empty:
    postcodes = postcodes.to_crs(hex_grid.crs)
    hex_grid = gpd.sjoin(hex_grid, postcodes[['plz', 'geometry']], how='left', predicate='intersects')
    hex_grid.drop(columns='index_right', inplace=True)

print(f'{len(hex_grid)} hexagons created')

# project back to WGS84 for Folium
map_df = hex_grid.to_crs(4326)
m = folium.Map(location=[52.52, 13.40], zoom_start=10, tiles='CartoDB positron')

# colour by district (categorical)
districts_list = map_df['district'].unique()
# Define a color palette for districts
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
          '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']
palette = (colors * ((len(districts_list) // len(colors)) + 1))[:len(districts_list)]
color_map = dict(zip(districts_list, palette))

for _, r in map_df.iterrows():
    folium.GeoJson(
        r.geometry,
        style_function=lambda x, col=color_map[r.district]: {
            'fillColor': col,
            'color': 'grey',
            'weight': 0.5,
            'fillOpacity': 0.6
        }
    ).add_to(m)

m.save('berlin_hex_district.html')
hex_grid.to_file('berlin_hex_500m.geojson', driver='GeoJSON')

# H3 grid generation removed (not used / not functioning).
# H3-related code intentionally removed to focus on district and postcode hex grids.

# ------------------------------------------------------------------
# 4.  Load resilience and change intensity data
# ------------------------------------------------------------------
resilience_file = 'final_berlin_h3_comparison.csv'
if os.path.exists(resilience_file):
    print(f"\n=== LOADING RESILIENCE DATA ===")
    resilience_df = pd.read_csv(resilience_file, encoding='utf-8')
    print(f"Loaded resilience data for {len(resilience_df)} districts")

    # Normalize district names to improve merge matching (remove accents, lower-case, strip)
    def normalize_name(s):
        if s is None:
            return ''
        s = str(s)
        s = s.strip().lower()
        # normalize unicode (remove accents)
        s = unicodedata.normalize('NFD', s)
        s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
        # replace common punctuation and whitespace variants
        s = s.replace('\u2013', '-')  # en-dash
        s = s.replace('\u2014', '-')  # em-dash
        s = s.replace('–', '-')
        s = s.replace('—', '-')
        s = s.replace("'", '')
        s = s.replace(' ', '_')
        s = s.replace('/', '_')
        return s

    resilience_df['district_key'] = resilience_df['District'].apply(normalize_name)
    map_df['district_key'] = map_df['district'].apply(normalize_name)

    # Debug: show unmatched keys
    csv_keys = set(resilience_df['district_key'].unique())
    map_keys = set(map_df['district_key'].unique())
    common = csv_keys & map_keys
    print(f"Resilience keys in CSV: {len(csv_keys)}; district keys in map: {len(map_keys)}; common: {len(common)}")

    # Merge on normalized key
    map_df_with_resilience = map_df.merge(
        resilience_df[['district_key', 'Resilience_Score', 'Estimated_Change_Intensity']],
        left_on='district_key',
        right_on='district_key',
        how='left'
    )
    # report how many hexes got a match
    matched = map_df_with_resilience['Resilience_Score'].notna().sum()
    print(f"Hexagons matched with resilience values: {matched} / {len(map_df_with_resilience)}")
else:
    print(f"\n⚠ Resilience data file '{resilience_file}' not found. Skipping resilience maps.")
    map_df_with_resilience = None

# ------------------------------------------------------------------
# 5.  Visualize maps
# ------------------------------------------------------------------
print("\n=== DISTRICT-LEVEL HEXAGON GRID MAP ===")
print(f"Saved to: berlin_hex_district.html")
print(f"Total hexagons: {len(map_df)}")

# Create postcode-level map if available
if hex_grid_postcode is not None:
    print("\n=== POSTAL CODE-LEVEL HEXAGON GRID MAP ===")
    map_df_postcode = hex_grid_postcode.to_crs(4326)
    m_postcode = folium.Map(location=[52.52, 13.40], zoom_start=10, tiles='CartoDB positron')
    
    # Color by postal code
    plz_list = map_df_postcode['plz'].unique()
    postcode_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                       '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']
    postcode_palette = (postcode_colors * ((len(plz_list) // len(postcode_colors)) + 1))[:len(plz_list)]
    postcode_color_map = dict(zip(plz_list, postcode_palette))
    
    for _, r in map_df_postcode.iterrows():
        folium.GeoJson(
            r.geometry,
            style_function=lambda x, col=postcode_color_map[r['plz']]: {
                'fillColor': col,
                'color': 'darkblue',
                'weight': 0.5,
                'fillOpacity': 0.5
            }
        ).add_to(m_postcode)
    
    m_postcode.save('berlin_hex_postcode.html')
    print(f"Saved to: berlin_hex_postcode.html")
    print(f"Total postcode hexagons: {len(map_df_postcode)}")

# H3 maps removed (not used)

# ------------------------------------------------------------------
# Create resilience and change intensity maps
# ------------------------------------------------------------------
if map_df_with_resilience is not None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Helper function to get color from value
    def get_color_from_value(value, vmin, vmax, cmap='RdYlGn'):
        """Map value to color using matplotlib colormap"""
        if pd.isna(value):
            return '#cccccc'
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = cm.get_cmap(cmap)
        rgba = cmap_obj(norm(value))
        return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
    
    # Create Resilience Score Map (Green = Stable, Red = Volatile)
    print("\n=== RESILIENCE SCORE MAP ===")
    m_resilience = folium.Map(location=[52.52, 13.40], zoom_start=10, tiles='CartoDB positron')
    
    # compute ranges safely (skip NaNs)
    if map_df_with_resilience['Resilience_Score'].dropna().empty:
        resilience_min = resilience_max = None
    else:
        resilience_min = map_df_with_resilience['Resilience_Score'].dropna().min()
        resilience_max = map_df_with_resilience['Resilience_Score'].dropna().max()
    
    for _, r in map_df_with_resilience.iterrows():
        resilience = r['Resilience_Score']
        color = get_color_from_value(resilience, resilience_min if resilience_min is not None else 0, resilience_max if resilience_max is not None else 1, cmap='RdYlGn')
        # safe formatting for missing values
        if pd.notna(resilience):
            res_str = f"{resilience:.2f}"
        else:
            res_str = "N/A"
        popup_text = f"District: {r['district']}<br>Resilience: {res_str}"
        tooltip_text = f"Resilience: {res_str}"

        folium.GeoJson(
            r.geometry,
            style_function=lambda x, col=color: {
                'fillColor': col,
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.7
            },
            popup=popup_text,
            tooltip=tooltip_text
        ).add_to(m_resilience)
    
    # Save resilience map to berlin_hex.html (overwrite existing hex map)
    m_resilience.save('berlin_hex.html')
    print(f"Saved to: berlin_hex.html")
    if resilience_min is None or resilience_max is None:
        print("Resilience Range: N/A")
    else:
        print(f"Resilience Range: {resilience_min:.2f} - {resilience_max:.2f}")
    
    # Create Change Intensity Map (Red = High Change, Green = Stable)
    print("\n=== CHANGE INTENSITY MAP ===")
    m_intensity = folium.Map(location=[52.52, 13.40], zoom_start=10, tiles='CartoDB positron')
    
    # compute safe intensity range
    if map_df_with_resilience['Estimated_Change_Intensity'].dropna().empty:
        intensity_min = intensity_max = None
    else:
        intensity_min = map_df_with_resilience['Estimated_Change_Intensity'].dropna().min()
        intensity_max = map_df_with_resilience['Estimated_Change_Intensity'].dropna().max()
    
    for _, r in map_df_with_resilience.iterrows():
        intensity = r['Estimated_Change_Intensity']
        # Reverse cmap so red = high change, green = low change
        color = get_color_from_value(intensity, intensity_min if intensity_min is not None else 0, intensity_max if intensity_max is not None else 1, cmap='YlOrRd')
        
        # safe formatting for missing intensity
        if pd.notna(intensity):
            intens_str = f"{intensity:.2f}"
        else:
            intens_str = "N/A"
        folium.GeoJson(
            r.geometry,
            style_function=lambda x, col=color, intens=intensity: {
                'fillColor': col,
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.7
            },
            popup=f"District: {r['district']}<br>Change Intensity: {intens_str}",
            tooltip=f"Change Intensity: {intens_str}"
        ).add_to(m_intensity)
    
    m_intensity.save('berlin_hex_change_intensity.html')
    print(f"Saved to: berlin_hex_change_intensity.html")
    if intensity_min is None or intensity_max is None:
        print("Change Intensity Range: N/A")
    else:
        print(f"Change Intensity Range: {intensity_min:.2f} - {intensity_max:.2f}")

# Create comparison map with all layers
print("\n=== COMPARISON MAP (All Layers) ===")
m_compare = folium.Map(location=[52.52, 13.40], zoom_start=10, tiles='CartoDB positron')

# Add district-level hex grid
fg_district = folium.FeatureGroup(name='District Hexagons (500m)', show=True)
for _, r in map_df.iterrows():
    folium.GeoJson(
        r.geometry,
        style_function=lambda x, col=color_map[r.district]: {
            'fillColor': col,
            'color': 'grey',
            'weight': 0.5,
            'fillOpacity': 0.4
        }
    ).add_to(fg_district)
fg_district.add_to(m_compare)

# Add postcode-level hex grid if available
if hex_grid_postcode is not None:
    fg_postcode = folium.FeatureGroup(name='Postcode Hexagons (500m)', show=False)
    for _, r in map_df_postcode.iterrows():
        folium.GeoJson(
            r.geometry,
            style_function=lambda x, col=postcode_color_map[r['plz']]: {
                'fillColor': col,
                'color': 'darkblue',
                'weight': 0.5,
                'fillOpacity': 0.3
            }
        ).add_to(fg_postcode)
    fg_postcode.add_to(m_compare)

# H3 layer omitted from comparison map

folium.LayerControl().add_to(m_compare)
m_compare.save('berlin_grid_comparison.html')
print(f"Saved to: berlin_grid_comparison.html")

print("\n" + "="*70)
print("✓ COMPREHENSIVE SUMMARY - All maps created successfully!")
print("="*70)
print("\n📊 HEXAGON GRIDS (500m):")
print(f"  ├─ District-Level: berlin_hex_district_500m.geojson")
print(f"  │  └─ Map: berlin_hex_district.html ({len(hex_grid_district)} hexagons)")
if hex_grid_postcode is not None:
    print(f"  ├─ Postal Code-Level: berlin_hex_postcode_500m.geojson")
    print(f"  │  └─ Map: berlin_hex_postcode.html ({len(hex_grid_postcode)} hexagons)")
print("  └─ H3 Grid: removed (not generated)")

print("\n🎨 THEMATIC ANALYSIS MAPS:")
if map_df_with_resilience is not None:
    print(f"  ├─ Resilience Score Map: berlin_hex.html")
    print(f"  │  └─ Color: Green (Stable) → Red (Volatile)")
    print(f"  ├─ Change Intensity Map: berlin_hex_change_intensity.html")
    print(f"  │  └─ Color: Green (Stable) → Red (High Change)")
else:
    print(f"  ├─ Resilience maps not generated (data file missing)")

print(f"  └─ Comparison Map (All Layers): berlin_grid_comparison.html")

print("\n📁 DATA FILES:")
print(f"  ├─ District hexagons: berlin_hex_district_500m.geojson")
if hex_grid_postcode is not None:
    print(f"  ├─ Postcode hexagons: berlin_hex_postcode_500m.geojson")
print(f"  └─ H3 hexagons: removed (not generated)")

print("="*70)
