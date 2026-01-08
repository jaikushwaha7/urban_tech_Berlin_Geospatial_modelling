import h3
import geopandas as gpd
import folium
from shapely.geometry import Polygon, mapping
import os
import numpy as np
import pandas as pd

# 1. Load Berlin Administrative Data
# Source: ODIS Berlin / OpenStreetMap
districts_url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
postcodes_url = "https://raw.githubusercontent.com/codeforberlin/plz-geojson/master/berlin_plz.geojson"

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
        logging.warning('Failed to load postcodes; using districts as a proxy for postal codes')
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

def get_h3_heatmap(target_gdf, h3_res=9):
    """Generates H3 hexagons within the target GeoDataFrame and assigns random values for heatmap."""
    hexagons = []
    for geom in target_gdf.geometry:
        # Convert geometry to H3 polyfill format
        if geom.geom_type == 'Polygon':
            coords = [[[lat, lng] for lng, lat in geom.exterior.coords]]
        else: # MultiPolygon
            coords = [[[lat, lng] for lng, lat in p.exterior.coords] for p in geom.geoms]
        
        for poly_coords in coords:
            # Try using available H3 API: prefer `polyfill`, fall back to `polygon_to_cells` if present
            if hasattr(h3, 'polyfill'):
                hexs = h3.polyfill({"type": "Polygon", "coordinates": [poly_coords]}, h3_res)
            elif hasattr(h3, 'polygon_to_cells'):
                # polygon_to_cells expects an H3 polygon shape; convert coords to (lat, lng) pairs
                latlngs = [(lat, lng) for lng, lat in poly_coords]
                try:
                    poly_obj = h3.LatLngPoly(vertices=latlngs)
                    hexs = h3.polygon_to_cells(poly_obj, h3_res)
                except Exception:
                    hexs = []
            else:
                # No polygon fill function available; skip
                hexs = []

            hexagons.extend(list(hexs))
    
    # Remove duplicates and create a DataFrame
    hexagons = list(set(hexagons))
    df_h3 = gpd.GeoDataFrame({'h3_id': hexagons})
    
    # Assign Change Intensity (Mocked from your NDVI data analysis)
    df_h3['intensity'] = np.random.uniform(0, 1, len(df_h3))
    
    # Convert H3 to Polygons for plotting
    df_h3['geometry'] = df_h3['h3_id'].apply(lambda x: Polygon([(lng, lat) for lat, lng in h3.h3_to_boundary(x)]))
    df_h3.crs = "EPSG:4326"
    return df_h3

# 2. Generate the H3 Grid (Resolution 9 is ~0.1km2 per hex)
h3_layer = get_h3_heatmap(postcodes, h3_res=9)

# 3. Create the Interactive Folium Map
m = folium.Map(location=[52.52, 13.40], zoom_start=10, tiles='cartodbpositron')

# Layer 1: H3 Hexagon Heatmap
folium.Choropleth(
    geo_data=h3_layer,
    data=h3_layer,
    columns=['h3_id', 'intensity'],
    key_on='feature.properties.h3_id',
    fill_color='YlGnBu',
    fill_alpha=0.6,
    line_weight=0.5,
    name='H3 Hexagon Heatmap'
).add_to(m)

# Layer 2: Postal Code Boundaries
folium.GeoJson(
    postcodes,
    name="Postal Codes (PLZ)",
    style_function=lambda x: {'fillColor': 'none', 'color': 'red', 'weight': 1, 'dashArray': '5, 5'}
).add_to(m)

# Layer 3: District Boundaries (Bezirke)
folium.GeoJson(
    districts,
    name="Districts (Bezirke)",
    style_function=lambda x: {'fillColor': 'none', 'color': 'black', 'weight': 2}
).add_to(m)

folium.LayerControl().add_to(m)
m.save("berlin_h3_nested_map.html")