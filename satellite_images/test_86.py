import geopandas as gpd
import shapely.geometry as sg
import numpy as np
import folium
from tqdm import tqdm   # nice progress bar (optional)
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


# optional: spatial join with postcodes if you want plz column
if 'postcodes' in locals() and not postcodes.empty:
    postcodes = postcodes.to_crs(hex_grid.crs)
    hex_grid = gpd.sjoin(hex_grid, postcodes[['plz', 'geometry']], how='left', predicate='intersects')
    hex_grid.drop(columns='index_right', inplace=True)

print(f'{len(hex_grid)} hexagons created')
hex_grid.head()

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

m

m.save('berlin_hex.html')
hex_grid.to_file('berlin_hex_500m.geojson', driver='GeoJSON')# ------------------------------------------------------------------
# 3.  Generate H3 grid for comparison
# ------------------------------------------------------------------
def get_h3_grid_for_districts(districts_gdf, h3_res=9):
    """Generates H3 hexagons within the target GeoDataFrame."""
    # Convert to WGS84 for H3
    districts_wgs84 = districts_gdf.to_crs(4326)
    
    hexagons = set()
    for idx, dist in districts_wgs84.iterrows():
        geom = dist.geometry  # Already in WGS84
        
        # Use shapely geometry directly with h3
        try:
            if geom.geom_type == 'Polygon':
                hexs = h3.polygon_to_cells(geom, h3_res)
                print(f"District {idx}: Found {len(hexs)} hexagons")
                hexagons.update(hexs)
            elif geom.geom_type == 'MultiPolygon':
                for i, poly in enumerate(geom.geoms):
                    hexs = h3.polygon_to_cells(poly, h3_res)
                    print(f"District {idx} (poly {i}): Found {len(hexs)} hexagons")
                    hexagons.update(hexs)
        except Exception as e:
            print(f"Warning: Failed to convert district {idx}: {e}")
            continue
    
    print(f"Total hexagons before dedup: {len(hexagons)}")
    # Convert to list and create DataFrame
    hexagons = list(hexagons)
    print(f"Total unique hexagons: {len(hexagons)}")
    df_h3 = gpd.GeoDataFrame({'h3_id': hexagons})
    
    # Convert H3 to Polygons for plotting
    df_h3['geometry'] = df_h3['h3_id'].apply(lambda x: Polygon([(lng, lat) for lat, lng in h3.h3_to_boundary(x)]))
    df_h3.crs = "EPSG:4326"
    return df_h3

h3_grid_gdf = get_h3_grid_for_districts(districts, h3_res=9)
h3_grid_gdf.to_file('h3_grid_res9_kring.geojson', driver='GeoJSON')
print(f'{len(h3_grid_gdf)} H3 hexagons (res 9) created and saved to h3_grid_res9_kring.geojson')

# ------------------------------------------------------------------
# 4.  Visualize maps
# ------------------------------------------------------------------
print("\n=== HEXAGON GRID MAP ===")
print(f"Saved to: berlin_hex.html")
print(f"Total hexagons: {len(map_df)}")

# Create H3 map for comparison
print("\n=== H3 GRID MAP ===")
m_h3 = folium.Map(location=[52.52, 13.40], zoom_start=10, tiles='CartoDB positron')
for _, r in h3_grid_gdf.iterrows():
    folium.GeoJson(
        mapping(r.geometry),
        style_function=lambda x: {
            'fillColor': '#1f77b4',
            'color': 'navy',
            'weight': 0.5,
            'fillOpacity': 0.5
        }
    ).add_to(m_h3)
m_h3.save('berlin_h3_grid.html')
print(f"Saved to: berlin_h3_grid.html")
print(f"Total H3 hexagons: {len(h3_grid_gdf)}")

# Create comparison map with both layers
print("\n=== COMPARISON MAP ===")
m_compare = folium.Map(location=[52.52, 13.40], zoom_start=10, tiles='CartoDB positron')

# Add hex grid
folium.FeatureGroup(name='Hexagon Grid (500m)', show=True).add_to(m_compare)
for _, r in map_df.iterrows():
    folium.GeoJson(
        r.geometry,
        style_function=lambda x, col=color_map[r.district]: {
            'fillColor': col,
            'color': 'grey',
            'weight': 0.5,
            'fillOpacity': 0.4
        }
    ).add_to(m_compare)

# Add H3 grid
folium.FeatureGroup(name='H3 Grid (res 9)', show=False).add_to(m_compare)
for _, r in h3_grid_gdf.iterrows():
    folium.GeoJson(
        mapping(r.geometry),
        style_function=lambda x: {
            'fillColor': 'red',
            'color': 'darkred',
            'weight': 0.5,
            'fillOpacity': 0.3
        }
    ).add_to(m_compare)

folium.LayerControl().add_to(m_compare)
m_compare.save('berlin_grid_comparison.html')
print(f"Saved to: berlin_grid_comparison.html")

print("\n✓ All maps created successfully!")
