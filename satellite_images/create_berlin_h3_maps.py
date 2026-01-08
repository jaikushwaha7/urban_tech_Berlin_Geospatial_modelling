import os
import pandas as pd
import geopandas as gpd
import folium
import numpy as np

# Paths (relative to this script)
BASE = os.path.dirname(__file__)
CSV_FP = os.path.join(BASE, 'final_berlin_h3_comparison.csv')
BOUNDARIES_DIR = os.path.join(os.path.dirname(BASE), 'berlin_heat_data', 'boundaries')

# Candidate filenames
district_candidates = [
    os.path.join(BOUNDARIES_DIR, 'berlin_bezirke.geojson'),
    os.path.join(BOUNDARIES_DIR, 'berlin_boundary.geojson'),
    os.path.join(BASE, 'berlin_bezirke.geojson'),
    os.path.join(BASE, 'berlin_boundary.geojson'),
]
postcodes_candidates = [
    os.path.join(BOUNDARIES_DIR, 'berlin_plz.geojson'),
    os.path.join(BASE, 'berlin_plz.geojson'),
]
h3_grid_candidates = [
    os.path.join(BOUNDARIES_DIR, 'h3_grid_res9_kring.geojson'),
    os.path.join(BASE, 'h3_grid_res9_kring.geojson'),
]

def pick_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_inputs():
    if not os.path.exists(CSV_FP):
        raise FileNotFoundError(f"CSV not found: {CSV_FP}")
    df = pd.read_csv(CSV_FP)

    districts_fp = pick_existing(district_candidates)
    if districts_fp is None:
        raise FileNotFoundError('District geojson not found; expected one of: ' + ','.join(district_candidates))
    districts = gpd.read_file(districts_fp)

    postcodes_fp = pick_existing(postcodes_candidates)
    postcodes = gpd.read_file(postcodes_fp) if postcodes_fp else None

    h3_grid_fp = pick_existing(h3_grid_candidates)
    h3_grid = gpd.read_file(h3_grid_fp) if h3_grid_fp else None

    return df, districts, postcodes, h3_grid


def build_district_map(df, districts_gdf, out_html='berlin_district_change_map.html'):
    # Try to find matching key between districts_gdf and df
    left_key = None
    for candidate in ['name', 'NAME', 'district', 'District', 'bezirk']:
        if candidate in districts_gdf.columns:
            left_key = candidate
            break
    if left_key is None:
        # Attempt to use first string column
        for c in districts_gdf.columns:
            if districts_gdf[c].dtype == object:
                left_key = c
                break
    if left_key is None:
        raise RuntimeError('No suitable join key found in districts GeoDataFrame')

    map_df = districts_gdf.merge(df, how='left', left_on=left_key, right_on='District')
    # Fill NaNs
    map_df['Change_Ratio'] = map_df['Change_Ratio'].fillna(0)

    m = folium.Map(location=[52.52, 13.40], zoom_start=11, tiles='cartodbpositron')
    folium.Choropleth(
        geo_data=map_df,
        data=map_df,
        columns=[left_key, 'Change_Ratio'],
        key_on=f'feature.properties.{left_key}',
        fill_color='YlOrRd',
        legend_name='Change Intensity per H3 Unit',
        name='District Change'
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(out_html)
    print('Saved district map to', out_html)
    return out_html


def build_postcode_map(h3_grid, districts_gdf, postcodes_gdf, out_html='berlin_postcode_change_map.html'):
    if h3_grid is None:
        raise RuntimeError('H3 grid geojson not available to aggregate to postcodes')
    if postcodes_gdf is None:
        raise RuntimeError('Postcodes geojson not available')

    # Ensure same CRS
    if districts_gdf.crs != h3_grid.crs:
        h3_grid = h3_grid.to_crs(districts_gdf.crs)
    if postcodes_gdf.crs != districts_gdf.crs:
        postcodes_gdf = postcodes_gdf.to_crs(districts_gdf.crs)

    # Spatial join: assign district attributes (Change_Ratio) to H3 hexes
    # For this we need districts to have Change_Ratio
    if 'Change_Ratio' not in districts_gdf.columns:
        raise RuntimeError('districts_gdf must contain Change_Ratio column (merge CSV first)')

    # Spatial join h3_grid with districts
    h3_with_district = gpd.sjoin(h3_grid, districts_gdf[[ 'geometry', 'Change_Ratio' ]], how='left', predicate='intersects')
    # Now aggregate to postcodes by intersecting h3 polygons with postcodes
    h3_post = gpd.sjoin(h3_with_district, postcodes_gdf[['geometry']], how='inner', predicate='intersects')

    # Compute mean Change_Ratio per postcode geometry index
    agg = h3_post.groupby('index_right').agg({'Change_Ratio': 'mean'}).rename(columns={'Change_Ratio':'mean_change_ratio'})
    postcodes_gdf = postcodes_gdf.reset_index().set_index('index').join(agg)
    postcodes_gdf['mean_change_ratio'] = postcodes_gdf['mean_change_ratio'].fillna(0)

    m = folium.Map(location=[52.52, 13.40], zoom_start=11, tiles='cartodbpositron')
    folium.Choropleth(
        geo_data=postcodes_gdf,
        data=postcodes_gdf,
        columns=[postcodes_gdf.index.name or 'index', 'mean_change_ratio'],
        key_on='feature.properties.' + (postcodes_gdf.index.name or 'index'),
        fill_color='YlGnBu',
        legend_name='Average Change per Postcode (from H3 aggregates)',
        name='Postcode Change'
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(out_html)
    print('Saved postcode map to', out_html)
    return out_html


if __name__ == '__main__':
    df, districts, postcodes, h3_grid = load_inputs()

    # Prepare districts: merge CSV
    districts = districts.merge(df, how='left', left_on='name', right_on='District') if 'name' in districts.columns else districts.merge(df, how='left', left_on=districts.columns[0], right_on='District')
    districts['Change_Ratio'] = districts['Change_Ratio'].fillna(0)

    # Save district map
    try:
        build_district_map(df, districts, out_html='Berlin_Change_Map_Districts.html')
    except Exception as e:
        print('Failed to build district map:', e)

    # Build postcode map via H3 aggregation if possible
    try:
        if postcodes is not None and h3_grid is not None:
            build_postcode_map(h3_grid, districts, postcodes, out_html='Berlin_Change_Map_Postcodes.html')
        else:
            print('Skipping postcode map; missing postcodes or h3_grid geojson')
    except Exception as e:
        print('Failed to build postcode map:', e)
