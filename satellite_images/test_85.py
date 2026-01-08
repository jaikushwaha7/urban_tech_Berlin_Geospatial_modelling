import os
import geopandas as gpd
import pandas as pd
import folium
import matplotlib.pyplot as plt

# Load your H3 Summary
h3_data = pd.read_csv('final_berlin_h3_comparison.csv')

# Helper: search for candidate geometry files
BASE = os.path.dirname(__file__)
BOUND_DIR = os.path.normpath(os.path.join(BASE, '..', 'berlin_heat_data', 'boundaries'))

def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

district_candidates = [
    os.path.join(BOUND_DIR, 'berlin_bezirke.geojson'),
    os.path.join(BOUND_DIR, 'berlin_boundary.geojson'),
    os.path.join(BASE, 'berlin_bezirke.geojson'),
]

postcode_candidates = [
    os.path.join(BOUND_DIR, 'berlin_plz.geojson'),
    os.path.join(BASE, 'berlin_plz.geojson'),
]

district_fp = first_existing(district_candidates)
postcode_fp = first_existing(postcode_candidates)

if district_fp:
    districts = gpd.read_file(district_fp)
    # find a suitable join key in districts
    join_key = None
    for k in ['name', 'NAME', 'bezirk', 'district']:
        if k in districts.columns:
            join_key = k
            break
    if join_key is None:
        # pick first string column
        for c in districts.columns:
            if districts[c].dtype == object:
                join_key = c
                break

    if join_key is None:
        print('No suitable join key found in district geometries; falling back to bar chart.')
        district_fp = None
    else:
        # Merge and create choropleth
        map_df = districts.merge(h3_data, left_on=join_key, right_on='District', how='left')
        map_df['Change_Ratio'] = map_df['Change_Ratio'].fillna(0)

        # stringify any datetime columns to avoid serialization errors
        for col in map_df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(map_df[col]):
                    map_df[col] = map_df[col].astype(str)
            except Exception:
                continue

        # Ensure active geometry
        if 'geometry' in map_df.columns:
            map_df = gpd.GeoDataFrame(map_df, geometry='geometry', crs=map_df.crs)

        m = folium.Map(location=[52.52, 13.40], zoom_start=11, tiles='cartodbpositron')
        folium.Choropleth(
            geo_data=map_df.__geo_interface__,
            data=map_df,
            columns=[join_key, 'Change_Ratio'],
            key_on=f'feature.properties.{join_key}',
            fill_color='YlOrRd',
            legend_name='Change Intensity per H3 Unit'
        ).add_to(m)
        m.save('Berlin_Change_Map_2025.html')
        print('Saved Berlin_Change_Map_2025.html')

if not district_fp:
    # Fallback: produce a bar chart of Change_Ratio per district
    df = h3_data.copy()
    df = df.sort_values('Change_Ratio', ascending=False)
    plt.figure(figsize=(12,6))
    plt.bar(df['District'], df['Change_Ratio'], color='crimson')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Change_Ratio')
    plt.title('Berlin District Change Intensity (fallback)')
    plt.tight_layout()
    plt.savefig('Berlin_Change_BarChart.png')
    print('District geometries not found; saved fallback bar chart: Berlin_Change_BarChart.png')

# Optional: if postcode geometries exist, create a simple postcode overlay
if postcode_fp and district_fp:
    postcodes = gpd.read_file(postcode_fp)
    # create a lightweight postcode choropleth using aggregated district values where possible
    try:
        # spatial join require consistent CRS
        postcodes = postcodes.to_crs(map_df.crs)
        h3_with_district = gpd.sjoin(map_df[['geometry','Change_Ratio']], postcodes, how='inner', predicate='intersects')
        agg = h3_with_district.groupby('index_right').agg({'Change_Ratio':'mean'}).rename(columns={'Change_Ratio':'mean_change'})
        postcodes = postcodes.reset_index().set_index('index').join(agg)
        postcodes['mean_change'] = postcodes['mean_change'].fillna(0)
        m2 = folium.Map(location=[52.52, 13.40], zoom_start=11, tiles='cartodbpositron')
        folium.Choropleth(
            geo_data=postcodes,
            data=postcodes,
            columns=[postcodes.index.name or 'index', 'mean_change'],
            key_on='feature.properties.' + (postcodes.index.name or 'index'),
            fill_color='YlGnBu',
            legend_name='Postcode mean change (from districts)',
        ).add_to(m2)
        m2.save('Berlin_Postcode_Change_Map.html')
        print('Saved Berlin_Postcode_Change_Map.html')
    except Exception as e:
        print('Failed to create postcode map:', e)