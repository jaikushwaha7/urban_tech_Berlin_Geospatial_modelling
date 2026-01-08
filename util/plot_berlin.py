# generate berlin on map using berlin_boundary.geojson using matplotlib and folium
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import pandas as pd
import hashlib
import numpy as np

# Optional: for consistent categorical coloring
def hash_to_int(s, n_colors=20):
    return int(hashlib.md5(str(s).encode()).hexdigest(), 16) % n_colors

# ------------------------------
# Berlin Boundary (GeoJSON)
# ------------------------------
def plot_berlin_matplotlib(geojson_path="../data/boundaries/berlin_boundary.geojson", output_path="berlin_matplotlib.png"):
    try:
        berlin_boundary = gpd.read_file(geojson_path)
    except FileNotFoundError:
        print(f"Error: GeoJSON file not found at {geojson_path}")
        return
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    berlin_boundary.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=1)
    ax.set_title("Berlin Boundary (Matplotlib)", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Matplotlib map saved to {output_path}")


def plot_berlin_folium(geojson_path="../data/boundaries/berlin_boundary.geojson", output_path="berlin_folium.html"):
    try:
        berlin_boundary = gpd.read_file(geojson_path)
    except FileNotFoundError:
        print(f"Error: GeoJSON file not found at {geojson_path}")
        return
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        return

    if not berlin_boundary.empty and not berlin_boundary.geometry.is_empty.all():
        centroid = berlin_boundary.geometry.unary_union.centroid
        location = [centroid.y, centroid.x]
    else:
        location = [52.52, 13.405]  # fallback

    m = folium.Map(location=location, zoom_start=11, tiles="CartoDB positron")

    folium.GeoJson(
        berlin_boundary.__geo_interface__,
        name="Berlin Boundary",
        style_function=lambda x: {
            'fillColor': '#d3d3d3',
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.6
        },
        tooltip=folium.Tooltip("Berlin City Boundary")
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"✅ Folium map saved to {output_path}")


# ------------------------------
# Districts (CSV → WKT)
# ------------------------------
def load_gdf_from_csv_wkt(csv_path, geom_col='geometry', crs="EPSG:4326", encoding='utf-8'):
    """
    Load GeoDataFrame from CSV with WKT geometry (semicolon-delimited).
    Handles column renaming for Berlin-specific files.
    """
    try:
        # Use sep=';' and engine='python' to handle WKT commas & irregular quoting
        df = pd.read_csv(csv_path, sep=';', engine='python', encoding=encoding, skipinitialspace=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except Exception as e:
        raise Exception(f"Failed to read CSV (sep=';'): {e}")

    # Normalize column names: handle 'Bezirk' → 'district', 'PLZ' → 'postal_code'
    df.columns = df.columns.str.strip()  # Remove whitespace
    if 'Bezirk' in df.columns:
        df = df.rename(columns={'Bezirk': 'district'})
    elif 'PLZ' in df.columns:
        df = df.rename(columns={'PLZ': 'postal_code'})
    
    # Ensure required columns exist
    if geom_col not in df.columns:
        raise ValueError(f"Geometry column '{geom_col}' not found. Available: {list(df.columns)}")
    
    # Drop rows with missing geometry
    df = df.dropna(subset=[geom_col]).copy()

    # Parse WKT safely — handle potential 'MULTIPOLYGON Z' or 'SRID=...' prefixes
    def clean_wkt(wkt):
        if pd.isna(wkt):
            return None
        wkt = str(wkt).strip()
        # Remove SRID prefix if present: "SRID=4326;POLYGON (...)"
        if wkt.startswith('SRID='):
            wkt = wkt.split(';', 1)[-1]
        return wkt

    df[geom_col] = df[geom_col].apply(clean_wkt)
    
    try:
        geometry = gpd.GeoSeries.from_wkt(df[geom_col], crs=crs)
    except Exception as e:
        # Try to print first few problematic rows
        invalids = []
        for i, wkt in enumerate(df[geom_col].head(10)):
            try:
                _ = gpd.GeoSeries.from_wkt([wkt], crs=crs)
            except:
                invalids.append((i, wkt[:100] + '...'))
        raise Exception(f"Error parsing WKT. First few invalid entries: {invalids[:3]}\nUnderlying error: {e}")

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    print(f"✅ Loaded {len(gdf)} valid features from {csv_path}")
    return gdf


def plot_berlin_districts_matplotlib(csv_path="../data/geodata_berlin_dis.csv", output_path="berlin_districts_matplotlib.png"):
    try:
        gdf_districts = load_gdf_from_csv_wkt(csv_path)
    except Exception as e:
        print(f"❌ Districts (Matplotlib): {e}")
        return

    # Assign consistent colors via hashing
    gdf_districts['color_idx'] = gdf_districts['district'].apply(lambda x: hash_to_int(x, n_colors=12))
    cmap = plt.cm.get_cmap('tab20', 20)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    for idx, row in gdf_districts.iterrows():
        ax.fill(*row.geometry.exterior.xy, 
                color=cmap(row['color_idx']/19), 
                edgecolor='black', linewidth=0.8, alpha=0.8)

    # Label districts (avoid overlap if possible)
    for idx, row in gdf_districts.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, row['district'],
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.set_title("Berlin Districts (Matplotlib)", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Matplotlib district map saved to {output_path}")


def plot_berlin_districts_folium(csv_path="../data/geodata_berlin_dis.csv", output_path="berlin_districts_folium.html"):
    try:
        gdf_districts = load_gdf_from_csv_wkt(csv_path)
    except Exception as e:
        print(f"❌ Districts (Folium): {e}")
        return

    m = folium.Map(location=[52.52, 13.405], zoom_start=10, tiles="CartoDB positron")

    # Define style per district
    def style_fn(feature):
        name = feature['properties']['district']
        color_idx = hash_to_int(name, n_colors=12)
        hex_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        rgb = (hex_colors[color_idx][:3] * 255).astype(int)
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
        return {
            'fillColor': hex_color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7,
            'dashArray': '2'
        }

    folium.GeoJson(
        gdf_districts.to_json(),
        name="Districts",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=['district'], aliases=['District:']),
        highlight_function=lambda x: {'weight': 3, 'color': 'gold', 'fillOpacity': 0.9}
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"✅ Folium district map saved to {output_path}")


# ------------------------------
# Postcodes (CSV → WKT)
# ------------------------------
def plot_berlin_postcodes_matplotlib(csv_path="../data/geodata_berlin_plz.csv", output_path="berlin_postcodes_matplotlib.png"):
    try:
        gdf_plz = load_gdf_from_csv_wkt(csv_path)
    except Exception as e:
        print(f"❌ Postcodes (Matplotlib): {e}")
        return

    gdf_plz['color_idx'] = gdf_plz['postal_code'].apply(lambda x: hash_to_int(x, n_colors=20))
    cmap = plt.cm.get_cmap('tab20', 20)

    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    for idx, row in gdf_plz.iterrows():
        if row.geometry.geom_type == 'Polygon':
            ax.fill(*row.geometry.exterior.xy,
                    color=cmap(row['color_idx']/19),
                    edgecolor='gray', linewidth=0.3, alpha=0.8)
        elif row.geometry.geom_type == 'MultiPolygon':
            for part in row.geometry.geoms:
                ax.fill(*part.exterior.xy,
                        color=cmap(row['color_idx']/19),
                        edgecolor='gray', linewidth=0.3, alpha=0.8)

    # Optional: only label larger postcodes or use offset
    for idx, row in gdf_plz.iterrows():
        if row.geometry.area > 1e-4:  # Adjust threshold as needed
            centroid = row.geometry.centroid
            ax.text(centroid.x, centroid.y, str(row['postal_code']),
                    fontsize=6, ha='center', va='center',
                    color='black', alpha=0.9)

    ax.set_title("Berlin Postcodes (Matplotlib)", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Matplotlib postcode map saved to {output_path}")


def plot_berlin_postcodes_folium(csv_path="../data/geodata_berlin_plz.csv", output_path="berlin_postcodes_folium.html"):
    try:
        gdf_plz = load_gdf_from_csv_wkt(csv_path)
    except Exception as e:
        print(f"❌ Postcodes (Folium): {e}")
        return

    m = folium.Map(location=[52.52, 13.405], zoom_start=11, tiles="CartoDB positron")

    def style_fn(feature):
        plz = str(feature['properties']['postal_code'])
        color_idx = hash_to_int(plz, n_colors=20)
        hex_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        rgb = (hex_colors[color_idx][:3] * 255).astype(int)
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
        return {
            'fillColor': hex_color,
            'color': '#666',
            'weight': 0.7,
            'fillOpacity': 0.7
        }

    folium.GeoJson(
        gdf_plz.to_json(),
        name="Postcodes",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=['postal_code'], aliases=['PostalCodes:']),
        highlight_function=lambda x: {'weight': 2, 'color': 'red', 'fillOpacity': 0.95}
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"✅ Folium postcode map saved to {output_path}")


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    print("🚀 Generating Berlin maps...\n")

    # Boundary
    plot_berlin_matplotlib()
    plot_berlin_folium()

    # Districts
    plot_berlin_districts_matplotlib()
    plot_berlin_districts_folium()

    # Postcodes
    plot_berlin_postcodes_matplotlib()
    plot_berlin_postcodes_folium()

    print("\n🎉 All maps generated successfully!")