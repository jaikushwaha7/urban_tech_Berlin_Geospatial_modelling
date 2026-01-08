import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon
import json
from datetime import datetime
import os
from shapely import wkt

# Page configuration
st.set_page_config(
    page_title="Berlin Urban Heat Analysis",
    page_icon="🌡️",
    layout="wide"
)

# Constants
BERLIN_CENTER = [52.52, 13.405]

# Hexagon generation functions (from file 2)
def hexagon(center, radius):
    """
    Return a POLYGON for a regular hexagon with *flat top*.
    radius = distance centre ⟺ side (also called 'hex size').
    """
    cx, cy = center
    angle = np.linspace(0, 2*np.pi, 7)
    x = cx + radius * np.cos(angle)
    y = cy + radius * np.sin(angle + np.pi/6)
    return Polygon(np.column_stack([x, y]))

def hex_grid_over_polygon(poly, radius):
    """
    Return list of hexagons whose centroid is inside `poly`.
    radius in the same units as the CRS (metres).
    """
    minx, miny, maxx, maxy = poly.bounds
    dx = 3/2 * radius
    dy = np.sqrt(3) * radius
    
    xcoords = np.arange(minx - dx, maxx + dx, dx)
    ycoords = np.arange(miny - dy, maxy + dy, dy)
    
    centers = []
    for j, y in enumerate(ycoords):
        offset = 0 if j % 2 == 0 else dx/2
        for x in xcoords:
            centers.append((x + offset, y))
    
    hexagons = [hexagon(c, radius) for c in centers if poly.contains(Point(c))]
    return hexagons

# Data loading functions
@st.cache_data
def load_berlin_districts():
    """Load Berlin district boundaries"""
    try:
        url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
        districts = gpd.read_file(url)
        
        for col in districts.columns:
            if pd.api.types.is_datetime64_any_dtype(districts[col]):
                districts[col] = districts[col].astype(str)
        
        return districts
    except Exception as e:
        st.error(f"Error loading districts: {e}")
        return None

@st.cache_data
def load_berlin_postcodes():
    """Load Berlin postal code boundaries"""
    try:
        url = "https://tsb-opendata.s3.eu-central-1.amazonaws.com/plz/plz.geojson"
        postcodes = gpd.read_file(url)
        postcodes = postcodes.rename(columns={"plz": "postal_code"})
        postcodes["postal_code"] = postcodes["postal_code"].astype(str)
        return postcodes
    except Exception as e:
        st.warning(f"Could not load postcodes: {e}")
        return None

@st.cache_data
def load_environmental_data():
    """Load sample environmental data"""

    # make postal code list for all codes in berlin
    # can use the data from berlin population where al berlin postal code are there
    # Load Berlin postal codes from a local file or URL
    try:
        berlin_plz_path = os.path.join(os.path.dirname(__file__), 'berlin_plz.geojson')
        if os.path.exists(berlin_plz_path):
            berlin_plz = gpd.read_file(berlin_plz_path)
        else:
            berlin_plz = gpd.read_file("https://tsb-opendata.s3.eu-central-1.amazonaws.com/plz/plz.geojson")
        
        # Extract unique postal codes
        plz_list = berlin_plz['plz'].unique().tolist()
    except Exception as e:
        st.warning(f"Could not load Berlin postal codes for environmental data generation: {e}. Using a default list.")
        
    data = {
        'postal_code': plz_list,
        'temperature_change': np.random.uniform(1.5, 3.0, len(plz_list)),
        'pollution_index': np.random.uniform(0.4, 1.0, len(plz_list)),
        'population_density': np.random.uniform(8000, 16000, len(plz_list)),
        'traffic_supply': np.random.uniform(0.5, 1.0, len(plz_list)),
        'no2_avg': np.random.uniform(20, 50, len(plz_list)),
        'pm10_avg': np.random.uniform(15, 40, len(plz_list))
    }
    
    return pd.DataFrame(data)

@st.cache_data
def generate_hex_grid_district(districts_geojson, radius_m):
    """Generate hexagonal grid at district level"""
    districts = gpd.GeoDataFrame.from_features(districts_geojson)
    
    # Convert to metric CRS if needed
    if districts.crs is None or districts.crs.is_geographic:
        districts = districts.set_crs('EPSG:4326', allow_override=True)
        districts = districts.to_crs('EPSG:25833')
    
    hex_rows = []
    for _, dist in districts.iterrows():
        pol = dist.geometry
        if pol.is_empty or pol is None:
            continue
        
        hex_polys = hex_grid_over_polygon(pol, radius_m)
        district_name = dist.get('name', dist.get('Gemeinde_name', 'unknown'))
        district_id = dist.get('id', dist.get('BEZ', -1))
        
        for h in hex_polys:
            hex_rows.append({
                'geometry': h,
                'district': district_name,
                'district_id': district_id,
                'level': 'district'
            })
    
    hex_grid = gpd.GeoDataFrame(hex_rows, crs='EPSG:25833')
    return hex_grid

@st.cache_data
def generate_hex_grid_postcode(postcodes_geojson, radius_m):
    """Generate hexagonal grid at postal code level"""
    postcodes = gpd.GeoDataFrame.from_features(postcodes_geojson)
    
    # Convert to metric CRS
    if postcodes.crs is None or postcodes.crs.is_geographic:
        postcodes = postcodes.set_crs('EPSG:4326', allow_override=True)
        postcodes = postcodes.to_crs('EPSG:25833')
    
    hex_rows = []
    for _, postcode in postcodes.iterrows():
        pol = postcode.geometry
        if pol.is_empty or pol is None:
            continue
        
        hex_polys = hex_grid_over_polygon(pol, radius_m)
        plz = postcode.get('postal_code', postcode.get('plz', 'unknown'))
        
        for h in hex_polys:
            hex_rows.append({
                'geometry': h,
                'postal_code': str(plz),
                'level': 'postcode'
            })
    
    hex_grid = gpd.GeoDataFrame(hex_rows, crs='EPSG:25833')
    return hex_grid

def merge_environmental_data(hex_grid, env_data):
    """Merge environmental data with hexagon grid"""
    if 'postal_code' in hex_grid.columns:
        hex_grid = hex_grid.merge(env_data, on='postal_code', how='left')
    return hex_grid

# Visualization functions
def get_color_for_value(value, vmin, vmax, colorscale='RdYlGn_r'):
    """Get hex color for a value"""
    if pd.isna(value):
        return '#cccccc'
    
    normalized = (value - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)
    
    if colorscale == 'RdYlGn_r':
        r = int(255 * normalized)
        g = int(255 * (1 - normalized))
        b = 50
    elif colorscale == 'Blues':
        r = int(100 + 155 * normalized)
        g = int(150 + 105 * normalized)
        b = int(200 + 55 * normalized)
    else:
        r = g = b = int(255 * normalized)
    
    return f'#{r:02x}{g:02x}{b:02x}'

def create_folium_map(hex_grid, data_layer, show_districts, show_postcodes, districts, postcodes):
    """Create interactive Folium map"""
    m = folium.Map(
        location=BERLIN_CENTER,
        zoom_start=11,
        tiles='CartoDB positron'
    )
    
    # Add district boundaries
    if show_districts and districts is not None:
        folium.GeoJson(
            districts,
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': '#2c3e50',
                'weight': 2,
                'fillOpacity': 0
            },
            tooltip=folium.GeoJsonTooltip(fields=['name'], labels=False)
        ).add_to(m)
    
    # Add postal code boundaries
    if show_postcodes and postcodes is not None:
        folium.GeoJson(
            postcodes,
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': '#e74c3c',
                'weight': 1,
                'fillOpacity': 0
            },
            tooltip=folium.GeoJsonTooltip(fields=['postal_code'], labels=False)
        ).add_to(m)
    
    # Add hexagons
    if data_layer != 'none' and data_layer in hex_grid.columns:
        # Convert to WGS84 for Folium
        hex_grid_wgs84 = hex_grid.to_crs('EPSG:4326')
        values = hex_grid_wgs84[data_layer].dropna()
        
        if len(values) > 0:
            vmin, vmax = values.min(), values.max()
            
            for idx, row in hex_grid_wgs84.iterrows():
                value = row[data_layer]
                color = get_color_for_value(value, vmin, vmax)
                
                location_id = row.get('postal_code', row.get('district', 'N/A'))
                tooltip_text = f"""
                <b>Location:</b> {location_id}<br>
                <b>{data_layer}:</b> {value:.2f}" if pd.notna(value) else f"<b>{data_layer}:</b> N/A"
                """
                
                folium.GeoJson(
                    row['geometry'],
                    style_function=lambda x, col=color: {
                        'fillColor': col,
                        'color': 'black',
                        'weight': 0.5,
                        'fillOpacity': 0.6
                    },
                    tooltip=tooltip_text
                ).add_to(m)
    
    return m

def add_heatmap_to_map(m, gdf, value_col):
    # Use centroid coordinates and value for heatmap
    heat_data = [
        [row.geometry.centroid.y, row.geometry.centroid.x, row[value_col]]
        for _, row in gdf.iterrows() if pd.notna(row[value_col])
    ]
    HeatMap(heat_data, radius=15, blur=10, min_opacity=0.4).add_to(m)
    return m

def create_correlation_plot(hex_grid, x_var, y_var):
    """Create scatter plot showing correlation"""
    fig = px.scatter(
        hex_grid,
        x=x_var,
        y=y_var,
        color='postal_code' if 'postal_code' in hex_grid.columns else 'district',
        title=f'{y_var} vs {x_var}',
        trendline='ols'
    )
    return fig

def create_distribution_plot(hex_grid, var):
    """Create distribution plot"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=hex_grid[var].dropna(),
        nbinsx=30,
        name=var
    ))
    fig.update_layout(
        title=f'Distribution of {var}',
        xaxis_title=var,
        yaxis_title='Count'
    )
    return fig

# Main app
def main():
    st.title("🌡️ Berlin Urban Heat & Environmental Analysis")
    st.markdown("**Multi-source spatial analysis with hexagonal grids**")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    # Grid configuration
    st.sidebar.subheader("🔷 Grid Configuration")
    
    grid_level = st.sidebar.radio(
        "Grid Level",
        options=['district', 'postcode'],
        format_func=lambda x: {
            'district': '🏛️ District Level',
            'postcode': '📮 Postal Code Level'
        }[x]
    )
    
    # Hexagon size in meters
    hex_size = st.sidebar.slider(
        "Hexagon Size (meters)",
        min_value=200,
        max_value=1000,
        value=500,
        step=50,
        help="Size of each hexagon in meters"
    )
    
    # Data layer selection
    data_layer = st.sidebar.selectbox(
        "📊 Data Layer",
        options=['none', 'temperature_change', 'pollution_index', 'population_density', 'traffic_supply'],
        format_func=lambda x: {
            'none': 'None',
            'temperature_change': '🌡️ Temperature Change (ΔLST)',
            'pollution_index': '💨 Pollution Index',
            'population_density': '👥 Population Density',
            'traffic_supply': '🚇 Traffic Supply Index'
        }[x]
    )
    
    # Layer visibility
    st.sidebar.subheader("🗺️ Map Layers")
    show_hexgrid = st.sidebar.checkbox("Show Hexagonal Grid", value=True)
    show_districts = st.sidebar.checkbox("Show Districts", value=True)
    show_postcodes = st.sidebar.checkbox("Show Postal Codes", value=False)
    
    # Analysis options
    st.sidebar.subheader("📈 Analysis")
    show_statistics = st.sidebar.checkbox("Show Statistics", value=True)
    show_correlations = st.sidebar.checkbox("Show Correlations", value=False)
    show_distributions = st.sidebar.checkbox("Show Distributions", value=False)
    
    # Load data
    with st.spinner("Loading geographic data..."):
        districts = load_berlin_districts()
        postcodes = load_berlin_postcodes()
        env_data = load_environmental_data()
    
    if districts is None:
        st.error("Failed to load district data. Please check your connection.")
        return
    
    # Generate hexagon grid
    with st.spinner(f"Generating {grid_level}-level hexagonal grid..."):
        if grid_level == 'district':
            hex_grid = generate_hex_grid_district(districts.__geo_interface__, hex_size)
        else:
            if postcodes is not None:
                hex_grid = generate_hex_grid_postcode(postcodes.__geo_interface__, hex_size)
            else:
                st.error("Postal code data unavailable. Falling back to district level.")
                hex_grid = generate_hex_grid_district(districts.__geo_interface__, hex_size)
        
        # Merge environmental data
        hex_grid = merge_environmental_data(hex_grid, env_data)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Interactive Map")
        
        if show_hexgrid:
            folium_map = create_folium_map(
                hex_grid, 
                data_layer, 
                show_districts, 
                show_postcodes,
                districts,
                postcodes
            )
            st_folium(folium_map, width=800, height=600)
        else:
            st.info("Enable 'Show Hexagonal Grid' in the sidebar to display the map")
    
    with col2:
        st.subheader("📊 Summary Statistics")
        
        # Overall stats
        st.metric("Total Hexagons", f"{len(hex_grid):,}")
        st.metric("Grid Level", grid_level.capitalize())
        st.metric("Hexagon Size", f"{hex_size}m")
        
        # Data layer stats
        if data_layer != 'none' and data_layer in hex_grid.columns:
            values = hex_grid[data_layer].dropna()
            if len(values) > 0:
                st.markdown(f"**{data_layer}**")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Min", f"{values.min():.2f}")
                col_b.metric("Mean", f"{values.mean():.2f}")
                col_c.metric("Max", f"{values.max():.2f}")
        
        # Export button
        if st.button("📥 Export Data (JSON)"):
            # Convert to WGS84 for export
            hex_grid_export = hex_grid.to_crs('EPSG:4326')
            
            export_data = {
                'metadata': {
                    'total_hexagons': len(hex_grid_export),
                    'hex_size_meters': hex_size,
                    'grid_level': grid_level,
                    'timestamp': datetime.now().isoformat()
                },
                'hexagons': hex_grid_export.drop(columns=['geometry']).to_dict('records')
            }
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"berlin_heat_{grid_level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Additional analysis sections
    if show_statistics:
        st.markdown("---")
        st.subheader("📈 Detailed Statistics")
        
        numeric_cols = ['temperature_change', 'pollution_index', 'population_density', 'traffic_supply']
        available_cols = [col for col in numeric_cols if col in hex_grid.columns]
        
        if available_cols:
            summary_stats = hex_grid[available_cols].describe()
            st.dataframe(summary_stats)
    
    if show_correlations and len(hex_grid) > 0:
        st.markdown("---")
        st.subheader("🔗 Correlation Analysis")
        
        numeric_cols = ['temperature_change', 'pollution_index', 'population_density', 'traffic_supply']
        available_cols = [col for col in numeric_cols if col in hex_grid.columns]
        
        if len(available_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_correlation_plot(hex_grid, available_cols[0], available_cols[1])
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                if len(available_cols) >= 3:
                    fig2 = create_correlation_plot(hex_grid, available_cols[2], available_cols[1])
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Correlation matrix
            corr_matrix = hex_grid[available_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                title="Correlation Matrix",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    if show_distributions and len(hex_grid) > 0:
        st.markdown("---")
        st.subheader("📊 Distribution Analysis")
        
        numeric_cols = ['temperature_change', 'pollution_index', 'population_density', 'traffic_supply']
        available_cols = [col for col in numeric_cols if col in hex_grid.columns]
        
        if available_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_distribution_plot(hex_grid, available_cols[0])
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                if len(available_cols) >= 2:
                    fig2 = create_distribution_plot(hex_grid, available_cols[1])
                    st.plotly_chart(fig2, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **📊 Data Sources:**
    - District boundaries: Berlin Open Data
    - Postal codes: TSB Open Data
    - LST: Sentinel/Landsat satellite imagery
    - Environmental data: Sample/placeholder data
    """)

# Add tabs to Streamlit app
main_tab, heatmap_tab = st.tabs(["Main", "Heatmap Viewer"])

with heatmap_tab:
    st.header("District and Postal Code Heatmaps")

    # Load population data
    pop_df = pd.read_csv("data/berlin_population_2022_english.csv")
    pop_df["postal_code"] = pop_df["postal_code"].astype(str)
    pop_df["area_name"] = pop_df["area_name"].astype(str)

    # Load districts from CSV
    dis_df = pd.read_csv("data/geodata_berlin_dis.csv", sep=";")
    dis_df["area_name"] = dis_df["Bezirk"].astype(str) if "Bezirk" in dis_df.columns else dis_df[dis_df.columns[0]].astype(str)
    dis_df["geometry"] = dis_df["geometry"].apply(wkt.loads)
    districts = gpd.GeoDataFrame(dis_df, geometry="geometry", crs="EPSG:4326")
    # Merge population by area_name
    districts = districts.merge(pop_df[["area_name", "population_2022"]], on="area_name", how="left")

    # Load postcodes from CSV
    plz_df = pd.read_csv("data/geodata_berlin_plz.csv", sep=";")
    plz_df["postal_code"] = plz_df["PLZ"].astype(str) if "PLZ" in plz_df.columns else plz_df[plz_df.columns[0]].astype(str)
    plz_df["geometry"] = plz_df["geometry"].apply(wkt.loads)
    postcodes = gpd.GeoDataFrame(plz_df, geometry="geometry", crs="EPSG:4326")
    # Merge population by postal_code
    postcodes = postcodes.merge(pop_df[["postal_code", "population_2022"]], on="postal_code", how="left")

    # District heatmap
    m_district = folium.Map(location=[52.52, 13.405], zoom_start=10)
    add_heatmap_to_map(m_district, districts, "population_2022")
    st.subheader("District Heatmap (Population 2022)")
    st_folium(m_district, width=700, height=500)

    # Postal code heatmap
    m_postcode = folium.Map(location=[52.52, 13.405], zoom_start=10)
    add_heatmap_to_map(m_postcode, postcodes, "population_2022")
    st.subheader("Postal Code Heatmap (Population 2022)")
    st_folium(m_postcode, width=700, height=500)

if __name__ == "__main__":
    main()