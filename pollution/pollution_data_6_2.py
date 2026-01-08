import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap, DualMap, Fullscreen
import json
from branca.colormap import linear
import matplotlib.pyplot as plt
import numpy as np

# --------------------
# 1. LOAD AND PREPARE DATA
# --------------------
print("Loading data...")

# Load your KPI data
district_kpi = pd.read_csv("berlin_2025_district_air_quality_kpi.csv")
plz_kpi = pd.read_csv("berlin_2025_plz_air_quality_kpi.csv")
plz_kpi["postal_code"] = plz_kpi["postal_code"].astype(str)

# Load GeoJSONs
districts_url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
postcodes_url = "https://tsb-opendata.s3.eu-central-1.amazonaws.com/plz/plz.geojson"

gdf_districts = gpd.read_file(districts_url).rename(columns={"name": "district"})
gdf_plz = gpd.read_file(postcodes_url).rename(columns={"plz": "postal_code"})
gdf_plz["postal_code"] = gdf_plz["postal_code"].astype(str)

# Merge with 2025 data
gdf_districts_2025 = gdf_districts.merge(
    district_kpi[district_kpi["year"] == 2025],
    on="district",
    how="left"
)

gdf_plz_2025 = gdf_plz.merge(
    plz_kpi[plz_kpi["year"] == 2025],
    on="postal_code",
    how="left"
)

# Calculate centroids for markers
gdf_districts_2025['centroid'] = gdf_districts_2025.geometry.centroid
gdf_districts_2025['centroid_lat'] = gdf_districts_2025.centroid.y
gdf_districts_2025['centroid_lon'] = gdf_districts_2025.centroid.x

# --------------------
# 2. CREATE INTERACTIVE FOLIUM MAP WITH MULTIPLE VIEWS
# --------------------

def create_berlin_air_quality_dashboard():
    """Create a comprehensive air quality dashboard with multiple map views"""
    
    # Base map
    m = folium.Map(
        location=[52.52, 13.405],
        zoom_start=10,
        tiles="cartodbpositron",
        control_scale=True
    )
    
    # Add fullscreen control
    Fullscreen().add_to(m)
    
    # --------------------
    # A. DISTRICT LAYER - NO2 Concentrations
    # --------------------
    # Create color scale for NO2
    min_no2 = gdf_districts_2025['mean_no2'].min()
    max_no2 = gdf_districts_2025['mean_no2'].max()
    
    colormap_no2 = linear.YlOrRd_09.scale(min_no2, max_no2)
    colormap_no2.caption = 'NO₂ Concentration (µg/m³)'
    
    # Add choropleth for districts
    folium.Choropleth(
        geo_data=gdf_districts_2025.__geo_interface__,
        data=gdf_districts_2025,
        columns=["district", "mean_no2"],
        key_on="feature.properties.district",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.5,
        line_weight=2,
        legend_name="Mean NO₂ (µg/m³) – Berlin Districts 2025",
        name="NO₂ by District",
        highlight=True,
        bins=7,
        reset=True
    ).add_to(m)
    
    # Add detailed GeoJson with tooltips for districts
    folium.GeoJson(
        gdf_districts_2025,
        name="District Details",
        style_function=lambda feature: {
            'fillColor': colormap_no2(feature['properties']['mean_no2']),
            'color': 'black',
            'weight': 1.5,
            'fillOpacity': 0.6,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["district", "mean_no2", "mean_pm10", "worst_aqi", "days_above_eu_limit"],
            aliases=["District:", "NO₂ (µg/m³):", "PM10 (µg/m³):", "Worst AQI:", "Days > EU Limit:"],
            localize=True,
            sticky=True,
            labels=True,
            style="""
                background-color: #F0EFEF;
                border: 2px solid black;
                border-radius: 3px;
                box-shadow: 3px;
            """
        ),
        popup=folium.GeoJsonPopup(
            fields=["district", "mean_no2", "mean_pm10", "worst_aqi", "days_above_eu_limit"],
            aliases=["District:", "NO₂:", "PM10:", "Worst AQI:", "Days > EU Limit:"]
        )
    ).add_to(m)
    
    # --------------------
    # B. POSTAL CODE LAYER - AQI Heatmap
    # --------------------
    # Create color scale for AQI (inverted: green=good, red=bad)
    colormap_aqi = linear.RdYlGn_11.scale(
        gdf_plz_2025['mean_aqi'].min(),
        gdf_plz_2025['mean_aqi'].max()
    )
    colormap_aqi.caption = 'Air Quality Index (Lower is Better)'
    
    # Add postal code choropleth
    folium.Choropleth(
        geo_data=gdf_plz_2025.__geo_interface__,
        data=gdf_plz_2025,
        columns=["postal_code", "mean_aqi"],
        key_on="feature.properties.postal_code",
        fill_color="RdYlGn",
        fill_opacity=0.5,
        line_opacity=0.2,
        legend_name="Mean AQI – Postal Codes 2025",
        name="AQI by Postal Code",
        overlay=True,
        show=False,  # Hidden by default
        bins=7,
        reset=True
    ).add_to(m)
    
    # Add detailed GeoJson for postal codes
    folium.GeoJson(
        gdf_plz_2025,
        name="Postal Code Details",
        style_function=lambda feature: {
            'fillColor': colormap_aqi(feature['properties']['mean_aqi']),
            'color': 'gray',
            'weight': 0.5,
            'fillOpacity': 0.4,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["postal_code", "district", "mean_aqi", "pm10_exceedance_days", "station_count"],
            aliases=["Postal Code:", "District:", "Mean AQI:", "PM10 Exceed Days:", "Stations:"],
            localize=True,
            sticky=True
        ),
        show=False  # Hidden by default
    ).add_to(m)
    
    # --------------------
    # C. ADD MARKERS FOR KEY POINTS
    # --------------------
    marker_cluster = MarkerCluster(name="District Centers").add_to(m)
    
    for idx, row in gdf_districts_2025.iterrows():
        # Determine marker color based on AQI
        aqi = row['worst_aqi']
        if pd.isna(aqi):
            color = 'gray'
        elif aqi <= 25:
            color = 'green'
        elif aqi <= 50:
            color = 'lightgreen'
        elif aqi <= 75:
            color = 'orange'
        else:
            color = 'red'
        
        # Create popup HTML
        popup_html = f"""
        <div style="width: 250px;">
            <h4>{row['district']}</h4>
            <hr>
            <b>NO₂:</b> {row['mean_no2']:.1f} µg/m³<br>
            <b>PM10:</b> {row['mean_pm10']:.1f} µg/m³<br>
            <b>Worst AQI:</b> {row['worst_aqi']}<br>
            <b>Days > EU Limit:</b> {row['days_above_eu_limit']}<br>
            <hr>
            <small>Click for details</small>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['centroid_lat'], row['centroid_lon']],
            radius=8,
            popup=folium.Popup(popup_html, max_width=300),
            color='black',
            weight=1,
            fill_color=color,
            fill_opacity=0.8,
            name=f"District Center - {row['district']}"
        ).add_to(marker_cluster)
    
    # --------------------
    # D. ADD HEATMAP FOR POLLUTION HOTSPOTS
    # --------------------
    # Create heatmap data points
    heat_data = []
    for idx, row in gdf_plz_2025.iterrows():
        centroid = row.geometry.centroid
        aqi = row['mean_aqi']
        if not pd.isna(aqi):
            # Weight by AQI (higher AQI = more intense)
            weight = min(aqi / 100, 1.0)
            heat_data.append([centroid.y, centroid.x, weight])
    
    HeatMap(
        heat_data,
        name="Pollution Hotspots",
        min_opacity=0.3,
        max_zoom=13,
        radius=20,
        blur=15,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'},
        show=False
    ).add_to(m)
    
    # --------------------
    # E. ADD LEGEND AND CONTROLS
    # --------------------
    # Add AQI legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);">
    
    <h4 style="margin-top:0;">Air Quality Index</h4>
    <div style="background-color: #00E400; padding: 2px; margin: 2px;">0-25: Good</div>
    <div style="background-color: #FFFF00; padding: 2px; margin: 2px;">26-50: Moderate</div>
    <div style="background-color: #FF7E00; padding: 2px; margin: 2px;">51-75: Unhealthy (Sensitive)</div>
    <div style="background-color: #FF0000; padding: 2px; margin: 2px;">76-100: Unhealthy</div>
    <div style="background-color: #8F3F97; padding: 2px; margin: 2px;">100+: Very Unhealthy</div>
    
    <hr style="margin: 8px 0;">
    <b>EU Limits:</b><br>
    • NO₂: 40 µg/m³<br>
    • PM10: 50 µg/m³
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add colorbars
    colormap_no2.add_to(m)
    colormap_aqi.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add title
    title_html = '''
    <h3 align="center" style="font-size:20px; margin-top:10px;">
    <b>Berlin Air Quality 2025 - May to August</b>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

# --------------------
# 3. CREATE COMPARISON MAP (2025 vs EU Limits)
# --------------------
def create_comparison_map():
    """Create a dual map comparing actual levels vs EU limits"""
    
    # Create dual map
    m_dual = DualMap(location=[52.52, 13.405], tiles="cartodbpositron", zoom_start=10)
    
    # LEFT MAP: Actual NO2 levels
    folium.Choropleth(
        geo_data=gdf_districts_2025.__geo_interface__,
        data=gdf_districts_2025,
        columns=["district", "mean_no2"],
        key_on="feature.properties.district",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.5,
        legend_name="Actual NO₂ (µg/m³)",
        name="Actual Levels"
    ).add_to(m_dual.m1)
    
    # RIGHT MAP: Comparison to EU limit (40 µg/m³)
    gdf_districts_2025['above_limit'] = gdf_districts_2025['mean_no2'] > 40
    
    def style_comparison(feature):
        return {
            'fillColor': '#ff3333' if feature['properties']['above_limit'] else '#33cc33',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6,
        }
    
    folium.GeoJson(
        gdf_districts_2025,
        name="EU Limit Comparison",
        style_function=style_comparison,
        tooltip=folium.GeoJsonTooltip(
            fields=["district", "mean_no2"],
            aliases=["District:", "NO₂:"],
            localize=True
        )
    ).add_to(m_dual.m2)
    
    # Add titles
    title_left = '<h4>Actual NO₂ Levels</h4>'
    title_right = '<h4>EU Limit (40µg/m³) Compliance</h4><div style="background:#33cc33;display:inline-block;width:20px;height:20px;"></div> Within Limit<div style="background:#ff3333;display:inline-block;width:20px;height:20px;margin-left:10px;"></div> Exceeds Limit'
    
    m_dual.m1.get_root().html.add_child(folium.Element(title_left))
    m_dual.m2.get_root().html.add_child(folium.Element(title_right))
    
    return m_dual

# --------------------
# 4. GENERATE ALL MAPS
# --------------------
print("Creating interactive maps...")

# Create main dashboard
main_map = create_berlin_air_quality_dashboard()
main_map.save("berlin_air_quality_dashboard.html")
print("✅ Main dashboard saved as 'berlin_air_quality_dashboard.html'")

# Create comparison map
comparison_map = create_comparison_map()
comparison_map.save("berlin_air_comparison_map.html")
print("✅ Comparison map saved as 'berlin_air_comparison_map.html'")

# --------------------
# 5. CREATE SIMPLE SINGLE-MAP VERSION
# --------------------
def create_simple_map():
    """Create a simple, clean map for quick viewing"""
    
    m_simple = folium.Map(location=[52.52, 13.405], zoom_start=10, tiles="cartodbpositron")
    
    # Simple AQI visualization
    folium.Choropleth(
        geo_data=gdf_districts_2025.__geo_interface__,
        data=gdf_districts_2025,
        columns=["district", "worst_aqi"],
        key_on="feature.properties.district",
        fill_color="RdYlGn_r",  # Reversed: red=bad, green=good
        fill_opacity=0.7,
        line_opacity=0.5,
        legend_name="Worst AQI 2025",
        bins=5
    ).add_to(m_simple)
    
    # Simple tooltips
    folium.GeoJson(
        gdf_districts_2025,
        tooltip=folium.GeoJsonTooltip(
            fields=["district", "worst_aqi", "mean_no2"],
            aliases=["District", "AQI", "NO₂"]
        )
    ).add_to(m_simple)
    
    m_simple.save("berlin_air_simple.html")
    print("✅ Simple map saved as 'berlin_air_simple.html'")
    
    return m_simple

create_simple_map()

# --------------------
# 6. PRINT DATA SUMMARY
# --------------------
print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)

print(f"\nDistricts analyzed: {len(gdf_districts_2025)}")
print(f"Postal codes analyzed: {len(gdf_plz_2025)}")

print("\nTop 5 districts by NO₂:")
top_no2 = gdf_districts_2025.nlargest(5, 'mean_no2')[['district', 'mean_no2']]
print(top_no2.to_string(index=False))

print("\nTop 5 districts by AQI (worst):")
top_aqi = gdf_districts_2025.nlargest(5, 'worst_aqi')[['district', 'worst_aqi']]
print(top_aqi.to_string(index=False))

print("\n" + "="*60)
print("MAPS CREATED SUCCESSFULLY!")
print("="*60)
print("1. berlin_air_quality_dashboard.html - Full interactive dashboard")
print("2. berlin_air_comparison_map.html   - Dual map comparison")
print("3. berlin_air_simple.html           - Simple AQI map")
print("\nOpen these files in your web browser to view the maps.")