import pandas as pd
import geopandas as gpd
import folium
import json
from pandas.api import types as pdt

# --------------------
# Load KPI data
# --------------------
district_kpi = pd.read_csv("berlin_2025_district_air_quality_kpi.csv")

# --------------------
# Load GeoJSON
# --------------------
districts_url = (
    "https://raw.githubusercontent.com/funkeinteraktiv/"
    "Berlin-Geodaten/master/berlin_bezirke.geojson"
)

gdf_districts = gpd.read_file(districts_url)
gdf_districts = gdf_districts.rename(columns={"name": "district"})

# Join KPIs
gdf_districts = gdf_districts.merge(
    district_kpi[district_kpi["year"] == 2025],
    on="district",
    how="left"
)


def sanitize_gdf_for_folium(gdf: gpd.GeoDataFrame) -> dict:
    """Return a GeoJSON-like dict where datetime-like columns are stringified.

    This prevents Timestamp objects from reaching folium's json.dumps.
    """
    g = gdf.copy()
    for col in g.columns:
        if col == g.geometry.name:
            continue
        try:
            if pdt.is_datetime64_any_dtype(g[col].dtype):
                g[col] = g[col].astype(str)
        except Exception:
            # Fallback: stringify any remaining non-serializable objects
            g[col] = g[col].apply(lambda v: None if v is None else str(v))
    return json.loads(g.to_json())

# --------------------
# Create Folium map
# --------------------
m_district = folium.Map(
    location=[52.52, 13.405],
    zoom_start=10,
    tiles="cartodbpositron"
)

district_geojson = sanitize_gdf_for_folium(gdf_districts)

folium.Choropleth(
    geo_data=district_geojson,
    data=gdf_districts,
    columns=["district", "mean_no2"],
    key_on="feature.properties.district",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.3,
    legend_name="Mean NO₂ (µg/m³) – 2025"
).add_to(m_district)

folium.GeoJson(
    district_geojson,
    tooltip=folium.GeoJsonTooltip(
        fields=["district", "mean_no2", "mean_pm10", "worst_aqi"],
        aliases=["District", "Mean NO₂", "Mean PM10", "Worst AQI"],
        localize=True
    )
).add_to(m_district)

m_district.save("berlin_district_air_quality_map.html")


# --------------------
# Load PLZ KPIs
# --------------------
plz_kpi = pd.read_csv("berlin_2025_plz_air_quality_kpi.csv")
plz_kpi["postal_code"] = plz_kpi["postal_code"].astype(str)

# --------------------
# Load PLZ GeoJSON
# --------------------
postcodes_url = (
    "https://tsb-opendata.s3.eu-central-1.amazonaws.com/plz/plz.geojson"
)

gdf_plz = gpd.read_file(postcodes_url)
gdf_plz = gdf_plz.rename(columns={"plz": "postal_code"})
gdf_plz["postal_code"] = gdf_plz["postal_code"].astype(str)

# Join KPIs
gdf_plz = gdf_plz.merge(
    plz_kpi[plz_kpi["year"] == 2025],
    on="postal_code",
    how="left"
)

plz_geojson = sanitize_gdf_for_folium(gdf_plz)

# --------------------
# Create PLZ map
# --------------------
m_plz = folium.Map(
    location=[52.52, 13.405],
    zoom_start=10,
    tiles="cartodbpositron"
)

folium.Choropleth(
    geo_data=plz_geojson,
    data=gdf_plz,
    columns=["postal_code", "mean_aqi"],
    key_on="feature.properties.postal_code",
    fill_color="RdYlGn_r",
    fill_opacity=0.6,
    line_opacity=0.2,
    legend_name="Mean AQI – 2025"
).add_to(m_plz)

folium.GeoJson(
    plz_geojson,
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "postal_code",
            "district",
            "mean_aqi",
            "pm10_exceedance_days"
        ],
        aliases=[
            "Postal Code",
            "District",
            "Mean AQI",
            "PM10 Exceedance Days"
        ],
        localize=True
    )
).add_to(m_plz)

m_plz.save("berlin_plz_air_quality_map.html")
