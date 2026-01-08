import pandas as pd

# --------------------
# Load pollution data
# --------------------
pollution = pd.read_csv(
    "../pollution/berlin_summer_2025_plz_estimates.csv",
    parse_dates=["date"]
)

pollution["postal_code"] = pollution["postal_code"].astype(str)

# --------------------
# Load PLZ traffic supply (from GTFS stops)
# --------------------
traffic_supply = pd.read_csv(
    "plz_traffic_supply_index.csv"   # output from GTFS pipeline
)
# rename plz column to postal_code
traffic_supply = traffic_supply.rename(columns={"plz": "postal_code"})

traffic_supply["postal_code"] = traffic_supply["postal_code"].astype(str)

# --------------------
# Merge
# --------------------
df = pollution.merge(
    traffic_supply,
    on="postal_code",
    how="left"
)

# Fill missing (edge PLZs)
df["traffic_supply_index"] = df["traffic_supply_index"].fillna(0)


plz_summary = (
    df
    .groupby(["postal_code", "district"])
    .agg(
        mean_no2=("no2", "mean"),
        mean_pm10=("pm10", "mean"),
        mean_aqi=("aqi", "mean"),
        traffic_supply_index=("traffic_supply_index", "mean")
    )
    .reset_index()
)

plz_summary["mean_no2"] = plz_summary["mean_no2"].round(1)
plz_summary["mean_pm10"] = plz_summary["mean_pm10"].round(1)
plz_summary["mean_aqi"] = plz_summary["mean_aqi"].round(0)


import geopandas as gpd

postcodes_url = (
    "https://tsb-opendata.s3.eu-central-1.amazonaws.com/plz/plz.geojson"
)

gdf_plz = gpd.read_file(postcodes_url)
gdf_plz = gdf_plz.rename(columns={"plz": "postal_code"})
gdf_plz["postal_code"] = gdf_plz["postal_code"].astype(str)

# Join summary data
gdf_plz = gdf_plz.merge(
    plz_summary,
    on="postal_code",
    how="left"
)

# Save the gdf_plz with summary data
gdf_plz.to_file("berlin_plz_traffic_supply.geojson", driver="GeoJSON")

import folium

m_supply = folium.Map(
    location=[52.52, 13.405],
    zoom_start=10,
    tiles="cartodbpositron"
)

folium.Choropleth(
    geo_data=gdf_plz,
    data=gdf_plz,
    columns=["postal_code", "traffic_supply_index"],
    key_on="feature.properties.postal_code",
    fill_color="Blues",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Traffic Supply Index (GTFS-based)"
).add_to(m_supply)

folium.GeoJson(
    gdf_plz,
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "postal_code",
            "district",
            "traffic_supply_index",
            "mean_no2",
            "mean_pm10"
        ],
        aliases=[
            "PLZ",
            "District",
            "Traffic Supply Index",
            "Mean NO₂ (µg/m³)",
            "Mean PM10 (µg/m³)"
        ],
        localize=True
    )
).add_to(m_supply)

m_supply.save("berlin_plz_traffic_supply_map.html")


gdf_plz["traffic_weighted_no2"] = (
    gdf_plz["mean_no2"] * (0.6 + 0.8 * gdf_plz["traffic_supply_index"])
).round(1)

m_no2 = folium.Map(
    location=[52.52, 13.405],
    zoom_start=10,
    tiles="cartodbpositron"
)

folium.Choropleth(
    geo_data=gdf_plz,
    data=gdf_plz,
    columns=["postal_code", "traffic_weighted_no2"],
    key_on="feature.properties.postal_code",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Traffic-weighted NO₂ (µg/m³)"
).add_to(m_no2)

folium.GeoJson(
    gdf_plz,
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "postal_code",
            "district",
            "mean_no2",
            "traffic_supply_index",
            "traffic_weighted_no2"
        ],
        aliases=[
            "PLZ",
            "District",
            "Mean NO₂",
            "Traffic Supply Index",
            "Traffic-weighted NO₂"
        ]
    )
).add_to(m_no2)

m_no2.save("berlin_plz_traffic_weighted_no2_map.html")
