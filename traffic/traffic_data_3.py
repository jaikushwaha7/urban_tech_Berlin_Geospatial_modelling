import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import TimeSliderChoropleth
import requests

# --------------------
# Load pollution data
# --------------------
pollution = pd.read_csv("../pollution/berlin_summer_2025_plz_estimates.csv", parse_dates=["date"])
pollution["postal_code"] = pollution["postal_code"].astype(str)

# --------------------
# Load traffic supply
# --------------------
traffic_supply = pd.read_csv("plz_traffic_supply_index.csv")

# rename the plz to postal_code
traffic_supply = traffic_supply.rename(columns={"plz": "postal_code"})

traffic_supply["postal_code"] = traffic_supply["postal_code"].astype(str)

# --------------------
# Merge pollution + traffic
# --------------------
df = pollution.merge(
    traffic_supply,
    on="postal_code",
    how="left"
).fillna(0)

# --------------------
# Load population data (open Berlin PLZ population CSV)
# Example: open data portal or downloaded CSV
# We'll simulate a download for example purposes
# --------------------
# URL: replace with real source if available
pop_url = "https://opendata.arcgis.com/datasets/berlin-plz-population.csv"

# Uncomment this if URL works:
# population = pd.read_csv(pop_url)
# For now, simulate population:
population = pd.DataFrame({
    "postal_code": df["postal_code"].unique(),
    "population": [1000 + i%50*100 for i in range(len(df["postal_code"].unique()))]
})
population["postal_code"] = population["postal_code"].astype(str)

# Merge population
df = df.merge(population, on="postal_code", how="left")

# --------------------
# Calculate population-weighted NO2
df["pop_weighted_no2"] = df["no2"] * df["population"]


df["month"] = df["date"].dt.to_period("M")

plz_monthly = (
    df.groupby(["postal_code", "district", "month"])
    .agg(
        mean_no2=("no2", "mean"),
        mean_pm10=("pm10", "mean"),
        mean_aqi=("aqi", "mean"),
        traffic_supply_index=("traffic_supply_index", "mean"),
        pop_weighted_no2=("pop_weighted_no2", "sum"),
        population=("population", "sum")
    )
    .reset_index()
)

# Calculate population-weighted exposure per PLZ
plz_monthly["pop_weighted_no2_per_capita"] = (
    plz_monthly["pop_weighted_no2"] / plz_monthly["population"]
).round(1)


postcodes_url = "https://tsb-opendata.s3.eu-central-1.amazonaws.com/plz/plz.geojson"
gdf_plz = gpd.read_file(postcodes_url)
gdf_plz = gdf_plz.rename(columns={"plz": "postal_code"})
gdf_plz["postal_code"] = gdf_plz["postal_code"].astype(str)


# Prepare GeoDataFrames for each month
gdfs_by_month = {}
for month, grp in plz_monthly.groupby("month"):
    gdf = gdf_plz.merge(grp, on="postal_code", how="left")
    gdfs_by_month[str(month)] = gdf


from branca.colormap import linear

# Base map
m = folium.Map(location=[52.52, 13.405], zoom_start=10, tiles="cartodbpositron")

# Color scales
aqi_colormap = linear.RdYlGn_03.scale(0, 100)
traffic_colormap = linear.Blues_09.scale(0, 1)

# Create separate layer groups
layer_aqi = folium.FeatureGroup(name="Mean AQI")
layer_traffic = folium.FeatureGroup(name="Traffic Supply Index")

for month, gdf in gdfs_by_month.items():
    for _, row in gdf.iterrows():
        if pd.notna(row["mean_aqi"]):
            # AQI circle
            folium.CircleMarker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                radius=6,
                color=aqi_colormap(row["mean_aqi"]),
                fill=True,
                fill_opacity=0.7,
                popup=(
                    f"PLZ: {row['postal_code']}<br>"
                    f"District: {row['district']}<br>"
                    f"Month: {month}<br>"
                    f"Mean AQI: {row['mean_aqi']}<br>"
                    f"Mean NO2: {row['mean_no2']:.1f} µg/m³<br>"
                    f"Traffic Supply: {row['traffic_supply_index']:.2f}"
                )
            ).add_to(layer_aqi)

        if pd.notna(row["traffic_supply_index"]):
            # Traffic supply circle
            folium.CircleMarker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                radius=6,
                color=traffic_colormap(row["traffic_supply_index"]),
                fill=True,
                fill_opacity=0.5,
                popup=(
                    f"PLZ: {row['postal_code']}<br>"
                    f"District: {row['district']}<br>"
                    f"Month: {month}<br>"
                    f"Traffic Supply Index: {row['traffic_supply_index']:.2f}"
                )
            ).add_to(layer_traffic)

layer_aqi.add_to(m)
layer_traffic.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# Save map
m.save("berlin_plz_aqi_traffic_timeslider.html")


district_monthly = (
    plz_monthly.groupby(["district", "month"])
    .agg(
        mean_no2=("mean_no2", "mean"),
        mean_pm10=("mean_pm10", "mean"),
        mean_aqi=("mean_aqi", "mean"),
        traffic_supply_index=("traffic_supply_index", "mean"),
        pop_weighted_no2_per_capita=("pop_weighted_no2_per_capita", "mean")
    )
    .reset_index()
)

district_monthly.to_csv("berlin_district_monthly_air_quality.csv", index=False)
