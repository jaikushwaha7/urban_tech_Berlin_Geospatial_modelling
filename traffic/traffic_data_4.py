import pandas as pd
import geopandas as gpd
import json
from datetime import datetime
import folium
from folium.plugins import TimeSliderChoropleth
from branca.colormap import linear

# --------------------
# Load pollution + traffic + population data
# --------------------
pollution = pd.read_csv("../pollution/berlin_summer_2025_plz_estimates.csv", parse_dates=["date"])
pollution["postal_code"] = pollution["postal_code"].astype(str)

traffic_supply = pd.read_csv("plz_traffic_supply_index.csv")
# rename the plz to postal_code
traffic_supply = traffic_supply.rename(columns={"plz": "postal_code"})
traffic_supply["postal_code"] = traffic_supply["postal_code"].astype(str)

# Example population data (replace with real)
population = pd.DataFrame({
    "postal_code": pollution["postal_code"].unique(),
    "population": [1000 + i%50*100 for i in range(len(pollution["postal_code"].unique()))]
})
population["postal_code"] = population["postal_code"].astype(str)

# Merge all
df = pollution.merge(traffic_supply, on="postal_code", how="left").merge(population, on="postal_code", how="left")
df["traffic_supply_index"] = df["traffic_supply_index"].fillna(0)

# Population-weighted NO2
df["pop_weighted_no2"] = df["no2"] * df["population"]

# --------------------
# Aggregate monthly
# --------------------
df["month"] = df["date"].dt.to_period("M")
plz_monthly = df.groupby(["postal_code", "district", "month"]).agg(
    mean_no2=("no2", "mean"),
    mean_pm10=("pm10", "mean"),
    mean_aqi=("aqi", "mean"),
    traffic_supply_index=("traffic_supply_index", "mean"),
    pop_weighted_no2=("pop_weighted_no2", "sum"),
    population=("population", "sum")
).reset_index()

# Population-weighted NO2 per capita
plz_monthly["pop_weighted_no2_per_capita"] = (plz_monthly["pop_weighted_no2"] / plz_monthly["population"]).round(1)



postcodes_url = "https://tsb-opendata.s3.eu-central-1.amazonaws.com/plz/plz.geojson"
gdf_plz = gpd.read_file(postcodes_url)
gdf_plz = gdf_plz.rename(columns={"plz": "postal_code"})
gdf_plz["postal_code"] = gdf_plz["postal_code"].astype(str)


gdfs_by_month = {}
for month, grp in plz_monthly.groupby("month"):
    gdf = gdf_plz.merge(grp, on="postal_code", how="left")
    gdfs_by_month[str(month)] = gdf


def style_feature(feature, color, opacity=0.7):
    return {
        "fillColor": color,
        "color": "black",
        "weight": 0.3,
        "fillOpacity": opacity
    }

from shapely.geometry import mapping

# Create time-indexed GeoJSON with required 'id' and 'times' properties
time_indexed_features = []
for i, (month, gdf) in enumerate(gdfs_by_month.items()):
    for _, row in gdf.iterrows():
        if pd.notna(row["mean_aqi"]):
            feature = {
                "type": "Feature",
                "geometry": mapping(row["geometry"]),
                "properties": {
                    "time": str(month),
                    "mean_aqi": row["mean_aqi"],
                    "mean_no2": row["mean_no2"],
                    "traffic_supply_index": row["traffic_supply_index"],
                    "district": row["district"],
                    "postal_code": row["postal_code"]
                },
                "id": str(len(time_indexed_features)),
                "times": [str(month)]
            }
            time_indexed_features.append(feature)

time_geojson = {"type": "FeatureCollection", "features": time_indexed_features}

m = folium.Map(location=[52.52, 13.405], zoom_start=10, tiles="cartodbpositron")

# Colormap for AQI
colormap_aqi = linear.RdYlGn_03.scale(0, 100)
colormap_aqi.caption = "Mean AQI"

# Build styledict: {feature_id: {timestamp: style_dict}}
styledict = {}
for feature in time_geojson["features"]:
    feature_id = feature["id"]
    timestamp = feature["times"][0]
    aqi = feature["properties"]["mean_aqi"]
    color = "#ffffff" if pd.isna(aqi) else colormap_aqi(aqi)
    styledict[feature_id] = {
        timestamp: {
            "color": color,
            "opacity": 0.7
        }
    }

TimeSliderChoropleth(
    data=time_geojson,
    styledict=styledict
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# ...existing code...
m.save("berlin_plz_aqi_choropleth_timeslider.html")
