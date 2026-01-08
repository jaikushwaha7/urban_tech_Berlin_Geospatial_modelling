import pandas as pd
import numpy as np
import geopandas as gpd

# insert documentation here if needed
"""
This script generates synthetic air quality data for Berlin postal codes (PLZ)
for the summer of 2025 (May 1st to August 31st).

It performs the following steps:
1.  **Loads Geographic Data**: Reads GeoJSON files for Berlin districts and
    postal code boundaries.
2.  **Spatial Join**: Joins postal code polygons with district polygons to
    associate each postal code with a district.
3.  **Data Generation**: Creates a DataFrame with a daily entry for each
    postal code, covering the specified date range.
4.  **Synthetic Pollutant Values**: Generates synthetic NO2 and PM10 values
    with some realistic variations:
    *   Lower NO2 on weekends.
    *   Slightly higher NO2 in July.
5.  **AQI Calculation**: Computes a simplified Air Quality Index (AQI) based
    on the generated NO2 and PM10 values.
6.  **Data Export**: Saves the generated air quality estimates to a CSV file
    (`berlin_summer_2025_plz_estimates.csv`).

The output CSV includes columns for `date`, `postal_code`, `district`, `no2`,
`pm10`, and `aqi`. This data can be used for further analysis and visualization
of air quality trends across Berlin's postal code areas.
"""


# ------------------------------
# Load GeoJSON polygons
# ------------------------------
districts_url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
postcodes_url = "https://tsb-opendata.s3.eu-central-1.amazonaws.com/plz/plz.geojson"

# Read district boundaries
gdf_districts = gpd.read_file(districts_url)

# Read postal code boundaries
gdf_plz = gpd.read_file(postcodes_url)

# Ensure same CRS
gdf_plz = gdf_plz.to_crs(gdf_districts.crs)

# ------------------------------
# Spatial join PLZ → District
# ------------------------------
gdf_plz = gpd.sjoin(gdf_plz, gdf_districts, how="left", predicate="intersects")

# Optional: rename columns to something friendly
gdf_plz = gdf_plz.rename(columns={
    "plz": "postal_code",
    "name": "district"
})

# Now we have each postal code mapped to a Berlin district

# ------------------------------
# Create air quality estimates
# ------------------------------
dates = pd.date_range("2025-05-01", "2025-08-31", freq="D")

# create a date range for may -august from 2020 till 2025
all_dates = pd.date_range("2020-05-01", "2025-08-31", freq="D")
summer_dates = all_dates[(all_dates.month >= 5) & (all_dates.month <= 8)]



# Build full grid: date × postal code
df = (
    pd.MultiIndex.from_product(
        [dates, gdf_plz["postal_code"]],
        names=["date", "postal_code"]
    )
    .to_frame(index=False)
)

# Ensure postal_code is a string on both sides
gdf_plz["postal_code"] = gdf_plz["postal_code"].astype(str)
# Create a unique mapping: pick the modal district for each postal code (handles multiple intersects)
plz_to_district = (
    gdf_plz.groupby("postal_code")["district"]
    .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
)

# Add district info to df (ensure df postal_code is string)
df["postal_code"] = df["postal_code"].astype(str)
df["district"] = df["postal_code"].map(plz_to_district)

# Report postal codes without a mapped district
missing = df["district"].isna().sum()
if missing:
    print(f"Warning: {missing} rows have no mapped district (missing PLZ or unmatched).")

# Base pollutant generation
np.random.seed(42)
df["no2"] = np.random.uniform(18, 35, size=len(df))
df["pm10"] = np.random.uniform(15, 25, size=len(df))

# Weekend effect
weekend_mask = df["date"].dt.weekday >= 5
df.loc[weekend_mask, "no2"] *= 0.7

# July season bump
july_mask = df["date"].dt.month == 7
df.loc[july_mask, "no2"] *= 1.1

# AQI calculation
df["aqi"] = (((df["no2"] / 40) * 50 + (df["pm10"] / 50) * 50) / 2).round().astype(int)

df["no2"] = df["no2"].round(1)
df["pm10"] = df["pm10"].round(1)

# ------------------------------
# Save results
# ------------------------------
df.to_csv("berlin_summer_2025_plz_estimates.csv", index=False)
print(f"Created {len(df)} records with PLZ and district mapping")
print(df.head())


