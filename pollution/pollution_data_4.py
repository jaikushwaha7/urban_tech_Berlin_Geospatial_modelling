import pandas as pd
import numpy as np

# data berlin_summer_2025_plz_estimates.csv call
df = pd.read_csv("berlin_summer_2025_plz_estimates.csv")

# Ensure `date` column is datetimelike so `.dt` accessor works
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    missing_dates = df["date"].isna().sum()
    if missing_dates:
        print(f"Warning: {missing_dates} rows have invalid or missing dates after parsing.")
else:
    raise KeyError("Expected 'date' column in berlin_summer_2025_plz_estimates.csv")

# Ensure postal_code is a string (helps later grouping/mapping)
if "postal_code" in df.columns:
    df["postal_code"] = df["postal_code"].astype(str)
else:
    print("Warning: 'postal_code' column not found in input CSV.")

# AQI sub-index functions
def caqi_no2(no2):
    """
    Simplified CAQI sub-index for NO2 (µg/m³)
    """
    bins = [0, 40, 90, 120, 230, np.inf]
    scores = [25, 50, 75, 100, 150]
    return pd.cut(no2, bins=bins, labels=scores, right=True).astype(float)


def caqi_pm10(pm10):
    """
    Simplified CAQI sub-index for PM10 (µg/m³)
    """
    bins = [0, 20, 35, 50, 100, np.inf]
    scores = [25, 50, 75, 100, 150]
    return pd.cut(pm10, bins=bins, labels=scores, right=True).astype(float)

# Daily AQI (max-pollutant logic)
df["aqi_no2"] = caqi_no2(df["no2"])
df["aqi_pm10"] = caqi_pm10(df["pm10"])

# Final AQI = worst pollutant
df["aqi"] = df[["aqi_no2", "aqi_pm10"]].max(axis=1).astype(int)

# Annual KPI aggregation (PLZ + District)
annual_kpi = (
    df
    .assign(year=df["date"].dt.year)
    .groupby(["year", "postal_code", "district"])
    .agg(
        no2_annual_mean=("no2", "mean"),
        pm10_annual_mean=("pm10", "mean"),
        pm10_exceedance_days=("pm10", lambda x: (x > 50).sum()),
        max_aqi=("aqi", "max"),
        mean_aqi=("aqi", "mean")
    )
    .reset_index()
)
# EU compliance flags (very useful KPI)
annual_kpi["no2_compliant"] = annual_kpi["no2_annual_mean"] < 40
annual_kpi["pm10_compliant"] = annual_kpi["pm10_annual_mean"] < 40
annual_kpi["pm10_exceedance_compliant"] = annual_kpi["pm10_exceedance_days"] <= 35

# district-level KPI rollup
district_kpi = (
    annual_kpi
    .groupby(["year", "district"])
    .agg(
        mean_no2=("no2_annual_mean", "mean"),
        mean_pm10=("pm10_annual_mean", "mean"),
        total_pm10_exceedance_days=("pm10_exceedance_days", "sum"),
        worst_aqi=("max_aqi", "max")
    )
    .reset_index()
)


annual_kpi.to_csv("berlin_2025_plz_air_quality_kpi.csv", index=False)
district_kpi.to_csv("berlin_2025_district_air_quality_kpi.csv", index=False)
