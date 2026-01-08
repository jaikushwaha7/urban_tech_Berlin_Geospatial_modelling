import requests
import pandas as pd


"""
This script fetches historical weather data for Berlin from the Open-Meteo Archive API.

The `get_recent_summers` function retrieves daily maximum temperature, minimum temperature,
and precipitation sum for the period between May 1, 2020, and August 31, 2025.
The data is filtered to include only the summer months (May to August) and saved
to a CSV file named "berlin_summer_2020_2025.csv".

The API endpoint used is: https://archive-api.open-meteo.com/v1/archive
Parameters include:
- latitude: 52.52 (for Berlin)
- longitude: 13.41 (for Berlin)
- start_date: 2020-05-01
- end_date: 2025-08-31 (Note: data for future dates might be limited or based on forecasts)
- daily: temperature_2m_max, temperature_2m_min, precipitation_sum
- timezone: Europe/Berlin

The output CSV file contains columns for date, year, month, daily max temperature,
daily min temperature, and daily precipitation sum.
"""


# Simple one-call version for 2024-2025
def get_recent_summers():
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "start_date": "2020-05-01",
        "end_date": "2025-08-31",  # Note: 2025 data may be limited
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Europe/Berlin"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    
    # Keep only May-August (in case 2025 data extends beyond August)
    df = df[(df["month"] >= 5) & (df["month"] <= 8)]
    
    df.to_csv("berlin_summer_2020_2025.csv", index=False)
    print(f"Saved {len(df)} days to berlin_summer_2020_2025.csv")
    return df

# Run it
df_simple = get_recent_summers()
print(df_simple.head())