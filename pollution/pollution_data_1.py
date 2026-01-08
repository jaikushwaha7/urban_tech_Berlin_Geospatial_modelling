import requests
import pandas as pd

# 1. Inspect https://luftdaten.berlin.de/api/doc for station codes & components
# Example: station code "mc027" (Marienfelde) and component "no2"

BASE_URL = "https://luftdaten.berlin.de/api"

def get_berlin_live_no2(station_code="mc027", component="no2"):
    """
    Live (current day) NO2 data from a Berlin station.
    """
    url = f"{BASE_URL}/stations/{station_code}/data"
    params = {
        "core": component,     # pollutant/component, e.g. "no2", "pm10", "pm25", "o3"
        "period": "1h",        # 1-hour values
        "span": "currentday"   # current day data
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()  # returns a list of measurements

    df = pd.DataFrame(data)
    # Optional: convert timestamp column to datetime if present
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df

if __name__ == "__main__":
    df_live = get_berlin_live_no2()
    print(df_live.head())
