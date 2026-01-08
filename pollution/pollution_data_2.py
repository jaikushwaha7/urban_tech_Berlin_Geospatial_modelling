import requests
import pandas as pd

# SIMPLEST: Get and save Berlin air quality data
url = "https://luftdaten.berlin.de/api/stations/mc174/data"
params = {"core": "no2", "period": "1h", "span": "last24h"}

response = requests.get(url, params=params)
data = response.json()

# Convert to pandas DataFrame
df = pd.DataFrame(data)
print(f"Got {len(df)} measurements")
print(df.head())

# Save to CSV
df.to_csv("berlin_air_simple.csv", index=False)
print("Saved to berlin_air_simple.csv")