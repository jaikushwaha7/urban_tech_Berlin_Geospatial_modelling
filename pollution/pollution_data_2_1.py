import requests
import pandas as pd
from io import StringIO

BASE = "https://luftdaten.berlin.de/api"

def get_simple_pollution():
    """Simple version: Get NO2 for all summer months 2020-2025"""
    
    all_data = []
    
    for year in range(2020, 2026):
        print(f"Getting {year}...")
        
        url = f"{BASE}/core/no2.csv"
        params = {
            "stationgroup": "all",
            "period": "1d",
            "timespan": "custom",
            "start[date]": f"{year}-05-01",
            "end[date]": f"{year}-08-31"
        }
        
        try:
            r = requests.get(url, params=params)
            try:
                df = pd.read_csv(StringIO(r.text))
            except Exception as e:
                print(f"  Error reading CSV for {year}: {e}")
                print(f"  Response text (first 200 chars):\n{r.text[:200]}")
                continue

            # Process the data
            melted = []
            for i, row in df.iterrows():
                station = row.iloc[0]
                for col in df.columns[1:]:
                    value = row[col]
                    if pd.notna(value):
                        try:
                            date = pd.to_datetime(col, format='%d.%m.%Y')
                            melted.append({
                                'date': date,
                                'station': station,
                                'no2': float(value)
                            })
                        except:
                            continue

            year_df = pd.DataFrame(melted)
            year_df['year'] = year
            all_data.append(year_df)
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        df_all.to_csv('berlin_no2_simple_2020_2025.csv', index=False)
        print(f"\n✅ Saved {len(df_all)} records to 'berlin_no2_simple_2020_2025.csv'")
        return df_all
    
    return pd.DataFrame()

# Run the simple version
df = get_simple_pollution()
if not df.empty:
    print(f"\nData shape: {df.shape}")
    print(df.head())