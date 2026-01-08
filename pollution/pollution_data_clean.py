import pandas as pd
# data source : https://luftdaten.berlin.de/station/mc174?period=24h&timespan=custom&start%5Bdate%5D=01.02.2020&end%5Bdate%5D=06.01.2026
# import the file pollution_data_20200201-20260106.csv and correct the column names
# Date;PM10;PM2.5;NO2;NO;NOx;O3;CO;Toluol;Kohlenmonoxid;Schwefeldioxid
df = pd.read_csv(
    "pollution_data_20200201-20260106.csv",
    sep=";",
    parse_dates=["Date"]
)

df = df.rename(columns={
    "Date": "date",
    "PM10": "pm10",
    "PM2.5": "pm25",
    "NO2": "no2",
    "NO": "no",
    "NOx": "nox",
    "O3": "o3",
    "CO": "co",
    "Toluol": "toluene",
    "Kohlenmonoxid": "carbon_monoxide",
    "Schwefeldioxid": "sulfur_dioxide"
})



print("Original columns:", df.columns.tolist())
print("First 5 rows of the cleaned data:")
print(df.head())

# base file date format is 05.02.2020 so bring at same level
df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")


# filter to these dates
all_dates = pd.date_range("2020-05-01", "2025-08-31", freq="D")

summer_dates = all_dates[(all_dates.month >= 5) & (all_dates.month <= 8)]

# make same format summer dates(2020-05-01) as df date(01.05.2020) column
summer_dates = pd.to_datetime(summer_dates)
df["date"] = df["date"].dt.normalize()


df_filtered = df[df["date"].isin(summer_dates)]

# Save the cleaned and filtered data
df_filtered.to_csv("berlin_pollution_cleaned_2020_2025_summer.csv", index=False)

print(f"\nCleaned and filtered data saved to 'berlin_pollution_cleaned_2020_2025_summer.csv'")
print(f"Number of records: {len(df_filtered)}")
print("\nFirst 5 rows of the filtered data:")
print(df_filtered.head())
