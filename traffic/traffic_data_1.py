import pandas as pd
import zipfile
# insert documentation of traffic data from vbb website
"""
This script processes GTFS (General Transit Feed Specification) data for Berlin
to calculate a 'traffic supply index' for each postal code (PLZ).

The GTFS data, typically provided as a ZIP archive containing several CSV files,
describes public transport schedules, routes, and stop locations.

The main steps are:
1. Load GTFS data from a specified ZIP file.
2. Identify active public transport services within a defined date range
   (e.g., May to August 2024).
3. Geocode public transport stops and spatially join them with Berlin's
   postal code (PLZ) polygons.
4. Calculate the number of stops within each PLZ.
5. Normalize the stop count to create a 'traffic supply index',
   representing the relative density of public transport stops per PLZ.
6. Save the resulting traffic supply index per PLZ to a CSV file.

The 'traffic supply index' can be used as a proxy for public transport
accessibility and traffic intensity in different areas of Berlin.
"""

# FIRST: Load your GTFS data
gtfs_path = "gtfs-2024.zip"  # Change this to your file

def load_gtfs(gtfs_path):
    with zipfile.ZipFile(gtfs_path) as z:
        return {
            "routes": pd.read_csv(z.open("routes.txt")),
            "trips": pd.read_csv(z.open("trips.txt")),
            "stop_times": pd.read_csv(z.open("stop_times.txt")),
            "stops": pd.read_csv(z.open("stops.txt")),
            "calendar": pd.read_csv(z.open("calendar.txt")),
            "calendar_dates": pd.read_csv(z.open("calendar_dates.txt")),
        }
    
    

gtfs = load_gtfs(gtfs_path)
# SECOND: Now you can use the functions
def active_services(calendar, start, end):
    calendar["start_date"] = pd.to_datetime(calendar["start_date"], format="%Y%m%d")
    calendar["end_date"] = pd.to_datetime(calendar["end_date"], format="%Y%m%d")
    return calendar[
        (calendar["start_date"] <= end) &
        (calendar["end_date"] >= start)
    ]["service_id"]

start = pd.Timestamp("2024-05-01")
end = pd.Timestamp("2024-08-31")

services = active_services(gtfs["calendar"], start, end)  # Use calendar, not gtfs["calendar"]
trips_active = gtfs["trips"][gtfs["trips"]["service_id"].isin(services)]

daily_trip_count = trips_active.groupby("route_id").size()

print(f"Active services: {len(services)}")
print(f"Active trips: {len(trips_active)}")

import geopandas as gpd

stops_gdf = gpd.GeoDataFrame(
    gtfs["stops"],
    geometry=gpd.points_from_xy(gtfs["stops"].stop_lon, gtfs["stops"].stop_lat),
    crs="EPSG:4326"
)

# Join stops to Berlin PLZ polygons (same GeoJSON you already use)
plz_gdf = gpd.read_file("plz.geojson").to_crs("EPSG:4326")

stops_plz = gpd.sjoin(stops_gdf, plz_gdf, predicate="within")

plz_stop_density = (
    stops_plz
    .groupby("plz")
    .size()
    .rename("stop_count")
    .reset_index()
)

plz_stop_density["traffic_supply_index"] = (
    plz_stop_density["stop_count"] /
    plz_stop_density["stop_count"].max()
)

# save to csv
plz_stop_density.to_csv("traffic_stop_density.csv", index=False)

