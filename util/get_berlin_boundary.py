import osmnx as ox
berlin = ox.geocode_to_gdf("Berlin, Germany", which_result=1)
berlin.to_file("../data/boundaries/berlin_boundary.geojson", driver="GeoJSON")