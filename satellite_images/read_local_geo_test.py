import geopandas as gpd
p = '../berlin_heat_data/boundaries/berlin_boundary.geojson'
print('Trying to read:', p)
g = gpd.read_file(p)
print('Rows,Cols:', g.shape)
print('CRS:', g.crs)
