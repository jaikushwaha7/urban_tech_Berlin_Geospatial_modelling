import os
import geopandas as gpd
bdir = os.path.join(r'E:\Study\BHT\Semester3\urbanTech\Project_1','berlin_heat_data','boundaries')
print('Files in', bdir)
for f in os.listdir(bdir):
    p=os.path.join(bdir,f)
    print(' -',f, 'exists:', os.path.exists(p))
    try:
        g=gpd.read_file(p)
        print('   -> rows, cols:', g.shape, 'columns:', list(g.columns)[:10])
    except Exception as e:
        print('   -> failed to read:', e)
