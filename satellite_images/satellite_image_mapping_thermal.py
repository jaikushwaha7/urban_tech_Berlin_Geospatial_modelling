import geopandas as gpd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
# Load Berlin boundaries
# Using your provided GeoJSON logic
districts_url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
districts = gpd.read_file(districts_url)

def generate_thermal_delta_map(thermal_start_path, thermal_end_path, districts_gdf):
    """
    Calculates Delta LST (Land Surface Temperature Change) across Berlin Districts.
    Red = Significant Warming (Heat Island intensifying)
    Blue = Cooling (Urban Greening/Water effect)
    """
    # Load Thermal Visualizations (LST)
    # Brighter pixels typically represent higher temperatures
    img_start = cv2.imread(thermal_start_path, cv2.IMREAD_GRAYSCALE)
    img_end = cv2.imread(thermal_end_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate Thermal Delta (Temperature Shift)
    # Positive values = area got hotter; Negative = area got cooler
    thermal_delta = img_end.astype(float) - img_start.astype(float)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Mapping coordinates (based on Berlin's 10km altitude reference)
    extent = [13.088345, 13.7611609, 52.3382448, 52.6755087] 
    
    # Plot Thermal Change Heatmap 
    # 'coolwarm' is standard for temperature delta (Blue: cool, White: neutral, Red: hot)
    im = ax.imshow(thermal_delta, cmap='coolwarm', extent=extent, alpha=0.9)
    
    # Overlay District Boundaries
    districts_gdf.boundary.plot(ax=ax, color='#333333', linewidth=1.5, alpha=0.7)
    
    # Add District Labels with Thermal Anomaly Stats
    for idx, row in districts_gdf.iterrows():
        centroid = row.geometry.centroid
        # Optional: Add the name and a label
        ax.text(centroid.x, centroid.y, row['name'], fontsize=8, 
                ha='center', weight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Thermal Colorbar configuration
    cbar = plt.colorbar(im, label='Temperature Intensity Change (ΔLST)', shrink=0.8)
    plt.title('Berlin District-Level Thermal Anomaly Detection (ΔLST)', fontsize=15, pad=20)
    
    # Contextual annotation
    ax.text(13.1, 52.35, "Scale Reference: 10km Altitude\nRed: Intensifying Heat Island\nBlue: Effective Cooling", 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.axis('off')
    plt.savefig('berlin_thermal_delta_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

# Execution using your thermal images
generate_thermal_delta_map('./Thermal/20200816.png', './Thermal/20250701.png', districts)

def generate_district_lst_analysis(thermal_img_path, districts_gdf):
    """
    Translates thermal image intensity into a District-level LST table and map.
    """
    # 1. Load Thermal Image (Sentinel-3 reference)
    # 10km altitude view; brighter = hotter
    img = cv2.imread(thermal_img_path, cv2.IMREAD_GRAYSCALE)
    img_height, img_width = img.shape
    
    # 2. Define Spatial Extent (Matching the 10km reference provided)
    # [Lon_Min, Lon_Max, Lat_Min, Lat_Max]
    extent = [13.088, 13.761, 52.338, 52.675]
    
    # 3. Calculate Mean LST per District
    # We map the image pixels to the district polygons
    district_results = []
    
    for idx, row in districts_gdf.iterrows():
        # Get district mask (simplified for thermal approximation)
        centroid = row.geometry.centroid
        
        # Map Lon/Lat to Pixel X/Y
        px = int((centroid.x - extent[0]) / (extent[1] - extent[0]) * img_width)
        py = int((1 - (centroid.y - extent[2]) / (extent[3] - extent[2])) * img_height)
        
        # Sample a 5x5 window around the center to get average local LST
        sample_window = img[max(0, py-2):min(img_height, py+2), 
                            max(0, px-2):min(img_width, px+2)]
        mean_intensity = np.mean(sample_window)
        
        district_results.append({
            'district': row['name'],
            'mean_lst_intensity': round(mean_intensity, 2),
            'thermal_anomaly': "High" if mean_intensity > 180 else "Normal"
        })

    # Convert to DataFrame
    lst_table = pd.DataFrame(district_results)
    
    # 4. Plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot Thermal Heatmap as background
    im = ax.imshow(img, cmap='magma', extent=extent, alpha=0.9)
    
    # Overlay Boundaries
    districts_gdf.boundary.plot(ax=ax, color='white', linewidth=1.5, alpha=0.6)
    
    # Add Heat Labels
    for idx, row in districts_gdf.iterrows():
        centroid = row.geometry.centroid
        intensity = lst_table.loc[lst_table['district'] == row['name'], 'mean_lst_intensity'].values[0]
        ax.text(centroid.x, centroid.y, f"{row['name']}\n{intensity}", 
                fontsize=8, ha='center', color='white', weight='bold',
                bbox=dict(facecolor='black', alpha=0.3, edgecolor='none'))

    plt.colorbar(im, label='LST Intensity (Pixel Value 0-255)')
    plt.title('Berlin District-Level Land Surface Temperature (LST) Map')
    plt.axis('off')
    
    # Save Outputs
    plt.savefig('berlin_district_lst_map.png', bbox_inches='tight', dpi=300)
    lst_table.to_csv('berlin_district_lst_data.csv', index=False)
    
    print("LST Analysis Complete. Table and Map generated.")
    return lst_table

# Execute
# Replace 'thermal_berlin.png' with your thermal satellite image file
lst_data = generate_district_lst_analysis('berlin_thermal_delta_plot.png', districts)