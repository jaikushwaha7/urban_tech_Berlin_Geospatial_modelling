import pandas as pd
import numpy as np

## insert documentation here if needed

# 1. Define Image Metadata & Temporal Analysis (Based on your nvdi_001-009 results)
image_analysis_data = {
    "Filename": ["nvdi_001.png", "nvdi_002.png", "nvdi_003.png", "nvdi_004.png", "nvdi_005.png",
                 "nvdi_006.png", "nvdi_007.png", "nvdi_008.png", "nvdi_009.png"],
    "Observation_Date": ["July 2023", "Sept 2023", "March 2024", "May 2025", "July 2025", 
                         "July 2025", "Aug 2025", "Aug 2025", "Aug 18, 2025"],
    "Avg_NDVI_Intensity": [104.32, 99.95, 100.76, 118.22, 100.16, 113.62, 106.58, 97.89, 96.6],
    "Green_Coverage_Pct": [17.0, 14.71, 14.21, 26.01, 10.78, 20.96, 19.44, 14.41, 15.23],
    "Condition": ["Baseline", "Stable", "Pre-Flush", "Peak Growth", "Heat Stress", 
                  "Recovery", "Stable", "Early Autumn", "Current Baseline"]
}

df_temporal = pd.DataFrame(image_analysis_data)

# 2. Define District Resilience Ranking (Pretrained Environmental Classification)
district_data = {
    "District": [
        "Treptow-Köpenick", "Spandau", "Steglitz-Zehlendorf", "Pankow",
        "Reinickendorf", "Marzahn-Hellersdorf", "Lichtenberg", "Charlottenburg-Wilmersdorf",
        "Tempelhof-Schöneberg", "Neukölln", "Friedrichshain-Kreuzberg", "Mitte"
    ],
    "Resilience_Tier": [
        "High", "High", "High", "Medium", "Medium", "Moderate",
        "Moderate", "Low-Medium", "Low", "Critical", "Critical", "Critical"
    ],
    "Primary_Land_Cover": [
        "Müggelheim Forest", "Forested West", "Grunewald Zone", "Mixed Ag/Res",
        "Tegel Forest", "Gardens", "Landscape Parks", "Schlosspark",
        "Tempelhofer Feld", "Urban Built-up", "Dense Block Structure", "Minimal Veg"
    ],
    "Resilience_Index": [0.92, 0.88, 0.85, 0.74, 0.69, 0.61, 0.58, 0.49, 0.38, 0.22, 0.18, 0.15]
}

df_districts = pd.DataFrame(district_data)
df_districts["Rank"] = df_districts["Resilience_Index"].rank(ascending=False).astype(int)
df_districts = df_districts.sort_values("Rank")

# 3. Export to Excel with Multiple Sheets
output_filename = "Berlin_Ecological_Analysis_Final.xlsx"

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    # Sheet 1: Image Timeline and Metadata
    df_temporal.to_excel(writer, sheet_name='Temporal_NDVI_Data', index=False)
    
    # Sheet 2: District Resilience Ranking
    df_districts.to_excel(writer, sheet_name='District_Rankings', index=False)
    
    # Sheet 3: Summary Statistics
    summary_stats = pd.DataFrame({
        "Metric": ["Max Green Coverage", "Min Green Coverage", "Resilience Average"],
        "Value": [f"{df_temporal['Green_Coverage_Pct'].max()}%", 
                  f"{df_temporal['Green_Coverage_Pct'].min()}%", 
                  f"{round(df_districts['Resilience_Index'].mean(), 2)}"]
    })
    summary_stats.to_excel(writer, sheet_name='Project_Summary', index=False)

print(f"Final Comparison Report saved as: {output_filename}")