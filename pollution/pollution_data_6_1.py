import matplotlib.pyplot as plt
import pandas as pd

# Simple bar chart
district_kpi = pd.read_csv("berlin_2025_district_air_quality_kpi.csv")
district_2025 = district_kpi[district_kpi["year"] == 2025]

# Sort by NO2
district_2025 = district_2025.sort_values('mean_no2', ascending=False)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. NO2 by district
bars = axes[0,0].barh(district_2025['district'], district_2025['mean_no2'])
axes[0,0].axvline(x=40, color='red', linestyle='--', label='EU Limit (40 µg/m³)')
axes[0,0].set_xlabel('NO₂ (µg/m³)')
axes[0,0].set_title('NO₂ Concentration by District (2025)')
axes[0,0].legend()

# 2. AQI by district
axes[0,1].barh(district_2025['district'], district_2025['worst_aqi'], color='orange')
axes[0,1].set_xlabel('AQI')
axes[0,1].set_title('Worst AQI by District (2025)')

# 3. Days above limit
axes[1,0].barh(district_2025['district'], district_2025['days_above_eu_limit'], color='red')
axes[1,0].set_xlabel('Days Above EU Limit')
axes[1,0].set_title('Days Exceeding NO₂ Limit')

# 4. Summary statistics
stats_text = f"""
Data Summary (2025, May-August):
• Districts: {len(district_2025)}
• Avg NO₂: {district_2025['mean_no2'].mean():.1f} µg/m³
• Max NO₂: {district_2025['mean_no2'].max():.1f} µg/m³
• Districts > EU limit: {(district_2025['mean_no2'] > 40).sum()}
• Avg AQI: {district_2025['worst_aqi'].mean():.1f}
"""
axes[1,1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
axes[1,1].axis('off')

plt.tight_layout()
plt.savefig('berlin_air_quality_charts.png', dpi=150, bbox_inches='tight')
print("✅ Charts saved as 'berlin_air_quality_charts.png'")
plt.show()