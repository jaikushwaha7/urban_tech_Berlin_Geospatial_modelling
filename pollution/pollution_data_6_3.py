import pandas as pd

# Load and prepare data
district_kpi = pd.read_csv("berlin_2025_district_air_quality_kpi.csv")
plz_kpi = pd.read_csv("berlin_2025_plz_air_quality_kpi.csv")

district_2025 = district_kpi[district_kpi["year"] == 2025]
plz_2025 = plz_kpi[plz_kpi["year"] == 2025]

# Create HTML dashboard
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Berlin Air Quality 2025</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .card {{ border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f5f5f5; }}
        .bad {{ background-color: #ffcccc; }}
        .good {{ background-color: #ccffcc; }}
        .moderate {{ background-color: #ffffcc; }}
        h1, h2 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Berlin Air Quality Analysis 2025 (May-August)</h1>
    
    <div class="dashboard">
        <div class="card">
            <h2>District Summary</h2>
            <p>Analyzed {len(district_2025)} districts</p>
            <p>Average NO₂: {district_2025['mean_no2'].mean():.1f} µg/m³</p>
            <p>EU Limit (40 µg/m³): {sum(district_2025['mean_no2'] > 40)} districts exceed</p>
        </div>
        
        <div class="card">
            <h2>AQI Distribution</h2>
            <p>Average AQI: {district_2025['worst_aqi'].mean():.1f}</p>
            <p>Postal codes analyzed: {len(plz_2025)}</p>
            <p>Data period: May 1 - August 31, 2025</p>
        </div>
    </div>
    
    <h2>District Data</h2>
    <table>
        <tr>
            <th>District</th>
            <th>NO₂ (µg/m³)</th>
            <th>PM10 (µg/m³)</th>
            <th>Worst AQI</th>
            <th>Days > EU Limit</th>
        </tr>
"""

# Add district rows
for _, row in district_2025.sort_values('mean_no2', ascending=False).iterrows():
    no2_class = "bad" if row['mean_no2'] > 40 else "good"
    aqi_class = "bad" if row['worst_aqi'] > 75 else "moderate" if row['worst_aqi'] > 50 else "good"
    
    html_content += f"""
        <tr>
            <td>{row['district']}</td>
            <td class="{no2_class}">{row['mean_no2']:.1f}</td>
            <td>{row['mean_pm10']:.1f}</td>
            <td class="{aqi_class}">{row['worst_aqi']}</td>
            <td>{row['days_above_eu_limit']}</td>
        </tr>
    """

html_content += """
    </table>
    
    <h2>Top 10 Postal Codes by AQI</h2>
    <table>
        <tr>
            <th>Postal Code</th>
            <th>District</th>
            <th>Mean AQI</th>
            <th>PM10 Exceedance Days</th>
        </tr>
"""

# Add postal code rows
for _, row in plz_2025.nlargest(10, 'mean_aqi').iterrows():
    html_content += f"""
        <tr>
            <td>{row['postal_code']}</td>
            <td>{row['district']}</td>
            <td>{row['mean_aqi']:.1f}</td>
            <td>{row['pm10_exceedance_days']}</td>
        </tr>
    """

html_content += """
    </table>
    
    <div style="margin-top: 40px; padding: 20px; background-color: #f5f5f5;">
        <h3>AQI Color Guide:</h3>
        <p><span style="background-color: #ccffcc; padding: 5px;">Green (0-50):</span> Good air quality</p>
        <p><span style="background-color: #ffffcc; padding: 5px;">Yellow (51-75):</span> Moderate air quality</p>
        <p><span style="background-color: #ffcccc; padding: 5px;">Red (76+):</span> Poor air quality</p>
        <p><strong>EU Limits:</strong> NO₂: 40 µg/m³, PM10: 50 µg/m³ (24h average)</p>
    </div>
</body>
</html>
"""

# Save HTML file
with open('berlin_air_quality_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✅ HTML dashboard saved as 'berlin_air_quality_dashboard.html'")
print("✅ Open the file in any web browser to view the interactive dashboard")