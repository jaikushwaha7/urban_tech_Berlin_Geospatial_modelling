import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
h3_df = pd.read_csv('berlin_district_h3_analysis.csv')
change_df = pd.read_csv('change_score.csv')

# Preprocessing change_df for temporal plot
change_df['Green Coverage %'] = change_df['Green Coverage %'].str.replace('%', '').astype(float)
change_df['Change Score (Δ)'] = change_df['Change Score (Δ)'].astype(float)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 1. Temporal Analysis: Green Coverage Over Time
axes[0, 0].plot(change_df['Acquisition Period'], change_df['Green Coverage %'], marker='o', color='forestgreen', linewidth=2)
axes[0, 0].set_title('Berlin Vegetation Dynamics (2023 - 2025)', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Green Coverage %')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# 2. H3 Analysis: Change Ratio by District (Spatial Heatmap Proxy)
h3_sorted = h3_df.sort_values('Change_Ratio', ascending=True)
sns.barplot(x='Change_Ratio', y='District', data=h3_sorted, palette='RdYlGn_r', ax=axes[0, 1])
axes[0, 1].set_title('H3-Normalized Change Intensity by District', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Change Ratio (Intensity per H3 Unit)')

# 3. Resilience vs Change Score
sns.scatterplot(x='Resilience_Score', y='Change_Ratio', size='H3_Units', hue='District', 
                data=h3_df, sizes=(100, 1000), alpha=0.7, palette='viridis', ax=axes[1, 0])
axes[1, 0].set_title('Ecological Resilience vs. Change Intensity', fontsize=14, fontweight='bold')
axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# 4. Temporal Change Score (Delta)
axes[1, 1].bar(change_df['Acquisition Period'], change_df['Change Score (Δ)'], 
               color=(change_df['Change Score (Δ)'] > 0).map({True: 'g', False: 'r'}), alpha=0.6)
axes[1, 1].set_title('Monthly NDVI Flux (Change Score Δ)', fontsize=14, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].axhline(0, color='black', linewidth=1)

plt.savefig('berlin_ndvi_h3_analysis_report.png', bbox_inches='tight')

# Generate a summary CSV for the user
h3_df['Interpretation'] = h3_df['Change_Ratio'].apply(lambda x: 'High Volatility' if x > 0.2 else ('Moderate' if x > 0.1 else 'Stable'))
h3_df.to_csv('final_berlin_h3_comparison.csv', index=False)

print("Visualizations generated: berlin_ndvi_h3_analysis_report.png")
print("Summary table created: final_berlin_h3_comparison.csv")