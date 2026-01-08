import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from our previous analysis
data = {
    "District": ["Treptow-Köpenick", "Spandau", "Steglitz-Zehlendorf", "Pankow", "Reinickendorf", "Mitte"],
    "Resilience": [0.92, 0.88, 0.85, 0.74, 0.69, 0.15],
    "Change_Intensity": [4.21, 5.03, 1.52, 11.80, 2.29, 15.23]
}
df = pd.DataFrame(data).sort_values("Resilience", ascending=False)

# --- Chart 1: Professional Bar Chart ---
plt.figure(figsize=(10, 6))
colors = plt.cm.RdYlGn(df['Resilience'])
plt.bar(df['District'], df['Resilience'], color=colors, edgecolor='black')
plt.title('Berlin District Ecological Resilience Ranking (2025)', fontsize=14)
plt.ylabel('Resilience Score (0.0 - 1.0)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('district_resilience_bar.png')

# --- Chart 2: Modern Radar Chart ---
def make_radar(df):
    labels = df['District'].values
    stats = df['Resilience'].values
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='green', alpha=0.25)
    ax.plot(angles, stats, color='green', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title('Berlin District Resilience Profile', size=15, color='darkgreen', y=1.1)
    plt.savefig('district_radar_chart.png')

make_radar(df)