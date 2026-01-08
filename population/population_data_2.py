import pandas as pd

# Create a simpler version with just key data
simple_df = df[['regional_code', 'area_name', 'population_2022', 'population_density_per_sqkm']].copy()

# Clean the data
simple_df['population_2022'] = pd.to_numeric(
    simple_df['population_2022'].astype(str).str.replace('.', ''), 
    errors='coerce'
)
simple_df['population_density_per_sqkm'] = pd.to_numeric(
    simple_df['population_density_per_sqkm'].astype(str).str.replace('.', ''), 
    errors='coerce'
)

# Save simplified version
simple_df.to_csv('berlin_population_simple_english.csv', index=False, encoding='utf-8')

print("\n✅ Simplified version saved as 'berlin_population_simple_english.csv'")
print("\nSimplified data:")
print(simple_df.head(10))