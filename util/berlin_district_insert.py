import pandas as pd
import io

# 1. The raw data provided by the user
df = pd.read_csv('berlin_district_rankings_complete.csv')

# 2. Define the Mapping Dictionary (Neighborhood -> District)
district_mapping = {
    'Marzahn': 'Marzahn-Hellersdorf', 'Gatow': 'Spandau', 'Grünau': 'Treptow-Köpenick',
    'Falkenberg': 'Lichtenberg', 'Stadtrandsiedlung Malchow': 'Pankow', 'Müggelheim': 'Treptow-Köpenick',
    'Tegel': 'Reinickendorf', 'Wannsee': 'Steglitz-Zehlendorf', 'Malchow': 'Lichtenberg',
    'Blankenfelde': 'Pankow', 'Hakenfelde': 'Spandau', 'Schmöckwitz': 'Treptow-Köpenick',
    'Wartenberg': 'Lichtenberg', 'Rosenthal': 'Pankow', 'Rahnsdorf': 'Treptow-Köpenick',
    'Blankenburg': 'Pankow', 'Lübars': 'Reinickendorf', 'Nikolassee': 'Steglitz-Zehlendorf',
    'Friedrichshagen': 'Treptow-Köpenick', 'Kladow': 'Spandau', 'Biesdorf': 'Marzahn-Hellersdorf',
    'Heinersdorf': 'Pankow', 'Siemensstadt': 'Spandau', 'Mahlsdorf': 'Marzahn-Hellersdorf',
    'Grunewald': 'Charlottenburg-Wilmersdorf', 'Bohnsdorf': 'Treptow-Köpenick', 'Heiligensee': 'Reinickendorf',
    'Tiergarten': 'Mitte', 'Konradshöhe': 'Reinickendorf', 'Dahlem': 'Steglitz-Zehlendorf',
    'Buch': 'Pankow', 'Französisch Buchholz': 'Pankow', 'Adlershof': 'Treptow-Köpenick',
    'Frohnau': 'Reinickendorf', 'Kaulsdorf': 'Marzahn-Hellersdorf', 'Marienfelde': 'Tempelhof-Schöneberg',
    'Charlottenburg-Nord': 'Charlottenburg-Wilmersdorf', 'Westend': 'Charlottenburg-Wilmersdorf',
    'Waidmannslust': 'Reinickendorf', 'Borsigwalde': 'Reinickendorf', 'Schlachtensee': 'Steglitz-Zehlendorf',
    'Niederschöneweide': 'Treptow-Köpenick', 'Zehlendorf': 'Steglitz-Zehlendorf', 'Karow': 'Pankow',
    'Johannisthal': 'Treptow-Köpenick', 'Plänterwald': 'Treptow-Köpenick', 'Wilhelmstadt': 'Spandau',
    'Hermsdorf': 'Reinickendorf', 'Karlshorst': 'Lichtenberg', 'Altglienicke': 'Treptow-Köpenick',
    'Lichtenrade': 'Tempelhof-Schöneberg', 'Wilhelmsruh': 'Pankow', 'Oberschöneweide': 'Treptow-Köpenick',
    'Mitte': 'Mitte', 'Baumschulenweg': 'Treptow-Köpenick', 'Märkisches Viertel': 'Reinickendorf',
    'Britz': 'Neukölln', 'Lichterfelde': 'Steglitz-Zehlendorf', 'Wittenau': 'Reinickendorf',
    'Tempelhof': 'Tempelhof-Schöneberg', 'Falkenhagener Feld': 'Spandau', 'Staaken': 'Spandau',
    'Kreuzberg': 'Friedrichshain-Kreuzberg', 'Rudow': 'Neukölln', 'Rummelsburg': 'Lichtenberg',
    'Niederschönhausen': 'Pankow', 'Haselhorst': 'Spandau', 'Alt-Treptow': 'Treptow-Köpenick',
    'Spandau': 'Spandau', 'Lichtenberg': 'Lichtenberg', 'Moabit': 'Mitte',
    'Reinickendorf': 'Reinickendorf', 'Schmargendorf': 'Charlottenburg-Wilmersdorf', 'Köpenick': 'Treptow-Köpenick',
    'Alt-Hohenschönhausen': 'Lichtenberg', 'Mariendorf': 'Tempelhof-Schöneberg', 'Hellersdorf': 'Marzahn-Hellersdorf',
    'Wedding': 'Mitte', 'Buckow': 'Neukölln', 'Lankwitz': 'Steglitz-Zehlendorf',
    'Pankow': 'Pankow', 'Prenzlauer Berg': 'Pankow', 'Friedrichshain': 'Friedrichshain-Kreuzberg',
    'Weißensee': 'Pankow', 'Schöneberg': 'Tempelhof-Schöneberg', 'Steglitz': 'Steglitz-Zehlendorf',
    'Neukölln': 'Neukölln', 'Hansaviertel': 'Mitte', 'Halensee': 'Charlottenburg-Wilmersdorf',
    'Charlottenburg': 'Charlottenburg-Wilmersdorf', 'Gesundbrunnen': 'Mitte', 'Friedrichsfelde': 'Lichtenberg',
    'Fennpfuhl': 'Lichtenberg', 'Neu-Hohenschönhausen': 'Lichtenberg', 'Friedenau': 'Tempelhof-Schöneberg',
    'Wilmersdorf': 'Charlottenburg-Wilmersdorf', 'Gropiusstadt': 'Neukölln'
}



# Apply the mapping
df['district'] = df['area_name'].map(district_mapping)

# 4. Reorder Columns (Place 'district' next to 'area_name')
cols = list(df.columns)
# Pop 'district' and insert it at index 3
cols.insert(3, cols.pop(cols.index('district')))
df = df[cols]

# 5. Export to CSV
df.to_csv('berlin_district_rankings_complete.csv', index=False)

print("File 'berlin_heat_risk_with_districts.csv' has been generated successfully.")

# insert district to berlin_enriched_data_with_scores.csv file and berlin_priority_intervention_zones.csv file 
berlin_enriched_data = pd.read_csv('berlin_enriched_data_with_scores.csv')
berlin_enriched_data['district'] = berlin_enriched_data['area_name'].map(district_mapping)
berlin_enriched_data.to_csv('berlin_enriched_data_with_scores.csv', index=False)

berlin_priority_intervention_zones = pd.read_csv('berlin_priority_intervention_zones.csv')
berlin_priority_intervention_zones['district'] = berlin_priority_intervention_zones['area_name'].map(district_mapping)
berlin_priority_intervention_zones.to_csv('berlin_priority_intervention_zones.csv', index=False)

# merge ../data/change_ratio.csv and ../data/retention on district/District
# on df,berlin_enriched_data_with_scores and berlin_priority_intervention_zones
change_ratio_df = pd.read_csv('../data/change_ratio.csv')
retention_df = pd.read_csv('../data/retention_rate.csv')

# Merge change_ratio_df and retention_df
merged_change_data = pd.merge(change_ratio_df, retention_df, left_on='District', right_on='District', how='inner')

# Merge with df (berlin_district_rankings_complete)
df = pd.merge(df, merged_change_data, left_on='district', right_on='District', how='left')
df = df.drop(columns=['District']) # Drop redundant 'District' column from merged_change_data
df.to_csv('berlin_district_rankings_complete.csv', index=False)

# Merge with berlin_enriched_data_with_scores
berlin_enriched_data = pd.merge(berlin_enriched_data, merged_change_data, left_on='district', right_on='District', how='left')
berlin_enriched_data = berlin_enriched_data.drop(columns=['District'])
berlin_enriched_data.to_csv('berlin_enriched_data_with_scores.csv', index=False)

# Merge with berlin_priority_intervention_zones
berlin_priority_intervention_zones = pd.merge(berlin_priority_intervention_zones, merged_change_data, left_on='district', right_on='District', how='left')
berlin_priority_intervention_zones = berlin_priority_intervention_zones.drop(columns=['District'])
berlin_priority_intervention_zones.to_csv('berlin_priority_intervention_zones.csv', index=False)

print("\nMerged change_ratio.csv and retention.csv data into all relevant files.")

