import pandas as pd

# Your data (copy-paste the entire table here)
data = """MERKMAL	Einwohnerzahl	Einwohnerzahl	Einwohnerzahl	Einwohnerzahl
SPALTE	15.05.2022	09.05.2011	Veränderung zu Zensus 2011	Einwohnerdichte
INFO_KASTEN	Anzahl	Anzahl	Anzahl	Einw. je km²
Regionalschluessel	Name_Gebietseinheit	Werte	Werte	Werte	Werte
110000010101	Moabit	75771	 	 	7081
110000010102	Märkisches Viertel	39341	 	 	5096
110000010103	Hansaviertel	5629	 	 	10621
110000010104	Tiergarten	13696	 	 	2649
110000010105	Wedding	78099	 	 	8461
110000010106	Gesundbrunnen	86701	 	 	14144
110000020201	Friedrichshain	126291	 	 	12913
110000020202	Köpenick	67140	 	 	6456
110000030301	Prenzlauer Berg	159369	 	 	14488
110000030302	Weißensee	54380	 	 	6858
110000030303	Blankenburg	6705	 	 	1112
110000030304	Heinersdorf	7623	 	 	1930
110000030305	Karow	18926	 	 	2846
110000030306	Stadtrandsiedlung Malchow	1078	 	 	190
110000030307	Pankow	64557	 	 	11406
110000030308	Blankenfelde	2013	 	 	150
110000030309	Buch	16399	 	 	901
110000030310	Französisch Buchholz	20858	 	 	1738
110000030311	Niederschönhausen	31138	 	 	4798
110000030312	Rosenthal	9674	 	 	1974
110000030313	Wilhelmsruh	7676	 	 	5603
110000040401	Charlottenburg	119191	 	 	11244
110000040402	Wilmersdorf	94261	 	 	13165
110000040403	Schmargendorf	22327	 	 	6219
110000040404	Grünau	7391	 	 	331
110000040405	Westend	38287	 	 	2836
110000040406	Charlottenburg-Nord	17766	 	 	2865
110000040407	Halensee	14595	 	 	11492
110000050501	Spandau	37536	 	 	4674
110000050502	Haselhorst	18049	 	 	3816
110000050503	Siemensstadt	11594	 	 	2048
110000050504	Staaken	44526	 	 	4085
110000050505	Gatow	3431	 	 	340
110000050506	Kladow	15906	 	 	1075
110000050507	Hakenfelde	30903	 	 	1515
110000050508	Falkenhagener Feld	37362	 	 	5431
110000050509	Wilhelmstadt	38453	 	 	3697
110000060601	Steglitz	71578	 	 	10542
110000060602	Lichterfelde	81594	 	 	4483
110000060603	Lankwitz	42291	 	 	6050
110000060604	Zehlendorf	52876	 	 	2813
110000060605	Dahlem	15919	 	 	1897
110000060606	Nikolassee	11107	 	 	567
110000060607	Wannsee	9760	 	 	412
110000060608	Schlachtensee	10004	 	 	2470
110000070701	Schöneberg	115670	 	 	10912
110000070702	Friedenau	27090	 	 	16418
110000070703	Tempelhof	57371	 	 	4703
110000070704	Mariendorf	49993	 	 	5330
110000070705	Marienfelde	30576	 	 	3342
110000070706	Lichtenrade	50590	 	 	5009
110000080801	Neukölln	150733	 	 	12883
110000080802	Britz	40252	 	 	3246
110000080803	Buckow	37718	 	 	5940
110000080804	Rudow	40736	 	 	3452
110000080805	Gropiusstadt	35584	 	 	13377
110000090901	Altglienicke	29905	 	 	3790
110000090902	Plänterwald	11118	 	 	3694
110000090903	Baumschulenweg	18813	 	 	3903
110000090904	Johannisthal	20224	 	 	3092
110000090905	Niederschöneweide	12946	 	 	3709
110000090906	Alt-Hohenschönhausen	48842	 	 	5235
110000090907	Adlershof	19753	 	 	3233
110000090908	Bohnsdorf	12310	 	 	1888
110000090909	Oberschöneweide	23416	 	 	3789
110000090910	Kreuzberg	137741	 	 	3947
110000090911	Friedrichshagen	18425	 	 	1316
110000090912	Rahnsdorf	10378	 	 	483
110000090913	Grunewald	10576	 	 	1158
110000090914	Müggelheim	6801	 	 	306
110000090915	Schmöckwitz	4439	 	 	260
110000101001	Mitte	97425	 	 	4996
110000101002	Biesdorf	28894	 	 	2330
110000101003	Kaulsdorf	18910	 	 	2146
110000101004	Mahlsdorf	29531	 	 	2289
110000101005	Hellersdorf	83175	 	 	10269
110000111101	Friedrichsfelde	55137	 	 	9935
110000111102	Karlshorst	27527	 	 	4171
110000111103	Lichtenberg	39622	 	 	5488
110000111104	Falkenberg	2891	 	 	945
110000111106	Malchow	663	 	 	431
110000111107	Wartenberg	2635	 	 	381
110000111109	Neu-Hohenschönhausen	55279	 	 	10713
110000111110	Alt-Treptow	12472	 	 	5399
110000111111	Fennpfuhl	32389	 	 	15278
110000111112	Rummelsburg	25551	 	 	5653
110000121201	Reinickendorf	76495	 	 	7285
110000121202	Tegel	35474	 	 	1053
110000121203	Konradshöhe	5768	 	 	2622
110000121204	Heiligensee	17506	 	 	1636
110000121205	Frohnau	16430	 	 	2106
110000121206	Hermsdorf	16429	 	 	2693
110000121207	Waidmannslust	10506	 	 	4568
110000121208	Lübars	4885	 	 	977
110000121209	Wittenau	23435	 	 	3972
110000121210	Marzahn	109498	 	 	34218
110000121211	Borsigwalde	6676	 	 	3338"""

# Split data into lines
lines = data.strip().split('\n')

# The first 4 lines are headers, the rest is data
headers = lines[:4]
data_lines = lines[4:]


# Parse headers (limit to number of columns in data)
num_cols = len(data_lines[0].split('\t'))
header_dict = {}
for i, line in enumerate(headers):
    parts = line.split('\t')
    for j in range(num_cols):
        part = parts[j] if j < len(parts) else ''
        if i == 0:
            header_dict[j] = [part]
        else:
            header_dict[j].append(part)

# Create column names from headers
column_names = []
for j in sorted(header_dict.keys()):
    if j == 0:  # First column
        column_names.append('regional_code')
    elif j == 1:  # Second column
        column_names.append('area_name')
    else:
        # Combine header rows for data columns
        col_parts = [h for h in header_dict[j] if h.strip()]
        if col_parts:
            # Use the most specific header
            column_names.append(col_parts[-1])

# English translations for column names
english_names = {
    'Regionalschluessel': 'regional_code',
    'Name_Gebietseinheit': 'area_name',
    'Werte': 'values',
    'Anzahl': 'count',
    '15.05.2022': 'population_2022',
    '09.05.2011': 'population_2011',
    'Veränderung zu Zensus 2011': 'change_from_2011',
    'Einw. je km²': 'population_density_per_sqkm',
    'Einwohnerdichte': 'population_density'
}

# Translate column names
translated_columns = []
for col in column_names:
    # Check for direct match
    if col in english_names:
        translated_columns.append(english_names[col])
    else:
        # Try partial match
        translated = col
        for de, en in english_names.items():
            if de in col:
                translated = en
                break
        translated_columns.append(translated)



print("Original columns:", column_names)
print("Translated columns:", translated_columns)

# ...existing code...



# Parse data (pad/truncate to num_cols)
parsed_data = []
for line in data_lines:
    parts = line.split('\t')
    if len(parts) < num_cols:
        parts += [''] * (num_cols - len(parts))
    elif len(parts) > num_cols:
        parts = parts[:num_cols]
    parsed_data.append(parts)

# Show actual DataFrame columns after creation
parsed_data_preview = parsed_data[:3]
print("Sample parsed data:", parsed_data_preview)


# Create DataFrame
df = pd.DataFrame(parsed_data, columns=translated_columns)
print("DataFrame columns:", list(df.columns))

# Convert numeric columns to proper types (only if present)
numeric_columns = ['population_2022', 'population_density_per_sqkm']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].str.replace('.', ''), errors='coerce')
    else:
        print(f"Warning: column '{col}' not found in DataFrame; skipping conversion.")

# Clean up the data
df = df.replace('', pd.NA)

# rename the column name with  []'regional_code' 'area_name' 'values' 'values' 'values' 'values'] to []'regional_code', 'area_name', 'population_2022', 'population_density_per_sqkm']
# Re-assign column names explicitly after translation logic
# This ensures correct mapping even if some intermediate steps are complex
df.columns = ['regional_code', 'area_name', 'population_2022', 'population_2011', 'change_from_2011', 'population_density_per_sqkm']

# drop population_2011 and change_from_2011 column
df = df.drop(columns=['population_2011', 'change_from_2011'])

# create a column postal code using regional code using last 5 digits
df['regional_code'] = df['regional_code'].astype(str)

df['postal_code'] = df['regional_code'].astype(str).str[-5:] # last 5 digits
# df['postal_code'] = df['postal_code'].apply(lambda x: x if x.isdigit() else None)



# Save to CSV
df.to_csv('berlin_population_2022_english.csv', index=False, encoding='utf-8')

print(f"\n✅ Converted {len(df)} records to CSV")
print("\nFirst few rows:")
print(df.head())
print(f"\nSaved as 'berlin_population_2022_english.csv'")




