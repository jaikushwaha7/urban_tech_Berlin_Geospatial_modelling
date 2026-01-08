# Data Integration for Berlin Urban Heat Study (MVP_V3)

## 🗃️ Data Sources & Processing

| Data Type      | Source                                 | Key Variables                        | Processing Logic                                                                 |
|--------------- |----------------------------------------|--------------------------------------|---------------------------------------------------------------------------------|
| Weather        | Open-Meteo API (2020–2025)             | temperature_2m_max, precipitation_sum| Monthly aggregation → May–August ΔTmax per year → Berlin-wide averages          |
| Air Pollution  | Luftdaten.info + EPA                   | no2, pm10, pm25, o3                  | Monthly means → Z-score normalized pollution index → 0–1 scaling                |
| Population     | Berlin Census 2022 (Amt für Statistik) | population_2022, population_density  | Numeric cleanup, postal code standardization (5-digit PLZ), area name retention |
| Traffic Supply | BVG/Berlin Mobility API                | stop_count, traffic_supply_index      | Normalized stop count → index (0–1). Missing values imputed                     |

---

## 🛰️ 1. Satellite Data Acquisition & Processing
**Source:** Copernicus Sentinel-2 Level-2A (via Copernicus Open Access Hub)
- **Temporal Scope:** May 1, 2020 – August 31, 2025 (summer seasons only)
- **Geographic Scope:** Berlin bounding box
- **Cloud Removal:** s2cloudless + DEM correction (≤10 km AGL)
- **Cloud Threshold:** ≤20% residual cloud cover per image
- **Output:** Monthly composite NDVI rasters, summarized per image tile

### NDVI Summary Metrics (per image)
| Field                | Description                        | Example      |
|----------------------|------------------------------------|--------------|
| Filename             | Image ID                           | nvdi_001.png |
| Size                 | Pixel dimensions                   | (512, 512)   |
| Format               | PNG (8-bit scaled NDVI)            | PNG          |
| Avg_NDVI_Intensity   | Mean pixel value (0–255 scaled)    | 104.32       |
| Max_NDVI_Intensity   | Peak vegetation health             | 255          |
| Green_Coverage_Pct   | % pixels with NDVI > 128 (scaled)  | 17.0%        |

### Change Score Calculation
For sequential acquisitions, compute relative NDVI shift:

    Change Score (Δ) = (Green_Coverage_Pctₜ − Green_Coverage_Pctₜ₋₁)

Used to detect green infrastructure loss/gain.

| File ID   | Acquisition Period | Primary District Cluster | Green Coverage % | NDVI Intensity (Mean) | Change Score (Δ) |
|-----------|-------------------|-------------------------|------------------|----------------------|------------------|
| nvdi_001  | July 2023         | Mixed / Baseline        | 17.00%           | 104.32               | 0.00             |
| nvdi_002  | Sept 2023         | Transition              | 14.71%           | 99.95                | −2.29            |
| nvdi_003  | March 2024        | Pre-Spring              | 14.21%           | 100.76               | −0.50            |
| nvdi_004  | May 2025          | Forest-Heavy            | 26.01%           | 118.22               | +11.80           |
| nvdi_005  | July 2025         | Urban Core              | 10.78%           | 100.16               | −15.23           |

---

## 📊 2. Population Data Integration
**Source:** Amt für Statistik Berlin-Brandenburg
- Official state statistics office — data legally mandated, annually updated, spatially referenced to official PLZ and Bezirk boundaries.
- **Data Used:**
  - "Bevölkerung nach Berliner Bezirken und Ortsteilen" (2022 base, projected to 2025)
  - Gridded population (100m × 100m) via Gitternetz-Bevölkerungsmodell (where available)
- **Temporal Handling:**
  - 2020–2022: census actuals
  - 2023–2025: linear projection based on migration + birth/death rates (official forecast tables)
  - Documented uncertainty: ±1.2% per year (per StatBB methodology report)
- **Spatial Alignment:**
  - Population grids → resampled to match Sentinel-2 10m resolution via conservative area-weighted aggregation (no interpolation), then zonal stats to PLZ/H3 units.

---

## 🔗 Integration Workflow Summary
1. **Ingest** all raw data (weather, pollution, population, traffic, satellite)
2. **Clean & standardize** (handle missing values, normalize, align spatial units)
3. **Aggregate** to common spatial units (PLZ, H3, district)
4. **Compute indices** (NDVI, pollution, traffic supply, population density)
5. **Export** integrated datasets for analytics and visualization

---

## 📁 Output Files
- `berlin_integrated_data.csv` — Main integrated dataset
- `final_berlin_h3_comparison.csv` — H3 grid-based summary
- `change_score.csv` — NDVI change scores
- `berlin_population_2022_english.csv` — Population reference

---

## 📞 Contact
For questions or data requests, contact the project maintainer via GitHub Issues.
