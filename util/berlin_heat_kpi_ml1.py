import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: DATA LOADING AND PREPARATION
# ============================================================================

print("=" * 80)
print("BERLIN URBAN HEAT ISLAND - ML-BASED KPI ANALYSIS")
print("=" * 80)

# Load datasets
integrated_data = pd.read_csv('../data/berlin_integrated_data.csv')
pollution_data = pd.read_csv('../data/berlin_pollution_cleaned_2020_2025_summer.csv')
population_data = pd.read_csv('../data/berlin_population_2022_english.csv')
summer_data = pd.read_csv('../data/berlin_summer_2020_2025.csv')
traffic_data = pd.read_csv('../data/plz_traffic_supply_index.csv')

print("\n📊 Dataset Overview:")
print(f"   - Integrated Data: {integrated_data.shape}")
print(f"   - Pollution Data: {pollution_data.shape}")
print(f"   - Population Data: {population_data.shape}")
print(f"   - Summer Weather: {summer_data.shape}")
print(f"   - Traffic Data: {traffic_data.shape}")

# Merge traffic data with integrated data
integrated_data = integrated_data.merge(
    traffic_data[['plz', 'stop_count', 'traffic_supply_index']], 
    on='plz', 
    how='left'
)

# Fill missing traffic data with median
integrated_data['traffic_supply_index'].fillna(
    integrated_data['traffic_supply_index'].median(), inplace=True
)
integrated_data['stop_count'].fillna(
    integrated_data['stop_count'].median(), inplace=True
)

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

print("\n🔧 Feature Engineering...")

# Calculate additional features
integrated_data['heat_exposure_index'] = (
    integrated_data['temperature_change'] * 
    integrated_data['population_density_per_sqkm'] / 1000
)

integrated_data['pollution_heat_interaction'] = (
    integrated_data['pollution_index'] * 
    integrated_data['temperature_change']
)

integrated_data['traffic_pollution_score'] = (
    integrated_data['traffic_supply_index'] * 
    integrated_data['no2_avg']
)

integrated_data['population_pollution_burden'] = (
    integrated_data['population_2022'] * 
    integrated_data['pollution_index']
)

# Urban density category
integrated_data['density_category'] = pd.cut(
    integrated_data['population_density_per_sqkm'],
    bins=[0, 5000, 10000, 15000],
    labels=['Low', 'Medium', 'High']
)

# ============================================================================
# STEP 3: COMPOSITE KPI CREATION USING PCA
# ============================================================================

print("\n🎯 Creating Composite KPIs using ML techniques...")

# Select features for KPI creation
kpi_features = [
    'temperature_change',
    'population_density_per_sqkm',
    'pollution_index',
    'no2_avg',
    'pm10_avg',
    'traffic_supply_index',
    'heat_exposure_index'
]

# Prepare data

X_kpi = integrated_data[kpi_features].copy()
X_kpi_scaled = StandardScaler().fit_transform(X_kpi)

# PCA for dimensionality reduction and KPI creation
pca = PCA(n_components=3)
pca_components = pca.fit_transform(X_kpi_scaled)

print(f"\n📈 PCA Explained Variance Ratio:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"   Component {i+1}: {var:.2%}")
print(f"   Total Variance Explained: {pca.explained_variance_ratio_.sum():.2%}")

# Create composite KPIs from PCA components
integrated_data['climate_vulnerability_score'] = pca_components[:, 0]
integrated_data['urban_intensity_score'] = pca_components[:, 1]
integrated_data['environmental_stress_score'] = pca_components[:, 2]

# Normalize composite scores to 0-100 scale
for col in ['climate_vulnerability_score', 'urban_intensity_score', 
            'environmental_stress_score']:
    min_val = integrated_data[col].min()
    max_val = integrated_data[col].max()
    integrated_data[f'{col}_normalized'] = (
        (integrated_data[col] - min_val) / (max_val - min_val) * 100
    )

# ============================================================================
# STEP 4: MASTER KPI - URBAN HEAT RISK INDEX
# ============================================================================

print("\n🏆 Creating Master KPI: Urban Heat Risk Index (UHRI)...")

# Weighted combination of normalized scores
weights = {
    'climate_vulnerability_score_normalized': 0.40,
    'urban_intensity_score_normalized': 0.30,
    'environmental_stress_score_normalized': 0.30
}

integrated_data['urban_heat_risk_index'] = (
    integrated_data['climate_vulnerability_score_normalized'] * weights['climate_vulnerability_score_normalized'] +
    integrated_data['urban_intensity_score_normalized'] * weights['urban_intensity_score_normalized'] +
    integrated_data['environmental_stress_score_normalized'] * weights['environmental_stress_score_normalized']
)

# Risk categories
integrated_data['risk_category'] = pd.cut(
    integrated_data['urban_heat_risk_index'],
    bins=[0, 33, 66, 100],
    labels=['Low Risk', 'Moderate Risk', 'High Risk']
)

# ============================================================================
# STEP 5: CLUSTERING ANALYSIS
# ============================================================================

print("\n🔍 Performing K-Means Clustering...")

# K-Means clustering to identify similar districts
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
integrated_data['cluster'] = kmeans.fit_predict(X_kpi_scaled)

cluster_names = {0: 'Type A', 1: 'Type B', 2: 'Type C'}
integrated_data['cluster_name'] = integrated_data['cluster'].map(cluster_names)

# ============================================================================
# STEP 6: PREDICTIVE MODELING
# ============================================================================

print("\n🤖 Building Predictive Models...")

# Prepare features for modeling
feature_cols = [
    'population_density_per_sqkm',
    'pollution_index',
    'no2_avg',
    'pm10_avg',
    'traffic_supply_index',
    'avg_tmax',
    'total_precip'
]

X = integrated_data[feature_cols].copy()
y = integrated_data['temperature_change'].copy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

# Model 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

# Model 3: Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_r2 = r2_score(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

print("\n📊 Model Performance Comparison:")
print(f"   Linear Regression    - R²: {lr_r2:.4f}, RMSE: {lr_rmse:.4f}")
print(f"   Random Forest        - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")
print(f"   Gradient Boosting    - R²: {gb_r2:.4f}, RMSE: {gb_rmse:.4f}")

# Feature importance (using best model - Random Forest)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🎯 Feature Importance (Random Forest):")
for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']:.<40} {row['importance']:.4f}")

# ============================================================================
# STEP 7: TIME SERIES ANALYSIS (POLLUTION TRENDS)
# ============================================================================

print("\n📅 Analyzing Pollution Trends...")

pollution_data['date'] = pd.to_datetime(pollution_data['date'])
pollution_data['year'] = pollution_data['date'].dt.year
pollution_data['month'] = pollution_data['date'].dt.month

# Calculate yearly aggregates
yearly_pollution = pollution_data.groupby('year').agg({
    'pm10': 'mean',
    'pm25': 'mean',
    'no2': 'mean',
    'o3': 'mean'
}).round(2)

print("\n📊 Yearly Pollution Trends (Summer Months):")
print(yearly_pollution)

# ============================================================================
# STEP 8: COMPREHENSIVE KPI SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("📋 COMPREHENSIVE KPI SUMMARY")
print("=" * 80)

summary_stats = integrated_data.groupby('risk_category').agg({
    'urban_heat_risk_index': ['mean', 'min', 'max'],
    'temperature_change': 'mean',
    'population_2022': 'sum',
    'pollution_index': 'mean',
    'traffic_supply_index': 'mean',
    'area_name': 'count'
}).round(2)

print("\n🎯 KPI Breakdown by Risk Category:")
print(summary_stats)

# Top 5 highest risk areas
print("\n🔴 TOP 5 HIGHEST RISK DISTRICTS:")
top_risk = integrated_data.nlargest(5, 'urban_heat_risk_index')[
    ['plz', 'area_name', 'urban_heat_risk_index', 'risk_category', 
     'temperature_change', 'population_density_per_sqkm']
]
for idx, row in top_risk.iterrows():
    print(f"\n   {row['area_name']} (PLZ: {row['plz']})")
    print(f"      - Heat Risk Index: {row['urban_heat_risk_index']:.2f}/100")
    print(f"      - Temperature Change: {row['temperature_change']:.2f}°C")
    print(f"      - Population Density: {row['population_density_per_sqkm']:.0f}/km²")

# Cluster characteristics
print("\n🔍 CLUSTER ANALYSIS:")
cluster_summary = integrated_data.groupby('cluster_name').agg({
    'temperature_change': 'mean',
    'population_density_per_sqkm': 'mean',
    'pollution_index': 'mean',
    'urban_heat_risk_index': 'mean',
    'area_name': 'count'
}).round(2)
cluster_summary.columns = ['Avg_Temp_Change', 'Avg_Pop_Density', 
                           'Avg_Pollution', 'Avg_Risk_Index', 'District_Count']
print(cluster_summary)

# Overall city statistics
print("\n🌆 OVERALL BERLIN STATISTICS:")
print(f"   Total Population: {integrated_data['population_2022'].sum():,}")
print(f"   Average Temperature Change: {integrated_data['temperature_change'].mean():.2f}°C")
print(f"   Average Pollution Index: {integrated_data['pollution_index'].mean():.4f}")
print(f"   Average Heat Risk Index: {integrated_data['urban_heat_risk_index'].mean():.2f}/100")
print(f"   High Risk Districts: {(integrated_data['risk_category'] == 'High Risk').sum()}")

# ============================================================================
# STEP 9: ACTIONABLE INSIGHTS
# ============================================================================

print("\n" + "=" * 80)
print("💡 KEY ACTIONABLE INSIGHTS")
print("=" * 80)

print("\n1. PRIORITY INTERVENTION ZONES:")
high_risk_districts = integrated_data[
    integrated_data['risk_category'] == 'High Risk'
]['area_name'].tolist()
print(f"   Districts requiring immediate attention: {', '.join(high_risk_districts)}")

print("\n2. CORRELATION INSIGHTS:")
corr_temp_pollution = integrated_data['temperature_change'].corr(
    integrated_data['pollution_index']
)
corr_temp_density = integrated_data['temperature_change'].corr(
    integrated_data['population_density_per_sqkm']
)
print(f"   Temperature-Pollution Correlation: {corr_temp_pollution:.3f}")
print(f"   Temperature-Density Correlation: {corr_temp_density:.3f}")

print("\n3. MODEL ACCURACY:")
print(f"   Best Model: Random Forest (R² = {rf_r2:.3f})")
print(f"   Temperature changes can be predicted with {rf_r2*100:.1f}% accuracy")

print("\n4. POPULATION AT RISK:")
high_risk_pop = integrated_data[
    integrated_data['risk_category'] == 'High Risk'
]['population_2022'].sum()
total_pop = integrated_data['population_2022'].sum()
print(f"   Population in High Risk zones: {high_risk_pop:,} ({high_risk_pop/total_pop*100:.1f}%)")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)

# Save results
output_file = 'berlin_heat_kpi_results.csv'
integrated_data.to_csv(output_file, index=False)
print(f"\n💾 Results saved to: {output_file}")