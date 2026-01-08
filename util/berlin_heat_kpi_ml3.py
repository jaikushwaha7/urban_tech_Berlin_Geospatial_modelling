"""
Berlin Urban Heat Analysis - Complete Pipeline
1. Data Integration Pipeline (from provided code)
2. Machine Learning KPI Creation
3. Evidence-Based District Rankings
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA PIPELINE (Adapted from provided code)
# ============================================================================

class BerlinDataPipeline:
    """
    Main data pipeline for Berlin urban heat analysis
    Loads and processes all data sources for hexagonal grid mapping
    """
    
    def __init__(self, data_dir='./data/'):
        self.data_dir = Path(data_dir)
        self.weather_data = None
        self.pollution_data = None
        self.population_data = None
        self.traffic_data = None
        self.integrated_data = None
        
    def load_weather_data(self, filepath='../data/berlin_summer_2020_2025.csv'):
        """Load and process weather data from Open-Meteo"""
        print("📊 Loading weather data...")
        
        try:
            df = pd.read_csv(self.data_dir / filepath)
            df['date'] = pd.to_datetime(df['time'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            
            # Calculate monthly aggregates
            monthly = df.groupby(['year', 'month']).agg({
                'temperature_2m_max': 'mean',
                'temperature_2m_min': 'mean',
                'precipitation_sum': 'sum'
            }).reset_index()
            
            # Calculate temperature change (May to August for each year)
            temp_changes = []
            for year in monthly['year'].unique():
                year_data = monthly[monthly['year'] == year]
                may_data = year_data[year_data['month'] == 5]
                aug_data = year_data[year_data['month'] == 8]
                
                if not may_data.empty and not aug_data.empty:
                    temp_change = (
                        aug_data['temperature_2m_max'].values[0] - 
                        may_data['temperature_2m_max'].values[0]
                    )
                    temp_changes.append({
                        'year': year,
                        'temp_change_may_aug': temp_change,
                        'avg_tmax': year_data['temperature_2m_max'].mean(),
                        'total_precip': year_data['precipitation_sum'].sum()
                    })
            
            self.weather_data = pd.DataFrame(temp_changes)
            print(f"✓ Loaded weather data: {len(self.weather_data)} years")
            return self.weather_data
            
        except Exception as e:
            print(f"✗ Error loading weather data: {e}")
            return None
    
    def load_pollution_data(self, filepath='../data/berlin_pollution_cleaned_2020_2025_summer.csv'):
        """Load and process pollution data"""
        print("📊 Loading pollution data...")
        
        try:
            df = pd.read_csv(self.data_dir / filepath)
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            
            # Calculate monthly aggregates
            monthly = df.groupby(['year', 'month']).agg({
                'no2': 'mean',
                'pm10': 'mean',
                'pm25': 'mean',
                'o3': 'mean'
            }).reset_index()
            
            # Calculate pollution index (normalized NO2 + PM10)
            monthly['pollution_index'] = (
                stats.zscore(monthly['no2'].fillna(0)) + 
                stats.zscore(monthly['pm10'].fillna(0))
            ) / 2
            
            # Normalize to 0-1 range
            monthly['pollution_index'] = (
                (monthly['pollution_index'] - monthly['pollution_index'].min()) /
                (monthly['pollution_index'].max() - monthly['pollution_index'].min())
            )
            
            self.pollution_data = monthly
            print(f"✓ Loaded pollution data: {len(self.pollution_data)} records")
            return self.pollution_data
            
        except Exception as e:
            print(f"✗ Error loading pollution data: {e}")
            return None
    
    def load_population_data(self, filepath='../data/berlin_population_2022_english.csv'):
        """Load and process population census data"""
        print("📊 Loading population data...")
        
        try:
            df = pd.read_csv(self.data_dir / filepath)
            
            # Clean and process
            df['population_2022'] = pd.to_numeric(
                df['population_2022'].astype(str).str.replace('.', ''), 
                errors='coerce'
            )
            df['population_density_per_sqkm'] = pd.to_numeric(
                df['population_density_per_sqkm'].astype(str).str.replace('.', ''), 
                errors='coerce'
            )
            
            # Ensure postal_code is string with proper formatting
            df['postal_code'] = df['postal_code'].astype(str).str.zfill(5)
            
            # Select relevant columns
            population = df[['postal_code', 'area_name', 'population_2022', 
                            'population_density_per_sqkm']].copy()
            
            self.population_data = population
            print(f"✓ Loaded population data: {len(self.population_data)} areas")
            return self.population_data
            
        except Exception as e:
            print(f"✗ Error loading population data: {e}")
            return None
    
    def load_traffic_data(self, filepath='../data/plz_traffic_supply_index.csv'):
        """Load and process traffic supply data"""
        print("📊 Loading traffic data...")
        
        try:
            df = pd.read_csv(self.data_dir / filepath)
            
            # Ensure plz is string
            df['plz'] = df['plz'].astype(str).str.zfill(5)
            
            # Calculate traffic supply index if not present
            if 'traffic_supply_index' not in df.columns:
                df['traffic_supply_index'] = (
                    df['stop_count'] / df['stop_count'].max()
                )
            
            self.traffic_data = df
            print(f"✓ Loaded traffic data: {len(self.traffic_data)} postal codes")
            return self.traffic_data
            
        except Exception as e:
            print(f"✗ Error loading traffic data: {e}")
            return None
    
    def integrate_data_by_plz(self):
        """Integrate all data sources by postal code"""
        print("\n🔗 Integrating data by postal code...")
        
        if self.population_data is None:
            print("✗ Population data not loaded")
            return None
        
        # Start with population data as base
        integrated = self.population_data.copy()
        integrated.rename(columns={'postal_code': 'plz'}, inplace=True)
        
        # Add traffic data
        if self.traffic_data is not None:
            integrated = integrated.merge(
                self.traffic_data[['plz', 'stop_count', 'traffic_supply_index']],
                on='plz',
                how='left'
            )
            # Fill missing traffic data
            integrated['traffic_supply_index'].fillna(
                integrated['traffic_supply_index'].median(), inplace=True
            )
            integrated['stop_count'].fillna(
                integrated['stop_count'].median(), inplace=True
            )
        
        # Add average weather data (applies to all Berlin)
        if self.weather_data is not None:
            avg_weather = self.weather_data.mean()
            integrated['avg_temp_change'] = avg_weather['temp_change_may_aug']
            integrated['avg_tmax'] = avg_weather['avg_tmax']
            integrated['total_precip'] = avg_weather['total_precip']
        
        # Add average pollution data (single station, applies to all)
        if self.pollution_data is not None:
            avg_pollution = self.pollution_data.groupby('year').mean().mean()
            integrated['no2_avg'] = avg_pollution['no2']
            integrated['pm10_avg'] = avg_pollution['pm10']
            integrated['pollution_index'] = avg_pollution['pollution_index']
        
        # Add simulated temperature change per PLZ (in real scenario, from satellite LST)
        # This simulates ΔLST based on population density and pollution
        if 'population_density_per_sqkm' in integrated.columns:
            # Higher density -> higher temperature change
            density_factor = integrated['population_density_per_sqkm'] / integrated['population_density_per_sqkm'].max()
            base_temp_change = integrated.get('avg_temp_change', 2.0)
            integrated['temperature_change'] = base_temp_change * (0.7 + 0.6 * density_factor)
        
        self.integrated_data = integrated
        print(f"✓ Integrated data: {len(self.integrated_data)} postal codes")
        print(f"  Columns: {list(self.integrated_data.columns)}")
        
        return self.integrated_data

# ============================================================================
# PART 2: ML-BASED KPI CREATION & RANKING SYSTEM
# ============================================================================

class BerlinHeatRankingSystem:
    """
    ML-based ranking system for Berlin urban heat analysis
    Creates composite KPIs and evidence-based district rankings
    """
    
    def __init__(self, integrated_data):
        self.data = integrated_data.copy()
        self.scaler = StandardScaler()
        self.pca = None
        self.rf_model = None
        self.feature_importance = None
        
    def engineer_features(self):
        """Create advanced features for ML analysis"""
        print("\n🔧 Engineering Features...")
        
        # 1. Heat Exposure Metrics
        self.data['heat_exposure_index'] = (
            self.data['temperature_change'] * 
            self.data['population_density_per_sqkm'] / 1000
        )
        
        self.data['absolute_heat_impact'] = (
            self.data['temperature_change'] * 
            self.data['population_2022'] / 10000
        )
        
        # 2. Pollution-Temperature Interactions
        self.data['pollution_heat_interaction'] = (
            self.data['pollution_index'] * 
            self.data['temperature_change']
        )
        
        self.data['no2_heat_score'] = (
            self.data['no2_avg'] * 
            self.data['temperature_change']
        )
        
        # 3. Traffic-Environmental Impact
        self.data['traffic_pollution_score'] = (
            self.data['traffic_supply_index'] * 
            self.data['no2_avg']
        )
        
        # 4. Population Vulnerability Metrics
        self.data['population_pollution_burden'] = (
            self.data['population_2022'] * 
            self.data['pollution_index']
        )
        
        self.data['vulnerable_population_score'] = (
            self.data['population_density_per_sqkm'] * 
            self.data['pollution_index'] * 
            self.data['temperature_change']
        )
        
        # 5. Precipitation Deficit Impact
        self.data['heat_precip_ratio'] = (
            self.data['avg_tmax'] / 
            (self.data['total_precip'] + 1)
        )
        
        print(f"✓ Created {8} engineered features")
        
    def create_composite_kpis(self):
        """Create composite KPIs using PCA"""
        print("\n🎯 Creating Composite KPIs with PCA...")
        
        # Select features for PCA
        kpi_features = [
            'temperature_change',
            'population_density_per_sqkm',
            'pollution_index',
            'no2_avg',
            'pm10_avg',
            'traffic_supply_index',
            'heat_exposure_index',
            'pollution_heat_interaction',
            'vulnerable_population_score'
        ]
        
        # check for nan and fill
        self.data[kpi_features] = self.data[kpi_features].fillna(0)

        # Standardize features
        X = self.data[kpi_features].copy()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        self.pca = PCA(n_components=4)
        components = self.pca.fit_transform(X_scaled)
        
        # Create interpretable scores
        self.data['climate_vulnerability_score'] = components[:, 0]
        self.data['urban_intensity_score'] = components[:, 1]
        self.data['environmental_stress_score'] = components[:, 2]
        self.data['socio_environmental_score'] = components[:, 3]
        
        # Normalize to 0-100 scale
        score_cols = [
            'climate_vulnerability_score',
            'urban_intensity_score',
            'environmental_stress_score',
            'socio_environmental_score'
        ]
        
        for col in score_cols:
            min_val = self.data[col].min()
            max_val = self.data[col].max()
            self.data[f'{col}_norm'] = (
                (self.data[col] - min_val) / (max_val - min_val) * 100
            )
        
        print(f"✓ PCA Explained Variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        for i, var in enumerate(self.pca.explained_variance_ratio_):
            print(f"   Component {i+1}: {var:.2%}")
    
    def calculate_master_kpi(self):
        """Calculate Urban Heat Risk Index (UHRI)"""
        print("\n🏆 Calculating Master KPI: Urban Heat Risk Index...")
        
        # Evidence-based weights
        weights = {
            'climate_vulnerability_score_norm': 0.35,
            'urban_intensity_score_norm': 0.25,
            'environmental_stress_score_norm': 0.25,
            'socio_environmental_score_norm': 0.15
        }
        
        self.data['urban_heat_risk_index'] = (
            self.data['climate_vulnerability_score_norm'] * weights['climate_vulnerability_score_norm'] +
            self.data['urban_intensity_score_norm'] * weights['urban_intensity_score_norm'] +
            self.data['environmental_stress_score_norm'] * weights['environmental_stress_score_norm'] +
            self.data['socio_environmental_score_norm'] * weights['socio_environmental_score_norm']
        )
        
        # Risk categories
        self.data['risk_category'] = pd.cut(
            self.data['urban_heat_risk_index'],
            bins=[0, 30, 60, 100],
            labels=['Low Risk', 'Moderate Risk', 'High Risk']
        )
        
        # Priority ranking
        self.data['priority_rank'] = self.data['urban_heat_risk_index'].rank(
            ascending=False, method='dense'
        ).astype(int)
        
        print(f"✓ UHRI calculated for {len(self.data)} districts")
    
    def train_predictive_models(self):
        """Train ML models for validation"""
        print("\n🤖 Training Predictive Models...")
        
        # Prepare features
        feature_cols = [
            'population_density_per_sqkm',
            'pollution_index',
            'no2_avg',
            'pm10_avg',
            'traffic_supply_index',
            'avg_tmax',
            'total_precip'
        ]
        
        X = self.data[feature_cols].copy()
        y = self.data['temperature_change'].copy()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        self.rf_model.fit(X_train_scaled, y_train)
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_r2 = r2_score(y_test, rf_pred)
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        gb_r2 = r2_score(y_test, gb_pred)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   Random Forest R²: {rf_r2:.4f}")
        print(f"   Gradient Boosting R²: {gb_r2:.4f}")
        
        return rf_r2, gb_r2
    
    def perform_clustering(self):
        """Cluster districts into typologies"""
        print("\n🔍 Clustering Districts...")
        
        # Features for clustering
        cluster_features = [
            'temperature_change',
            'population_density_per_sqkm',
            'pollution_index',
            'no2_avg',
            'pm10_avg',
            'traffic_supply_index',
            'heat_exposure_index',
            'pollution_heat_interaction',
            'vulnerable_population_score'
        ]
        
        X = self.data[cluster_features].copy()
        X_scaled = StandardScaler().fit_transform(X)
        
        # K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.data['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Assign meaningful names
        cluster_chars = self.data.groupby('cluster').agg({
            'population_density_per_sqkm': 'mean'
        }).sort_values('population_density_per_sqkm', ascending=False)
        
        cluster_names = {
            cluster_chars.index[0]: 'Urban Core (High Density)',
            cluster_chars.index[1]: 'Mixed Urban (Medium Density)',
            cluster_chars.index[2]: 'Suburban (Low Density)'
        }
        
        self.data['district_type'] = self.data['cluster'].map(cluster_names)
        print(f"✓ Created 3 district typologies")
    
    def generate_rankings(self):
        """Generate comprehensive rankings"""
        print("\n📊 Generating Evidence-Based Rankings...")
        
        ranking_cols = [
            'priority_rank',
            'plz',
            'area_name',
            'urban_heat_risk_index',
            'risk_category',
            'district_type',
            'temperature_change',
            'population_2022',
            'population_density_per_sqkm',
            'pollution_index',
            'no2_avg',
            'pm10_avg',
            'traffic_supply_index',
            'heat_exposure_index',
            'vulnerable_population_score'
        ]
        
        rankings = self.data[ranking_cols].copy()
        rankings = rankings.sort_values('priority_rank')
        
        # Round numerical values
        for col in rankings.columns:
            if rankings[col].dtype in ['float64', 'float32']:
                rankings[col] = rankings[col].round(2)
        
        return rankings
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 80)
        print("📋 BERLIN URBAN HEAT - COMPREHENSIVE SUMMARY")
        print("=" * 80)
        
        total_pop = self.data['population_2022'].sum()
        high_risk_pop = self.data[
            self.data['risk_category'] == 'High Risk'
        ]['population_2022'].sum()
        
        print(f"\n🌆 OVERALL STATISTICS:")
        print(f"   Total Districts: {len(self.data)}")
        print(f"   Total Population: {total_pop:,.0f}")
        print(f"   Avg Temperature Change: {self.data['temperature_change'].mean():.2f}°C")
        print(f"   Avg Heat Risk Index: {self.data['urban_heat_risk_index'].mean():.2f}/100")
        
        print(f"\n🚦 RISK DISTRIBUTION:")
        for category in ['High Risk', 'Moderate Risk', 'Low Risk']:
            count = (self.data['risk_category'] == category).sum()
            pop = self.data[self.data['risk_category'] == category]['population_2022'].sum()
            print(f"   {category:.<20} {count:>3} districts ({pop:>8,.0f} residents)")
        
        print(f"\n🔝 TOP 5 HIGHEST RISK DISTRICTS:")
        top_5 = self.data.nsmallest(5, 'priority_rank')[
            ['priority_rank', 'plz', 'area_name', 'urban_heat_risk_index', 
             'temperature_change', 'population_density_per_sqkm']
        ]
        for _, row in top_5.iterrows():
            print(f"\n   #{row['priority_rank']} | {row['area_name']} (PLZ: {row['plz']})")
            print(f"      Risk Index: {row['urban_heat_risk_index']:.1f} | "
                  f"Temp Change: {row['temperature_change']:.2f}°C | "
                  f"Pop Density: {row['population_density_per_sqkm']:.0f}/km²")
        
        print(f"\n🎯 FEATURE IMPORTANCE (Top 5):")
        for _, row in self.feature_importance.head(5).iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"   {row['feature']:.<35} {bar} {row['importance']:.4f}")
        
        print(f"\n🏙️ DISTRICT TYPOLOGIES:")
        for dtype in self.data['district_type'].unique():
            count = (self.data['district_type'] == dtype).sum()
            avg_risk = self.data[self.data['district_type']==dtype]['urban_heat_risk_index'].mean()
            print(f"   {dtype:.<35} {count} districts (Avg Risk: {avg_risk:.1f})")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute complete pipeline"""
    print("=" * 80)
    print("🌡️ BERLIN URBAN HEAT ANALYSIS - COMPLETE PIPELINE")
    print("=" * 80)
    
    # ========== STEP 1: DATA INTEGRATION ==========
    print("\n" + "=" * 80)
    print("STEP 1: DATA INTEGRATION PIPELINE")
    print("=" * 80)
    
    pipeline = BerlinDataPipeline(data_dir='./')
    
    # Load all data sources
    pipeline.load_weather_data()
    pipeline.load_pollution_data()
    pipeline.load_population_data()
    pipeline.load_traffic_data()
    
    # Integrate data
    integrated_data = pipeline.integrate_data_by_plz()
    
    if integrated_data is None:
        print("\n❌ Pipeline failed - check data files")
        return
    
    # Save integrated data
    integrated_data.to_csv('berlin_integrated_data.csv', index=False)
    print(f"\n💾 Saved: berlin_integrated_data.csv")
    
    # ========== STEP 2: ML-BASED RANKING SYSTEM ==========
    print("\n" + "=" * 80)
    print("STEP 2: ML-BASED KPI CREATION & RANKING")
    print("=" * 80)
    
    ranking_system = BerlinHeatRankingSystem(integrated_data)
    
    # Execute ranking pipeline
    ranking_system.engineer_features()
    ranking_system.create_composite_kpis()
    ranking_system.calculate_master_kpi()
    ranking_system.train_predictive_models()
    ranking_system.perform_clustering()
    
    # Generate rankings
    rankings = ranking_system.generate_rankings()
    
    # ========== STEP 3: SAVE RESULTS ==========
    print("\n" + "=" * 80)
    print("STEP 3: SAVING RESULTS")
    print("=" * 80)
    
    # Save complete rankings
    rankings.to_csv('berlin_district_rankings_complete.csv', index=False)
    print(f"✓ Saved: berlin_district_rankings_complete.csv")
    
    # Save top priority zones
    top_10 = rankings.head(10)
    top_10.to_csv('berlin_priority_intervention_zones.csv', index=False)
    print(f"✓ Saved: berlin_priority_intervention_zones.csv")
    
    # Save full enriched dataset
    ranking_system.data.to_csv('berlin_enriched_data_with_scores.csv', index=False)
    print(f"✓ Saved: berlin_enriched_data_with_scores.csv")
    
    # ========== STEP 4: DISPLAY SUMMARY ==========
    ranking_system.print_summary()
    
    # ========== STEP 5: DISPLAY COMPLETE RANKINGS TABLE ==========
    print("\n" + "=" * 80)
    print("📊 COMPLETE DISTRICT RANKINGS TABLE")
    print("=" * 80)
    print(rankings.to_string(index=False))
    
    # ========== STEP 6: POSTAL CODE LOOKUP ==========
    print("\n" + "=" * 80)
    print("🔍 POSTAL CODE QUICK LOOKUP")
    print("=" * 80)
    
    for _, row in rankings.iterrows():
        print(f"PLZ {row['plz']} | {row['area_name']:.<25} | "
              f"Rank #{row['priority_rank']:>2} | "
              f"Risk: {row['urban_heat_risk_index']:>5.1f} | "
              f"Category: {row['risk_category']}")
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE PIPELINE EXECUTION FINISHED")
    print("=" * 80)
    print("\n📁 Output Files Generated:")
    print("   1. berlin_integrated_data.csv              (Base integrated data)")
    print("   2. berlin_district_rankings_complete.csv   (All rankings)")
    print("   3. berlin_priority_intervention_zones.csv  (Top 10 priority)")
    print("   4. berlin_enriched_data_with_scores.csv    (Full dataset with all scores)")


if __name__ == "__main__":
    main()