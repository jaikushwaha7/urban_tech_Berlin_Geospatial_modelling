"""
Berlin Urban Heat Data Pipeline
Integrates weather, pollution, population, and traffic data for hexagonal grid analysis
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import json
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class BerlinDataPipeline:
    """
    Main data pipeline for Berlin urban heat analysis
    Loads and processes all data sources for hexagonal grid mapping
    """
    
    def __init__(self, data_dir='./'):
        self.data_dir = Path(data_dir)
        self.weather_data = None
        self.pollution_data = None
        self.population_data = None
        self.traffic_data = None
        self.integrated_data = None
        
    def load_weather_data(self, filepath='berlin_summer_2020_2025.csv'):
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
    
    def load_pollution_data(self, filepath='berlin_pollution_cleaned_2020_2025_summer.csv'):
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
    
    def load_population_data(self, filepath='berlin_population_2022_english.csv'):
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
    
    def load_traffic_data(self, filepath='traffic_stop_density.csv'):
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
    
    def export_for_streamlit(self, output_path='berlin_integrated_data.csv'):
        """Export integrated data for Streamlit app"""
        if self.integrated_data is None:
            print("✗ No integrated data to export")
            return
        
        try:
            filepath = self.data_dir / output_path
            self.integrated_data.to_csv(filepath, index=False)
            print(f"\n✓ Exported to: {filepath}")
            print(f"  Records: {len(self.integrated_data)}")
            print(f"  Columns: {len(self.integrated_data.columns)}")
        except Exception as e:
            print(f"✗ Error exporting data: {e}")
    
    def export_for_webapp(self, output_path='berlin_hex_data.json'):
        """Export as JSON for web applications"""
        if self.integrated_data is None:
            print("✗ No integrated data to export")
            return
        
        try:
            # Convert to dictionary format
            hex_data = {}
            for _, row in self.integrated_data.iterrows():
                plz = row['plz']
                hex_data[plz] = {
                    'temperature_change': float(row.get('temperature_change', 0)),
                    'pollution_index': float(row.get('pollution_index', 0)),
                    'population_density': float(row.get('population_density_per_sqkm', 0)),
                    'traffic_supply': float(row.get('traffic_supply_index', 0)),
                    'no2_avg': float(row.get('no2_avg', 0)),
                    'pm10_avg': float(row.get('pm10_avg', 0)),
                }
            
            filepath = self.data_dir / output_path
            with open(filepath, 'w') as f:
                json.dump(hex_data, f, indent=2)
            
            print(f"\n✓ Exported JSON to: {filepath}")
            print(f"  Postal codes: {len(hex_data)}")
        except Exception as e:
            print(f"✗ Error exporting JSON: {e}")
    
    def generate_summary_report(self):
        """Generate summary statistics report"""
        if self.integrated_data is None:
            print("✗ No data to summarize")
            return
        
        print("\n" + "="*70)
        print("📊 BERLIN URBAN HEAT DATA SUMMARY")
        print("="*70)
        
        # Data sources
        print("\n📁 Data Sources Loaded:")
        print(f"  ✓ Weather: {len(self.weather_data) if self.weather_data is not None else 0} years")
        print(f"  ✓ Pollution: {len(self.pollution_data) if self.pollution_data is not None else 0} records")
        print(f"  ✓ Population: {len(self.population_data) if self.population_data is not None else 0} areas")
        print(f"  ✓ Traffic: {len(self.traffic_data) if self.traffic_data is not None else 0} postal codes")
        
        # Key statistics
        print("\n📈 Key Statistics (by Postal Code):")
        stats_cols = ['temperature_change', 'pollution_index', 
                      'population_density_per_sqkm', 'traffic_supply_index']
        
        for col in stats_cols:
            if col in self.integrated_data.columns:
                values = self.integrated_data[col].dropna()
                if len(values) > 0:
                    print(f"\n  {col}:")
                    print(f"    Min:  {values.min():.2f}")
                    print(f"    Mean: {values.mean():.2f}")
                    print(f"    Max:  {values.max():.2f}")
        
        # Correlation analysis
        print("\n🔗 Correlation with Temperature Change:")
        if 'temperature_change' in self.integrated_data.columns:
            for col in ['pollution_index', 'population_density_per_sqkm', 'traffic_supply_index']:
                if col in self.integrated_data.columns:
                    corr = self.integrated_data[['temperature_change', col]].corr().iloc[0, 1]
                    print(f"    {col}: {corr:.3f}")
        
        print("\n" + "="*70)


def main():
    """Main execution function"""
    print("🌡️ Berlin Urban Heat Data Pipeline")
    print("="*70)
    
    # Initialize pipeline
    pipeline = BerlinDataPipeline(data_dir='./data/')
    
    # Load all data sources
    pipeline.load_weather_data()
    pipeline.load_pollution_data()
    pipeline.load_population_data()
    pipeline.load_traffic_data()
    
    # Integrate data
    integrated = pipeline.integrate_data_by_plz()
    
    if integrated is not None:
        # Export for different uses
        pipeline.export_for_streamlit('berlin_integrated_data.csv')
        pipeline.export_for_webapp('berlin_hex_data.json')
        
        # Generate report
        pipeline.generate_summary_report()
        
        print("\n✅ Pipeline execution complete!")
        print("\nNext steps:")
        print("  1. Run Streamlit app: streamlit run streamlit_app.py")
        print("  2. Use berlin_integrated_data.csv for analysis")
        print("  3. Use berlin_hex_data.json for web mapping")
    else:
        print("\n❌ Pipeline failed - check data files")


if __name__ == "__main__":
    main()