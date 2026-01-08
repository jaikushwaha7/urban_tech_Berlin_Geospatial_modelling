import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point
import json
from datetime import datetime
from scipy import stats
import os
# Page configuration
st.set_page_config(
    page_title="Berlin Urban Heat Analysis",
    page_icon="🌡️",
    layout="wide"
)

# Constants
BERLIN_CENTER = [52.52, 13.405]

# Data loading functions
@st.cache_data
def load_berlin_districts():
    """Load Berlin district boundaries"""
    try:
        url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
        districts = gpd.read_file(url)
        return districts
    except Exception as e:
        st.error(f"Error loading districts: {e}")
        return None

@st.cache_data
def load_berlin_postcodes():
    """Load Berlin postal code boundaries"""
    try:
        url = "https://tsb-opendata.s3.eu-central-1.amazonaws.com/plz/plz.geojson"
        postcodes = gpd.read_file(url)
        postcodes = postcodes.rename(columns={"plz": "postal_code"})
        postcodes["postal_code"] = postcodes["postal_code"].astype(str)
        return postcodes
    except Exception as e:
        st.warning(f"Could not load postcodes: {e}")
        return None

@st.cache_data
def load_environmental_data():
    """Load sample environmental data with seasonal LST"""
    np.random.seed(42)
    # make postal code list for all codes in berlin
    # can use the data from berlin population where al berlin postal code are there
    # Load Berlin postal codes from a local file or URL
    try:
        berlin_plz_path = os.path.join(os.path.dirname(__file__), 'berlin_plz.geojson')
        if os.path.exists(berlin_plz_path):
            berlin_plz = gpd.read_file(berlin_plz_path)
        else:
            berlin_plz = gpd.read_file("https://tsb-opendata.s3.eu-central-1.amazonaws.com/plz/plz.geojson")
        
        # Extract unique postal codes
        plz_list = berlin_plz['plz'].unique().tolist()
    except Exception as e:
        st.warning(f"Could not load Berlin postal codes for environmental data generation: {e}. Using a default list.")
        
    
    data = {
        'postal_code': plz_list,
        'lst_may': np.random.uniform(18, 24, len(plz_list)),
        'lst_august': np.random.uniform(28, 36, len(plz_list)),
        'no2_avg': np.random.uniform(20, 50, len(plz_list)),
        'pm10_avg': np.random.uniform(15, 40, len(plz_list)),
        'population_density': np.random.uniform(5000, 18000, len(plz_list)),
        'traffic_supply': np.random.uniform(0.3, 1.0, len(plz_list)),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived metrics
    df['delta_lst'] = df['lst_august'] - df['lst_may']
    df['pollution_index'] = stats.zscore(df['no2_avg']) + stats.zscore(df['pm10_avg'])
    df['population_weighted_lst'] = df['delta_lst'] * (df['population_density'] / df['population_density'].max())
    
    return df

def merge_geo_data(postcodes, env_data):
    """Merge environmental data with postal code geometries"""
    merged = postcodes.merge(env_data, on='postal_code', how='inner')
    # Calculate centroids for point visualization
    merged['centroid'] = merged.geometry.centroid
    merged['lat'] = merged.centroid.y
    merged['lon'] = merged.centroid.x
    return merged

# Visualization functions
def create_point_map(gdf, metric, title):
    """Create map with colored points"""
    m = folium.Map(
        location=BERLIN_CENTER,
        zoom_start=11,
        tiles='CartoDB positron'
    )
    
    # Add postal code boundaries
    # Add each point as a CircleMarker
    for idx, row in gdf.iterrows():
        if row["geometry"].geom_type == "Point":
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=8,
                popup=f"<b>PLZ:</b> {row['postal_code']}<br><b>{metric}:</b> {row.get(metric, '')}",
                color="blue",
                fill=True,
                fillColor="blue",
                fillOpacity=0.7,
                weight=2
            ).add_to(m)

    # Normalize values for color mapping
    values = gdf[metric].dropna()
    if len(values) > 0:
        vmin, vmax = values.min(), values.max()
        
        for idx, row in gdf.iterrows():
            value = row[metric]
            if pd.notna(value):
                # Color scale: blue (low) to red (high)
                normalized = (value - vmin) / (vmax - vmin)
                color = f'#{int(255*normalized):02x}{int(100*(1-normalized)):02x}{int(255*(1-normalized)):02x}'
                
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=8,
                    popup=f"<b>PLZ:</b> {row['postal_code']}<br><b>{metric}:</b> {value:.2f}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
    
    return m

def create_heatmap(gdf, metric):
    """Create heatmap visualization"""
    m = folium.Map(
        location=BERLIN_CENTER,
        zoom_start=11,
        tiles='CartoDB positron'
    )
    
    # Prepare heat data
    heat_data = []
    for idx, row in gdf.iterrows():
        if pd.notna(row[metric]):
            heat_data.append([row['lat'], row['lon'], float(row[metric])])
    
    if heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_opacity=0.8,
            radius=25,
            blur=20,
            gradient={0.4: 'blue', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
        ).add_to(m)
    
    return m

def create_choropleth_map(gdf, metric):
    """Create choropleth map"""
    m = folium.Map(
        location=BERLIN_CENTER,
        zoom_start=11,
        tiles='CartoDB positron'
    )
    
    values = gdf[metric].dropna()
    if len(values) > 0:
        vmin, vmax = values.min(), values.max()
        
        for idx, row in gdf.iterrows():
            value = row[metric]
            if pd.notna(value):
                normalized = (value - vmin) / (vmax - vmin)
                # Red-Yellow-Green scale (inverted for heat)
                r = int(255 * normalized)
                g = int(255 * (1 - normalized))
                color = f'#{r:02x}{g:02x}50'
                
                folium.GeoJson(
                    row.geometry,
                    style_function=lambda x, col=color: {
                        'fillColor': col,
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.6
                    },
                    tooltip=f"PLZ: {row['postal_code']}<br>{metric}: {value:.2f}"
                ).add_to(m)
    
    return m

def run_regression_analysis(df, target, predictors):
    """Run multiple regression analysis"""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    
    # Prepare data
    X = df[predictors].dropna()
    y = df.loc[X.index, target]
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Create results dataframe
    results = pd.DataFrame({
        'Predictor': predictors,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    return {
        'model': model,
        'results': results,
        'r2': r2,
        'rmse': rmse,
        'intercept': model.intercept_
    }

# Main app
def main():
    st.title("🌡️ Berlin Urban Heat & Environmental Analysis")
    st.markdown("**Multi-source spatial analysis with advanced visualizations**")
    
    # Load data
    with st.spinner("Loading data..."):
        districts = load_berlin_districts()
        postcodes = load_berlin_postcodes()
        env_data = load_environmental_data()
        
        if postcodes is not None and env_data is not None:
            gdf = merge_geo_data(postcodes, env_data)
        else:
            st.error("Failed to load required data")
            return
    
    # Create tabs
    tabs = st.tabs([
        "🗺️ Interactive Maps", 
        "📊 Statistical Analysis",
        "❓ Research Questions",
        "📈 Detailed Statistics"
    ])
    
    # ========== TAB 1: INTERACTIVE MAPS ==========
    with tabs[0]:
        st.header("Interactive Spatial Visualizations")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("⚙️ Settings")
            
            viz_type = st.radio(
                "Visualization Type",
                options=['points', 'heatmap', 'choropleth'],
                format_func=lambda x: {
                    'points': '📍 Point Map',
                    'heatmap': '🔥 Heatmap',
                    'choropleth': '🗾 Choropleth'
                }[x]
            )
            
            metric = st.selectbox(
                "Select Metric",
                options=['delta_lst', 'pollution_index', 'population_density', 
                        'traffic_supply', 'no2_avg', 'pm10_avg', 'population_weighted_lst'],
                format_func=lambda x: {
                    'delta_lst': '🌡️ Temperature Change (ΔLST)',
                    'pollution_index': '💨 Pollution Index',
                    'population_density': '👥 Population Density',
                    'traffic_supply': '🚗 Traffic Supply',
                    'no2_avg': 'NO₂ Average',
                    'pm10_avg': 'PM10 Average',
                    'population_weighted_lst': '👥🌡️ Pop-Weighted ΔLST'
                }[x]
            )
            
            show_districts = st.checkbox("Show District Boundaries", value=False)
            
            # Summary stats
            st.markdown("---")
            st.subheader("📊 Summary")
            values = gdf[metric].dropna()
            if len(values) > 0:
                st.metric("Min", f"{values.min():.2f}")
                st.metric("Mean", f"{values.mean():.2f}")
                st.metric("Max", f"{values.max():.2f}")
                st.metric("Std Dev", f"{values.std():.2f}")
        
        with col2:
            st.subheader(f"Map: {metric}")
            
            # Create map based on visualization type
            if viz_type == 'points':
                m = create_point_map(gdf, metric, f"Point Map: {metric}")
            elif viz_type == 'heatmap':
                m = create_heatmap(gdf, metric)
            else:  # choropleth
                m = create_choropleth_map(gdf, metric)
            
            # Add district boundaries if requested
            if show_districts and districts is not None:
                folium.GeoJson(
                    districts,
                    style_function=lambda x: {
                        'fillColor': 'transparent',
                        'color': '#2c3e50',
                        'weight': 2,
                        'fillOpacity': 0
                    }
                ).add_to(m)
            
            st_folium(m, width=900, height=600)
    
    # ========== TAB 2: STATISTICAL ANALYSIS ==========
    with tabs[1]:
        st.header("Statistical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Correlation Matrix")
            corr_vars = ['delta_lst', 'pollution_index', 'population_density', 'traffic_supply']
            corr_matrix = gdf[corr_vars].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f',
                title="Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect='auto',
                labels=dict(color="Correlation")
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.subheader("📊 Distribution Comparison")
            var_to_plot = st.selectbox(
                "Select variable",
                options=['delta_lst', 'pollution_index', 'population_density', 'traffic_supply'],
                key='dist_var'
            )
            
            fig_hist = px.histogram(
                gdf,
                x=var_to_plot,
                nbins=20,
                title=f"Distribution of {var_to_plot}",
                marginal="box"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 Regression Analysis")
        
        # Run regression
        predictors = ['pollution_index', 'population_density', 'traffic_supply']
        reg_results = run_regression_analysis(gdf, 'delta_lst', predictors)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{reg_results['r2']:.3f}")
        col2.metric("RMSE", f"{reg_results['rmse']:.3f}")
        col3.metric("Intercept", f"{reg_results['intercept']:.3f}")
        
        st.markdown("**Regression Coefficients:**")
        st.dataframe(reg_results['results'], use_container_width=True)
        
        # Scatter plots
        st.subheader("📉 Predictor vs Target Relationships")
        
        cols = st.columns(3)
        for i, pred in enumerate(predictors):
            with cols[i]:
                fig = px.scatter(
                    gdf,
                    x=pred,
                    y='delta_lst',
                    trendline='ols',
                    title=f"ΔLST vs {pred}",
                    labels={'delta_lst': 'ΔLST (°C)'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 3: RESEARCH QUESTIONS ==========
    with tabs[2]:
        st.header("❓ Five Core Research Questions")
        st.markdown("**Urban Heat Drivers in Berlin Using Multi-Source Data**")
        
        # Question 1
        with st.expander("**Question 1: Where does LST increase the most during summer?**", expanded=True):
            st.markdown("""
            **Metric:** ΔLST = LST(August) − LST(May)
            
            **Data Source:** Satellite imagery (Sentinel / Landsat / MODIS)
            
            **Aggregation:** Per postal code (PLZ)
            
            **Why it matters:** Identifies spatial heat hotspots and establishes baseline patterns before attribution
            """)
            
            # Top hotspots
            top_hotspots = gdf.nlargest(10, 'delta_lst')[['postal_code', 'delta_lst', 'lst_may', 'lst_august']]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Top 10 Heat Hotspots:**")
                st.dataframe(top_hotspots.reset_index(drop=True), use_container_width=True)
            
            with col2:
                fig = px.bar(
                    top_hotspots,
                    x='postal_code',
                    y='delta_lst',
                    title='Top 10 Postal Codes by Temperature Change',
                    labels={'delta_lst': 'ΔLST (°C)', 'postal_code': 'Postal Code'},
                    color='delta_lst',
                    color_continuous_scale='RdYlBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Question 2
        with st.expander("**Question 2: How strongly are pollution levels associated with summer heating?**"):
            st.markdown("""
            **Metrics:**
            - Monthly mean NO₂
            - Monthly mean PM10
            - Pollution Index = z(NO₂) + z(PM10)
            
            **Evaluation:** Regression coefficient β₁ in: `ΔLST ~ Pollution_index + controls`
            
            **Why it matters:** Links combustion-related activity to heat amplification
            """)
            
            # Run pollution-focused regression
            pollution_reg = run_regression_analysis(gdf, 'delta_lst', ['pollution_index'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Pollution Coefficient (β₁)", f"{pollution_reg['results'].iloc[0]['Coefficient']:.4f}")
                st.metric("R² Score", f"{pollution_reg['r2']:.3f}")
                st.markdown(f"""
                **Interpretation:** For every 1-unit increase in pollution index, 
                ΔLST changes by {pollution_reg['results'].iloc[0]['Coefficient']:.4f}°C.
                """)
            
            with col2:
                fig = px.scatter(
                    gdf,
                    x='pollution_index',
                    y='delta_lst',
                    trendline='ols',
                    title='Pollution Index vs Temperature Change',
                    labels={'pollution_index': 'Pollution Index', 'delta_lst': 'ΔLST (°C)'},
                    color='no2_avg',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Question 3
        with st.expander("**Question 3: Does population density increase heat exposure?**"):
            st.markdown("""
            **Metrics:**
            - Population density (people / km²)
            - Population-weighted ΔLST
            
            **Evaluation:**
            - β₂ (population density effect)
            - Ranking PLZs by population-weighted ΔLST
            
            **Why it matters:** Distinguishes hot places from places where people are most affected
            """)
            
            # Population analysis
            pop_reg = run_regression_analysis(gdf, 'delta_lst', ['population_density'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Population Density Coefficient (β₂)", f"{pop_reg['results'].iloc[0]['Coefficient']:.6f}")
                
                # Top exposed areas
                top_exposed = gdf.nlargest(10, 'population_weighted_lst')[
                    ['postal_code', 'population_weighted_lst', 'population_density', 'delta_lst']
                ]
                st.markdown("**Most Exposed Areas (Population-Weighted):**")
                st.dataframe(top_exposed.reset_index(drop=True), use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    gdf,
                    x='population_density',
                    y='delta_lst',
                    size='population_weighted_lst',
                    title='Population Density vs Temperature Change',
                    labels={
                        'population_density': 'Population Density (people/km²)',
                        'delta_lst': 'ΔLST (°C)',
                        'population_weighted_lst': 'Pop-Weighted ΔLST'
                    },
                    color='population_weighted_lst',
                    color_continuous_scale='OrRd'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Question 4
        with st.expander("**Question 4: What is the contribution of traffic intensity?**"):
            st.markdown("""
            **Metrics:**
            - Traffic supply index (GTFS-based)
            - Stop density as mobility proxy
            
            **Evaluation:**
            - β₃ (traffic effect) in ΔLST model
            - Comparison: β_traffic vs β_pollution vs β_population
            
            **Why it matters:** Informs traffic management and heat mitigation strategies
            """)
            
            # Multi-factor regression
            multi_reg = run_regression_analysis(
                gdf, 
                'delta_lst', 
                ['pollution_index', 'population_density', 'traffic_supply']
            )
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Effect Size Comparison:**")
                comparison_df = multi_reg['results'].copy()
                comparison_df['Effect_Rank'] = range(1, len(comparison_df) + 1)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Find traffic coefficient
                traffic_coef = multi_reg['results'][
                    multi_reg['results']['Predictor'] == 'traffic_supply'
                ]['Coefficient'].values[0]
                st.metric("Traffic Coefficient (β₃)", f"{traffic_coef:.4f}")
            
            with col2:
                fig = px.bar(
                    multi_reg['results'],
                    x='Predictor',
                    y='Coefficient',
                    title='Relative Effect Sizes (Regression Coefficients)',
                    labels={'Coefficient': 'β Coefficient', 'Predictor': 'Factor'},
                    color='Abs_Coefficient',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Question 5
        with st.expander("**Question 5: Do modeled heat changes match observations?**"):
            st.markdown("""
            **Validation Metrics:**
            - ΔLST_sat = LST_Aug − LST_May (observed)
            - ΔLST_model = ŷ_Aug − ŷ_May (predicted)
            
            **Validation Methods:**
            - Spatial correlation (Pearson r ≥ 0.6)
            - RMSE
            - Overlap of hottest PLZs (top 20%)
            
            **Supporting Evidence:**
            - RGB change maps (urban surface change)
            - NDVI loss (reduced cooling capacity)
            """)
            
            # Model predictions
            predictors = ['pollution_index', 'population_density', 'traffic_supply']
            X = gdf[predictors].dropna()
            y_true = gdf.loc[X.index, 'delta_lst']
            
            model = run_regression_analysis(gdf, 'delta_lst', predictors)
            y_pred = model['model'].predict(X)
            
            # Calculate validation metrics
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            
            # Top 20% overlap
            top_20_pct = int(len(gdf) * 0.2)
            top_observed = set(gdf.nlargest(top_20_pct, 'delta_lst')['postal_code'])
            
            gdf_temp = gdf.loc[X.index].copy()
            gdf_temp['predicted_lst'] = y_pred
            top_predicted = set(gdf_temp.nlargest(top_20_pct, 'predicted_lst')['postal_code'])
            overlap = len(top_observed & top_predicted) / top_20_pct * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Spatial Correlation (r)", f"{correlation:.3f}", 
                         delta="✓ Good" if correlation >= 0.6 else "⚠ Needs improvement")
                st.metric("RMSE", f"{rmse:.3f}°C")
                st.metric("Top 20% Overlap", f"{overlap:.1f}%")
            
            with col2:
                fig = px.scatter(
                    x=y_true,
                    y=y_pred,
                    title='Observed vs Predicted ΔLST',
                    labels={'x': 'Observed ΔLST (°C)', 'y': 'Predicted ΔLST (°C)'},
                    trendline='ols'
                )
                fig.add_shape(
                    type='line',
                    x0=y_true.min(), y0=y_true.min(),
                    x1=y_true.max(), y1=y_true.max(),
                    line=dict(color='red', dash='dash'),
                    name='Perfect prediction'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 4: DETAILED STATISTICS ==========
    with tabs[3]:
        st.header("📈 Detailed Statistics")
        
        st.subheader("Complete Dataset")
        st.dataframe(gdf.drop(columns=['geometry', 'centroid']), use_container_width=True)
        
        st.subheader("Descriptive Statistics")
        numeric_cols = ['delta_lst', 'pollution_index', 'population_density', 
                       'traffic_supply', 'no2_avg', 'pm10_avg', 'population_weighted_lst']
        st.dataframe(gdf[numeric_cols].describe(), use_container_width=True)
        
        # Export options
        st.subheader("📥 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = gdf.drop(columns=['geometry', 'centroid']).to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"berlin_heat_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = gdf.drop(columns=['geometry', 'centroid']).to_json(orient='records', indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"berlin_heat_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **📊 Data Sources:**
    - District boundaries: Berlin Open Data (funkeinteraktiv)
    - Postal codes: TSB Open Data
    - LST: Sentinel/Landsat satellite imagery (simulated)
    - Pollution: luftdaten.berlin.de (simulated)
    - Population: Census data (simulated)
    - Traffic: GTFS VBB data (simulated)
    
    *Note: Current data is simulated for demonstration. Replace with actual data sources.*
    """)

if __name__ == "__main__":
    main()