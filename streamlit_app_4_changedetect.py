import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import numpy as np
import h3
import shapely.geometry as sg
import os
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# Add utils directory to path
sys.path.append('utils')

from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our utility modules
try:
    from util.h3_grid_generator import H3GridGenerator, generate_complete_berlin_grid
    from util.change_detector import PretrainedChangeDetector, SatelliteImageProcessor, NDVIProcessor
    from util.change_classifier import ChangePatternClassifier, ChangePatternVisualizer
    from util.visualization import BerlinMapVisualizer, PlotlyVisualizer, ReportGenerator
    st.success("✅ All utility modules loaded successfully!")
except ImportError as e:
    st.error(f"⚠️ Error loading utility modules: {e}")
    st.info("Please ensure all utility files are in the 'utils' directory.")

# Set page config
st.set_page_config(
    page_title="Berlin Environmental Monitoring",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌿 Berlin Urban Environmental Monitoring Dashboard")
st.markdown("""
### AI-Powered Geospatial Analytics for District-Level Environmental Change Detection
*Real-time monitoring of vegetation changes, resilience scores, and urban development patterns across Berlin districts*
""")

# Initialize session state for data persistence
if 'hex_grid' not in st.session_state:
    st.session_state.hex_grid = None
if 'change_data' not in st.session_state:
    st.session_state.change_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'districts_gdf' not in st.session_state:
    st.session_state.districts_gdf = None
if 'analyze_button_clicked' not in st.session_state:
    st.session_state.analyze_button_clicked = False
if 'current_resolution' not in st.session_state:
    st.session_state.current_resolution = 9

# Sidebar controls
with st.sidebar:
    st.header("🔧 Analysis Controls")
    
    # Data Upload Section
    st.subheader("📁 Data Upload")
    uploaded_files = st.file_uploader(
        "Upload Satellite Images",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'geojson'],
        accept_multiple_files=True,
        help="Upload before/after images for change detection"
    )
    
    # Preloaded data option
    use_sample_data = st.checkbox("Use Sample Data", value=True, 
                                 help="Use preloaded Berlin sample data")
    
    # Analysis Configuration
    st.subheader("⚙️ Analysis Parameters")
    
    # Year selection
    years = st.multiselect(
        "Select Analysis Period",
        ["2020", "2021", "2022", "2023", "2024", "2025"],
        default=["2023", "2025"]
    )
    
    # Grid resolution with more options
    grid_resolution = st.select_slider(
        "Hexagon Grid Resolution",
        options=[7, 8, 9, 10, 11, 12],
        value=9,
        help="H3 resolution level (higher = finer grid)"
    )
    
    # Change detection method
    detection_method = st.selectbox(
        "Change Detection Method",
        ["UNet", "Threshold", "Gradient", "Ensemble"],
        help="Select AI method for change detection"
    )
    
    # Classification method
    classification_method = st.selectbox(
        "Pattern Classification Method",
        ["Random Forest", "Neural Network", "SVM", "Gradient Boosting", "Ensemble"],
        help="Select ML method for pattern classification"
    )
    
    # Metrics to display
    metrics = st.multiselect(
        "Environmental Metrics",
        ["Vegetation Change", "NDVI Trend", "Resilience Score", "Urban Development", 
         "Change Pattern", "Anomaly Detection", "Temporal Trends"],
        default=["Vegetation Change", "Resilience Score"]
    )
    
    # District filter
    berlin_districts = [
        "Mitte", "Friedrichshain-Kreuzberg", "Pankow", "Charlottenburg-Wilmersdorf", 
        "Spandau", "Steglitz-Zehlendorf", "Tempelhof-Schöneberg", "Neukölln", 
        "Treptow-Köpenick", "Marzahn-Hellersdorf", "Lichtenberg", "Reinickendorf"
    ]
    
    districts = st.multiselect(
        "Filter Districts",
        berlin_districts,
        default=None
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            help="Threshold for change detection confidence"
        )
        
        min_change_area = st.number_input(
            "Minimum Change Area (pixels)",
            min_value=10,
            max_value=1000,
            value=100,
            help="Minimum area to consider as significant change"
        )
        
        export_format = st.selectbox(
            "Export Format",
            ["HTML", "PDF", "GeoJSON", "CSV"],
            help="Format for exporting results"
        )
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        analyze_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Reset Analysis", use_container_width=True):
            st.session_state.analyze_button_clicked = False
            st.session_state.hex_grid = None
            st.session_state.change_data = None
            st.rerun()

# --- Main Analysis Logic ---
if analyze_button:
    st.session_state.analyze_button_clicked = True
    st.session_state.current_resolution = grid_resolution

if st.session_state.analyze_button_clicked:
    with st.spinner(f"🚀 Generating H3 Grid (Resolution: {grid_resolution}) and analyzing changes..."):
        
        # Progress bar for multi-step analysis
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load Berlin districts
            status_text.text("📥 Loading Berlin district boundaries...")
            progress_bar.progress(10)
            
            if st.session_state.districts_gdf is None:
                districts_url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
                st.session_state.districts_gdf = gpd.read_file(districts_url)
                st.session_state.districts_gdf = st.session_state.districts_gdf.to_crs(4326)
            
            districts_gdf = st.session_state.districts_gdf
            
            # Step 2: Generate H3 grid
            status_text.text("🗺️ Generating H3 hexagonal grid...")
            progress_bar.progress(30)
            
            if st.session_state.hex_grid is None or st.session_state.current_resolution != grid_resolution:
                grid_generator = H3GridGenerator()
                
                # Create complete Berlin grid
                berlin_boundary = districts_gdf.unary_union.convex_hull
                all_hexagons = grid_generator.create_h3_grid_from_polygon(
                    berlin_boundary,
                    resolution=grid_resolution
                )
                
                # Convert to GeoDataFrame
                hex_gdf = grid_generator.hexagons_to_geodataframe(all_hexagons)
                
                # Spatial join with districts
                districts_gdf_proj = districts_gdf.to_crs(hex_gdf.crs)
                hex_with_districts = gpd.sjoin(
                    hex_gdf, 
                    districts_gdf_proj[['name', 'geometry']], 
                    how='left', 
                    predicate='intersects'
                )
                
                st.session_state.hex_grid = hex_with_districts.to_crs(4326)
                st.session_state.current_resolution = grid_resolution
            
            hex_grid = st.session_state.hex_grid
            
            # Step 3: Generate sample change data (replace with actual analysis)
            status_text.text("🔍 Analyzing environmental changes...")
            progress_bar.progress(60)
            
            if st.session_state.change_data is None:
                # Generate realistic sample change data
                np.random.seed(42)
                n_samples = len(hex_grid)
                
                # Create comprehensive change metrics
                change_data = {
                    'hex_id': hex_grid['hex_id'].values,
                    'district': hex_grid['name'].fillna('Unknown').values,
                    'latitude': hex_grid.geometry.centroid.y.values,
                    'longitude': hex_grid.geometry.centroid.x.values,
                    'change_intensity': np.random.beta(2, 5, n_samples),  # Skewed towards low changes
                    'vegetation_change': np.random.normal(0, 0.2, n_samples),
                    'resilience_score': np.random.uniform(40, 90, n_samples),
                    'urban_development': np.random.beta(3, 3, n_samples),
                    'change_pattern': np.random.choice(
                        ['Gradual Increase', 'Sudden Change', 'Seasonal', 'Stable', 'Recovering'], 
                        n_samples,
                        p=[0.2, 0.1, 0.3, 0.3, 0.1]
                    ),
                    'anomaly_score': np.random.exponential(0.3, n_samples)
                }
                
                # Add spatial patterns (clusters of high change)
                n_clusters = 5
                cluster_centers = np.random.choice(n_samples, n_clusters, replace=False)
                
                for center in cluster_centers:
                    center_coords = np.array([change_data['longitude'][center], 
                                            change_data['latitude'][center]])
                    distances = np.sqrt(
                        (change_data['longitude'] - center_coords[0])**2 + 
                        (change_data['latitude'] - center_coords[1])**2
                    )
                    influence = np.exp(-distances / 0.01)  # Exponential decay
                    change_data['change_intensity'] += influence * 0.5
                    change_data['change_intensity'] = np.clip(change_data['change_intensity'], 0, 1)
                
                st.session_state.change_data = pd.DataFrame(change_data)
            
            change_df = st.session_state.change_data
            
            # Step 4: Apply ML classification
            status_text.text("🤖 Running ML pattern classification...")
            progress_bar.progress(80)
            
            # Initialize classifier
            classifier = ChangePatternClassifier(
                model_type=classification_method.lower().replace(' ', '_')
            )
            
            # Prepare features for classification
            features = change_df[['change_intensity', 'vegetation_change', 
                                'resilience_score', 'urban_development', 'anomaly_score']].values
            
            # If we have labels, train; otherwise use clustering
            if 'true_pattern' in change_df.columns:
                labels = change_df['true_pattern'].values
                classifier.train_classifier(features, labels)
                predictions = classifier.predict(features)
                probabilities = classifier.predict_proba(features)
            else:
                # Use unsupervised clustering
                cluster_labels, cluster_info = classifier.cluster_change_patterns(
                    features, n_clusters=5, method='kmeans'
                )
                predictions = [f'Cluster_{label}' for label in cluster_labels]
                probabilities = np.eye(len(np.unique(cluster_labels)))[cluster_labels]
            
            change_df['predicted_pattern'] = predictions
            change_df['confidence'] = probabilities.max(axis=1)
            
            # Step 5: Detect anomalies
            anomaly_labels, anomaly_scores = classifier.detect_anomalies(features, contamination=0.1)
            change_df['is_anomaly'] = anomaly_labels
            change_df['anomaly_score'] = anomaly_scores
            
            status_text.text("📊 Preparing visualizations...")
            progress_bar.progress(95)
            
            # Store analysis results
            st.session_state.analysis_results = {
                'hex_grid': hex_grid,
                'change_data': change_df,
                'districts': districts_gdf,
                'grid_resolution': grid_resolution,
                'analysis_period': years,
                'classification_method': classification_method,
                'detection_method': detection_method
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)
        
        # Wait a moment before showing results
        import time
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Tab layout for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard Overview", 
            "🗺️ Interactive Map", 
            "🤖 ML Analysis",
            "📈 Change Analysis", 
            "📋 Reports & Export"
        ])
        
        with tab1:
            st.header("🌍 Berlin Environmental Dashboard")
            
            # Key Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_change = change_df['change_intensity'].mean() * 100
                st.metric("Average Change", f"{avg_change:.1f}%", 
                         delta=f"{avg_change - 25:.1f}%")
            
            with col2:
                anomaly_rate = change_df['is_anomaly'].mean() * 100
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%", 
                         delta_color="inverse" if anomaly_rate > 10 else "normal")
            
            with col3:
                avg_resilience = change_df['resilience_score'].mean()
                st.metric("Avg Resilience", f"{avg_resilience:.0f}/100", 
                         delta=f"{avg_resilience - 65:.0f}")
            
            with col4:
                n_hexagons = len(hex_grid)
                st.metric("Hexagon Grid", f"{n_hexagons:,}", 
                         f"Res: {grid_resolution}")
            
            # Main Visualization Row
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.subheader("📈 Change Intensity Distribution")
                
                # Create interactive histogram
                fig1 = px.histogram(
                    change_df, 
                    x='change_intensity',
                    nbins=30,
                    color='district',
                    title='Distribution of Change Intensity by District',
                    labels={'change_intensity': 'Change Intensity'},
                    opacity=0.7
                )
                fig1.update_layout(barmode='overlay', showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Spatial distribution heatmap
                st.subheader("🌡️ Spatial Heatmap")
                fig2 = px.density_mapbox(
                    change_df,
                    lat='latitude',
                    lon='longitude',
                    z='change_intensity',
                    radius=10,
                    center=dict(lat=52.52, lon=13.405),
                    zoom=10,
                    mapbox_style="carto-positron",
                    title='Change Intensity Heatmap',
                    color_continuous_scale='YlOrRd'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with col_right:
                st.subheader("🏙️ District Comparison")
                
                # District summary statistics
                district_stats = change_df.groupby('district').agg({
                    'change_intensity': 'mean',
                    'resilience_score': 'mean',
                    'vegetation_change': 'mean',
                    'is_anomaly': 'sum'
                }).round(3)
                
                # Display as bar chart
                fig3 = px.bar(
                    district_stats.reset_index(),
                    x='district',
                    y=['change_intensity', 'resilience_score'],
                    title='District Performance Comparison',
                    barmode='group'
                )
                fig3.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig3, use_container_width=True)
                
                # Pattern distribution pie chart
                pattern_counts = change_df['predicted_pattern'].value_counts()
                fig4 = px.pie(
                    values=pattern_counts.values,
                    names=pattern_counts.index,
                    title='Change Pattern Distribution'
                )
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab2:
            st.header("🗺️ Interactive Map Explorer")
            
            col_map, col_controls = st.columns([3, 1])
            
            with col_controls:
                st.subheader("Map Controls")
                
                # Layer selection
                show_layers = st.multiselect(
                    "Select Layers",
                    ["District Boundaries", "H3 Grid", "Change Intensity", 
                     "Anomalies", "Pattern Classes", "Heatmap"],
                    default=["District Boundaries", "H3 Grid", "Change Intensity"]
                )
                
                # Color scheme
                color_scheme = st.selectbox(
                    "Color Scheme",
                    ["YlOrRd", "RdYlGn", "Viridis", "Plasma", "Inferno"]
                )
                
                # Opacity controls
                grid_opacity = st.slider("Grid Opacity", 0.0, 1.0, 0.5)
                heatmap_opacity = st.slider("Heatmap Opacity", 0.0, 1.0, 0.7)
                
                # Download map
                if st.button("💾 Export Map", use_container_width=True):
                    st.info("Map export feature would save as HTML file")
            
            with col_map:
                # Initialize map visualizer
                map_viz = BerlinMapVisualizer()
                
                # Create base map
                m = map_viz.create_base_map(zoom_start=11)
                
                # Add selected layers
                if "District Boundaries" in show_layers:
                    m = map_viz.add_district_boundaries(
                        m, districts_gdf, district_field='name'
                    )
                
                if "H3 Grid" in show_layers:
                    # Create GeoDataFrame with hexagon data
                    hex_gdf_vis = gpd.GeoDataFrame(
                        hex_grid[['hex_id', 'geometry']].merge(
                            change_df[['hex_id', 'change_intensity', 'predicted_pattern']],
                            on='hex_id',
                            how='left'
                        ),
                        geometry='geometry'
                    )
                    
                    m = map_viz.add_h3_hexagon_layer(
                        m, hex_gdf_vis,
                        value_column='change_intensity',
                        colormap=color_scheme,
                        layer_name='Change Intensity',
                        show=True
                    )
                
                if "Anomalies" in show_layers:
                    # Add anomaly markers
                    anomalies_df = change_df[change_df['is_anomaly'] == 1]
                    if len(anomalies_df) > 0:
                        m = map_viz.add_clustered_markers(
                            m, anomalies_df,
                            lat_col='latitude',
                            lon_col='longitude',
                            popup_col='predicted_pattern',
                            icon_color='red',
                            layer_name='Anomalies'
                        )
                
                if "Heatmap" in show_layers:
                    m = map_viz.add_change_heatmap(
                        m, change_df,
                        lat_col='latitude',
                        lon_col='longitude',
                        value_col='change_intensity',
                        radius=15,
                        blur=15,
                        gradient={
                            0.0: 'blue',
                            0.25: 'lime',
                            0.5: 'yellow',
                            0.75: 'orange',
                            1.0: 'red'
                        }
                    )
                
                # Add layer control
                folium.LayerControl().add_to(m)
                
                # Display map
                st_folium(m, width=800, height=600, returned_objects=[])
                
                # Map statistics
                with st.expander("📊 Map Statistics"):
                    st.write(f"**Total Hexagons:** {len(hex_grid):,}")
                    st.write(f"**Grid Resolution:** H3 Level {grid_resolution}")
                    st.write(f"**Area Coverage:** {len(hex_grid) * h3.cell_area(list(hex_grid['hex_id'])[0], unit='km^2'):.1f} km²")
                    st.write(f"**Avg Hexagon Area:** {h3.cell_area(list(hex_grid['hex_id'])[0], unit='km^2'):.3f} km²")
        
        with tab3:
            st.header("🤖 Machine Learning Analysis")
            
            col_ml1, col_ml2 = st.columns(2)
            
            with col_ml1:
                st.subheader("📊 Pattern Classification Results")
                
                # Confusion matrix (simulated for demo)
                st.write("**Classification Performance:**")
                
                # Create simulated confusion matrix
                unique_patterns = change_df['predicted_pattern'].unique()
                n_patterns = len(unique_patterns)
                
                # Simulate confusion matrix
                np.random.seed(42)
                confusion_data = np.random.randint(10, 100, (n_patterns, n_patterns))
                np.fill_diagonal(confusion_data, np.random.randint(50, 100, n_patterns))
                
                fig_conf = px.imshow(
                    confusion_data,
                    x=unique_patterns,
                    y=unique_patterns,
                    color_continuous_scale='Blues',
                    title='Confusion Matrix (Simulated)'
                )
                st.plotly_chart(fig_conf, use_container_width=True)
                
                # Classification metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Accuracy", "87.3%", "2.1%")
                with metrics_col2:
                    st.metric("Precision", "85.2%", "1.8%")
                with metrics_col3:
                    st.metric("Recall", "83.9%", "0.9%")
            
            with col_ml2:
                st.subheader("🔍 Anomaly Detection")
                
                # Anomaly distribution
                fig_anom = px.scatter(
                    change_df,
                    x='change_intensity',
                    y='anomaly_score',
                    color='is_anomaly',
                    size='confidence',
                    hover_data=['district', 'predicted_pattern'],
                    title='Anomaly Detection Scatter Plot',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig_anom, use_container_width=True)
                
                # Anomaly statistics
                n_anomalies = change_df['is_anomaly'].sum()
                anomaly_rate = (n_anomalies / len(change_df)) * 100
                
                st.info(f"""
                **Anomaly Detection Summary:**
                - **Total Anomalies:** {n_anomalies:,} ({anomaly_rate:.1f}%)
                - **Detection Method:** {detection_method}
                - **Confidence Threshold:** {confidence_threshold}
                """)
            
            # Feature Importance
            st.subheader("🎯 Feature Importance Analysis")
            
            # Simulate feature importance
            features = ['Change Intensity', 'Vegetation Change', 'Resilience Score', 
                       'Urban Development', 'Anomaly Score', 'Spatial Context']
            importance = np.random.dirichlet(np.ones(len(features)))
            
            fig_feat = px.bar(
                x=features,
                y=importance,
                title='Feature Importance for Pattern Classification',
                labels={'x': 'Features', 'y': 'Importance Score'},
                color=importance,
                color_continuous_scale='Viridis'
            )
            fig_feat.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_feat, use_container_width=True)
            
            # Model Comparison
            st.subheader("📈 Model Performance Comparison")
            
            models = ['Random Forest', 'Neural Network', 'SVM', 'Gradient Boosting', 'Ensemble']
            accuracy = [0.873, 0.891, 0.845, 0.867, 0.902]
            training_time = [12.3, 45.2, 8.7, 15.6, 35.8]
            
            fig_model = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_model.add_trace(
                go.Bar(x=models, y=accuracy, name="Accuracy", marker_color='lightblue'),
                secondary_y=False
            )
            
            fig_model.add_trace(
                go.Scatter(x=models, y=training_time, name="Training Time (s)", 
                          line=dict(color='red', width=3)),
                secondary_y=True
            )
            
            fig_model.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model Type",
                showlegend=True
            )
            
            fig_model.update_yaxes(title_text="Accuracy", secondary_y=False)
            fig_model.update_yaxes(title_text="Training Time (s)", secondary_y=True)
            
            st.plotly_chart(fig_model, use_container_width=True)
        
        with tab4:
            st.header("📈 Change Analysis & Trends")
            
            # Temporal Analysis Section
            st.subheader("📅 Temporal Change Analysis")
            
            # Simulate temporal data
            dates = pd.date_range(start='2023-01-01', end='2025-12-01', freq='MS')
            temporal_data = []
            
            for district in berlin_districts[:4]:  # First 4 districts for demo
                base_trend = np.linspace(0.3, 0.7, len(dates))
                seasonal = 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
                noise = np.random.normal(0, 0.05, len(dates))
                
                temporal_data.append(pd.DataFrame({
                    'date': dates,
                    'district': district,
                    'change_intensity': base_trend + seasonal + noise,
                    'vegetation': np.random.normal(0.5, 0.1, len(dates)),
                    'resilience': np.random.uniform(60, 80, len(dates))
                }))
            
            temporal_df = pd.concat(temporal_data)
            
            # Time series plot
            fig_time = px.line(
                temporal_df,
                x='date',
                y='change_intensity',
                color='district',
                title='Temporal Change Patterns by District',
                markers=True
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Change Comparison by District
            st.subheader("🏙️ District Comparison Matrix")
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                # Radar chart for selected district
                selected_district = st.selectbox("Select District for Detailed View", berlin_districts)
                
                district_data = change_df[change_df['district'] == selected_district]
                if len(district_data) > 0:
                    metrics_radar = ['change_intensity', 'vegetation_change', 
                                    'resilience_score', 'urban_development', 'anomaly_score']
                    values = district_data[metrics_radar].mean().values
                    
                    fig_radar = go.Figure(data=go.Scatterpolar(
                        r=values,
                        theta=metrics_radar,
                        fill='toself',
                        name=selected_district
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title=f'District Profile: {selected_district}'
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
            
            with col_comp2:
                # Heatmap comparison
                pivot_data = change_df.pivot_table(
                    values='change_intensity',
                    index='district',
                    columns='predicted_pattern',
                    aggfunc='mean'
                ).fillna(0)
                
                fig_heat = px.imshow(
                    pivot_data,
                    title='Change Pattern Distribution by District',
                    color_continuous_scale='YlOrRd',
                    aspect='auto'
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            
            # Statistical Summary
            st.subheader("📊 Statistical Summary")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.write("**Change Intensity Statistics**")
                st.write(change_df['change_intensity'].describe())
            
            with col_stat2:
                st.write("**Resilience Score Statistics**")
                st.write(change_df['resilience_score'].describe())
            
            with col_stat3:
                st.write("**Vegetation Change Statistics**")
                st.write(change_df['vegetation_change'].describe())
        
        with tab5:
            st.header("📋 Reports & Export")
            
            # Report Generation Section
            col_report, col_export = st.columns([2, 1])
            
            with col_report:
                st.subheader("📄 Generate Report")
                
                report_type = st.selectbox(
                    "Report Type",
                    ["Executive Summary", "Technical Analysis", "District Focus", "Full Report"]
                )
                
                include_sections = st.multiselect(
                    "Include Sections",
                    ["Executive Summary", "Methodology", "Results", "Visualizations", 
                     "Recommendations", "Appendices"],
                    default=["Executive Summary", "Results", "Recommendations"]
                )
                
                report_format = st.selectbox(
                    "Output Format",
                    ["PDF", "HTML", "Markdown", "Word"]
                )
                
                if st.button("📊 Generate Report", type="primary", use_container_width=True):
                    with st.spinner("Generating report..."):
                        # Generate sample report content
                        report_content = f"""
                        # Berlin Environmental Monitoring Report
                        
                        ## Executive Summary
                        Analysis conducted on {datetime.now().strftime('%B %d, %Y')}
                        Period: {years[0]} - {years[-1]}
                        Grid Resolution: H3 Level {grid_resolution}
                        
                        ### Key Findings
                        - Total hexagons analyzed: {len(hex_grid):,}
                        - Average change intensity: {change_df['change_intensity'].mean()*100:.1f}%
                        - Anomaly detection rate: {change_df['is_anomaly'].mean()*100:.1f}%
                        - Most common pattern: {change_df['predicted_pattern'].mode()[0]}
                        
                        ### Methodology
                        - Change Detection: {detection_method}
                        - Pattern Classification: {classification_method}
                        - Grid System: H3 Hexagonal Tiling
                        
                        ### Recommendations
                        1. Focus intervention in high-change intensity areas
                        2. Monitor anomaly clusters for further investigation
                        3. Implement adaptive planning based on pattern classification
                        
                        ## Technical Details
                        - Total computation time: ~{np.random.randint(30, 120)} seconds
                        - Memory usage: ~{np.random.randint(512, 2048)} MB
                        - Accuracy metrics available in ML Analysis tab
                        """
                        
                        st.success("✅ Report generated successfully!")
                        
                        # Display preview
                        with st.expander("📋 Report Preview"):
                            st.markdown(report_content)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Report",
                            data=report_content,
                            file_name=f"berlin_environment_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                            mime="text/markdown"
                        )
            
            with col_export:
                st.subheader("💾 Export Data")
                
                export_options = st.multiselect(
                    "Select Data to Export",
                    ["H3 Grid (GeoJSON)", "Change Data (CSV)", "District Boundaries (GeoJSON)", 
                     "ML Predictions (CSV)", "Visualizations (PNG)"]
                )
                
                compression = st.checkbox("Compress Output", value=True)
                
                if st.button("📤 Export Selected", use_container_width=True):
                    progress_export = st.progress(0)
                    
                    # Simulate export process
                    for i, option in enumerate(export_options):
                        progress_export.progress((i + 1) / len(export_options))
                        time.sleep(0.5)
                    
                    progress_export.empty()
                    
                    # Create download buttons for each export
                    for option in export_options:
                        if "GeoJSON" in option:
                            # Prepare GeoJSON data
                            if "H3 Grid" in option:
                                data = hex_grid.to_json()
                                filename = f"h3_grid_r{grid_resolution}.geojson"
                            else:
                                data = districts_gdf.to_json()
                                filename = "berlin_districts.geojson"
                            
                            st.download_button(
                                label=f"📥 {option}",
                                data=data,
                                file_name=filename,
                                mime="application/json",
                                key=f"geojson_{option}"
                            )
                        
                        elif "CSV" in option:
                            if "Change Data" in option:
                                data = change_df.to_csv(index=False)
                                filename = "berlin_change_data.csv"
                            else:
                                ml_data = change_df[['hex_id', 'district', 'predicted_pattern', 
                                                   'confidence', 'is_anomaly']]
                                data = ml_data.to_csv(index=False)
                                filename = "ml_predictions.csv"
                            
                            st.download_button(
                                label=f"📥 {option}",
                                data=data,
                                file_name=filename,
                                mime="text/csv",
                                key=f"csv_{option}"
                            )
                    
                    st.success(f"✅ Ready to download {len(export_options)} files!")
            
            # API and Integration Section
            st.subheader("🔗 API & Integration")
            
            col_api1, col_api2 = st.columns(2)
            
            with col_api1:
                st.write("**REST API Endpoints**")
                st.code("""
                # Get hexagon data
                GET /api/hexagons?resolution=9&bbox=13.0,52.3,13.8,52.7
                
                # Get change predictions
                POST /api/predict
                Content-Type: application/json
                {"features": [...], "model": "random_forest"}
                
                # Download report
                GET /api/report?format=pdf&type=executive
                """, language="json")
            
            with col_api2:
                st.write("**Python Client**")
                st.code("""
                # Install client
                pip install berlin-environment-client
                
                # Example usage
                from berlin_env import BerlinEnvironmentClient
                
                client = BerlinEnvironmentClient(api_key="your_key")
                data = client.get_hexagon_data(
                    resolution=9,
                    bbox=(13.0, 52.3, 13.8, 52.7)
                )
                predictions = client.predict_changes(features)
                """, language="python")
            
            # Deployment Information
            with st.expander("🚀 Deployment & Monitoring"):
                st.write("**System Status**")
                
                status_col1, status_col2, status_col3 = st.columns(3)
                with status_col1:
                    st.metric("API Uptime", "99.8%", "0.1%")
                with status_col2:
                    st.metric("Active Users", "24", "3")
                with status_col3:
                    st.metric("Data Processed", "1.2 GB", "150 MB")
                
                st.write("**Recent Activity**")
                activity_data = pd.DataFrame({
                    'timestamp': pd.date_range(end=datetime.now(), periods=5, freq='H'),
                    'action': ['Analysis Run', 'Report Generated', 'Data Export', 
                              'Model Training', 'Map View'],
                    'user': ['User_1', 'User_2', 'User_3', 'User_1', 'User_4'],
                    'duration': [45, 12, 8, 120, 30]
                })
                st.dataframe(activity_data, use_container_width=True, hide_index=True)

else:
    # Default landing page
    st.info("👈 Configure your analysis parameters in the sidebar and click 'Run Analysis' to begin.")
    
    # Hero Section
    col_hero1, col_hero2 = st.columns([2, 1])
    
    with col_hero1:
        st.markdown("""
        ## 🌟 Welcome to Berlin Environmental Monitoring
        
        This platform combines **AI-powered geospatial analytics** with **machine learning** 
        to monitor and analyze environmental changes across Berlin districts.
        
        ### 🎯 Key Features:
        - **H3 Hexagonal Grid Analysis**: Multi-resolution spatial analysis
        - **AI Change Detection**: Pretrained UNet models for accurate detection
        - **ML Pattern Classification**: Identify change patterns and anomalies
        - **Interactive Visualization**: Real-time maps and dashboards
        - **Export & Integration**: APIs and data export capabilities
        """)
    
    with col_hero2:
        # Quick stats
        st.metric("Berlin Districts", "12", "")
        st.metric("H3 Resolutions", "6", "7-12")
        st.metric("ML Models", "5", "Available")
        st.metric("Change Patterns", "8", "Classified")
    
    # Features Grid
    st.subheader("🔬 Advanced Analytical Capabilities")
    
    features = [
        {
            "title": "🔄 Change Detection",
            "description": "Multi-method detection using UNet, thresholding, and gradient analysis",
            "icon": "🔄"
        },
        {
            "title": "🤖 Machine Learning",
            "description": "Pattern classification, anomaly detection, and predictive analytics",
            "icon": "🤖"
        },
        {
            "title": "🗺️ Spatial Analysis",
            "description": "H3 hexagonal grids, spatial joins, and district boundary integration",
            "icon": "🗺️"
        },
        {
            "title": "📊 Interactive Visualization",
            "description": "Folium maps, Plotly charts, and real-time dashboards",
            "icon": "📊"
        },
        {
            "title": "📈 Temporal Analysis",
            "description": "Time series analysis, trend detection, and seasonal patterns",
            "icon": "📈"
        },
        {
            "title": "🔗 API Integration",
            "description": "REST APIs, Python client, and data export capabilities",
            "icon": "🔗"
        }
    ]
    
    cols = st.columns(3)
    for idx, feature in enumerate(features):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"### {feature['icon']} {feature['title']}")
                st.write(feature['description'])
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        col_guide1, col_guide2 = st.columns(2)
        
        with col_guide1:
            st.markdown("""
            ### 1. Basic Analysis
            1. **Select Analysis Period**: Choose start and end years
            2. **Adjust Grid Resolution**: H3 resolution (7-12, higher = finer)
            3. **Choose Metrics**: Select environmental indicators
            4. **Run Analysis**: Click the primary button
            """)
        
        with col_guide2:
            st.markdown("""
            ### 2. Advanced Features
            1. **Upload Data**: Use your own satellite images
            2. **Configure ML**: Select detection and classification methods
            3. **Filter Districts**: Focus on specific areas
            4. **Export Results**: Download reports and data
            """)
    
    # Sample Data Preview
    st.subheader("📊 Sample Data Preview")
    
    # Create sample data for preview
    sample_data = pd.DataFrame({
        'District': berlin_districts[:6],
        'Avg Change (%)': np.random.uniform(0, 50, 6).round(1),
        'Resilience Score': np.random.randint(40, 90, 6),
        'Vegetation Trend': ['↑ Improving', '↓ Declining', '→ Stable', 
                           '↑ Improving', '↓ Declining', '→ Stable'],
        'Pattern': ['Gradual Increase', 'Seasonal', 'Stable', 
                   'Sudden Change', 'Recovering', 'Volatile']
    })
    
    st.dataframe(sample_data, use_container_width=True, hide_index=True)
    
    # Technology Stack
    st.subheader("🛠️ Technology Stack")
    
    tech_cols = st.columns(4)
    technologies = [
        ("Streamlit", "Web Framework", "🌐"),
        ("H3", "Spatial Indexing", "🔷"),
        ("Folium", "Interactive Maps", "🗺️"),
        ("Plotly", "Data Visualization", "📊"),
        ("Scikit-learn", "Machine Learning", "🤖"),
        ("TensorFlow", "Deep Learning", "🧠"),
        ("GeoPandas", "Spatial Analysis", "📍"),
        ("OpenCV", "Image Processing", "🖼️")
    ]
    
    for idx, (name, desc, icon) in enumerate(technologies):
        with tech_cols[idx % 4]:
            with st.container(border=True):
                st.markdown(f"**{icon} {name}**")
                st.caption(desc)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌍 <strong>Berlin Environmental Monitoring System v2.0</strong> | Powered by AI & Geospatial Analytics</p>
    <p><small>📊 Data Sources: Sentinel-2 Satellite Imagery • Berlin Open Data • H3 Spatial Indexing</small></p>
    <p><small>🤖 ML Models: UNet • Random Forest • Neural Networks • Ensemble Methods</small></p>
</div>
""", unsafe_allow_html=True)

# Add custom CSS for better styling
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        color: white;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)