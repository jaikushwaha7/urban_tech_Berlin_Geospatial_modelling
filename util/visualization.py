# utils/visualization.py
"""
Map Visualization Utilities for Berlin Environmental Monitoring
Interactive maps, heatmaps, and spatial visualizations
"""

import folium
from folium import plugins
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import branca.colormap as cmap
from typing import Dict, List, Tuple, Optional, Union
import json
import warnings
warnings.filterwarnings('ignore')

class BerlinMapVisualizer:
    """
    Interactive map visualization for Berlin environmental data
    """
    
    def __init__(self, berlin_center: Tuple[float, float] = (52.52, 13.405)):
        """
        Initialize map visualizer
        
        Args:
            berlin_center: Center coordinates for Berlin
        """
        self.berlin_center = berlin_center
        self.default_tiles = 'CartoDB positron'
        self.district_colors = self._get_district_colors()
        
    def _get_district_colors(self) -> Dict[str, str]:
        """Get distinct colors for Berlin districts"""
        # ColorBrewer Set3 colors
        colors = [
            '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
            '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
            '#ccebc5', '#ffed6f', '#a6cee3', '#1f78b4', '#b2df8a'
        ]
        
        districts = [
            'Mitte', 'Friedrichshain-Kreuzberg', 'Pankow', 
            'Charlottenburg-Wilmersdorf', 'Spandau', 
            'Steglitz-Zehlendorf', 'Tempelhof-Schöneberg', 
            'Neukölln', 'Treptow-Köpenick', 'Marzahn-Hellersdorf', 
            'Lichtenberg', 'Reinickendorf'
        ]
        
        return dict(zip(districts, colors))
    
    def create_base_map(
        self,
        zoom_start: int = 11,
        tiles: str = None,
        width: str = '100%',
        height: str = '100%'
    ) -> folium.Map:
        """
        Create base Folium map centered on Berlin
        
        Args:
            zoom_start: Initial zoom level
            tiles: Map tiles to use
            width: Map width
            height: Map height
            
        Returns:
            Folium map object
        """
        if tiles is None:
            tiles = self.default_tiles
        
        m = folium.Map(
            location=self.berlin_center,
            zoom_start=zoom_start,
            tiles=tiles,
            width=width,
            height=height,
            control_scale=True
        )
        
        return m
    
    def add_district_boundaries(
        self,
        map_obj: folium.Map,
        districts_gdf: gpd.GeoDataFrame,
        district_field: str = 'name',
        style: Dict = None,
        tooltip: bool = True,
        popup: bool = False
    ) -> folium.Map:
        """
        Add district boundaries to map
        
        Args:
            map_obj: Folium map
            districts_gdf: GeoDataFrame with districts
            district_field: Field containing district names
            style: Style dictionary for boundaries
            tooltip: Whether to add tooltips
            popup: Whether to add popups
            
        Returns:
            Updated map
        """
        if style is None:
            style = {
                'fillColor': '#3186cc',
                'color': '#000000',
                'weight': 2,
                'fillOpacity': 0.1
            }
        
        # Ensure GeoJSON is serializable
        districts_gdf = districts_gdf.copy()
        for col in districts_gdf.columns:
            if districts_gdf[col].dtype == 'object':
                districts_gdf[col] = districts_gdf[col].astype(str)
        
        # Create GeoJson layer
        geojson_data = districts_gdf.__geo_interface__
        
        # Style function
        def style_function(feature):
            district_name = feature['properties'].get(district_field, 'Unknown')
            color = self.district_colors.get(district_name, '#3186cc')
            
            return {
                'fillColor': color,
                'color': '#000000',
                'weight': 2,
                'fillOpacity': 0.3,
                'dashArray': '5, 5'
            }
        
        # Tooltip function
        tooltip_fields = [district_field] + [col for col in districts_gdf.columns 
                                           if col not in ['geometry', district_field]]
        
        # Add to map
        geojson_layer = folium.GeoJson(
            geojson_data,
            name='Berlin Districts',
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=[f.capitalize() for f in tooltip_fields],
                localize=True
            ) if tooltip else None,
            popup=folium.GeoJsonPopup(
                fields=tooltip_fields,
                aliases=[f.capitalize() for f in tooltip_fields],
                localize=True
            ) if popup else None
        )
        
        geojson_layer.add_to(map_obj)
        
        return map_obj
    
    def add_h3_hexagon_layer(
        self,
        map_obj: folium.Map,
        hex_gdf: gpd.GeoDataFrame,
        value_column: str = None,
        colormap: str = 'YlOrRd',
        layer_name: str = 'H3 Hexagons',
        show: bool = True
    ) -> folium.Map:
        """
        Add H3 hexagon layer to map
        
        Args:
            map_obj: Folium map
            hex_gdf: GeoDataFrame with hexagon geometries
            value_column: Column to use for coloring
            colormap: Matplotlib colormap name
            layer_name: Name for the layer
            show: Whether to show layer by default
            
        Returns:
            Updated map
        """
        # Create feature group
        fg = folium.FeatureGroup(name=layer_name, show=show)
        
        # Create colormap if value column provided
        if value_column is not None and value_column in hex_gdf.columns:
            values = hex_gdf[value_column].dropna()
            if len(values) > 0:
                vmin, vmax = values.min(), values.max()
                color_scale = cm.linear.YlOrRd_09.scale(vmin, vmax)
            else:
                color_scale = None
        else:
            color_scale = None
        
        # Add each hexagon
        for idx, row in hex_gdf.iterrows():
            # Default color
            fill_color = '#3186cc'
            fill_opacity = 0.5
            
            # Apply colormap if available
            if color_scale is not None and pd.notna(row.get(value_column)):
                fill_color = color_scale(row[value_column])
                fill_opacity = 0.7
            
            # Create polygon
            if hasattr(row.geometry, '__geo_interface__'):
                poly = folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x, color=fill_color: {
                        'fillColor': color,
                        'color': '#000000',
                        'weight': 1,
                        'fillOpacity': fill_opacity
                    },
                    tooltip=self._create_hexagon_tooltip(row, value_column)
                )
                poly.add_to(fg)
        
        fg.add_to(map_obj)
        
        # Add colormap legend if applicable
        if color_scale is not None:
            color_scale.caption = f'{value_column} Value'
            map_obj.add_child(color_scale)
        
        return map_obj
    
    def _create_hexagon_tooltip(
        self,
        row: pd.Series,
        value_column: str
    ) -> folium.GeoJsonTooltip:
        """Create tooltip for hexagon"""
        tooltip_fields = ['hex_id', 'district', 'area_km2']
        if value_column:
            tooltip_fields.append(value_column)
        
        # Filter available fields
        available_fields = [f for f in tooltip_fields if f in row.index]
        
        return folium.GeoJsonTooltip(
            fields=available_fields,
            aliases=[f.replace('_', ' ').title() for f in available_fields],
            localize=True
        )
    
    def add_change_heatmap(
        self,
        map_obj: folium.Map,
        change_data: pd.DataFrame,
        lat_col: str = 'lat',
        lon_col: str = 'lon',
        value_col: str = 'change_intensity',
        radius: int = 15,
        blur: int = 15,
        max_zoom: int = 13,
        gradient: Dict = None,
        layer_name: str = 'Change Heatmap'
    ) -> folium.Map:
        """
        Add heatmap layer for change intensity
        
        Args:
            map_obj: Folium map
            change_data: DataFrame with change data
            lat_col: Latitude column name
            lon_col: Longitude column name
            value_col: Value column for heat intensity
            radius: Heatmap point radius
            blur: Heatmap blur
            max_zoom: Maximum zoom for heatmap
            gradient: Color gradient dictionary
            layer_name: Layer name
            
        Returns:
            Updated map
        """
        if gradient is None:
            gradient = {
                0.0: 'blue',
                0.25: 'lime',
                0.5: 'yellow',
                0.75: 'orange',
                1.0: 'red'
            }
        
        # Prepare heatmap data
        heat_data = []
        for _, row in change_data.iterrows():
            if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                intensity = row.get(value_col, 1)
                heat_data.append([row[lat_col], row[lon_col], intensity])
        
        # Add heatmap layer
        plugins.HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            max_zoom=max_zoom,
            gradient=gradient,
            name=layer_name
        ).add_to(map_obj)
        
        return map_obj
    
    def add_clustered_markers(
        self,
        map_obj: folium.Map,
        marker_data: pd.DataFrame,
        lat_col: str = 'lat',
        lon_col: str = 'lon',
        popup_col: str = None,
        icon_color: str = 'blue',
        layer_name: str = 'Change Locations'
    ) -> folium.Map:
        """
        Add clustered markers to map
        
        Args:
            map_obj: Folium map
            marker_data: DataFrame with marker data
            lat_col: Latitude column
            lon_col: Longitude column
            popup_col: Column for popup text
            icon_color: Marker color
            layer_name: Layer name
            
        Returns:
            Updated map
        """
        # Create marker cluster
        marker_cluster = plugins.MarkerCluster(name=layer_name)
        
        # Add markers
        for _, row in marker_data.iterrows():
            if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                popup_text = str(row[popup_col]) if popup_col and popup_col in row else ''
                
                marker = folium.Marker(
                    location=[row[lat_col], row[lon_col]],
                    popup=popup_text,
                    icon=folium.Icon(color=icon_color, icon='info-sign')
                )
                marker.add_to(marker_cluster)
        
        marker_cluster.add_to(map_obj)
        
        return map_obj
    
    def create_interactive_choropleth(
        self,
        districts_gdf: gpd.GeoDataFrame,
        value_column: str,
        title: str = 'Berlin Environmental Metrics',
        colormap: str = 'YlOrRd',
        legend_name: str = 'Value'
    ) -> folium.Map:
        """
        Create interactive choropleth map
        
        Args:
            districts_gdf: District boundaries with data
            value_column: Column to visualize
            title: Map title
            colormap: Color scale
            legend_name: Legend title
            
        Returns:
            Interactive choropleth map
        """
        # Create base map
        m = self.create_base_map()
        
        # Create choropleth
        folium.Choropleth(
            geo_data=districts_gdf,
            name='choropleth',
            data=districts_gdf,
            columns=['name', value_column],
            key_on='feature.properties.name',
            fill_color=colormap,
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=legend_name,
            highlight=True
        ).add_to(m)
        
        # Add tooltips
        style_function = lambda x: {'fillColor': '#ffffff', 
                                   'color':'#000000', 
                                   'fillOpacity': 0.1, 
                                   'weight': 0.1}
        
        highlight_function = lambda x: {'fillColor': '#000000', 
                                       'color':'#000000', 
                                       'fillOpacity': 0.50, 
                                       'weight': 0.1}
        
        tooltip_fields = ['name', value_column] + \
                        [col for col in districts_gdf.columns 
                         if col not in ['name', value_column, 'geometry']]
        
        tooltip = folium.features.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[f.capitalize() for f in tooltip_fields],
            localize=True
        )
        
        geojson_layer = folium.features.GeoJson(
            districts_gdf,
            style_function=style_function,
            control=False,
            highlight_function=highlight_function,
            tooltip=tooltip
        )
        
        m.add_child(geojson_layer)
        m.keep_in_front(geojson_layer)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add title
        title_html = f'''
        <h3 align="center" style="font-size:16px">
            <b>{title}</b>
        </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def create_side_by_side_maps(
        self,
        left_gdf: gpd.GeoDataFrame,
        right_gdf: gpd.GeoDataFrame,
        left_title: str = 'Before',
        right_title: str = 'After',
        value_column: str = 'value'
    ) -> folium.Map:
        """
        Create side-by-side comparison maps
        
        Args:
            left_gdf: Left side GeoDataFrame
            right_gdf: Right side GeoDataFrame
            left_title: Left map title
            right_title: Right map title
            value_column: Value column for coloring
            
        Returns:
            Side-by-side map
        """
        # Create base map
        m = self.create_base_map()
        
        # Create synchronizer
        sync = plugins.Sync()
        m.add_child(sync)
        
        # Create left map
        m1 = self.create_base_map()
        m1 = self.add_district_boundaries(m1, left_gdf)
        if value_column in left_gdf.columns:
            m1 = self.create_interactive_choropleth(m1, left_gdf, value_column)
        
        # Create right map
        m2 = self.create_base_map()
        m2 = self.add_district_boundaries(m2, right_gdf)
        if value_column in right_gdf.columns:
            m2 = self.create_interactive_choropleth(m2, right_gdf, value_column)
        
        # Add titles
        title_html1 = f'<h4>{left_title}</h4>'
        title_html2 = f'<h4>{right_title}</h4>'
        
        m1.get_root().html.add_child(folium.Element(title_html1))
        m2.get_root().html.add_child(folium.Element(title_html2))
        
        # Sync the maps
        sync.add_child(m1)
        sync.add_child(m2)
        
        return m
    
    def save_map(self, map_obj: folium.Map, filename: str):
        """Save map to HTML file"""
        map_obj.save(filename)
        logger.info(f"Map saved to {filename}")
    
    def create_dashboard_layout(
        self,
        maps: Dict[str, folium.Map],
        titles: Dict[str, str],
        n_cols: int = 2
    ) -> str:
        """
        Create HTML dashboard with multiple maps
        
        Args:
            maps: Dictionary of maps {name: map_object}
            titles: Dictionary of titles {name: title}
            n_cols: Number of columns in layout
            
        Returns:
            HTML string for dashboard
        """
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Berlin Environmental Dashboard</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.css" />
            <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.Default.css" />
            <script src="https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster.js"></script>
            <script src="https://unpkg.com/leaflet-side-by-side@2.0.0/leaflet-side-by-side.js"></script>
            <style>
                body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
                .dashboard { 
                    display: grid; 
                    grid-template-columns: repeat(''' + str(n_cols) + ''', 1fr);
                    grid-gap: 20px; 
                    padding: 20px; 
                }
                .map-container { 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    overflow: hidden; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .map-title { 
                    background: #f8f9fa; 
                    padding: 10px; 
                    margin: 0; 
                    font-weight: bold;
                    border-bottom: 1px solid #ddd;
                }
                .map-frame { 
                    height: 400px; 
                    width: 100%; 
                }
                @media (max-width: 768px) {
                    .dashboard { grid-template-columns: 1fr; }
                }
            </style>
        </head>
        <body>
            <div class="dashboard">
        '''
        
        # Add each map
        for name, map_obj in maps.items():
            # Save map to temporary HTML
            temp_html = f'temp_{name}.html'
            self.save_map(map_obj, temp_html)
            
            # Read HTML content
            with open(temp_html, 'r') as f:
                map_html = f.read()
            
            # Extract map div and script
            import re
            div_pattern = r'<div id="[^"]+"[^>]*>.*?</div>'
            script_pattern = r'<script>.*?</script>'
            
            div_match = re.search(div_pattern, map_html, re.DOTALL)
            script_match = re.search(script_pattern, map_html, re.DOTALL)
            
            if div_match and script_match:
                div_content = div_match.group(0)
                script_content = script_match.group(0)
                
                # Add to template
                html_template += f'''
                <div class="map-container">
                    <h3 class="map-title">{titles.get(name, name)}</h3>
                    <div class="map-frame">{div_content}</div>
                    {script_content}
                </div>
                '''
        
        html_template += '''
            </div>
        </body>
        </html>
        '''
        
        return html_template


class PlotlyVisualizer:
    """
    Create interactive Plotly visualizations for environmental data
    """
    
    @staticmethod
    def create_change_timeseries(
        data: pd.DataFrame,
        time_column: str,
        value_column: str,
        district_column: str = 'district',
        title: str = 'Environmental Change Over Time'
    ) -> go.Figure:
        """
        Create interactive timeseries plot of changes
        
        Args:
            data: DataFrame with time series data
            time_column: Column with timestamps
            value_column: Column with values to plot
            district_column: Column with district names
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = px.line(
            data,
            x=time_column,
            y=value_column,
            color=district_column,
            title=title,
            markers=True,
            line_shape='spline',
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title=value_column.replace('_', ' ').title(),
            hovermode='x unified',
            legend_title='District'
        )
        
        return fig
    
    @staticmethod
    def create_district_comparison(
        data: pd.DataFrame,
        district_column: str = 'district',
        value_columns: List[str] = None,
        title: str = 'District Comparison'
    ) -> go.Figure:
        """
        Create bar chart comparing districts
        
        Args:
            data: DataFrame with district data
            district_column: Column with district names
            value_columns: Columns to compare
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if value_columns is None:
            value_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            value_columns = [col for col in value_columns if col != district_column]
        
        fig = go.Figure()
        
        for value_col in value_columns:
            fig.add_trace(go.Bar(
                x=data[district_column],
                y=data[value_col],
                name=value_col.replace('_', ' ').title(),
                text=data[value_col].round(2),
                textposition='auto'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='District',
            yaxis_title='Value',
            barmode='group',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_heatmap_matrix(
        data: pd.DataFrame,
        index_column: str,
        columns: List[str],
        title: str = 'Change Intensity Matrix',
        colorscale: str = 'RdYlGn'
    ) -> go.Figure:
        """
        Create heatmap of change intensity across districts
        
        Args:
            data: DataFrame with data
            index_column: Column for row index (usually districts)
            columns: Columns to include in heatmap
            title: Plot title
            colorscale: Colorscale for heatmap
            
        Returns:
            Plotly figure
        """
        # Pivot data for heatmap
        heatmap_data = data.pivot_table(
            values=columns[0] if len(columns) == 1 else 'value',
            index=index_column,
            columns=columns if len(columns) > 1 else None
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            colorscale=colorscale,
            hoverongaps=False,
            colorbar=dict(title='Intensity')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Metric' if len(columns) > 1 else columns[0],
            yaxis_title=index_column.replace('_', ' ').title(),
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_radar_chart(
        data: pd.DataFrame,
        district: str,
        metrics: List[str],
        title: str = 'District Profile'
    ) -> go.Figure:
        """
        Create radar chart for district profile
        
        Args:
            data: DataFrame with district data
            district: District name
            metrics: List of metrics to include
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Filter data for district
        district_data = data[data['district'] == district].iloc[0]
        
        # Get values for each metric
        values = [district_data[metric] for metric in metrics]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=district
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1]
                )
            ),
            title=f'{title} - {district}',
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_scatter_matrix(
        data: pd.DataFrame,
        dimensions: List[str],
        color_column: str = 'district',
        title: str = 'Feature Relationships'
    ) -> go.Figure:
        """
        Create scatter plot matrix
        
        Args:
            data: DataFrame with data
            dimensions: List of columns to include
            color_column: Column for coloring points
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = px.scatter_matrix(
            data,
            dimensions=dimensions,
            color=color_column,
            title=title,
            template='plotly_white'
        )
        
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(
            height=800,
            showlegend=True
        )
        
        return fig


class ReportGenerator:
    """
    Generate PDF reports with visualizations
    """
    
    @staticmethod
    def generate_pdf_report(
        visualizations: List[go.Figure],
        map_html: str,
        summary_stats: Dict,
        output_path: str = 'berlin_environment_report.pdf'
    ):
        """
        Generate PDF report with all visualizations
        
        Args:
            visualizations: List of Plotly figures
            map_html: HTML string for interactive map
            summary_stats: Dictionary with summary statistics
            output_path: Output PDF path
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.graphics.shapes import Drawing
            from reportlab.graphics.charts.lineplots import LinePlot
            from reportlab.graphics import renderPDF
            
            import tempfile
            import base64
            from io import BytesIO
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#2c3e50')
            )
            
            story.append(Paragraph("Berlin Environmental Monitoring Report", title_style))
            story.append(Spacer(1, 12))
            
            # Date
            from datetime import datetime
            date_str = datetime.now().strftime("%B %d, %Y")
            story.append(Paragraph(f"Report generated on: {date_str}", styles["Normal"]))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            summary_style = ParagraphStyle(
                'SummaryStyle',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.HexColor('#3498db')
            )
            
            story.append(Paragraph("Executive Summary", summary_style))
            
            # Add summary statistics as a table
            summary_data = [['Metric', 'Value']]
            for key, value in summary_stats.items():
                if isinstance(value, (int, float)):
                    summary_data.append([key.replace('_', ' ').title(), f"{value:.2f}"])
                else:
                    summary_data.append([key.replace('_', ' ').title(), str(value)])
            
            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Key Findings
            findings_style = ParagraphStyle(
                'FindingsStyle',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.HexColor('#e74c3c')
            )
            
            story.append(Paragraph("Key Findings", findings_style))
            
            findings = [
                "• Significant vegetation changes detected in central districts",
                "• Resilience scores vary across Berlin with clear spatial patterns",
                "• Change intensity shows correlation with urban development",
                "• Machine learning models achieved high accuracy in change detection"
            ]
            
            for finding in findings:
                story.append(Paragraph(finding, styles["Bullet"]))
                story.append(Spacer(1, 6))
            
            story.append(Spacer(1, 20))
            
            # Visualizations section
            viz_style = ParagraphStyle(
                'VizStyle',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.HexColor('#27ae60')
            )
            
            story.append(Paragraph("Data Visualizations", viz_style))
            
            # Add static images from Plotly figures
            for i, fig in enumerate(visualizations):
                # Save figure as PNG
                img_bytes = fig.to_image(format="png", width=800, height=600)
                img_buffer = BytesIO(img_bytes)
                
                # Create temporary image file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp.write(img_buffer.getbuffer())
                    tmp_path = tmp.name
                
                # Add image to PDF
                story.append(Image(tmp_path, width=6*inch, height=4*inch))
                story.append(Spacer(1, 12))
                
                # Add caption
                caption = Paragraph(f"Figure {i+1}: {fig.layout.title.text if fig.layout.title.text else 'Visualization'}", 
                                   styles["Italic"])
                story.append(caption)
                story.append(Spacer(1, 20))
            
            # Save map HTML separately
            map_output_path = output_path.replace('.pdf', '_map.html')
            with open(map_output_path, 'w') as f:
                f.write(map_html)
            
            map_notice = Paragraph(f"Interactive map saved as: {map_output_path}", styles["Italic"])
            story.append(map_notice)
            story.append(Spacer(1, 20))
            
            # Recommendations
            rec_style = ParagraphStyle(
                'RecStyle',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.HexColor('#f39c12')
            )
            
            story.append(Paragraph("Recommendations", rec_style))
            
            recommendations = [
                "1. Prioritize green infrastructure in districts with declining vegetation",
                "2. Implement targeted resilience programs based on district scores",
                "3. Enhance monitoring in areas with high change volatility",
                "4. Consider urban planning adjustments in rapidly changing areas"
            ]
            
            for rec in recommendations:
                story.append(Paragraph(rec, styles["Normal"]))
                story.append(Spacer(1, 6))
            
            story.append(Spacer(1, 20))
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated: {output_path}")
            
        except ImportError as e:
            logger.error(f"ReportLab not installed. Install with: pip install reportlab")
            raise e


# Example usage function
def example_visualization():
    """Example of using visualization utilities"""
    print("Creating Berlin map visualizations...")
    
    # Initialize visualizer
    visualizer = BerlinMapVisualizer()
    
    # Create base map
    m = visualizer.create_base_map(zoom_start=12)
    
    # Load sample districts (using dummy data)
    from shapely.geometry import Polygon
    import geopandas as gpd
    
    # Create sample districts
    districts = ['Mitte', 'Pankow', 'Friedrichshain-Kreuzberg']
    geometries = []
    data = []
    
    for i, district in enumerate(districts):
        # Create simple polygon for each district
        polygon = Polygon([
            (13.3 + i*0.1, 52.5),
            (13.4 + i*0.1, 52.5),
            (13.4 + i*0.1, 52.6),
            (13.3 + i*0.1, 52.6)
        ])
        geometries.append(polygon)
        data.append({
            'name': district,
            'change_intensity': np.random.uniform(0, 1),
            'resilience_score': np.random.uniform(50, 100)
        })
    
    districts_gdf = gpd.GeoDataFrame(data, geometry=geometries, crs='EPSG:4326')
    
    # Add district boundaries
    m = visualizer.add_district_boundaries(m, districts_gdf)
    
    # Create sample hexagon data
    hex_data = []
    for _ in range(50):
        hex_data.append({
            'geometry': Polygon([
                (np.random.uniform(13.3, 13.6), np.random.uniform(52.4, 52.7)),
                (np.random.uniform(13.3, 13.6), np.random.uniform(52.4, 52.7)),
                (np.random.uniform(13.3, 13.6), np.random.uniform(52.4, 52.7)),
                (np.random.uniform(13.3, 13.6), np.random.uniform(52.4, 52.7))
            ]),
            'hex_id': f'hex_{_}',
            'district': np.random.choice(districts),
            'change_value': np.random.uniform(0, 1)
        })
    
    hex_gdf = gpd.GeoDataFrame(hex_data, crs='EPSG:4326')
    
    # Add hexagon layer
    m = visualizer.add_h3_hexagon_layer(
        m, hex_gdf, 
        value_column='change_value',
        layer_name='Change Hexagons'
    )
    
    # Create sample heatmap data
    heatmap_data = pd.DataFrame({
        'lat': np.random.uniform(52.4, 52.7, 100),
        'lon': np.random.uniform(13.3, 13.6, 100),
        'change_intensity': np.random.uniform(0, 1, 100)
    })
    
    # Add heatmap
    m = visualizer.add_change_heatmap(m, heatmap_data)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    visualizer.save_map(m, 'berlin_example_map.html')
    print("Example map saved as 'berlin_example_map.html'")
    
    # Create Plotly visualizations
    plotly_viz = PlotlyVisualizer()
    
    # Sample time series data
    time_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=12, freq='M'),
        'district': np.repeat(districts, 4),
        'ndvi_change': np.random.uniform(-0.2, 0.2, 12)
    })
    
    fig1 = plotly_viz.create_change_timeseries(
        time_data, 'date', 'ndvi_change', 'district',
        'NDVI Change Over Time'
    )
    
    # Save Plotly figure
    fig1.write_html('timeseries_plot.html')
    print("Timeseries plot saved as 'timeseries_plot.html'")
    
    print("\nVisualization examples complete!")


if __name__ == "__main__":
    example_visualization()