# utils/h3_grid_generator.py
"""
H3 Grid Generation Utilities for Berlin Environmental Monitoring
Creates hexagon grids at multiple resolutions and integrates with district boundaries
"""

import h3
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point, shape
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class H3GridGenerator:
    """
    Advanced H3 grid generator for Berlin districts with multiple resolution support
    """
    
    def __init__(self):
        self.resolution_levels = {
            8: "District-level (approx 500m)",
            9: "Neighborhood-level (approx 200m)", 
            10: "Street-level (approx 100m)",
            11: "Building-level (approx 50m)",
            12: "High-detail (approx 20m)"
        }
        
    def create_h3_grid_from_polygon(
        self, 
        polygon: Polygon,
        resolution: int = 9,
        buffer_distance: float = 0
    ) -> List[str]:
        """
        Create H3 hexagons covering a polygon
        
        Args:
            polygon: Shapely polygon
            resolution: H3 resolution (8-12)
            buffer_distance: Buffer distance in degrees
            
        Returns:
            List of H3 hexagon IDs
        """
        try:
            # Buffer if needed
            if buffer_distance > 0:
                polygon = polygon.buffer(buffer_distance)
            
            # Get polygon bounding box
            min_lon, min_lat, max_lon, max_lat = polygon.bounds
            
            # Generate hexagons using polyfill
            geojson_poly = {
                "type": "Polygon",
                "coordinates": [list(polygon.exterior.coords)]
            }
            
            hexagons = h3.polyfill(
                geojson_poly,
                resolution,
                geo_json_conformant=True
            )
            
            # Filter hexagons whose centers are within the polygon
            filtered_hexagons = []
            for hex_id in hexagons:
                center = h3.h3_to_geo(hex_id)
                point = Point(center[1], center[0])  # Note: h3 returns (lat, lon)
                if polygon.contains(point):
                    filtered_hexagons.append(hex_id)
            
            logger.info(f"Generated {len(filtered_hexagons)} H3 hexagons at resolution {resolution}")
            return filtered_hexagons
            
        except Exception as e:
            logger.error(f"Error creating H3 grid: {e}")
            raise
    
    def create_district_h3_grids(
        self,
        districts_gdf: gpd.GeoDataFrame,
        resolution: int = 9,
        district_field: str = 'name'
    ) -> Dict[str, List[str]]:
        """
        Create H3 grids for each district
        
        Args:
            districts_gdf: GeoDataFrame of districts
            resolution: H3 resolution
            district_field: Field containing district names
            
        Returns:
            Dictionary mapping district names to lists of H3 hex IDs
        """
        district_grids = {}
        
        for idx, district in districts_gdf.iterrows():
            district_name = district.get(district_field, f"District_{idx}")
            polygon = district.geometry
            
            if polygon.is_empty:
                logger.warning(f"Empty geometry for district: {district_name}")
                continue
            
            try:
                hexagons = self.create_h3_grid_from_polygon(
                    polygon, 
                    resolution
                )
                district_grids[district_name] = hexagons
                logger.info(f"District '{district_name}': {len(hexagons)} hexagons")
                
            except Exception as e:
                logger.error(f"Error processing district {district_name}: {e}")
        
        return district_grids
    
    def hexagons_to_geodataframe(
        self,
        hexagon_ids: List[str],
        attributes: Optional[Dict[str, List]] = None
    ) -> gpd.GeoDataFrame:
        """
        Convert H3 hexagon IDs to GeoDataFrame with geometries
        
        Args:
            hexagon_ids: List of H3 hex IDs
            attributes: Optional dictionary of attributes for each hexagon
            
        Returns:
            GeoDataFrame with hexagon geometries
        """
        features = []
        
        for i, hex_id in enumerate(hexagon_ids):
            # Get hexagon boundary coordinates
            boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)
            
            # Create polygon (Note: h3 returns (lat, lon), GeoJSON expects (lon, lat))
            boundary = [(lon, lat) for lat, lon in boundary]
            polygon = Polygon(boundary)
            
            # Create feature
            feature = {
                'hex_id': hex_id,
                'geometry': polygon,
                'resolution': h3.h3_get_resolution(hex_id),
                'area_km2': h3.cell_area(hex_id, unit='km^2')
            }
            
            # Add custom attributes if provided
            if attributes:
                for key, values in attributes.items():
                    if i < len(values):
                        feature[key] = values[i]
            
            features.append(feature)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')
        
        # Calculate centroids
        gdf['centroid_lat'] = gdf.geometry.centroid.y
        gdf['centroid_lon'] = gdf.geometry.centroid.x
        
        return gdf
    
    def create_multi_resolution_grid(
        self,
        polygon: Polygon,
        resolutions: List[int] = [8, 9, 10]
    ) -> Dict[int, gpd.GeoDataFrame]:
        """
        Create grids at multiple resolutions for hierarchical analysis
        
        Args:
            polygon: Base polygon
            resolutions: List of H3 resolutions
            
        Returns:
            Dictionary mapping resolutions to GeoDataFrames
        """
        multi_grid = {}
        
        for res in resolutions:
            logger.info(f"Generating grid at resolution {res}")
            hex_ids = self.create_h3_grid_from_polygon(polygon, res)
            gdf = self.hexagons_to_geodataframe(hex_ids)
            multi_grid[res] = gdf
            
            # Add parent-child relationships for hierarchical analysis
            if res > min(resolutions):
                parent_res = res - 1
                gdf['parent_hex_id'] = gdf['hex_id'].apply(
                    lambda x: h3.h3_to_parent(x, parent_res)
                )
        
        return multi_grid
    
    def aggregate_hexagon_data(
        self,
        hex_gdf: gpd.GeoDataFrame,
        data_gdf: gpd.GeoDataFrame,
        data_column: str,
        aggregation_method: str = 'mean'
    ) -> gpd.GeoDataFrame:
        """
        Aggregate data from polygons to hexagons
        
        Args:
            hex_gdf: Hexagon GeoDataFrame
            data_gdf: Data GeoDataFrame with values
            data_column: Column to aggregate
            aggregation_method: 'mean', 'sum', 'max', 'min', 'median'
            
        Returns:
            Hexagon GeoDataFrame with aggregated values
        """
        # Ensure same CRS
        if hex_gdf.crs != data_gdf.crs:
            data_gdf = data_gdf.to_crs(hex_gdf.crs)
        
        # Perform spatial join
        joined = gpd.sjoin(hex_gdf, data_gdf[[data_column, 'geometry']], 
                          how='left', predicate='intersects')
        
        # Aggregate values
        if aggregation_method == 'mean':
            aggregated = joined.groupby('hex_id')[data_column].mean().reset_index()
        elif aggregation_method == 'sum':
            aggregated = joined.groupby('hex_id')[data_column].sum().reset_index()
        elif aggregation_method == 'max':
            aggregated = joined.groupby('hex_id')[data_column].max().reset_index()
        elif aggregation_method == 'min':
            aggregated = joined.groupby('hex_id')[data_column].min().reset_index()
        elif aggregation_method == 'median':
            aggregated = joined.groupby('hex_id')[data_column].median().reset_index()
        else:
            aggregated = joined.groupby('hex_id')[data_column].mean().reset_index()
        
        # Merge back with original hexagon data
        result = hex_gdf.merge(aggregated, on='hex_id', how='left')
        
        return result
    
    def create_change_grid(
        self,
        hex_gdf_year1: gpd.GeoDataFrame,
        hex_gdf_year2: gpd.GeoDataFrame,
        value_column: str = 'ndvi_value'
    ) -> gpd.GeoDataFrame:
        """
        Create grid showing changes between two time periods
        
        Args:
            hex_gdf_year1: Hexagon grid for year 1
            hex_gdf_year2: Hexagon grid for year 2
            value_column: Column to compare
            
        Returns:
            GeoDataFrame with change metrics
        """
        # Ensure same hexagon IDs
        common_hex_ids = set(hex_gdf_year1['hex_id']).intersection(
            set(hex_gdf_year2['hex_id'])
        )
        
        # Create change DataFrame
        change_data = []
        
        for hex_id in common_hex_ids:
            val1 = hex_gdf_year1.loc[hex_gdf_year1['hex_id'] == hex_id, value_column].values
            val2 = hex_gdf_year2.loc[hex_gdf_year2['hex_id'] == hex_id, value_column].values
            
            if len(val1) > 0 and len(val2) > 0:
                change = val2[0] - val1[0]
                percent_change = (change / val1[0] * 100) if val1[0] != 0 else 0
                
                change_data.append({
                    'hex_id': hex_id,
                    f'{value_column}_year1': val1[0],
                    f'{value_column}_year2': val2[0],
                    'change_absolute': change,
                    'change_percentage': percent_change,
                    'change_category': self._categorize_change(percent_change)
                })
        
        change_df = pd.DataFrame(change_data)
        
        # Merge with geometry from year 1
        change_gdf = hex_gdf_year1[['hex_id', 'geometry']].merge(
            change_df, on='hex_id', how='inner'
        )
        
        return change_gdf
    
    def _categorize_change(self, percent_change: float) -> str:
        """Categorize percentage change"""
        if percent_change > 20:
            return "Strong Increase"
        elif percent_change > 10:
            return "Moderate Increase"
        elif percent_change > 5:
            return "Slight Increase"
        elif percent_change < -20:
            return "Strong Decrease"
        elif percent_change < -10:
            return "Moderate Decrease"
        elif percent_change < -5:
            return "Slight Decrease"
        else:
            return "Stable"
    
    def export_grid(
        self,
        hex_gdf: gpd.GeoDataFrame,
        output_format: str = 'geojson',
        filename: str = 'h3_grid'
    ):
        """
        Export grid to various formats
        
        Args:
            hex_gdf: Hexagon GeoDataFrame
            output_format: 'geojson', 'shapefile', 'csv', 'parquet'
            filename: Output filename without extension
        """
        if output_format == 'geojson':
            output_path = f"{filename}.geojson"
            hex_gdf.to_file(output_path, driver='GeoJSON')
        elif output_format == 'shapefile':
            output_path = f"{filename}.shp"
            hex_gdf.to_file(output_path)
        elif output_format == 'csv':
            output_path = f"{filename}.csv"
            # Export without geometry for CSV
            df = pd.DataFrame(hex_gdf.drop(columns='geometry'))
            df.to_csv(output_path, index=False)
        elif output_format == 'parquet':
            output_path = f"{filename}.parquet"
            hex_gdf.to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
        
        logger.info(f"Grid exported to {output_path}")
        return output_path
    
    def validate_grid(
        self,
        hex_gdf: gpd.GeoDataFrame,
        reference_polygon: Polygon
    ) -> Dict[str, float]:
        """
        Validate grid coverage and quality
        
        Args:
            hex_gdf: Hexagon grid to validate
            reference_polygon: Reference polygon for validation
            
        Returns:
            Dictionary of validation metrics
        """
        validation_metrics = {}
        
        # Calculate coverage
        hex_union = hex_gdf.unary_union
        coverage_area = hex_union.area
        reference_area = reference_polygon.area
        
        validation_metrics['coverage_percentage'] = (coverage_area / reference_area) * 100
        validation_metrics['hexagon_count'] = len(hex_gdf)
        validation_metrics['average_area_km2'] = hex_gdf['area_km2'].mean()
        validation_metrics['area_variation'] = hex_gdf['area_km2'].std()
        
        # Check for gaps
        coverage_gap = reference_polygon.difference(hex_union)
        validation_metrics['gap_area_percentage'] = (coverage_gap.area / reference_area) * 100
        
        # Check for overlaps
        # Note: H3 hexagons shouldn't overlap by design
        
        logger.info(f"Validation metrics: {validation_metrics}")
        return validation_metrics


# Helper functions
def load_berlin_boundary() -> gpd.GeoDataFrame:
    """Load Berlin boundary from various sources"""
    try:
        # Try local file first
        local_path = '../data/boundaries/berlin_boundary.geojson'
        gdf = gpd.read_file(local_path)
        logger.info(f"Loaded Berlin boundary from local file: {local_path}")
    except:
        # Fallback to remote source
        remote_url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
        gdf = gpd.read_file(remote_url)
        gdf = gdf.dissolve()  # Combine all districts into single boundary
        logger.info("Loaded Berlin boundary from remote source")
    
    return gdf

def load_berlin_districts() -> gpd.GeoDataFrame:
    """Load Berlin districts GeoDataFrame"""
    districts_url = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
    districts = gpd.read_file(districts_url)
    return districts

def generate_complete_berlin_grid(
    resolution: int = 9,
    export: bool = True
) -> Tuple[gpd.GeoDataFrame, Dict[str, List[str]]]:
    """
    Generate complete H3 grid for Berlin
    
    Args:
        resolution: H3 resolution
        export: Whether to export results
        
    Returns:
        Tuple of (complete grid GeoDataFrame, district grids dictionary)
    """
    # Initialize generator
    generator = H3GridGenerator()
    
    # Load data
    districts = load_berlin_districts()
    berlin_boundary = load_berlin_boundary()
    
    # Generate district-level grids
    logger.info("Generating district-level H3 grids...")
    district_grids = generator.create_district_h3_grids(
        districts, 
        resolution=resolution
    )
    
    # Combine all hexagons
    all_hexagons = []
    for hex_list in district_grids.values():
        all_hexagons.extend(hex_list)
    all_hexagons = list(set(all_hexagons))  # Remove duplicates
    
    # Create complete GeoDataFrame
    logger.info("Creating complete grid GeoDataFrame...")
    complete_grid = generator.hexagons_to_geodataframe(all_hexagons)
    
    # Add district information
    district_mapping = []
    for district_name, hex_ids in district_grids.items():
        for hex_id in hex_ids:
            district_mapping.append({
                'hex_id': hex_id,
                'district': district_name
            })
    
    district_df = pd.DataFrame(district_mapping)
    complete_grid = complete_grid.merge(district_df, on='hex_id', how='left')
    
    # Validate grid
    validation = generator.validate_grid(
        complete_grid, 
        berlin_boundary.geometry.iloc[0]
    )
    
    if export:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generator.export_grid(
            complete_grid,
            output_format='geojson',
            filename=f'berlin_h3_grid_r{resolution}_{timestamp}'
        )
        
        # Export district mapping
        district_grids_df = pd.DataFrame([
            {'district': district, 'hex_count': len(hex_ids)}
            for district, hex_ids in district_grids.items()
        ])
        district_grids_df.to_csv(
            f'berlin_district_h3_counts_r{resolution}_{timestamp}.csv',
            index=False
        )
    
    return complete_grid, district_grids


if __name__ == "__main__":
    # Example usage
    print("Testing H3 Grid Generator...")
    
    # Generate sample grid
    grid, district_grids = generate_complete_berlin_grid(resolution=9)
    
    print(f"\nGenerated {len(grid)} hexagons")
    print(f"Coverage: {grid.geometry.area.sum():.2f} sq degrees")
    print(f"Districts covered: {len(district_grids)}")
    
    # Show first few rows
    print("\nFirst 5 hexagons:")
    print(grid[['hex_id', 'district', 'area_km2']].head())