"""
OpenStreetMap (OSM) feature extraction for spatial regions.

This module provides functionality to extract static features from OpenStreetMap
for spatial regions such as grid cells, administrative boundaries, or Voronoi regions.
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import os
import pyproj
from tqdm import tqdm
import pickle


class OSMFeatureExtractor:
    """Extract OpenStreetMap features for spatial regions."""

    def __init__(
        self,
        cache_dir=None,
        clear_cache=False,
        meter_crs="EPSG:3857",
        lat_lon_crs="EPSG:4326",
    ):
        """
        Initialize the OSM feature extractor.

        Args:
            cache_dir: Directory to cache downloaded OSM data
            clear_cache: Whether to ignore existing cache and download fresh data
            meter_crs: CRS to use for meter-based calculations
            lat_lon_crs: CRS to use for latitude/longitude coordinates (typically EPSG:4326)
        """
        self.cache_dir = cache_dir
        self.clear_cache = clear_cache
        self.meter_crs = meter_crs
        self.lat_lon_crs = lat_lon_crs

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def extract_features(
        self, regions_gdf, bounds=None, normalize=True, feature_types=None
    ):
        """
        Extract OSM features for the given regions.

        Args:
            regions_gdf: GeoDataFrame containing region polygons
            bounds: Optional bounding box to limit data extraction
            normalize: Whether to normalize features by region area
            feature_types: List of feature types to extract (e.g., 'poi', 'road', 'junction')
                           If None, extracts all available features

        Returns:
            DataFrame with OSM features for each region
        """
        print("Extracting OpenStreetMap features...")

        if bounds is None:
            # Use bounds of regions with some padding
            bbox = regions_gdf.total_bounds
            bounds = {
                "min_lat": bbox[1] - 0.01,
                "max_lat": bbox[3] + 0.01,
                "min_lon": bbox[0] - 0.01,
                "max_lon": bbox[2] + 0.01,
            }

        osm_data = self._download_osm_data(bounds)

        features_df = self._compute_region_features(
            regions_gdf, osm_data, normalize, feature_types
        )

        print(
            f"Extracted {len(features_df.columns)} OSM features for {len(features_df)} regions"
        )
        return features_df

    def _download_osm_data(self, bounds):
        """
        Download OSM data for the given bounds.

        Args:
            bounds: Dictionary or tuple with geographic bounds

        Returns:
            Dictionary with POIs, roads, and nodes GeoDataFrames
        """
        if isinstance(bounds, dict):
            if abs(bounds["min_lat"]) > 180 or abs(bounds["max_lat"]) > 180:
                transformer = pyproj.Transformer.from_crs(
                    self.meter_crs, self.lat_lon_crs, always_xy=True
                )
                west_lon, south_lat = transformer.transform(
                    bounds["min_lon"], bounds["min_lat"]
                )
                east_lon, north_lat = transformer.transform(
                    bounds["max_lon"], bounds["max_lat"]
                )
                north, south, east, west = north_lat, south_lat, east_lon, west_lon
            else:
                north = bounds["max_lat"]
                south = bounds["min_lat"]
                east = bounds["max_lon"]
                west = bounds["min_lon"]
        else:
            if isinstance(bounds, gpd.GeoDataFrame) or isinstance(
                bounds, gpd.GeoSeries
            ):
                if bounds.crs is not None and bounds.crs != self.lat_lon_crs:
                    bounds = bounds.to_crs(self.lat_lon_crs)
                bbox = bounds.total_bounds  # minx, miny, maxx, maxy
                west, south, east, north = bbox

        print(
            f"Downloading OSM data for bounds (lat/lon): N={north:.5f}, S={south:.5f}, E={east:.5f}, W={west:.5f}"
        )

        # Check cache first if enabled
        cache_path = None
        if self.cache_dir and not self.clear_cache:
            bounds_str = f"{north:.5f}_{south:.5f}_{east:.5f}_{west:.5f}".replace(
                ".", "p"
            )
            cache_path = os.path.join(self.cache_dir, f"osm_data_{bounds_str}.pkl")
            if os.path.exists(cache_path):
                print(f"Loading OSM data from cache: {cache_path}")
                try:
                    with open(cache_path, "rb") as f:
                        osm_data = pickle.load(f)
                    print(f"Successfully loaded OSM data from cache")
                    return osm_data
                except FileNotFoundError:
                    print(
                        f"Cache file not found: {cache_path}. Downloading fresh data..."
                    )
                except Exception as e:
                    print(f"Error loading from cache: {e}. Downloading fresh data...")

        # Download data if not cached or cache failed
        try:
            print("Downloading POIs...")
            poi_tags = {
                "amenity": True,
                "shop": True,
                "leisure": True,
                "tourism": True,
                "office": True,
                "education": True,
                "healthcare": True,
                "building": [
                    "school",
                    "university",
                    "hospital",
                    "church",
                    "cathedral",
                    "mosque",
                    "synagogue",
                    "temple",
                ],
            }

            try:
                pois = ox.features_from_bbox(
                    bbox=(north, south, east, west), tags=poi_tags
                )
                if len(pois) == 0:
                    print("No POIs found. Creating empty GeoDataFrame.")
                    pois = gpd.GeoDataFrame({"geometry": []}, crs=self.lat_lon_crs)
                else:
                    # Normalize and clean POI data
                    pois = pois.reset_index()
                    if "element_type" not in pois.columns:
                        pois["element_type"] = "node"
                    # Handle polygon POIs by converting to centroids
                    if not all(isinstance(geom, Point) for geom in pois.geometry):
                        pois = pois.to_crs(self.meter_crs)
                        pois["geometry"] = pois.geometry.centroid
                        pois = pois.to_crs(self.lat_lon_crs)

            except Exception as e:
                print(f"Error downloading POIs: {e}. Creating empty DataFrame.")
                pois = gpd.GeoDataFrame({"geometry": []}, crs=self.lat_lon_crs)

            print("Downloading road network...")
            try:
                G = ox.graph_from_bbox(
                    bbox=(north, south, east, west), network_type="drive", simplify=True
                )
                nodes, roads = ox.graph_to_gdfs(G)
            except Exception as e:
                print(
                    f"Error downloading road network: {e}. Creating empty DataFrames."
                )
                nodes = gpd.GeoDataFrame({"geometry": []}, crs=self.lat_lon_crs)
                roads = gpd.GeoDataFrame({"geometry": []}, crs=self.lat_lon_crs)

            if cache_path:
                print(f"Caching OSM data to: {cache_path}")
                try:
                    osm_data = {"pois": pois, "roads": roads, "nodes": nodes}
                    with open(cache_path, "wb") as f:
                        pickle.dump(osm_data, f)
                    print("Successfully cached OSM data to pickle file")
                except Exception as e:
                    print(f"Error caching OSM data: {e}")

            return {"pois": pois, "roads": roads, "nodes": nodes}

        except Exception as e:
            print(f"Error downloading OSM data: {e}")
            return {
                "pois": gpd.GeoDataFrame({"geometry": []}, crs=self.lat_lon_crs),
                "roads": gpd.GeoDataFrame({"geometry": []}, crs=self.lat_lon_crs),
                "nodes": gpd.GeoDataFrame({"geometry": []}, crs=self.lat_lon_crs),
            }

    def _classify_pois(self, pois_gdf):
        """
        Classify POIs into categories.

        Args:
            pois_gdf: GeoDataFrame containing POI information

        Returns:
            GeoDataFrame with added 'category' column
        """
        if len(pois_gdf) == 0:
            pois_gdf["category"] = ""
            return pois_gdf

        pois = pois_gdf.copy()

        category_mapping = {
            "shop": "shop",
            "restaurant": "food",
            "fast_food": "food",
            "cafe": "food",
            "bar": "food",
            "pub": "food",
            "food_court": "food",
            "school": "education",
            "university": "education",
            "college": "education",
            "library": "education",
            "kindergarten": "education",
            "hospital": "healthcare",
            "clinic": "healthcare",
            "doctors": "healthcare",
            "pharmacy": "healthcare",
            "dentist": "healthcare",
            "bank": "service",
            "post_office": "service",
            "police": "service",
            "fire_station": "service",
            "townhall": "service",
            "cinema": "entertainment",
            "theatre": "entertainment",
            "arts_centre": "entertainment",
            "museum": "entertainment",
            "park": "recreation",
            "playground": "recreation",
            "sports_centre": "recreation",
            "stadium": "recreation",
            "fitness_centre": "recreation",
            "hotel": "accommodation",
            "hostel": "accommodation",
            "guest_house": "accommodation",
            "parking": "transportation",
            "bus_station": "transportation",
            "train_station": "transportation",
            "subway_entrance": "transportation",
            "fuel": "transportation",
            "place_of_worship": "religious",
            "church": "religious",
            "mosque": "religious",
            "synagogue": "religious",
            "temple": "religious",
        }

        def get_category(poi):
            if "amenity" in poi and poi["amenity"] in category_mapping:
                return category_mapping[poi["amenity"]]

            if "shop" in poi and poi["shop"]:
                return "shop"

            if "building" in poi and poi["building"] in category_mapping:
                return category_mapping[poi["building"]]

            if "leisure" in poi and poi["leisure"]:
                return "recreation"

            if "tourism" in poi and poi["tourism"]:
                return "tourism"

            return "other"

        pois["category"] = pois.apply(get_category, axis=1)

        return pois

    def _compute_region_features(
        self, regions_gdf, osm_data, normalize=True, feature_types=None
    ):
        """
        Compute features for each region from OSM data.

        Args:
            regions_gdf: GeoDataFrame containing region geometries
            osm_data: Dictionary with 'pois', 'roads', and 'nodes'
            normalize: Whether to normalize features by region area
            feature_types: Types of features to extract (list containing 'poi', 'road', 'junction')

        Returns:
            DataFrame with region features
        """
        if feature_types is None:
            feature_types = ["poi", "road", "junction"]

        regions_projected = regions_gdf.to_crs(self.meter_crs)
        pois_projected = (
            osm_data["pois"].to_crs(self.meter_crs)
            if len(osm_data["pois"]) > 0
            else None
        )
        roads_projected = (
            osm_data["roads"].to_crs(self.meter_crs)
            if len(osm_data["roads"]) > 0
            else None
        )
        nodes_projected = (
            osm_data["nodes"].to_crs(self.meter_crs)
            if len(osm_data["nodes"]) > 0
            else None
        )

        if normalize:
            regions_projected["area_m2"] = regions_projected.geometry.area

        features = pd.DataFrame(index=regions_projected.index)

        # 1. POI densities by category
        if (
            "poi" in feature_types
            and pois_projected is not None
            and len(pois_projected) > 0
        ):
            pois_categorized = self._classify_pois(pois_projected)

            print("Calculating POI densities...")
            for category in tqdm(pois_categorized["category"].unique()):
                category_pois = pois_categorized[
                    pois_categorized["category"] == category
                ]

                poi_counts = gpd.sjoin(
                    category_pois, regions_projected, how="inner", predicate="within"
                )
                poi_counts = poi_counts.groupby("index_right").size()

                # Add counts to features, filling missing values with 0
                features[f"poi_{category}_count"] = poi_counts.reindex(
                    features.index, fill_value=0
                )

                # Normalize by area if requested
                if normalize:
                    features[f"poi_{category}_density"] = (
                        features[f"poi_{category}_count"] / regions_projected["area_m2"]
                    )
                    features = features.drop(columns=[f"poi_{category}_count"])

            total_poi_counts = gpd.sjoin(
                pois_projected, regions_projected, how="inner", predicate="within"
            )
            total_poi_counts = total_poi_counts.groupby("index_right").size()
            features["poi_total_count"] = total_poi_counts.reindex(
                features.index, fill_value=0
            )

            if normalize:
                features["poi_total_density"] = (
                    features["poi_total_count"] / regions_projected["area_m2"]
                )
                features = features.drop(columns=["poi_total_count"])

        # 2. Junction counts/density
        if (
            "junction" in feature_types
            and nodes_projected is not None
            and len(nodes_projected) > 0
        ):
            print("Calculating junction counts...")
            if "street_count" in nodes_projected.columns:
                junction_nodes = nodes_projected[nodes_projected["street_count"] > 1]
            else:
                # If street_count not available, use all nodes as potential junctions
                junction_nodes = nodes_projected

            # Count junctions per region
            junction_counts = gpd.sjoin(
                junction_nodes, regions_projected, how="inner", predicate="within"
            )
            junction_counts = junction_counts.groupby("index_right").size()

            features["junction_count"] = junction_counts.reindex(
                features.index, fill_value=0
            )

            if normalize:
                features["junction_density"] = (
                    features["junction_count"] / regions_projected["area_m2"]
                )
                features = features.drop(columns=["junction_count"])

        # 3. Road density and features
        if (
            "road" in feature_types
            and roads_projected is not None
            and len(roads_projected) > 0
        ):
            print("Calculating road density and features...")
            road_stats = []

            for idx, region in tqdm(
                regions_projected.iterrows(), total=len(regions_projected)
            ):
                # Clip roads to region
                region_roads = gpd.clip(roads_projected, region.geometry)

                # Calculate total road length
                total_length_m = region_roads.geometry.length.sum()

                # Calculate average road speed if available
                avg_speed = None
                if "maxspeed" in region_roads.columns:
                    speeds = []
                    for speed in region_roads["maxspeed"]:
                        if isinstance(speed, str):
                            try:
                                speeds.append(float(speed.split()[0]))
                            except (ValueError, IndexError):
                                pass
                        elif isinstance(speed, (int, float)):
                            speeds.append(speed)

                    if speeds:
                        avg_speed = sum(speeds) / len(speeds)

                road_stats.append(
                    {
                        "index": idx,
                        "road_length_m": total_length_m,
                        "avg_speed_kmh": avg_speed,
                    }
                )

            road_stats_df = pd.DataFrame(road_stats).set_index("index")

            # Add road features
            features["road_length_m"] = road_stats_df["road_length_m"]

            if "avg_speed_kmh" in road_stats_df.columns:
                features["avg_speed_kmh"] = road_stats_df["avg_speed_kmh"]

            if normalize:
                features["road_density_m_per_m2"] = (
                    features["road_length_m"] / regions_projected["area_m2"]
                )

        # Fill NaN values with mean or 0
        for col in features.columns:
            if features[col].isna().any():
                mean_val = features[col].mean()
                if pd.isna(mean_val):  # If mean is also NaN, use 0
                    features[col] = features[col].fillna(0)
                else:
                    features[col] = features[col].fillna(mean_val)

        return features


def extract_osm_features(
    regions_gdf,
    bounds=None,
    cache_dir=None,
    feature_types=None,
    normalize=True,
    meter_crs="EPSG:3857",
    lat_lon_crs="EPSG:4326"
):
    """
    Extract OpenStreetMap features for a set of spatial regions.

    Args:
        regions_gdf: GeoDataFrame containing region polygons
        bounds: Optional dictionary with geographical bounds {'min_lat', 'max_lat', 'min_lon', 'max_lon'}
        cache_dir: Directory to cache downloaded OSM data
        feature_types: List of feature types to extract ('poi', 'road', 'junction')
        normalize: Whether to normalize features by region area
        meter_crs: Coordinate reference system for meter-based calculations (e.g., UTM zone)
        lat_lon_crs: Coordinate reference system for lat/lon coordinates (default: EPSG:4326)

    Returns:
        DataFrame with OSM features for each region
    """
    extractor = OSMFeatureExtractor(
        cache_dir=cache_dir, meter_crs=meter_crs, lat_lon_crs=lat_lon_crs
    )

    return extractor.extract_features(
        regions_gdf=regions_gdf,
        bounds=bounds,
        normalize=normalize,
        feature_types=feature_types,
    )
