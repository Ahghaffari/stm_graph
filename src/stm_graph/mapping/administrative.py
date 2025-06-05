import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, box
from shapely import wkt
import matplotlib.pyplot as plt
import contextily as ctx
from typing import Tuple
from .base import BaseMapping
import json
from shapely.geometry import shape


class AdministrativeMapping(BaseMapping):
    """
    Administrative boundary-based spatial mapping.

    This class implements mapping using administrative boundaries from any source,
    supporting various file formats (ShapeFile, GeoJSON, CSV with geometry columns).
    """

    def __init__(
        self,
        admin_type="administrative",
        districts_file=None,
        districts_gdf=None,
        id_column=None,
        name_column=None,
        geometry_column=None,
        input_crs="EPSG:4326",
        meter_crs="EPSG:3857",
        testing_mode=False,
        test_bounds=None,
        **kwargs,
    ):
        """Initialize administrative mapping with the specified parameters."""
        super().__init__(name=f"administrative_{admin_type}", **kwargs)
        self.admin_type = admin_type
        self.districts_file = districts_file
        self.districts_gdf = districts_gdf
        self.id_column = id_column
        self.name_column = name_column
        self.geometry_column = geometry_column
        self.input_crs = input_crs
        self.meter_crs = meter_crs
        self.testing_mode = testing_mode
        self.test_bounds = test_bounds

    def create_mapping(
        self, points_gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, pd.Series]:
        """
        Create a mapping from points to administrative districts.

        Args:
            points_gdf: GeoDataFrame with point geometries

        Returns:
            Tuple of (districts_gdf, point_to_district)
            - districts_gdf: GeoDataFrame with district polygons
            - point_to_district: Series mapping point indices to district indices
        """
        print(f"Creating administrative mapping using {self.admin_type} districts...")

        districts_gdf = self._load_administrative_polygons()

        if points_gdf.crs is None:
            points_gdf = points_gdf.set_crs(self.input_crs)

        meter_districts_gdf = (
            districts_gdf.to_crs(self.meter_crs)
            if districts_gdf.crs != self.meter_crs
            else districts_gdf
        )

        point_to_district = self._assign_points_to_polygons(
            points_gdf, meter_districts_gdf
        )

        # Report mapping statistics
        mapped_points = (point_to_district >= 0).sum()
        total_points = len(point_to_district)
        mapping_percentage = (mapped_points / total_points) * 100

        print(
            f"Mapped {mapped_points} of {total_points} points ({mapping_percentage:.2f}%)"
        )
        print(
            f"Points assigned to {point_to_district.nunique() - (1 if -1 in point_to_district.values else 0)} districts"
        )

        return districts_gdf, point_to_district

    def _filter_by_test_bounds(self, gdf):
        """Filter GeoDataFrame to only include geometries within test bounds"""
        if not self.testing_mode or self.test_bounds is None:
            return gdf

        test_box = box(
            self.test_bounds["min_lon"],
            self.test_bounds["min_lat"],
            self.test_bounds["max_lon"],
            self.test_bounds["max_lat"],
        )
        test_box_gdf = gpd.GeoDataFrame(geometry=[test_box], crs=self.input_crs)

        if gdf.crs != test_box_gdf.crs:
            test_box_gdf = test_box_gdf.to_crs(gdf.crs)

        # Keep only districts that intersect with test box
        return gdf[gdf.intersects(test_box_gdf.iloc[0].geometry)].copy()

    def _parse_geometry_from_csv(self, csv_df):
        """
        Parse geometries from CSV file with various formats.
        Handles WKT, GeoJSON text, and basic coordinate parsing.
        """
        if self.geometry_column and self.geometry_column in csv_df.columns:
            try:
                geometries = csv_df[self.geometry_column].apply(
                    lambda x: wkt.loads(x) if isinstance(x, str) else None
                )
                if not geometries.isna().all():
                    return geometries
            except Exception as e:
                print(f"Failed to parse geometries as WKT: {e}")

            # Try parsing as GeoJSON-like text
            try:

                def parse_geojson(geom_str):
                    try:
                        if isinstance(geom_str, str):
                            geom_dict = json.loads(geom_str)
                            return shape(geom_dict)
                        return None
                    except:
                        return None

                geometries = csv_df[self.geometry_column].apply(parse_geojson)
                if not geometries.isna().all():
                    return geometries
            except Exception as e:
                print(f"Failed to parse geometries as GeoJSON: {e}")

        # Look for known geometry column patterns
        for geom_col in ["the_geom", "geometry", "wkt", "geom", "GEOMETRY", "SHAPE"]:
            if geom_col in csv_df.columns:
                try:
                    geometries = csv_df[geom_col].apply(
                        lambda x: wkt.loads(x) if isinstance(x, str) else None
                    )
                    if not geometries.isna().all():
                        return geometries
                except Exception:
                    pass

        # Try detecting common formats like MULTIPOLYGON ((...))
        if "the_geom" in csv_df.columns:
            try:

                def parse_polygon_str(geom_str):
                    if not isinstance(geom_str, str):
                        return None

                    try:
                        if "MULTIPOLYGON" in geom_str.upper():
                            coords_str = (
                                geom_str.replace("MULTIPOLYGON (((", "")
                                .replace("MULTIPOLYGON(((", "")
                                .replace("))))", "")
                                .replace(")))", "")
                            )

                            coord_pairs = [
                                pair.strip() for pair in coords_str.split(",")
                            ]
                            coords = []
                            for pair in coord_pairs:
                                values = pair.split()
                                if len(values) >= 2:
                                    coords.append((float(values[0]), float(values[1])))

                            return Polygon(coords) if coords else None

                        elif "POLYGON" in geom_str.upper():
                            coords_str = (
                                geom_str.replace("POLYGON ((", "")
                                .replace("POLYGON((", "")
                                .replace("))", "")
                                .replace(")", "")
                            )

                            coord_pairs = [
                                pair.strip() for pair in coords_str.split(",")
                            ]
                            coords = []
                            for pair in coord_pairs:
                                values = pair.split()
                                if len(values) >= 2:
                                    coords.append((float(values[0]), float(values[1])))

                            return Polygon(coords) if coords else None
                    except Exception as e:
                        print(f"Failed to parse geometry: {e}")
                        return None

                geometries = csv_df["the_geom"].apply(parse_polygon_str)
                if not geometries.isna().all():
                    return geometries
            except Exception as e:
                print(f"Failed to parse polygon strings: {e}")

        lat_candidates = ["latitude", "lat", "y", "LATITUDE", "LAT", "Y"]
        lon_candidates = [
            "longitude",
            "lon",
            "long",
            "x",
            "LONGITUDE",
            "LONG",
            "LON",
            "X",
        ]

        for lat_col in lat_candidates:
            if lat_col in csv_df.columns:
                for lon_col in lon_candidates:
                    if lon_col in csv_df.columns:
                        try:
                            return gpd.points_from_xy(
                                csv_df[lon_col].astype(float),
                                csv_df[lat_col].astype(float),
                            )
                        except Exception:
                            pass

        print(
            "Failed to find or parse geometry columns in CSV. Please specify geometry_column parameter."
        )
        return None

    def _load_administrative_polygons(self):
        """
        Load administrative polygon boundaries from file or GeoDataFrame.
        Supports ShapeFiles, GeoJSON, CSV with geometry columns, and direct GeoDataFrames.
        """
        if self.districts_gdf is not None:
            print("Using provided GeoDataFrame for administrative districts")
            districts_gdf = self.districts_gdf.copy()

            if districts_gdf.crs is None:
                districts_gdf = districts_gdf.set_crs(self.input_crs)

            return districts_gdf

        districts_file = self.districts_file

        if not districts_file:
            raise ValueError("No district file or GeoDataFrame provided")

        if not os.path.exists(districts_file):
            raise FileNotFoundError(f"Districts file not found: {districts_file}")

        print(f"Loading administrative districts from {districts_file}...")

        file_ext = os.path.splitext(districts_file)[1].lower()

        if file_ext in [".shp", ".geojson", ".json", ".gpkg"]:
            districts_gdf = gpd.read_file(districts_file)
        elif file_ext in [".csv", ".txt"]:
            df = pd.read_csv(districts_file)
            geometries = self._parse_geometry_from_csv(df)

            if geometries is None:
                raise ValueError(f"Could not parse geometries from {districts_file}")

            districts_gdf = gpd.GeoDataFrame(df, geometry=geometries)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        if self.name_column and self.name_column in districts_gdf.columns:
            districts_gdf["name"] = districts_gdf[self.name_column]
        elif "name" not in districts_gdf.columns:
            name_candidates = [
                "NAME",
                "Name",
                "LABEL",
                "Label",
                "DISTRICT",
                "District",
                "ID",
                "Id",
            ]
            for col in name_candidates:
                if col in districts_gdf.columns:
                    districts_gdf["name"] = districts_gdf[col]
                    break
            else:
                districts_gdf["name"] = [
                    f"District {i}" for i in range(len(districts_gdf))
                ]

        if districts_gdf.crs is None:
            districts_gdf = districts_gdf.set_crs(self.input_crs)

        if districts_gdf.crs.to_string() != self.input_crs:
            districts_gdf = districts_gdf.to_crs(self.input_crs)

        districts_gdf = districts_gdf.to_crs(self.meter_crs)

        districts_gdf = self._filter_by_test_bounds(districts_gdf)

        districts_gdf = districts_gdf[~districts_gdf.geometry.is_empty].copy()
        districts_gdf = districts_gdf[districts_gdf.geometry.is_valid].copy()
        districts_gdf.reset_index(drop=True, inplace=True)

        print(f"Loaded {len(districts_gdf)} administrative districts")

        return districts_gdf

    def _assign_points_to_polygons(self, points_gdf, polygons_gdf):
        """
        Spatially join each point to 'polygons_gdf', returning
        a Series 'point_to_polygon' with polygon IDs (index in polygons_gdf).
        Points not in any polygon get -1.
        """
        pts_proj = (
            points_gdf.to_crs(self.meter_crs)
            if points_gdf.crs != self.meter_crs
            else points_gdf
        )

        joined = gpd.sjoin(pts_proj, polygons_gdf, how="left", predicate="within")

        point_to_polygon = joined["index_right"].fillna(-1).astype(int)
        return point_to_polygon
