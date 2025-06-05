import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import os
from shapely.geometry import Polygon
from typing import Tuple
from .base import BaseMapping


class GridMapping(BaseMapping):
    """
    Grid-based spatial mapping.

    This class implements a regular grid mapping where the space is divided into
    equal-sized square cells.
    """

    def __init__(self, cell_size=3000.0, target_crs="EPSG:3857", **kwargs):
        """
        Initialize grid mapping.

        Args:
            cell_size: Size of grid cells in meters
            target_crs: Coordinate reference system for projected coordinates
            **kwargs: Additional parameters
        """
        super().__init__(name="grid", **kwargs)
        self.cell_size = cell_size
        self.target_crs = target_crs

    def create_mapping(
        self, points_gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, pd.Series]:
        """
        Create a grid mapping from points.

        Args:
            points_gdf: GeoDataFrame with point geometries

        Returns:
            Tuple of (grid_gdf, point_to_cell)
            - grid_gdf: GeoDataFrame with grid cell polygons
            - point_to_cell: Series mapping point indices to cell indices
        """
        if points_gdf.crs is None:
            points_gdf = points_gdf.set_crs(self.target_crs)

        if points_gdf.crs.to_string() != self.target_crs:
            points_gdf = points_gdf.to_crs(self.target_crs)

        points_xy = np.column_stack([points_gdf.geometry.x, points_gdf.geometry.y])

        # Create grid
        grid_gdf = self._create_square_grid(points_xy, self.cell_size)

        # Map points to grid cells
        joined = gpd.sjoin(points_gdf, grid_gdf, how="left", predicate="within")
        point_to_cell = joined["index_right"].fillna(-1).astype(int)

        return grid_gdf, point_to_cell

    def _create_square_grid(self, points_xy, cell_size=3000.0, padding_scale=0.001):
        """Create a square grid over the given points."""
        padding = cell_size * padding_scale
        min_x, min_y = np.min(points_xy, axis=0) - padding
        max_x, max_y = np.max(points_xy, axis=0) + padding
        nx = int(np.round((max_x - min_x) / cell_size))
        ny = int(np.round((max_y - min_y) / cell_size))
        x = np.linspace(min_x, min_x + nx * cell_size, nx + 1)
        y = np.linspace(min_y, min_y + ny * cell_size, ny + 1)

        polygons = []
        for i in range(nx):
            for j in range(ny):
                polygons.append(
                    Polygon(
                        [
                            (x[i], y[j]),
                            (x[i + 1], y[j]),
                            (x[i + 1], y[j + 1]),
                            (x[i], y[j + 1]),
                        ]
                    )
                )
        grid_gdf = gpd.GeoDataFrame(geometry=polygons, crs=self.target_crs).reset_index(
            drop=True
        )
        return grid_gdf
