import geopandas as gpd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import contextily as ctx
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, Union


class BaseMapping(ABC):
    """
    Base class for all spatial mapping methods.

    This abstract class defines the interface that all mapping implementations
    should follow, including creating mappings and visualizing results.
    """

    def __init__(self, name: str = "base", **kwargs):
        """
        Initialize the mapping method with optional parameters.

        Args:
            name: Identifier for the mapping method
            **kwargs: Additional parameters specific to the mapping method
        """
        self.name = name
        self.params = kwargs

    @abstractmethod
    def create_mapping(
        self, points_gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, pd.Series]:
        """
        Create a mapping from points to partitions.

        Args:
            points_gdf: GeoDataFrame with point data

        Returns:
            Tuple of (partition_gdf, point_to_partition)
            - partition_gdf: GeoDataFrame with partition geometries
            - point_to_partition: Series mapping point indices to partition indices
        """
        pass

    def visualize(
        self,
        points_gdf: gpd.GeoDataFrame,
        partition_gdf: gpd.GeoDataFrame,
        point_to_partition: pd.Series,
        out_dir: str = "visualizations",
        remove_empty: bool = True,
        testing_mode: bool = False,
        plot_crs: str = "EPSG:3857",
        file_format: str = "png",
        fig_dpi: int = 300,
        rasterized: bool = False,
        **kwargs,
    ) -> None:
        """
        Visualize the mapping.

        Args:
            points_gdf: GeoDataFrame with point data
            partition_gdf: GeoDataFrame with partition geometries
            point_to_partition: Series mapping point indices to partition indices
            out_dir: Directory to save visualization outputs
            remove_empty: Whether to remove empty regions in visualization
            testing_mode: Whether running in testing mode (affects visualization settings)
            plot_crs: CRS to use for plotting
            file_format: Format to save figures in (svg, pdf, png, etc)
            fig_dpi: DPI for saved figures
            rasterized: Whether to rasterize the plot for faster rendering
            **kwargs: Additional visualization parameters
        """
        out_dir = os.path.join(out_dir, "mapping")
        os.makedirs(out_dir, exist_ok=True)

        points_plot = points_gdf.to_crs(plot_crs)
        regions_plot = partition_gdf.to_crs(plot_crs)

        # Get bounds for consistent view across plots
        bounds = points_plot.total_bounds
        x_pad = (bounds[2] - bounds[0]) * 0.1
        y_pad = (bounds[3] - bounds[1]) * 0.1

        # Count points per region
        region_counts = point_to_partition.value_counts()
        regions_plot["point_count"] = 0

        for region_id, count in region_counts.items():
            if region_id >= 0 and region_id in regions_plot.index:
                regions_plot.loc[region_id, "point_count"] = count

        # Adjust visualization parameters based on test mode
        zoom = 15 if testing_mode else 12
        point_size = 3 if testing_mode else 0.5

        # Use a standard map style
        map_style = ctx.providers.CartoDB.PositronNoLabels

        # Get importance nodes for mapping methods like Voronoi (if available)
        importance_nodes = kwargs.get("importance_nodes", None)
        if importance_nodes is not None:
            nodes_plot = importance_nodes.to_crs(plot_crs)

        # Plot 1: All regions with point counts
        fig, ax = plt.subplots(figsize=(12, 12))

        ax.set_xlim([bounds[0] - x_pad, bounds[2] + x_pad])
        ax.set_ylim([bounds[1] - y_pad, bounds[3] + y_pad])

        try:
            ctx.add_basemap(ax, source=map_style, zoom=zoom, crs=plot_crs)
        except Exception as e:
            print(f"Warning: Could not add basemap: {e}")

        regions_plot.plot(
            column="point_count",
            ax=ax,
            alpha=0.5,
            edgecolor="black",
            linewidth=0.5,
            legend=True,
            legend_kwds={"label": f"Points per {self.name} region"},
            zorder=2,
            rasterized=rasterized,
        )

        points_plot.plot(ax=ax, color="red", markersize=point_size, alpha=0.5, zorder=3, rasterized=rasterized)

        plt.title(f"{self.name.title()} Mapping (All Regions)")
        plt.axis("off")
        plt.savefig(
            os.path.join(out_dir, f"all_regions.{file_format}"),
            bbox_inches="tight",
            format=file_format,
            dpi=fig_dpi
        )
        plt.close()

        # Plot 2: Only regions with points (if requested)
        if remove_empty:
            active_ids = [idx for idx in region_counts.index if idx >= 0]
            active_regions = (
                regions_plot.loc[active_ids].copy()
                if active_ids
                else regions_plot.copy()
            )

            fig, ax = plt.subplots(figsize=(12, 12))

            ax.set_xlim([bounds[0] - x_pad, bounds[2] + x_pad])
            ax.set_ylim([bounds[1] - y_pad, bounds[3] + y_pad])

            try:
                ctx.add_basemap(ax, source=map_style, zoom=zoom, crs=plot_crs)
            except Exception as e:
                print(f"Warning: Could not add basemap: {e}")

            active_regions.plot(
                column="point_count",
                ax=ax,
                alpha=0.5,
                edgecolor="black",
                linewidth=0.5,
                legend=True,
                legend_kwds={"label": f"Points per {self.name} region (Active only)"},
                zorder=2,
                rasterized=rasterized,
            )

            points_plot.plot(
                ax=ax, color="red", markersize=point_size, alpha=0.5, zorder=3, rasterized=rasterized
            )

            plt.title(f"{self.name.title()} Mapping - Active Regions Only")
            plt.axis("off")
            plt.savefig(
                os.path.join(out_dir, f"active_regions.{file_format}"),
                bbox_inches="tight",
                format=file_format,
                dpi=fig_dpi
            )
            plt.close()

        # Plot 3: Point density heatmap
        fig, ax = plt.subplots(figsize=(12, 12))

        ax.set_xlim([bounds[0] - x_pad, bounds[2] + x_pad])
        ax.set_ylim([bounds[1] - y_pad, bounds[3] + y_pad])

        try:
            ctx.add_basemap(ax, source=map_style, zoom=zoom, crs=plot_crs)
        except Exception as e:
            print(f"Warning: Could not add basemap: {e}")

        regions_plot.plot(
            column="point_count",
            cmap="viridis",
            ax=ax,
            legend=True,
            legend_kwds={"label": "Points per region"},
            edgecolor="black",
            linewidth=0.2,
            alpha=0.7,
            zorder=2,
            rasterized=rasterized,
        )

        plt.title(f"{self.name.title()} Region Point Density")
        plt.axis("off")
        plt.savefig(
            os.path.join(out_dir, f"point_density.{file_format}"),
            bbox_inches="tight",
            format=file_format,
            dpi=fig_dpi
        )
        plt.close()

        print(
            f"{self.name.title()} mapping visualizations saved to {out_dir} in {file_format.upper()} format!"
        )

    def compute_adjacency(self, partition_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """
        Compute adjacency matrix for partitions.

        Args:
            partition_gdf: GeoDataFrame with partition geometries

        Returns:
            numpy array adjacency matrix
        """
        n = len(partition_gdf)
        adj_matrix = np.zeros((n, n))

        for i, geom_i in enumerate(partition_gdf.geometry):
            for j in range(i + 1, n):
                geom_j = partition_gdf.geometry.iloc[j]
                if geom_i.touches(geom_j):
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

        return adj_matrix

    def filter_empty_partitions(
        self, partition_gdf: gpd.GeoDataFrame, point_to_partition: pd.Series
    ) -> Tuple[gpd.GeoDataFrame, pd.Series]:
        """
        Filter out partitions that don't contain any points.

        Args:
            partition_gdf: GeoDataFrame with partition geometries
            point_to_partition: Series mapping point indices to partition indices

        Returns:
            Tuple of (filtered_gdf, updated_point_to_partition)
        """
        active_partitions = set(point_to_partition[point_to_partition >= 0].unique())

        filtered_gdf = (
            partition_gdf.loc[list(active_partitions)].copy().reset_index(drop=True)
        )

        old_to_new = {
            old_idx: new_idx for new_idx, old_idx in enumerate(active_partitions)
        }

        updated_mapping = point_to_partition.copy()
        for i, val in enumerate(updated_mapping):
            if val >= 0:
                updated_mapping.iloc[i] = old_to_new.get(val, -1)

        return filtered_gdf, updated_mapping
