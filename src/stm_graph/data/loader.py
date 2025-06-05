"""
Preprocessing utilities for spatial-temporal event datasets.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path
from matplotlib.dates import DateFormatter
from .reader import read_event_dataset


def preprocess_dataset(
    data_path: str,
    dataset: str,
    time_col: str = "time",
    lat_col: str = "latitude",
    lng_col: str = "longitude",
    crs: str = "EPSG:4326",
    vis_crs: str = "EPSG:3857",
    testing_mode: bool = False,
    test_bounds: Optional[Union[str, List, Tuple, Dict]] = None,
    column_mapping: Optional[Dict] = None,
    filter_dates: Optional[Tuple[str, str]] = None,
    required_columns: Optional[List[str]] = None,
    drop_duplicates: bool = True,
    drop_na_columns: bool = True,
    time_format: Optional[str] = None,
    visualize: bool = True,
    output_dir: Optional[str] = None,
    show_background_map: bool = True,
    point_color: str = "red",
    point_alpha: float = 0.5,
    point_size: float = 1,
    fig_format: str = "png",
    fig_dpi: int = 1200,
    rasterized: bool = False,
) -> gpd.GeoDataFrame:
    """
    Preprocess a dataset according to specified parameters.

    Args:
        data_path: Path to the directory containing the dataset
        dataset: File name of the dataset
        time_col: Name of the column containing event timestamps
        lat_col: Name of the column containing latitude
        lng_col: Name of the column containing longitude
        crs: Coordinate reference system
        vis_crs: Coordinate reference system for visualization in meters
        testing_mode: Whether running in testing mode
        test_bounds: Geographic bounds for testing
        column_mapping: Dictionary mapping original column names to desired names
        filter_dates: Tuple of (start_date, end_date) strings to filter the dataset
        required_columns: List of columns that must be present in the dataset
        drop_duplicates: Whether to drop duplicate rows
        drop_na_columns: Whether to drop rows with NA values
        time_format: Format string for parsing timestamps
        visualize: Whether to generate and save visualizations
        output_dir: Directory to save visualizations (required if visualize=True)
        show_background_map: Whether to include a background map in spatial visualization
        point_color: Color for points in spatial visualization
        point_alpha: Transparency for points (0-1)
        point_size: Size of points in visualization
        fig_dpi: Resolution in dots per inch for raster formats
        fig_format: Format to save figures ('png', 'pdf', 'svg', 'eps')
        rasterized: Whether to rasterize the plot for faster rendering
        
    Returns:
        GeoDataFrame with preprocessed event data
    """
    # Construct full path to dataset
    data_file = os.path.join(data_path, dataset)

    # Parse test bounds if in testing mode
    bounds = None
    if testing_mode and test_bounds:
        if isinstance(test_bounds, str):
            bound_values = [float(x) for x in test_bounds.split(",")]
            bounds = {
                "min_lat": bound_values[0],
                "max_lat": bound_values[2],
                "min_lon": bound_values[1],
                "max_lon": bound_values[3],
            }
        elif isinstance(test_bounds, (list, tuple)) and len(test_bounds) >= 4:
            bounds = {
                "min_lat": test_bounds[0],
                "max_lat": test_bounds[2],
                "min_lon": test_bounds[1],
                "max_lon": test_bounds[3],
            }
        elif isinstance(test_bounds, dict):
            bounds = test_bounds

        print("Running in testing mode with bounds:", bounds)

    # Read and preprocess the dataset
    gdf = read_event_dataset(
        data_path=data_file,
        time_col=time_col,
        lat_col=lat_col,
        lon_col=lng_col,
        column_mapping=column_mapping,
        time_format=time_format,
        bounds=bounds,
        filter_dates=filter_dates,
        required_columns=required_columns,
        drop_duplicates=drop_duplicates,
        drop_na_columns=drop_na_columns,
        crs=crs,
    )

    # Generate visualizations if requested
    if visualize:
        if output_dir is None:
            print(
                "Warning: output_dir is required for visualization. Skipping visualization."
            )
        else:
            visualize_preprocessed_data(
                gdf=gdf,
                time_col=time_col,
                output_dir=output_dir,
                show_background_map=show_background_map,
                point_color=point_color,
                point_alpha=point_alpha,
                point_size=point_size,
                vis_crs=vis_crs,
                fig_format=fig_format,
                fig_dpi=fig_dpi,
                rasterized=rasterized
            )

    return gdf


def visualize_preprocessed_data(
    gdf: gpd.GeoDataFrame,
    time_col: str,
    output_dir: str,
    show_background_map: bool = True,
    point_color: str = "red",
    point_alpha: float = 0.5,
    point_size: float = 1,
    vis_crs: str = "EPSG:3857",
    fig_format: str = "png",
    fig_dpi: int = 1200,
    spatial_figsize: Tuple[int, int] = (12, 10),
    temporal_figsize: Tuple[int, int] = (12, 6),
    rasterized: bool = False,
) -> None:
    """
    Generate and save visualizations for the preprocessed dataset.

    Args:
        gdf: GeoDataFrame with preprocessed event data
        time_col: Name of the column containing event timestamps
        output_dir: Directory to save visualizations
        show_background_map: Whether to include a background map
        point_color: Color for points in spatial visualization
        point_alpha: Transparency for points (0-1)
        point_size: Size of points in visualization
        vis_crs: Coordinate reference system for visualization
        fig_format: Format to save figures ('png', 'pdf', 'svg', 'eps')
        fig_dpi: Resolution in dots per inch for raster formats
        spatial_figsize: Figure size (width, height) for spatial plot
        temporal_figsize: Figure size (width, height) for temporal plot
    """
    vis_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Generating visualizations in {vis_dir}...")

    # 1. Spatial visualization (point map)
    try:
        if show_background_map and gdf.crs != vis_crs:
            gdf_map = gdf.to_crs(vis_crs)
        else:
            gdf_map = gdf.copy()

        fig, ax = plt.subplots(figsize=spatial_figsize)
        
        gdf_map.plot(ax=ax, markersize=point_size, alpha=point_alpha, color=point_color, rasterized=rasterized)

        if show_background_map:
            try:
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
            except Exception as e:
                print(f"Warning: Could not add background map: {e}")

        ax.set_title(f"Spatial Distribution of Events ({len(gdf)} points)")
        ax.set_axis_off()
        plt.tight_layout()

        spatial_viz_path = os.path.join(vis_dir, f"spatial_distribution.{fig_format}")
        plt.savefig(
            spatial_viz_path, dpi=fig_dpi, bbox_inches="tight", format=fig_format
        )
        print(f"Spatial visualization saved to {spatial_viz_path}")
        plt.close()

    except Exception as e:
        print(f"Error creating spatial visualization: {e}")

    # 2. Temporal visualization (time series)
    try:
        if time_col in gdf.columns and pd.api.types.is_datetime64_any_dtype(gdf[time_col]):
            fig, ax = plt.subplots(figsize=temporal_figsize)
            vis_df = gdf.copy()
            vis_df[time_col] = pd.to_datetime(vis_df[time_col])
            
            try:
                vis_df["date"] = vis_df[time_col].dt.floor('D')
            except:
                vis_df["date"] = pd.to_datetime(vis_df[time_col].dt.strftime('%Y-%m-%d'))
            
            try:
                time_series = vis_df.groupby("date").size()
            except TypeError:
                try:
                    time_series = vis_df["date"].value_counts().sort_index()
                except:
                    counts = {}
                    for date in vis_df["date"]:
                        if date in counts:
                            counts[date] += 1
                        else:
                            counts[date] = 1
                    time_series = pd.Series(counts)

            if not isinstance(time_series.index, pd.DatetimeIndex):
                time_series.index = pd.to_datetime(time_series.index)

            time_series.plot(ax=ax, linewidth=2)

            ax.set_title("Event Counts Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Events")
            ax.grid(True, alpha=0.3)

            plt.gcf().autofmt_xdate()
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

            threshold = time_series.mean() + time_series.std()
            peaks = time_series[time_series > threshold]
            for date, count in peaks.items():
                ax.annotate(
                    f"{count}",
                    xy=(date, count),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

            plt.tight_layout()
            time_viz_path = os.path.join(vis_dir, f"temporal_distribution.{fig_format}")
            plt.savefig(
                time_viz_path, dpi=fig_dpi, bbox_inches="tight", format=fig_format
            )
            print(f"Temporal visualization saved to {time_viz_path}")
            plt.close()

        else:
            print(
                f"Warning: Time column '{time_col}' not found in data. Skipping temporal visualization."
            )
    except Exception as e:
        print(f"Error creating temporal visualization: {e}")


def move_batch_to_device(batch, device):
    """Move batch data to device."""
    result = {}
    for key, value in batch.items():
        if value is None:
            result[key] = None
        elif isinstance(value, np.ndarray):
            tensor_type = torch.long if key == "edge_index" else torch.float32
            result[key] = torch.tensor(value, dtype=tensor_type).to(device)
        elif isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            try:
                result[key] = torch.tensor(value, dtype=torch.float32).to(device)
            except:
                result[key] = value
                print(f"Warning: Could not move {key} to device")
    return result
