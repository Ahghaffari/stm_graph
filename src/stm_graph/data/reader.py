"""
Generalized dataset reader for spatial-temporal event datasets.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union


def read_event_dataset(
    data_path: str,
    time_col: str,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    column_mapping: Optional[Dict[str, str]] = None,
    time_format: Optional[str] = None,
    bounds: Optional[Dict[str, float]] = None,
    filter_dates: Optional[Tuple[str, str]] = None,
    required_columns: Optional[List[str]] = None,
    drop_duplicates: bool = True,
    drop_na_columns: bool = True,
    crs: str = "EPSG:4326",
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Read and perform initial preprocessing on a spatial-temporal event dataset.

    Args:
        data_path: Path to the CSV dataset
        time_col: Name of the column containing event timestamps
        lat_col: Name of the column containing latitude
        lon_col: Name of the column containing longitude
        column_mapping: Dictionary mapping original column names to desired names
        time_format: Format string for parsing timestamps (if None, uses pandas default)
        bounds: Dictionary with 'min_lat', 'max_lat', 'min_lon', 'max_lon' for spatial filtering
        filter_dates: Tuple of (start_date, end_date) strings to filter the dataset
        required_columns: List of columns that must be present after preprocessing
        drop_duplicates: Whether to drop duplicate rows
        drop_na_columns: Whether to drop columns that are all NaN
        crs: Coordinate reference system

    Returns:
        GeoDataFrame with preprocessed event data
    """
    print(f"Reading dataset from {data_path}...")

    # Determine file extension
    file_ext = os.path.splitext(data_path)[1].lower()
    if file_ext in [".xlsx", ".xls"]:
        print(f"Reading Excel file: {data_path}")
        df = pd.read_excel(data_path)
    elif file_ext in [".csv"]:
        print(f"Reading CSV file: {data_path}")
        df = pd.read_csv(data_path, low_memory=False)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")
    print(f"Original dataset shape: {df.shape}")

    if column_mapping:
        df = df.rename(columns=column_mapping)
        time_col = column_mapping.get(time_col, time_col)
        lat_col = column_mapping.get(lat_col, lat_col)
        lon_col = column_mapping.get(lon_col, lon_col)

    essential_columns = [time_col, lat_col, lon_col]
    missing_columns = [col for col in essential_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing from dataset: {missing_columns}")

    if time_col in df.columns:
        if time_format:
            df[time_col] = pd.to_datetime(
                df[time_col], format=time_format, errors="coerce"
            )
        else:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        df = df.dropna(subset=[time_col])

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    df = df.dropna(subset=[lat_col, lon_col])

    if bounds:
        df = filter_by_bounds(df, lat_col, lon_col, bounds)
        
    if filter_dates:
        start_date, end_date = filter_dates
        df = filter_by_time(df, time_col, start_date, end_date)

    if drop_duplicates and len(df) > 0:
        if "unique_key" in df.columns or "id" in df.columns:
            id_col = "unique_key" if "unique_key" in df.columns else "id"
            df = df.drop_duplicates(subset=[id_col])
        else:
            df = df.drop_duplicates()

    if drop_na_columns:
        df = df.dropna(axis=1, how="all")

    if required_columns:
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            raise ValueError(
                f"Required columns missing after preprocessing: {missing_required}"
            )

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=crs
    )

    if target_crs and crs != target_crs:
        print(f"Converting coordinates from {crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)

        df[lon_col] = gdf.geometry.x
        df[lat_col] = gdf.geometry.y

        gdf = gpd.GeoDataFrame(df, geometry=gdf.geometry, crs=target_crs)

    if time_col in gdf.columns:
        print("Sorting dataset by timestamp (ascending order)...")
        gdf = gdf.sort_values(by=time_col)

    print(f"Final dataset shape: {gdf.shape}")
    print(f"Time range: {gdf[time_col].min()} to {gdf[time_col].max()}")

    return gdf


def filter_by_bounds(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    bounds: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Filter dataframe by geographic bounds.

    Args:
        df: DataFrame containing coordinates
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        bounds: Dictionary with 'min_lat', 'max_lat', 'min_lon', 'max_lon'

    Returns:
        Filtered DataFrame
    """
    if bounds is None:
        return df

    filtered_df = df[
        (df[lat_col] >= bounds["min_lat"])
        & (df[lat_col] <= bounds["max_lat"])
        & (df[lon_col] >= bounds["min_lon"])
        & (df[lon_col] <= bounds["max_lon"])
    ]

    print(f"Filtered by bounds: {len(df)} → {len(filtered_df)} rows")
    return filtered_df


def filter_by_time(
    df: pd.DataFrame, time_col: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Filter dataframe by time range.

    Args:
        df: DataFrame containing timestamps
        time_col: Name of time column
        start_date: Start date string (inclusive)
        end_date: End date string (inclusive)

    Returns:
        Filtered DataFrame
    """
    original_len = len(df)

    # Apply start date filter if provided
    if start_date is not None:
        start = (
            pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        )
        df = df[df[time_col] >= start]

    # Apply end date filter if provided
    if end_date is not None:
        end = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
        df = df[df[time_col] <= end]

    filtered_len = len(df)

    print(f"Filtered by time range: {original_len} → {filtered_len} rows")
    if filtered_len > 0:
        print(f"New time range: {df[time_col].min()} to {df[time_col].max()}")

    return df
