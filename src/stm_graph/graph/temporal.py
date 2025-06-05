"""
Temporal graph data processing functions.

This module provides utilities to create temporal features, integrate static features,
normalize data, and build PyTorch Geometric Temporal datasets.
"""

import torch
import numpy as np
import pandas as pd
import os
import gc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


def save_static_temporal_data(dataset, out_file="temporal_graph_data.pt"):
    """Saves the StaticGraphTemporalSignal to a .pt file."""
    torch.save(dataset, out_file)
    print(f"Saved temporal graph dataset to: {out_file}")


def load_static_temporal_data(in_file="temporal_graph_data.pt"):
    """Loads a previously saved StaticGraphTemporalSignal."""
    return torch.load(in_file)


def build_time_binned_features_4d(
    df,
    cell_col="cell_id",
    time_col="created_time",
    bin_type="hourly",
    interval_hours=1,
    all_node_ids=None,
    history_window=24,
    use_time_features=True,
):
    """
    Creates time-binned features with explicit history dimension.
    Returns 4D array: [num_intervals, num_nodes, history_window, base_features].

    Args:
        df: DataFrame with time and cell data
        cell_col: Column name for cell IDs
        time_col: Column name for timestamps
        bin_type: "hourly" or "daily" binning
        interval_hours: Size of each time bin in hours (for hourly binning)
        all_node_ids: Optional list of all node IDs to include
        history_window: Number of historical time steps to include
        use_time_features: Whether to include time-based features

    Returns:
        4D numpy array of shape [num_intervals, num_nodes, history_window, base_features]
    """
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Create time bins based on bin_type
    if bin_type == "daily":
        df["time_bin"] = df[time_col].dt.floor("D")  # Daily binning at midnight
        print(f"Using daily time bins")
    elif bin_type == "weekly":
        df["time_bin"] = df[time_col].dt.floor(
            "W-MON"
        )  # Weekly binning starting Monday
        print(f"Using weekly time bins")
    else:
        df["time_bin"] = df[time_col].dt.floor(f"{interval_hours}h")  # Hourly binning
        print(f"Using hourly time bins of {interval_hours} hours each")

    try:
        grouped = df.groupby([cell_col, "time_bin"]).size().reset_index(name="count")
    except TypeError:
        try:
            count_col = df.columns[0]
            grouped = df.groupby([cell_col, "time_bin"])[count_col].count().reset_index(name="count")
        except:
            grouped = df.groupby([cell_col, "time_bin"]).apply(lambda x: len(x)).reset_index(name="count")
    pivot_df = grouped.pivot(index=cell_col, columns="time_bin", values="count").fillna(
        0
    )

    non_empty_mask = pivot_df.sum(axis=1) > 0
    pivot_df = pivot_df[non_empty_mask]

    if all_node_ids is not None:
        pivot_df = pivot_df.reindex(all_node_ids, fill_value=0)

    time_bins = pd.to_datetime(pivot_df.columns)

    most_recent = time_bins.max()
    time_deltas = (most_recent - time_bins).total_seconds() / 3600.0
    time_deltas = (
        time_deltas / time_deltas.max() if time_deltas.max() > 0 else time_deltas
    )

    if use_time_features:
        base_features = 8
    else:
        base_features = 1
        print("Time features disabled: using only count features")

    node_ids = pivot_df.index.to_numpy()
    count_matrix = pivot_df.values.T  # [T, N]

    num_timesteps = len(time_bins)
    num_nodes = len(node_ids)

    # Create 4D feature tensor [T, N, H, F]
    features_4d = np.zeros((num_timesteps, num_nodes, history_window, base_features))

    # Prepare time features for all time steps if needed
    if use_time_features:
        if bin_type == "daily":
            # For daily binning, focus on day of week, day of month, month features
            day_of_week_sin = np.sin(2 * np.pi * time_bins.dayofweek / 7.0)
            day_of_week_cos = np.cos(2 * np.pi * time_bins.dayofweek / 7.0)
            day_of_month_sin = np.sin(2 * np.pi * time_bins.day / 31.0)
            day_of_month_cos = np.cos(2 * np.pi * time_bins.day / 31.0)
            month_sin = np.sin(2 * np.pi * time_bins.month / 12.0)
            month_cos = np.cos(2 * np.pi * time_bins.month / 12.0)

            # Time features for each timestep [T, 7]
            all_time_features = np.stack(
                [
                    day_of_week_sin,
                    day_of_week_cos,
                    day_of_month_sin,
                    day_of_month_cos,
                    month_sin,
                    month_cos,
                    1 - time_deltas,  # Normalized time delta (1 = most recent)
                ],
                axis=1,
            )
        else:
            # For hourly binning, include hour of day features
            hour_sin = np.sin(2 * np.pi * time_bins.hour / 24.0)
            hour_cos = np.cos(2 * np.pi * time_bins.hour / 24.0)
            day_sin = np.sin(2 * np.pi * time_bins.dayofweek / 7.0)
            day_cos = np.cos(2 * np.pi * time_bins.dayofweek / 7.0)
            month_sin = np.sin(2 * np.pi * time_bins.month / 12.0)
            month_cos = np.cos(2 * np.pi * time_bins.month / 12.0)

            # Time features for each timestep [T, 7]
            all_time_features = np.stack(
                [
                    hour_sin,
                    hour_cos,
                    day_sin,
                    day_cos,
                    month_sin,
                    month_cos,
                    1 - time_deltas,  # Normalized time delta (1 = most recent)
                ],
                axis=1,
            )

    # Fill features for each timestep using historical data
    for t in range(num_timesteps):
        for h in range(history_window):
            t_hist = max(0, t - h)
            features_4d[t, :, h, 0] = count_matrix[t_hist]

            if use_time_features:
                for i in range(7):  # 7 time features
                    features_4d[t, :, h, i + 1] = all_time_features[t_hist, i]

    print(f"Feature dimensions:")
    print(f"- Total timesteps: {num_timesteps}")
    print(f"- Number of nodes: {num_nodes}")
    print(f"- History window: {history_window} steps")
    print(f"- Base features per step: {base_features}")
    print(f"- Shape: [timesteps, nodes, history, features] = {features_4d.shape}")
    print(f"- Time span: {time_bins.max() - time_bins.min()}")
    print(f"- Bin type: {bin_type}")

    print("\nFeature structure for each history step (h):")
    print(f"- Feature 0: Event counts")
    if use_time_features:
        if bin_type == "daily":
            print(f"- Features 1-2: Day of week (sin/cos)")
            print(f"- Features 3-4: Day of month (sin/cos)")
            print(f"- Features 5-6: Month (sin/cos)")
        else:
            print(f"- Features 1-2: Hour of day (sin/cos)")
            print(f"- Features 3-4: Day of week (sin/cos)")
            print(f"- Features 5-6: Month (sin/cos)")
        print(f"- Feature 7: Time delta")

    return features_4d


def convert_4d_to_3d_dataset(dataset, static_features_count=0):
    """
    Convert a temporal dataset from 4D format [T, N, H, F+S] to 3D format [T, N, H*F+S]
    where static features appear only once at the end of each feature vector.

    Args:
        dataset: A StaticGraphTemporalSignal dataset with 4D features [N, H, F+S]
        static_features_count: Number of static features at the end of each history step's features

    Returns:
        A new StaticGraphTemporalSignal dataset with 3D features
    """
    edge_index = dataset.edge_index
    edge_weight = dataset.edge_weight

    # Check if dataset is already in 3D format
    if not hasattr(dataset, "is_4d_format") or not dataset.is_4d_format:
        print("Dataset is already in 3D format, no conversion needed.")
        return dataset

    if isinstance(dataset.features[0], torch.Tensor):
        T = len(dataset.features)

        sample_feature = dataset.features[0]
        N, H, F = sample_feature.shape
        all_features = torch.stack([f for f in dataset.features])

        if static_features_count > 0:
            dynamic_features_count = F - static_features_count

            dynamic_part = all_features[:, :, :, :dynamic_features_count]
            static_part = all_features[
                :, :, 0, dynamic_features_count:
            ]  # Take static from first history step

            # Reshape dynamic features to 3D [T, N, H*dynamic_features]
            flattened_dynamic = dynamic_part.reshape(T, N, H * dynamic_features_count)

            features_3d = torch.zeros(
                T,
                N,
                H * dynamic_features_count + static_features_count,
                dtype=flattened_dynamic.dtype,
                device=flattened_dynamic.device,
            )

            features_3d[:, :, : H * dynamic_features_count] = flattened_dynamic

            features_3d[:, :, H * dynamic_features_count :] = static_part
        else:
            # No static features, just flatten history and feature dimensions
            features_3d = all_features.reshape(T, N, H * F)
    else:
        T = len(dataset.features)

        sample_feature = dataset.features[0]
        N, H, F = sample_feature.shape

        all_features = np.stack([np.array(f) for f in dataset.features])

        if static_features_count > 0:
            dynamic_features_count = F - static_features_count

            dynamic_part = all_features[:, :, :, :dynamic_features_count]
            static_part = all_features[:, :, 0, dynamic_features_count:]

            # Reshape dynamic features to 3D [T, N, H*dynamic_features]
            flattened_dynamic = dynamic_part.reshape(T, N, H * dynamic_features_count)

            features_3d = np.zeros(
                (T, N, H * dynamic_features_count + static_features_count),
                dtype=flattened_dynamic.dtype,
            )

            features_3d[:, :, : H * dynamic_features_count] = flattened_dynamic

            # Add static features at the end
            features_3d[:, :, H * dynamic_features_count :] = static_part
        else:
            # No static features, just flatten history and feature dimensions
            features_3d = all_features.reshape(T, N, H * F)

    features_3d_list = [features_3d[t] for t in range(T)]

    targets_3d = None
    if hasattr(dataset, "targets") and dataset.targets is not None:
        targets_3d = dataset.targets

    new_dataset = StaticGraphTemporalSignal(
        edge_index, edge_weight, features_3d_list, targets_3d
    )

    if hasattr(dataset, "history_window"):
        new_dataset.history_window = dataset.history_window
    if hasattr(dataset, "base_features"):
        new_dataset.base_features = dataset.base_features

    new_dataset.is_4d_format = False

    print(f"Converted dataset from 4D to 3D format")
    print(f"Original 4D shape: [T={T}, N={N}, H={H}, F={F}]")
    new_shape = features_3d_list[0].shape
    print(f"New 3D shape: [T={T}, N={new_shape[0]}, F={new_shape[1]}]")
    print(f"Static features: {static_features_count}")
    print(f"Dynamic features per history step: {F - static_features_count}")

    return new_dataset


def convert_3d_to_4d_dataset(
    dataset, history_window, features_per_timestep=None, static_features_count=0
):
    """
    Convert a temporal dataset from 3D format [T, N, F] to 4D format [T, N, H, F/H + static]

    Args:
        dataset: A StaticGraphTemporalSignal dataset with 3D features
        history_window: Number of historical timesteps in the data
        features_per_timestep: Number of features per timestep (excluding static).
                              If None, will be calculated as (F-static_features_count)/history_window
        static_features_count: Number of static features appended at the end of each feature vector

    Returns:
        A new StaticGraphTemporalSignal dataset with 4D features
    """
    # Copy dataset structure
    edge_index = dataset.edge_index
    edge_weight = dataset.edge_weight

    # Extract all features into a single tensor/array for efficient processing
    if isinstance(dataset.features[0], torch.Tensor):
        T = len(dataset.features)
        N, F = dataset.features[0].shape

        all_features = torch.stack(
            [
                torch.tensor(f) if not isinstance(f, torch.Tensor) else f
                for f in dataset.features
            ]
        )

        temporal_features = F - static_features_count
        if features_per_timestep is None:
            features_per_timestep = temporal_features // history_window

        if temporal_features % history_window != 0:
            raise ValueError(
                f"Temporal feature dimension {temporal_features} is not divisible by history_window {history_window}"
            )

        if static_features_count > 0:
            temporal_part = all_features[:, :, :-static_features_count]
            static_part = all_features[:, :, -static_features_count:]
        else:
            temporal_part = all_features
            static_part = None

        # Reshape temporal features to 4D [T, N, history_window, features_per_timestep]
        reshaped_temporal = temporal_part.reshape(
            T, N, history_window, features_per_timestep
        )

        if static_features_count > 0:
            # Create output tensor with space for temporal + static features
            features_4d = torch.zeros(
                T,
                N,
                history_window,
                features_per_timestep + static_features_count,
                dtype=reshaped_temporal.dtype,
                device=reshaped_temporal.device,
            )

            features_4d[:, :, :, :features_per_timestep] = reshaped_temporal

            for h in range(history_window):
                features_4d[:, :, h, features_per_timestep:] = static_part
        else:
            features_4d = reshaped_temporal

    else:
        T = len(dataset.features)
        N, F = dataset.features[0].shape

        all_features = np.stack([np.array(f) for f in dataset.features])

        temporal_features = F - static_features_count
        if features_per_timestep is None:
            features_per_timestep = temporal_features // history_window

        if temporal_features % history_window != 0:
            raise ValueError(
                f"Temporal feature dimension {temporal_features} is not divisible by history_window {history_window}"
            )

        # Split static and temporal features
        if static_features_count > 0:
            temporal_part = all_features[:, :, :-static_features_count]
            static_part = all_features[:, :, -static_features_count:]
        else:
            temporal_part = all_features
            static_part = None

        # Reshape temporal features to 4D [T, N, history_window, features_per_timestep]
        reshaped_temporal = temporal_part.reshape(
            T, N, history_window, features_per_timestep
        )

        if static_features_count > 0:
            features_4d = np.zeros(
                (T, N, history_window, features_per_timestep + static_features_count),
                dtype=reshaped_temporal.dtype,
            )

            features_4d[:, :, :, :features_per_timestep] = reshaped_temporal

            for h in range(history_window):
                features_4d[:, :, h, features_per_timestep:] = static_part
        else:
            features_4d = reshaped_temporal

    features_4d_list = [features_4d[t] for t in range(T)]

    targets_4d = None
    if hasattr(dataset, "targets") and dataset.targets is not None:
        targets_4d = dataset.targets

    new_dataset = StaticGraphTemporalSignal(
        edge_index, edge_weight, features_4d_list, targets_4d
    )

    if hasattr(dataset, "history_window"):
        new_dataset.history_window = dataset.history_window
    if hasattr(dataset, "base_features"):
        new_dataset.base_features = dataset.base_features

    new_dataset.is_4d_format = True

    print(f"Converted dataset from 3D to 4D format")
    print(f"Original shape: [T={T}, N={N}, F={F}]")
    new_shape = features_4d_list[0].shape
    print(f"New shape: [T={T}, N={new_shape[0]}, H={new_shape[1]}, F={new_shape[2]}]")
    print(f"Static features: {static_features_count}")

    return new_dataset


def integrate_static_features_4d(features_4d, static_features, node_ids):
    """
    Integrate static features into 4D time-dependent features.

    Args:
        features_4d: Features tensor of shape [T, N, H, F]
        static_features: DataFrame with static features for each node
        node_ids: List of node IDs corresponding to rows in features_4d

    Returns:
        Updated features tensor with static features [T, N, H, F+S]
    """
    if static_features is None or len(static_features) == 0:
        return features_4d

    print("Integrating static features into 4D temporal features...")

    T, N, H, F = features_4d.shape

    static_features_aligned = static_features.loc[node_ids]

    if len(static_features_aligned) != N:
        print(
            f"Warning: Static features count ({len(static_features_aligned)}) != Node count ({N})"
        )
        return features_4d

    static_array = static_features_aligned.values
    num_static_features = static_array.shape[1]

    integrated_features = np.zeros(
        (T, N, H, F + num_static_features), dtype=features_4d.dtype
    )

    integrated_features[:, :, :, :F] = features_4d

    for t in range(T):
        for h in range(H):
            integrated_features[t, :, h, F:] = static_array

    print(
        f"Added {num_static_features} static features to each time step and history window"
    )
    print(f"New features shape: {integrated_features.shape}")

    return integrated_features


def scale_features_4d(features_4d, scaler_type="minmax", per_feature=True):
    """
    Scale features to a standard range while preserving 4D structure.

    Args:
        features_4d: Features tensor of shape [T, N, H, F]
        scaler_type: Type of scaling ('minmax' or 'standard')
        per_feature: Whether to scale each feature independently

    Returns:
        Scaled features tensor [T, N, H, F]
    """
    T, N, H, F = features_4d.shape
    if scaler_type not in ["minmax", "standard"]:
        raise ValueError("scaler_type must be 'minmax' or 'standard'")

    print(f"Scaling 4D features using {scaler_type} scaling...")

    # Create appropriate scaler
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()

    features_2d = features_4d.reshape(-1, F)

    if per_feature:
        # Scale each feature independently
        for f in range(F):
            if np.min(features_2d[:, f]) != np.max(
                features_2d[:, f]
            ):  # Avoid constant features
                features_2d[:, f] = scaler.fit_transform(
                    features_2d[:, f].reshape(-1, 1)
                ).flatten()
    else:
        features_2d = scaler.fit_transform(features_2d)

    scaled_features = features_2d.reshape(T, N, H, F)

    return scaled_features


def prepare_forecasting_data_4d(
    features_4d, horizon=1, task="regression", first_feature_only=True
):
    """
    Prepare input features and target labels for forecasting tasks, preserving 4D structure.

    Args:
        features_4d: Features tensor of shape [T, N, H, F]
        horizon: Number of time steps ahead to forecast
        task: 'regression' or 'classification'
        first_feature_only: If True, uses only the first feature as the target

    Returns:
        Tuple of (x_4d, y_3d)
        - x_4d: Input features [T-horizon, N, H, F]
        - y_3d: Target labels [T-horizon, N, 1]
    """
    T, N, H, F = features_4d.shape

    if horizon >= T:
        raise ValueError(
            f"Horizon ({horizon}) must be less than total time steps ({T})"
        )

    x_4d = features_4d[:-horizon]

    if first_feature_only:
        y_3d = features_4d[horizon:, :, 0, 0:1]
    else:
        y_3d = features_4d[horizon:, :, 0, :]

    if task == "classification":
        y_3d = (y_3d > 0).astype(np.float32)
        positive_ratio = np.mean(y_3d)
        print(
            f"Converted to binary classification task: {positive_ratio*100:.2f}% positive samples"
        )

    print(
        f"Prepared 4D forecasting data: Input shape: {x_4d.shape}, Target shape: {y_3d.shape}"
    )
    return x_4d, y_3d


def convert_4d_to_3d_features(features_4d, static_features_count=0):
    """
    Convert 4D features tensor [T, N, H, F+S] to 3D format [T, N, H*F+S].

    Args:
        features_4d: 4D feature tensor
        static_features_count: Number of static features at the end of each history step

    Returns:
        3D feature tensor
    """
    T, N, H, F = features_4d.shape

    if static_features_count > 0:
        dynamic_features_count = F - static_features_count

        dynamic_part = features_4d[:, :, :, :dynamic_features_count]
        static_part = features_4d[
            :, :, 0, dynamic_features_count:
        ]  # Take static from first history step

        # Reshape dynamic features to 3D [T, N, H*dynamic_features]
        flattened_dynamic = dynamic_part.reshape(T, N, H * dynamic_features_count)

        # Create output array with space for flattened dynamic + static
        features_3d = np.zeros(
            (T, N, H * dynamic_features_count + static_features_count),
            dtype=features_4d.dtype,
        )

        features_3d[:, :, : H * dynamic_features_count] = flattened_dynamic

        features_3d[:, :, H * dynamic_features_count :] = static_part
    else:
        features_3d = features_4d.reshape(T, N, H * F)

    return features_3d


def build_static_temporal_data(
    edge_index,
    features_3d,
    edge_weights=None,
    labels_3d=None,
    downsample_factor=1,
    force_3d=False,
):
    """
    Build a StaticGraphTemporalSignal from features and edge index.

    Args:
        edge_index: Graph connectivity (2 x E)
        features_3d: Features tensor of shape [T, N, F] or [T, N, H, F] (4D)
        edge_weights: Optional edge weights (E)
        labels_3d: Target labels of shape [T, N, C]
        downsample_factor: Temporal downsampling factor

    Returns:
        StaticGraphTemporalSignal object
    """
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.LongTensor(edge_index)

    if edge_weights is not None and isinstance(edge_weights, np.ndarray):
        edge_weights = torch.FloatTensor(edge_weights)

    is_4d = len(features_3d.shape) == 4

    feature_list = []

    if is_4d:
        T, N, H, F = features_3d.shape
        for t in range(0, T, downsample_factor):
            feature_mat = torch.FloatTensor(features_3d[t])
            feature_list.append(feature_mat)
    else:
        T, N, F = features_3d.shape
        for t in range(0, T, downsample_factor):
            feature_mat = torch.FloatTensor(features_3d[t])
            feature_list.append(feature_mat)

    label_list = None
    if labels_3d is not None:
        label_list = []
        LT = labels_3d.shape[0]  # Number of label timesteps

        for t in range(0, LT, downsample_factor):
            label_mat = torch.FloatTensor(labels_3d[t])
            label_list.append(label_mat)

    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weights,
        features=feature_list,
        targets=label_list,
    )

    num_timesteps = len(feature_list)
    feature_shape = feature_list[0].shape
    edge_count = edge_index.shape[1]

    print(f"Created temporal graph dataset:")
    print(f"- Timesteps: {num_timesteps}")
    print(f"- Feature shape: {feature_shape}")
    print(f"- Edges: {edge_count}")
    if downsample_factor > 1:
        print(f"- Downsampled by factor of {downsample_factor}")

    dataset.is_4d_format = is_4d

    return dataset


def create_temporal_dataset(
    edge_index,
    augmented_df,
    edge_weights=None,
    node_ids=None,
    static_features=None,
    time_col="created_time",
    cell_col="cell_id",
    bin_type="hourly",
    interval_hours=1,
    history_window=24,
    use_time_features=True,
    task="regression",
    horizon=1,
    downsample_factor=1,
    normalize=True,
    scaler_type="minmax",
    output_format="4d",
    out_dir=None,
    dataset_name="temporal_dataset",
):
    """
    Complete pipeline to create and save a temporal graph dataset.

    Args:
        edge_index: Graph connectivity (2 x E)
        augmented_df: DataFrame with point data mapped to cells
        edge_weights: Optional edge weights (E)
        node_ids: List of node IDs
        static_features: Optional static features DataFrame
        time_col: Column name for timestamps
        cell_col: Column name for cell IDs
        bin_type: "hourly", "daily", or "weekly" binning
        interval_hours: Size of each time bin in hours
        history_window: Number of historical time steps to include
        use_time_features: Whether to include time-based features
        task: "regression" or "classification"
        horizon: Number of time steps ahead to forecast
        downsample_factor: Optional temporal downsampling factor
        normalize: Whether to normalize features
        scaler_type: "minmax" or "standard" scaling
        output_format: '4d' or '3d' format for output dataset
        out_dir: Output directory for saved files
        dataset_name: Base name for output files

    Returns:
        Tuple of (dataset, dataset_path, metadata) where:
        - dataset: StaticGraphTemporalSignal object
        - dataset_path: Path to saved dataset file
        - metadata: Dictionary with dataset information
    """
    # Build time-binned features with 4D structure
    features_4d = build_time_binned_features_4d(
        df=augmented_df,
        cell_col=cell_col,
        time_col=time_col,
        bin_type=bin_type,
        interval_hours=interval_hours,
        all_node_ids=node_ids,
        history_window=history_window,
        use_time_features=use_time_features,
    )

    # Integrate static features if provided
    if static_features is not None and len(static_features) > 0:
        features_4d = integrate_static_features_4d(
            features_4d=features_4d, static_features=static_features, node_ids=node_ids
        )

    # Normalize features if requested
    if normalize:
        features_4d = scale_features_4d(
            features_4d=features_4d, scaler_type=scaler_type, per_feature=True
        )

    # Prepare forecasting data with 4D structure
    x_4d, y_3d = prepare_forecasting_data_4d(
        features_4d=features_4d, horizon=horizon, task=task, first_feature_only=True
    )

    # Convert to 3D if requested
    if output_format.lower() == "3d":
        static_count = 0
        if static_features is not None:
            static_count = static_features.shape[1]
        features_for_dataset = convert_4d_to_3d_features(x_4d, static_count)
        format_description = "3D"
    else:
        features_for_dataset = x_4d
        format_description = "4D"

    T, N, H, F = features_4d.shape
    base_features = F

    # Build and save the dataset
    dataset = build_static_temporal_data(
        edge_index=edge_index,
        features_3d=features_for_dataset,
        edge_weights=edge_weights,
        labels_3d=y_3d,
        downsample_factor=downsample_factor,
    )

    # Save dataset if output directory is provided
    dataset_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        dataset_path = os.path.join(out_dir, f"{dataset_name}.pt")
        save_static_temporal_data(dataset, dataset_path)

        # Save metadata
        metadata = {
            "num_timesteps": features_for_dataset.shape[0],
            "num_nodes": features_for_dataset.shape[1],
            "feature_format": format_description,
            "history_window": history_window,
            "base_features": base_features,
            "num_edges": edge_index.shape[1],
            "bin_type": bin_type,
            "interval_hours": interval_hours,
            "task": task,
            "horizon": horizon,
            "has_static_features": static_features is not None,
            "normalized": normalize,
            "downsampled": downsample_factor > 1,
        }

        # Save metadata as JSON
        import json

        with open(os.path.join(out_dir, f"{dataset_name}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to {dataset_path}")
        print(
            f"Metadata saved to {os.path.join(out_dir, f'{dataset_name}_metadata.json')}"
        )

    dataset.format_4d = output_format.lower() == "4d"
    dataset.history_window = history_window
    dataset.base_features = base_features

    return dataset, dataset_path, metadata if "metadata" in locals() else None
