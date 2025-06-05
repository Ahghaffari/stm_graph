"""
Visualization utilities for spatial-temporal graph data.
Includes some sample time series plotting and spatial network visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import geopandas as gpd
import contextily as ctx
from datetime import datetime, timedelta


def plot_node_time_series(
    temporal_dataset,
    node_ids=None,
    num_nodes=None,
    feature_idx=0,
    selection_method="highest_activity",
    plot_type="2d",
    n_steps=None,
    start_time=None,
    time_delta=timedelta(hours=1),
    title=None,
    cmap="viridis",
    figsize=(12, 8),
    out_dir=None,
    filename="node_time_series",
    file_format="png",
    rasterized: bool = False,
    fig_dpi: int = 300,
):
    """
    Plot time series data for selected nodes from a temporal graph dataset.

    Args:
        temporal_dataset: PyTorch Geometric Temporal dataset
        node_ids: List of node IDs to plot (if None, uses num_nodes)
        num_nodes: Number of nodes to plot if node_ids is None
        selection_method: Method to select nodes ("highest_activity" or "random")
        feature_idx: Index of feature to plot (default: 0, usually event count)
        plot_type: '2d' or '3d' visualization
        n_steps: Number of time steps to include (None = all)
        start_time: Start time for x-axis (None = use indices)
        time_delta: Time delta between steps (default: 1 hour)
        title: Plot title
        cmap: Colormap for 3D plot
        figsize: Figure size (width, height) in inches
        out_dir: Output directory for saved visualization
        filename: Filename for saved plot
        file_format: File format for saving (png, svg, pdf)
        rasterized: Whether to rasterize the plot
        fig_dpi: DPI for the figure

    Returns:
        Path to saved plot file
    """
    if out_dir:
        os.makedirs(os.path.join(out_dir, "graph"), exist_ok=True)

    n_avail = temporal_dataset.features[0].shape[0]

    if node_ids is None:
        if num_nodes is None:
            num_nodes = 3
        num_nodes = min(num_nodes, n_avail)

        if selection_method == "highest_activity":
            activity = np.zeros(n_avail)
            for t in range(len(temporal_dataset.features)):
                feature_data = temporal_dataset.features[t][:, feature_idx]
                if hasattr(feature_data, "numpy"):
                    feature_data = feature_data.numpy()
                activity += feature_data

            top_indices = np.argsort(activity)[-num_nodes:]
            node_ids = top_indices.tolist()
            print(f"Selected {num_nodes} most active nodes: {node_ids}")
        else:
            # Random selection
            node_ids = np.random.choice(n_avail, size=num_nodes, replace=False).tolist()
            print(f"Randomly selected {num_nodes} nodes: {node_ids}")

    # Determine time steps to include
    if n_steps is None:
        n_steps = len(temporal_dataset.features)
    else:
        n_steps = min(n_steps, len(temporal_dataset.features))

    if start_time is not None:
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        time_array = [start_time + time_delta * i for i in range(n_steps)]
    else:
        time_array = list(range(n_steps))

    node_data = {}
    for node_id in node_ids:
        node_data[node_id] = [
            temporal_dataset.features[t][node_id, feature_idx].item()
            for t in range(n_steps)
        ]

    if plot_type.lower() == "2d":
        fig, ax = plt.subplots(figsize=figsize)

        for node_id, values in node_data.items():
            ax.plot(time_array, values, label=f"Node {node_id}", linewidth=2, rasterized=rasterized)

        if start_time is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            plt.xticks(rotation=45)

        plt.xlabel("Time" if start_time is None else "Date/Time")
        plt.ylabel(f"Feature {feature_idx} Value")
        plt.title(title or f"Time Series for Selected Nodes (Feature {feature_idx})")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    elif plot_type.lower() == "3d":
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        x = range(len(time_array))
        y = range(len(node_ids))
        X, Y = np.meshgrid(x, y)

        Z = np.zeros((len(node_ids), len(time_array)))
        for i, node_id in enumerate(node_ids):
            Z[i, :] = node_data[node_id]

        surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none", alpha=0.8)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Node ID")
        ax.set_zlabel(f"Feature {feature_idx} Value")
        ax.set_yticks(range(len(node_ids)))
        ax.set_yticklabels([f"Node {node_id}" for node_id in node_ids])

        ax.set_title(
            title or f"3D Time Series for Selected Nodes (Feature {feature_idx})"
        )

    else:
        raise ValueError("plot_type must be '2d' or '3d'")

    if out_dir:
        save_path = os.path.join(out_dir, "graph", f"{filename}.{file_format}")
        plt.savefig(save_path, bbox_inches="tight", dpi=fig_dpi)
        plt.close()
        print(f"Time series plot saved to {save_path}")
        return save_path
    else:
        plt.show()
        return None


def plot_spatial_network(
    regions_gdf,
    edge_index,
    edge_weights=None,
    node_values=None,
    node_ids=None,
    time_step=None,
    plot_crs="EPSG:3857",
    title=None,
    node_cmap="viridis",
    edge_cmap="plasma",
    map_style=ctx.providers.CartoDB.Positron,
    alpha_regions=0.7,
    alpha_edges=0.8,
    figsize=(15, 15),
    edge_width_scale=5.0,
    show_colorbar=True,
    out_dir=None,
    filename="spatial_network",
    file_format="png",
    rasterized: bool = False,
    fig_dpi: int = 300,
):
    """
    Plot the spatial network with regions colored by node values and edges by edge weights.

    Args:
        regions_gdf: GeoDataFrame with region polygons
        edge_index: Edge index tensor [2, E]
        edge_weights: Edge weight values (if None, all edges have equal weight)
        node_values: Values to use for coloring regions (if None, uses uniform color)
        node_ids: Optional mapping from node index to original region index
        time_step: Optional time step for temporal data
        plot_crs: CRS to use for plotting
        title: Plot title
        node_cmap: Colormap for nodes/regions
        edge_cmap: Colormap for edges
        map_style: Contextily map style
        alpha_regions: Transparency for regions
        alpha_edges: Transparency for edges
        figsize: Figure size (width, height) in inches
        edge_width_scale: Scale factor for edge widths
        show_colorbar: Whether to show colorbars
        out_dir: Output directory for saved visualization
        filename: Filename for saved plot
        file_format: File format for saving (png, svg, pdf)
        rasterized: Whether to rasterize the plot
        fig_dpi: DPI for the figure

    Returns:
        Path to saved plot file
    """
    if out_dir:
        os.makedirs(os.path.join(out_dir, "graph"), exist_ok=True)

    regions_plot = regions_gdf.copy().to_crs(plot_crs)

    if node_ids is not None:
        used_region_indices = set(node_ids)

        regions_plot = regions_plot.loc[
            regions_plot.index.isin(used_region_indices)
        ].copy()
        print(
            f"Filtered from {len(regions_gdf)} to {len(regions_plot)} active grid cells"
        )

        if node_values is not None:
            mapped_values = {}
            for i, idx in enumerate(node_ids):
                mapped_values[idx] = node_values[i]

            regions_plot["value"] = regions_plot.index.map(
                lambda idx: mapped_values.get(idx, 0)
            )
        else:
            regions_plot["value"] = 0
    elif node_values is not None:
        if len(node_values) == len(regions_plot):
            regions_plot["value"] = node_values

            if not isinstance(regions_plot["value"], (int, float)):
                regions_plot = regions_plot[regions_plot["value"] > 0].copy()
                print(f"Filtered to {len(regions_plot)} regions with non-zero values")
        else:
            print(
                f"Warning: node_values length ({len(node_values)}) != regions length ({len(regions_plot)})"
            )
            regions_plot["value"] = 0
    else:
        regions_plot["value"] = 0

    fig, ax = plt.subplots(figsize=figsize)

    bounds = regions_plot.total_bounds
    ax.set_xlim([bounds[0], bounds[2]])
    ax.set_ylim([bounds[1], bounds[3]])

    try:
        ctx.add_basemap(ax, source=map_style, crs=plot_crs)
    except Exception as e:
        print(f"Warning: Could not add basemap: {e}")

    region_plot = regions_plot.plot(
        column="value",
        ax=ax,
        cmap=node_cmap,
        alpha=alpha_regions,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
        zorder=2,
        rasterized=rasterized,
    )

    if show_colorbar and node_values is not None:
        cbar_ax = fig.add_axes(
            [0.91, 0.32, 0.02, 0.35]
        )  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=node_cmap)
        sm.set_array(regions_plot["value"])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(
            "Node Value" + (f" (t={time_step})" if time_step is not None else "")
        )

    if edge_index is not None and edge_index.shape[0] >= 2:
        if edge_weights is None:
            edge_weights = np.ones(edge_index.shape[1])

        if len(edge_weights) > 0:
            min_weight = edge_weights.min()
            max_weight = edge_weights.max()

            if max_weight > min_weight:
                if max_weight / (min_weight + 1e-10) > 100:
                    norm_weights = np.log1p(edge_weights) / np.log1p(max_weight)
                    print(
                        f"Using logarithmic normalization for edge weights (range: {min_weight:.4f}-{max_weight:.4f})"
                    )
                else:
                    norm_weights = (edge_weights - min_weight) / (
                        max_weight - min_weight
                    )
                    print(
                        f"Using linear normalization for edge weights (range: {min_weight:.4f}-{max_weight:.4f})"
                    )
            else:
                norm_weights = np.ones_like(edge_weights) * 0.5
                print(f"All edge weights are identical: {min_weight:.4f}")
        else:
            norm_weights = []

        centroids = regions_plot.geometry.centroid

        edge_color_map = plt.cm.get_cmap(edge_cmap)

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]

            if node_ids is not None:
                src = node_ids[src] if src < len(node_ids) else src
                dst = node_ids[dst] if dst < len(node_ids) else dst

            if src in centroids.index and dst in centroids.index:
                x1, y1 = centroids.loc[src].x, centroids.loc[src].y
                x2, y2 = centroids.loc[dst].x, centroids.loc[dst].y

                if i < len(norm_weights):
                    width = (norm_weights[i] * edge_width_scale) + 0.5
                    color = edge_color_map(norm_weights[i])
                else:
                    width = 1.0
                    color = "gray"

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=color,
                    linewidth=width,
                    alpha=alpha_edges,
                    zorder=3,
                    solid_capstyle="round",
                    rasterized=rasterized,
                )

        if show_colorbar and len(edge_weights) > 0:
            cbar_ax = fig.add_axes(
                [0.91, 0.72, 0.02, 0.15]
            )  # [left, bottom, width, height]
            sm = plt.cm.ScalarMappable(cmap=edge_cmap)
            sm.set_array(edge_weights)
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label("Edge Weight")

    plt.title(title or "Spatial Network with Region Values and Edges")
    plt.axis("off")

    if out_dir:
        save_path = os.path.join(out_dir, "graph", f"{filename}.{file_format}")
        plt.savefig(save_path, bbox_inches="tight", dpi=fig_dpi)
        plt.close()
        print(f"Spatial network plot saved to {save_path}")
        return save_path
    else:
        plt.show()
        return None


def plot_temporal_heatmap(
    temporal_dataset,
    node_ids=None,
    num_nodes=None,
    feature_idx=0,
    selection_method="highest_activity",
    n_steps=100,
    start_time=None,
    time_delta=timedelta(hours=1),
    title=None,
    cmap="viridis",
    figsize=(15, 10),
    out_dir=None,
    filename="temporal_heatmap",
    file_format="png",
    rasterized=False,
    fig_dpi=300,
):
    """
    Plot a heatmap of temporal data for selected nodes.

    Args:
        temporal_dataset: PyTorch Geometric Temporal dataset
        node_ids: List of node IDs to plot (if None, uses num_nodes)
        num_nodes: Number of nodes to plot if node_ids is None
        feature_idx: Index of feature to plot (default: 0, usually event count)
        selection_method: Method to select nodes ("highest_activity" or "random")
        n_steps: Maximum number of timesteps to include (for readability)
        start_time: Start time for x-axis (None = use indices)
        time_delta: Time delta between steps (default: 1 hour)
        title: Plot title
        cmap: Colormap for heatmap
        figsize: Figure size (width, height) in inches
        out_dir: Output directory for saved visualization
        filename: Filename for saved plot
        file_format: File format for saving (png, svg, pdf)
        rasterized: Whether to rasterize the plot
        fig_dpi: DPI for the figure

    Returns:
        Path to saved plot file
    """
    if out_dir:
        os.makedirs(os.path.join(out_dir, "graph"), exist_ok=True)

    n_avail = temporal_dataset.features[0].shape[0]

    if node_ids is None:
        if num_nodes is None:
            num_nodes = 10
        num_nodes = min(num_nodes, n_avail)

        if selection_method == "highest_activity":
            activity = np.zeros(n_avail)
            for t in range(len(temporal_dataset.features)):
                feature_data = temporal_dataset.features[t][:, feature_idx]
                if hasattr(feature_data, "numpy"):
                    feature_data = feature_data.numpy()
                activity += feature_data

            top_indices = np.argsort(activity)[-num_nodes:]
            node_ids = top_indices.tolist()
            print(f"Selected {num_nodes} most active nodes: {node_ids}")
        else:
            node_ids = np.random.choice(n_avail, size=num_nodes, replace=False).tolist()
            print(f"Randomly selected {num_nodes} nodes: {node_ids}")
    elif len(node_ids) > num_nodes and num_nodes is not None:
        node_ids = node_ids[:num_nodes]
        print(f"Warning: Limited to {num_nodes} nodes for readability")

    n_timesteps = min(len(temporal_dataset.features), n_steps)

    if start_time is not None:
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        time_array = [start_time + time_delta * i for i in range(n_timesteps)]
    else:
        time_array = list(range(n_timesteps))

    data = np.zeros((len(node_ids), n_timesteps))
    for i, node_id in enumerate(node_ids):
        for t in range(n_timesteps):
            data[i, t] = temporal_dataset.features[t][node_id, feature_idx].item()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect="auto", interpolation="none", rasterized=rasterized)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Feature {feature_idx} Value")

    ax.set_xlabel("Time" if start_time is None else "Date/Time")
    
    if start_time is not None:
        n_ticks = min(10, n_timesteps) 
        tick_indices = np.linspace(0, n_timesteps-1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        tick_labels = [time_array[i].strftime("%m-%d %H:%M") for i in tick_indices]
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    else:
        if n_timesteps > 20:
            n_ticks = 10
            tick_indices = np.linspace(0, n_timesteps-1, n_ticks, dtype=int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([str(i) for i in tick_indices])

    ax.set_ylabel("Node ID")
    ax.set_yticks(range(len(node_ids)))
    ax.set_yticklabels([f"Node {node_id}" for node_id in node_ids])

    ax.set_title(title or f"Temporal Heatmap for Selected Nodes (Feature {feature_idx})")

    plt.tight_layout()

    if out_dir:
        save_path = os.path.join(out_dir, "graph", f"{filename}.{file_format}")
        plt.savefig(save_path, bbox_inches="tight", dpi=fig_dpi)
        plt.close()
        print(f"Temporal heatmap saved to {save_path}")
        return save_path
    else:
        plt.show()
        return None
