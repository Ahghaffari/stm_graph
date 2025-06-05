"""
Graph construction and feature augmentation utilities.

This module provides functionality to construct graph structures from spatial partitions
and augment them with static and dynamic features.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def find_isolated_subgraphs(edge_index, num_nodes):
    """
    Find isolated subgraphs using connected components analysis.

    Args:
        edge_index: Edge index tensor of shape [2, E]
        num_nodes: Total number of nodes

    Returns:
        Tuple of (n_components, labels)
    """
    adj_matrix = np.zeros((num_nodes, num_nodes))
    adj_matrix[edge_index[0], edge_index[1]] = 1
    n_components, labels = connected_components(csr_matrix(adj_matrix), directed=False)
    return n_components, labels


def connect_subgraphs(
    partition_gdf, edge_index, labels, distance_threshold=float("inf")
):
    """
    Connect isolated subgraphs to nearest neighbors within threshold.

    Args:
        partition_gdf: GeoDataFrame of region polygons
        edge_index: Edge index tensor [2, E]
        labels: Component labels for each node
        distance_threshold: Maximum distance to connect components (default: infinity)

    Returns:
        Set of new edges to add
    """
    new_edges = set()
    existing_edges = set(zip(edge_index[0], edge_index[1]))

    potential_connections = []

    for comp_id in range(labels.max() + 1):
        comp_nodes = np.where(labels == comp_id)[0]

        for node_id in comp_nodes:
            node_geom = partition_gdf.iloc[node_id].geometry
            node_centroid = node_geom.centroid

            for other_id in range(len(partition_gdf)):
                if labels[other_id] != comp_id:
                    other_geom = partition_gdf.iloc[other_id].geometry
                    other_centroid = other_geom.centroid

                    dist = node_centroid.distance(other_centroid)
                    if dist < distance_threshold:
                        potential_connections.append((dist, node_id, other_id))

    potential_connections.sort()

    connected_comps = set()

    for dist, node_id, other_id in potential_connections:
        comp1, comp2 = labels[node_id], labels[other_id]

        if comp1 not in connected_comps or comp2 not in connected_comps:
            edge = (node_id, other_id)
            if edge not in existing_edges:
                new_edges.add(edge)
                new_edges.add(edge[::-1])
                connected_comps.add(comp1)
                connected_comps.add(comp2)

    # If any components still isolated, connect to absolute nearest neighbor
    remaining_comps = set(range(labels.max() + 1)) - connected_comps
    if remaining_comps:
        print(
            f"Warning: {len(remaining_comps)} components still isolated, connecting to nearest neighbors"
        )
        for comp_id in remaining_comps:
            comp_nodes = np.where(labels == comp_id)[0]
            min_dist = float("inf")
            best_edge = None

            for node_id in comp_nodes:
                node_centroid = partition_gdf.iloc[node_id].geometry.centroid
                for other_id in range(len(partition_gdf)):
                    if labels[other_id] != comp_id:
                        other_centroid = partition_gdf.iloc[other_id].geometry.centroid
                        dist = node_centroid.distance(other_centroid)
                        if dist < min_dist:
                            min_dist = dist
                            best_edge = (node_id, other_id)

            if best_edge:
                new_edges.add(best_edge)
                new_edges.add(best_edge[::-1])

    return new_edges


def build_polygon_adjacency(
    partition_gdf,
    polygon_id_col=None,
    tolerance=1e-6,
    enhanced=True,
    distance_threshold=float("inf"),
    use_meters=True,
):
    """
    Build adjacency graph from polygon geometries.

    Args:
        partition_gdf: GeoDataFrame with region polygons
        polygon_id_col: Column name for node IDs
        tolerance: Distance tolerance for neighborhood
        enhanced: If True, ensures fully connected graph
        distance_threshold: Maximum distance to connect isolated subgraphs
        use_meters: Whether to use meter-based CRS for distance calculations

    Returns:
        Tuple of (edge_index, node_ids)
    """
    partition_copy = partition_gdf.copy()

    if use_meters and partition_copy.crs is None or partition_copy.crs.is_geographic:
        try:
            partition_copy = partition_copy.to_crs(epsg=3857)
        except Exception as e:
            print(f"Warning: Failed to convert to meter-based CRS: {e}")

    partition_copy = partition_copy.reset_index(drop=True)

    partition_copy["geometry"] = partition_copy["geometry"].buffer(0).buffer(tolerance)

    # Find adjacent polygons
    edges = set()
    for idx, geom in enumerate(partition_copy.geometry):
        for nb_idx in partition_copy.sindex.intersection(geom.bounds):
            if nb_idx != idx:
                nb_geom = partition_copy.loc[nb_idx, "geometry"]
                is_neighbor = any(
                    [
                        geom.touches(nb_geom),
                        geom.intersects(nb_geom),
                        not geom.boundary.intersection(nb_geom.boundary).is_empty,
                        geom.distance(nb_geom) < tolerance,
                    ]
                )
                if is_neighbor:
                    edges.add(tuple(sorted([idx, nb_idx])))

    if not edges:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_src, edge_dst = [], []
        for a, b in edges:
            edge_src.extend([a, b])
            edge_dst.extend([b, a])

        edge_index = np.vstack([edge_src, edge_dst])

    if enhanced:
        n_components, labels = find_isolated_subgraphs(edge_index, len(partition_copy))

        if n_components > 1:
            print(f"Found {n_components} isolated subgraphs, connecting them...")
            new_edges = connect_subgraphs(
                partition_copy,
                edge_index,
                labels,
                distance_threshold=distance_threshold,
            )

            if new_edges:
                new_src, new_dst = zip(*new_edges)
                edge_src = np.append(edge_index[0], new_src)
                edge_dst = np.append(edge_index[1], new_dst)
                edge_index = np.array([edge_src, edge_dst], dtype=np.int64)

            n_components_final, _ = find_isolated_subgraphs(
                edge_index, len(partition_copy)
            )
            print(f"After connecting: {n_components_final} component(s)")

    node_ids = (
        partition_copy[polygon_id_col].tolist()
        if polygon_id_col and polygon_id_col in partition_copy.columns
        else partition_copy.index.tolist()
    )

    return edge_index, node_ids


def process_adjacency_matrix(adj_matrix, active_cell_ids=None):
    """
    Process adjacency matrix to get edge_index and weights.

    Args:
        adj_matrix: Numpy array of adjacency matrix
        active_cell_ids: List of active cell IDs (optional)

    Returns:
        Tuple of (edge_index, edge_weights)
    """
    if adj_matrix is None:
        return None, None

    if active_cell_ids is not None:
        adj_filtered = adj_matrix[active_cell_ids][:, active_cell_ids]
    else:
        adj_filtered = adj_matrix

    rows, cols = np.nonzero(adj_filtered)
    edge_index = np.stack([rows, cols])
    edge_weights = adj_filtered[rows, cols]

    return edge_index, edge_weights


def build_road_based_adjacency(
    partition_gdf, 
    road_edges_gdf, 
    enhanced=True,
    distance_threshold=float("inf"),
    buffer_distance=0 
):
    """
    Build adjacency graph based on shared road lengths between regions.
    
    Args:
        partition_gdf: GeoDataFrame with region polygons
        road_edges_gdf: GeoDataFrame with road segments
        enhanced: If True, ensures fully connected graph
        distance_threshold: Maximum distance to connect isolated subgraphs
        buffer_distance: Distance to buffer road geometries for intersection (in CRS units)
        
    Returns:
        Tuple of (edge_index, edge_weights)
    """
    if road_edges_gdf.crs != partition_gdf.crs:
        road_edges_gdf = road_edges_gdf.to_crs(partition_gdf.crs)
    
    n_regions = len(partition_gdf)
    adj_matrix = np.zeros((n_regions, n_regions))
    
    total_road_segments = len(road_edges_gdf)
    intersecting_roads = 0
    
    for _, edge in road_edges_gdf.iterrows():
        geom = edge.geometry.buffer(buffer_distance) if buffer_distance > 0 else edge.geometry
        intersecting = partition_gdf[partition_gdf.intersects(geom)]
        
        if len(intersecting) >= 2:
            intersecting_roads += 1
            for i, row1 in intersecting.iterrows():
                for j, row2 in intersecting.iterrows():
                    if i < j: 
                        shared_length = edge.geometry.length
                        adj_matrix[i, j] += shared_length
                        adj_matrix[j, i] += shared_length
    print(f"Road-based adjacency: {intersecting_roads}/{total_road_segments} road segments connect multiple regions ({(intersecting_roads/total_road_segments)*100:.2f}%)")
    
    if intersecting_roads == 0:
        print("WARNING: No road segments intersect multiple regions!")
        print("Consider using buffer_distance > 0 or check alignment of road data with regions")
    
    rows, cols = np.nonzero(adj_matrix)
    edge_index = np.stack([rows, cols])
    edge_weights = adj_matrix[rows, cols]
    
    if enhanced:
        n_components, labels = find_isolated_subgraphs(edge_index, n_regions)
        
        if n_components > 1:
            print(f"Found {n_components} isolated subgraphs, connecting them...")
            new_edges = connect_subgraphs(
                partition_gdf,
                edge_index,
                labels,
                distance_threshold=distance_threshold
            )
            
            if new_edges:
                new_src, new_dst = zip(*new_edges)
                edge_src = np.append(edge_index[0], new_src)
                edge_dst = np.append(edge_index[1], new_dst)
                edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
                new_weights = np.ones(len(new_src))
                edge_weights = np.append(edge_weights, new_weights)
                
            n_components_final, _ = find_isolated_subgraphs(edge_index, n_regions)
            print(f"After connecting: {n_components_final} component(s)")
    
    return edge_index, edge_weights


def build_graph_and_augment(
    grid_gdf,
    points_gdf,
    point_to_cell,
    adj_matrix=None,
    remove_empty_nodes=True,
    road_edges_gdf=None,
    adjacency_type="spatial",
    out_dir=None,
    save_flag=True,
    static_features=None,
    meter_crs="EPSG:3857",
):
    """
    Build graph and augment with features.

    Args:
        grid_gdf: GeoDataFrame of spatial regions
        points_gdf: GeoDataFrame of points
        point_to_cell: Series mapping points to regions
        adj_matrix: Optional weighted adjacency matrix
        remove_empty_nodes: Whether to remove empty regions
        road_edges_gdf: GeoDataFrame with road segments (required for road_based adjacency)
        adjacency_type: Type of adjacency to use ("spatial" or "road_based")
        out_dir: Output directory for saving
        save_flag: Whether to save outputs
        static_features: Optional DataFrame of static features
        meter_crs: CRS for meter-based calculations

    Returns:
        Dictionary with graph components:
            - edge_index: Edge index tensor [2, E]
            - edge_weight: Edge weight tensor [E]
            - node_features: Node feature matrix [N, F]
            - augmented_df: DataFrame with cell_id mapping
            - num_nodes: Number of nodes
            - node_ids: Original node IDs
    """
    cell_counts = point_to_cell.value_counts()
    
    if -1 in cell_counts.index:
        cell_counts = cell_counts.drop(-1)
    
    if remove_empty_nodes:
        print("Removing empty regions...")
        print(f"Before: {len(grid_gdf)} regions")
        
        active_cell_ids = cell_counts.index.tolist()
        active_cell_ids.sort()
        
        grid_filtered = grid_gdf.loc[active_cell_ids].reset_index(drop=True)

        if adj_matrix is not None:
            edge_index, edge_weights = process_adjacency_matrix(adj_matrix, active_cell_ids)
        elif adjacency_type == "road_based" and road_edges_gdf is not None:
            if hasattr(road_edges_gdf, 'to_crs'):
                road_edges_filtered = road_edges_gdf.to_crs(grid_filtered.crs)
            else:
                road_edges_filtered = road_edges_gdf

            edge_index, edge_weights = build_road_based_adjacency(
                grid_filtered, road_edges_filtered, enhanced=True, buffer_distance=20
            )
        else:
            edge_index, node_ids = build_polygon_adjacency(
                grid_filtered, use_meters=True, enhanced=True
            )
            edge_weights = np.ones(edge_index.shape[1])

        node_features = cell_counts.loc[active_cell_ids].values.reshape(-1, 1)

        old_to_new = {
            old_idx: new_idx for new_idx, old_idx in enumerate(active_cell_ids)
        }
        augmented_df = points_gdf.copy()
        augmented_df["cell_id"] = point_to_cell.map(old_to_new).fillna(-1).astype(int)

        if static_features is not None:
            static_features_filtered = static_features.loc[active_cell_ids].reset_index(
                drop=True
            )

            if len(static_features_filtered) == len(node_features):
                static_features_array = static_features_filtered.values
                node_features = np.hstack([node_features, static_features_array])
            else:
                print(
                    f"Warning: Static feature dimensions ({len(static_features_filtered)}) don't match node dimensions ({len(node_features)})"
                )

        print(f"After: {len(grid_filtered)} regions")
        node_ids = active_cell_ids

    else:
        print("Keeping all regions...")
        if adj_matrix is not None:
            edge_index, edge_weights = process_adjacency_matrix(adj_matrix)
        else:
            edge_index, node_ids = build_polygon_adjacency(
                grid_gdf, use_meters=True, enhanced=True
            )
            edge_weights = np.ones(edge_index.shape[1])

        node_features = np.zeros((len(grid_gdf), 1))
        for idx, count in cell_counts.items():
            if idx >= 0 and idx < len(node_features):
                node_features[idx, 0] = count

        if static_features is not None:
            if len(static_features) == len(node_features):
                static_features_array = static_features.values

                node_features = np.hstack([node_features, static_features_array])
            else:
                print(
                    f"Warning: Static feature dimensions ({len(static_features)}) don't match node dimensions ({len(node_features)})"
                )

        augmented_df = points_gdf.copy()
        augmented_df["cell_id"] = point_to_cell.fillna(-1).astype(int)
        node_ids = grid_gdf.index.tolist()

    edge_index = np.array(edge_index, dtype=np.int64)
    edge_weights = np.array(edge_weights, dtype=np.float32)
    node_features = np.array(node_features, dtype=np.float32)

    # Create dictionary with all graph components
    graph_data = {
        "edge_index": edge_index,
        "edge_weight": edge_weights,
        "node_features": node_features,
        "augmented_df": augmented_df,
        "num_nodes": len(node_features),
        "node_ids": node_ids,
    }

    # Save outputs if requested
    if save_flag and out_dir:
        print(f"Saving graph components to {out_dir}...")
        os.makedirs(out_dir, exist_ok=True)

        np.save(f"{out_dir}/edge_index.npy", edge_index)
        np.save(f"{out_dir}/edge_weights.npy", edge_weights)
        np.save(f"{out_dir}/node_features.npy", node_features)

        # Save augmented dataset
        if "geometry" in augmented_df.columns:
            augmented_df_save = augmented_df.drop(columns="geometry")
        else:
            augmented_df_save = augmented_df

        augmented_df_save.to_csv(f"{out_dir}/augmented_dataset.csv", index=False)

        # Save node IDs mapping
        pd.DataFrame({"original_id": node_ids, "new_id": range(len(node_ids))}).to_csv(
            f"{out_dir}/node_mapping.csv", index=False
        )

        if static_features is not None:
            if remove_empty_nodes:
                static_features.loc[active_cell_ids].reset_index().to_csv(
                    f"{out_dir}/static_features.csv", index=False
                )
            else:
                static_features.to_csv(f"{out_dir}/static_features.csv", index=False)

    return graph_data
