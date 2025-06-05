import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import contextily as ctx
from typing import Tuple
from shapely.geometry import Polygon, LineString, box
from scipy.spatial import Voronoi
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from .base import BaseMapping


class VoronoiDegreeMapping(BaseMapping):
    """
    Degree-based Voronoi spatial mapping.

    This class implements mapping using Voronoi cells generated from important
    network intersections, where importance is determined by node degree.

    The method is based on the work by Gan, Jing, et al.:
    "A Novel Voronoi-Based Spatio-Temporal Graph Convolutional Network for
    Traffic Crash Prediction Considering Geographical Spatial Distributions."
    IEEE Transactions on Intelligent Transportation Systems (2024).
    """

    def __init__(
        self,
        place_name=None,
        buffer_distance=0.01,
        small_cell_size=500,
        large_cell_size=5000,
        min_degree=3,
        input_crs="EPSG:4326",
        meter_crs="EPSG:3857",
        testing_mode=False,
        test_bounds=None,
        **kwargs,
    ):
        """Initialize Voronoi degree-based mapping with the specified parameters."""
        super().__init__(name="voronoi_degree", **kwargs)
        self.place_name = place_name
        self.small_cell_size = small_cell_size
        self.large_cell_size = large_cell_size
        self.min_degree = min_degree
        self.input_crs = input_crs
        self.meter_crs = meter_crs
        self.testing_mode = testing_mode
        self.test_bounds = test_bounds
        self.buffer_distance = buffer_distance
        self.adj_matrix = None
        self.importance_nodes = None

    def create_mapping(
        self, points_gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, pd.Series]:
        """
        Create a Voronoi degree-based mapping from points.

        Args:
            points_gdf: GeoDataFrame with point geometries

        Returns:
            Tuple of (voronoi_gdf, point_to_region)
            - voronoi_gdf: GeoDataFrame with Voronoi region polygons
            - point_to_region: Series mapping point indices to region indices
        """
        print("Creating Voronoi degree-based mapping...")

        if points_gdf.crs is None:
            points_gdf = points_gdf.set_crs(self.input_crs)

        # Create the degree-based Voronoi mapping
        voronoi_gdf, adj_matrix, nodes_gdf, point_to_region = (
            self._create_voronoi_degree_mapping(
                place_name=self.place_name,
                data_points_gdf=points_gdf,
                small_cell_size=self.small_cell_size,
                large_cell_size=self.large_cell_size,
                testing_mode=self.testing_mode,
                test_bounds=self.test_bounds,
            )
        )

        self.adj_matrix = adj_matrix
        self.importance_nodes = nodes_gdf
        self.data_points = points_gdf
        mapped_points = (point_to_region >= 0).sum()
        total_points = len(point_to_region)
        mapping_percentage = (mapped_points / total_points) * 100

        print(
            f"Mapped {mapped_points} of {total_points} points ({mapping_percentage:.2f}%)"
        )
        print(
            f"Points assigned to {point_to_region.nunique() - (1 if -1 in point_to_region.values else 0)} Voronoi regions"
        )

        return voronoi_gdf, point_to_region
    
    def calculate_node_connectivity(self, nodes_gdf, edges_gdf):
        """
        Calculate node degrees using NetworkX graph directly.
        """
        G = nx.Graph()
        
        for _, edge in edges_gdf.iterrows():
            G.add_edge(
                tuple(edge.geometry.coords[0]), 
                tuple(edge.geometry.coords[-1])
            )
        
        node_degrees = pd.Series(dict(G.degree()))
        
        result = pd.Series(index=nodes_gdf.index)
        for idx, node in nodes_gdf.iterrows():
            coord = tuple(node.geometry.coords[0])  
            result[idx] = node_degrees.get(coord, 0)
        return result

    def get_major_intersections(
        self,
        place_name=None,
        data_points_gdf=None,
        min_degree=3,
        testing_mode=False,
        test_bounds=None,
    ):
        """Get major road intersections based on node degree."""

        # If no place_name is provided, use the bounds of the data points
        if place_name is None and data_points_gdf is not None:
            print(
                "No place name provided. Using dataset bounds to fetch road network..."
            )

            if data_points_gdf.crs != self.input_crs:
                points_latlon = data_points_gdf.to_crs(self.input_crs)
            else:
                points_latlon = data_points_gdf

            bounds = (
                points_latlon.total_bounds
            )  # [minx, miny, maxx, maxy] or [min_lon, min_lat, max_lon, max_lat]

            print(f"Dataset bounds: {bounds}")

            # Add a buffer to ensure we get all relevant roads
            buffer = self.buffer_distance
            bbox = (
                bounds[1] - buffer,  # south (min_lat)
                bounds[0] - buffer,  # west (min_lon)
                bounds[3] + buffer,  # north (max_lat)
                bounds[2] + buffer,  # east (max_lon)
            )

            print(f"Fetching road network for bbox: {bbox}")
            try:
                G = ox.graph_from_bbox(
                    north=bbox[2],
                    south=bbox[0],
                    east=bbox[3],
                    west=bbox[1],
                    network_type="drive",
                    simplify=True,
                    clean_periphery=True,
                )
                print(
                    f"Successfully fetched road network with {len(G.nodes)} nodes and {len(G.edges)} edges"
                )
            except Exception as e:
                print(f"Error fetching road network: {e}")
                print("Trying with larger buffer...")
                # Try with a larger buffer if the first attempt fails
                buffer = buffer * 2
                bbox = (
                    bounds[1] - buffer,
                    bounds[0] - buffer,
                    bounds[3] + buffer,
                    bounds[2] + buffer,
                )
                G = ox.graph_from_bbox(
                    north=bbox[2],
                    south=bbox[0],
                    east=bbox[3],
                    west=bbox[1],
                    network_type="drive",
                    simplify=True,
                    clean_periphery=True,
                )

        elif testing_mode and test_bounds:
            print(f"Using test bounds: {test_bounds}")
            G = ox.graph_from_bbox(
                north=test_bounds["max_lat"],
                south=test_bounds["min_lat"],
                east=test_bounds["max_lon"],
                west=test_bounds["min_lon"],
                network_type="drive",
                simplify=True,
                clean_periphery=True,
            )

        elif place_name:
            print(f"Fetching road network for {place_name}...")
            G = ox.graph_from_place(place_name, network_type="drive")

        else:
            raise ValueError("Either place_name or data_points_gdf must be provided")

        nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

        node_degrees = self.calculate_node_connectivity(nodes, edges)
        major_nodes = nodes[
            nodes.index.isin(node_degrees[node_degrees >= min_degree].index)
        ]

        print(
            f"Found {len(major_nodes)} major intersections with degree >= {min_degree}"
        )

        return major_nodes, edges, node_degrees

    def calculate_shared_roads(self, voronoi_gdf, edges_gdf):
        """Calculate shared road lengths between adjacent Voronoi regions."""
        edges_gdf = edges_gdf.to_crs(voronoi_gdf.crs)

        n_regions = len(voronoi_gdf)
        adj_matrix = np.zeros((n_regions, n_regions))

        for _, edge in edges_gdf.iterrows():
            intersecting = voronoi_gdf[voronoi_gdf.intersects(edge.geometry)]
            for i, j in combinations(intersecting.index.tolist(), 2):
                if i != j:
                    shared_length = edge.geometry.length
                    adj_matrix[i, j] += shared_length
                    adj_matrix[j, i] += shared_length

        return adj_matrix
    
    def get_road_network(self):
        """Return the road network used for mapping."""
        # Get the road network for the current partition
        _, edges_gdf, _ = self.get_major_intersections(
            place_name=self.place_name,
            data_points_gdf=self.data_points,
            testing_mode=self.testing_mode,
            test_bounds=self.test_bounds
        )
        return edges_gdf

    def get_adjacency_matrix(self):
        """Return the pre-calculated adjacency matrix."""
        return self.adj_matrix

    def get_reference_thresholds(
        self, bounds, small_cell_size=500, large_cell_size=5000
    ):
        """
        Get reference node counts k1, k2 from different grid sizes of the study area.
        Args:
            bounds: tuple of (minx, miny, maxx, maxy) in meters
            small_cell_size: size in meters for fine grid
            large_cell_size: size in meters for coarse grid
        """
        minx, miny, maxx, maxy = bounds

        nx_small = int(np.ceil((maxx - minx) / small_cell_size))
        ny_small = int(np.ceil((maxy - miny) / small_cell_size))
        k1 = nx_small * ny_small

        nx_large = int(np.ceil((maxx - minx) / large_cell_size))
        ny_large = int(np.ceil((maxy - miny) / large_cell_size))
        k2 = nx_large * ny_large

        return k1, k2

    def determine_degree_thresholds(self, degrees, k1, k2):
        """
        Determine Dmin, Dmax thresholds based on reference counts k1, k2.
        Uses sorted degree distribution to find appropriate cutoffs.
        """
        sorted_degrees = np.sort(degrees.values)[::-1]

        D_min = sorted_degrees[min(k1, len(sorted_degrees) - 1)]
        D_max = sorted_degrees[min(k2, len(sorted_degrees) - 1)]

        return D_min, D_max

    def _create_voronoi_degree_mapping(
        self,
        place_name,
        data_points_gdf,
        small_cell_size=500,
        large_cell_size=5000,
        testing_mode=False,
        test_bounds=None,
    ):
        """Create degree-based Voronoi mapping using degree-based node filtering."""
        nodes_gdf, edges_gdf, node_degrees = self.get_major_intersections(
            place_name=place_name,
            data_points_gdf=data_points_gdf,
            min_degree=1,
            testing_mode=testing_mode,
            test_bounds=test_bounds,
        )

        nodes_meter = nodes_gdf.to_crs(self.meter_crs)
        points_meter = data_points_gdf.to_crs(self.meter_crs)

        # Get reference thresholds
        print("Determining reference thresholds...")
        area_bounds = points_meter.total_bounds
        k1, k2 = self.get_reference_thresholds(
            bounds=area_bounds,
            small_cell_size=small_cell_size,
            large_cell_size=large_cell_size,
        )
        print(f"Reference counts - k1: {k1} (fine grid), k2: {k2} (coarse grid)")

        # Determine degree thresholds
        D_min, D_max = self.determine_degree_thresholds(node_degrees, k1, k2)
        print(f"Degree thresholds - D_min: {D_min}, D_max: {D_max}")

        # Filter nodes by degree importance
        filtered_nodes = nodes_gdf[(node_degrees >= D_min) & (node_degrees <= D_max)]
        print(
            f"Filtered to {len(filtered_nodes)} nodes with degrees between {D_min} and {D_max}"
        )

        if len(filtered_nodes) < 4:
            print("Not enough nodes after filtering. Using nodes with degree >= D_min")
            filtered_nodes = nodes_gdf[node_degrees >= D_min]
            print(f"Using {len(filtered_nodes)} nodes with degree >= {D_min}")

            # If still not enough, use all nodes
            if len(filtered_nodes) < 4:
                print("Still not enough nodes. Using all available nodes.")
                filtered_nodes = nodes_gdf.copy()

        if testing_mode and test_bounds:
            bbox = box(
                test_bounds["min_lon"],
                test_bounds["min_lat"],
                test_bounds["max_lon"],
                test_bounds["max_lat"],
            )
            bounding_polygon = (
                gpd.GeoSeries([bbox], crs=self.input_crs).to_crs(self.meter_crs).iloc[0]
            )
        else:
            bounds = points_meter.total_bounds
            bounding_polygon = box(*bounds)

        # Generate Voronoi diagram
        filtered_nodes_meter = filtered_nodes.to_crs(self.meter_crs)
        coords = np.column_stack(
            [filtered_nodes_meter.geometry.x, filtered_nodes_meter.geometry.y]
        )
        vor = Voronoi(coords)

        # Convert to polygons and clip
        polygons = []
        node_indices = []  # Track which nodes correspond to which polygons

        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not region or -1 in region:
                continue
            try:
                poly = Polygon(
                    [(vor.vertices[j][0], vor.vertices[j][1]) for j in region]
                )
                if poly.is_valid:
                    clipped = poly.intersection(bounding_polygon)
                    if not clipped.is_empty and clipped.is_valid:
                        polygons.append(clipped)
                        node_indices.append(i)
            except Exception as e:
                print(f"Error processing region {i}: {e}")
                continue

        voronoi_gdf = gpd.GeoDataFrame(
            {"region_id": range(len(polygons)), "node_idx": node_indices},
            geometry=polygons,
            crs=self.meter_crs,
        )

        # Map nodes to voronoi regions
        filtered_nodes_meter = filtered_nodes_meter.iloc[node_indices].reset_index(
            drop=True
        )

        # Add degree information
        voronoi_gdf["degree"] = 0
        for i, node_idx in enumerate(filtered_nodes.index[node_indices]):
            if i < len(voronoi_gdf) and node_idx in node_degrees.index:
                voronoi_gdf.loc[i, "degree"] = node_degrees[node_idx]

        # Calculate adjacency matrix
        adj_matrix = self.calculate_shared_roads(
            voronoi_gdf, edges_gdf.to_crs(self.meter_crs)
        )

        # Map points to regions
        joined = gpd.sjoin(points_meter, voronoi_gdf, how="left", predicate="within")
        point_to_region = joined.index_right.fillna(-1).astype(int)

        voronoi_gdf = voronoi_gdf.to_crs(self.input_crs)

        return (
            voronoi_gdf,
            adj_matrix,
            filtered_nodes_meter.to_crs(self.input_crs),
            point_to_region,
        )

    def visualize(self, points_gdf, partition_gdf, point_to_partition, **kwargs):
        """Pass importance nodes to the base visualization method."""
        if hasattr(self, "importance_nodes") and self.importance_nodes is not None:
            kwargs["importance_nodes"] = self.importance_nodes

        super().visualize(points_gdf, partition_gdf, point_to_partition, **kwargs)
