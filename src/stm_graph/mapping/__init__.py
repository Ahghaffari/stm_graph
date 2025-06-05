from .base import BaseMapping
from .grid import GridMapping
from .administrative import AdministrativeMapping
from .voronoi import VoronoiDegreeMapping

__all__ = [
    "BaseMapping",
    "GridMapping",
    "AdministrativeMapping",
    "VoronoiDegreeMapping",
    "create_grid_mapping",
    "create_administrative_mapping",
    "create_voronoi_degree_mapping",
]


def create_grid_mapping(points_gdf, cell_size=3000.0, **kwargs):
    mapper = GridMapping(cell_size=cell_size, **kwargs)
    return mapper.create_mapping(points_gdf)


def create_administrative_mapping(
    points_gdf,
    admin_type="administrative",
    districts_file=None,
    districts_gdf=None,
    id_column=None,
    name_column=None,
    geometry_column=None,
    input_crs="EPSG:4326",
    meter_crs="EPSG:32618",
    testing_mode=False,
    test_bounds=None,
    **kwargs
):
    mapper = AdministrativeMapping(
        admin_type=admin_type,
        districts_file=districts_file,
        districts_gdf=districts_gdf,
        id_column=id_column,
        name_column=name_column,
        geometry_column=geometry_column,
        input_crs=input_crs,
        meter_crs=meter_crs,
        testing_mode=testing_mode,
        test_bounds=test_bounds,
        **kwargs
    )
    return mapper.create_mapping(points_gdf)


def create_voronoi_degree_mapping(
    points_gdf,
    place_name,
    small_cell_size=500,
    large_cell_size=5000,
    input_crs="EPSG:4326",
    meter_crs="EPSG:32618",
    testing_mode=False,
    test_bounds=None,
    **kwargs
):
    mapper = VoronoiDegreeMapping(
        place_name=place_name,
        small_cell_size=small_cell_size,
        large_cell_size=large_cell_size,
        input_crs=input_crs,
        meter_crs=meter_crs,
        testing_mode=testing_mode,
        test_bounds=test_bounds,
        **kwargs
    )
    return mapper.create_mapping(points_gdf)
