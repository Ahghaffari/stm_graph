"""
Spatial-Temporal Graph package for processing and modeling urban events.
"""

from stm_graph.mapping import AdministrativeMapping, GridMapping, VoronoiDegreeMapping

from stm_graph.features.osm import extract_osm_features

from stm_graph.graph.construction import build_graph_and_augment

from stm_graph.graph.plot import (
    plot_node_time_series,
    plot_spatial_network,
    plot_temporal_heatmap,
)

from stm_graph.graph.temporal import (
    build_time_binned_features_4d,
    integrate_static_features_4d,
    scale_features_4d,
    prepare_forecasting_data_4d,
    build_static_temporal_data,
    save_static_temporal_data,
    load_static_temporal_data,
    create_temporal_dataset,
    convert_3d_to_4d_dataset,
    convert_4d_to_3d_dataset,
)

from stm_graph.data.loader import preprocess_dataset, move_batch_to_device

from stm_graph.training.train import train_model, train_one_epoch
from stm_graph.training.evaluate import (
    evaluate_model,
    get_predictions,
    calculate_classification_metrics,
)

from stm_graph.models.model_factory import (
    create_model,
    list_available_models,
    get_model_info,
    get_model_params,
    list_pytorch_geometric_temporal_models,
    get_pytorch_geometric_temporal_model_info,
    get_pytorch_geometric_temporal_model_params,
)

__all__ = [
    # Mapping classes
    "AdministrativeMapping",
    "GridMapping",
    "VoronoiDegreeMapping",
    # OSM features
    "extract_osm_features",
    # Graph construction
    "build_graph_and_augment",
    # Temporal utilities
    "build_time_binned_features",
    "integrate_static_features",
    "scale_features",
    "prepare_forecasting_data",
    "build_static_temporal_data",
    "save_static_temporal_data",
    "load_static_temporal_data",
    "create_temporal_dataset",
    # Data processing
    "preprocess_dataset",
    "move_batch_to_device",
    # Graph plotting utilities
    "plot_node_time_series",
    "plot_spatial_network",
    "plot_temporal_heatmap",
    # Model
    "create_model",
    "list_available_models",
    "get_model_info",
    "get_model_params",
    "list_pytorch_geometric_temporal_models",
    "get_pytorch_geometric_temporal_model_info",
    "get_pytorch_geometric_temporal_model_params",
    # Training utilities
    "train_model",
    "train_one_epoch",
    "evaluate_model",
    "get_predictions",
    "calculate_classification_metrics",
]
