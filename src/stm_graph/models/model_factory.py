"""
Factory for creating different spatial-temporal graph models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import importlib
import inspect

from .model_registry import ModelRegistry, register_models
from .model_adapters import create_pytorch_geometric_temporal_adapter
from .model_discovery import ModelDiscoveryService


register_models()


def create_model(
    model_name: str, source: str = "auto", task: str = "regression", **model_params
) -> nn.Module:
    """
    Create a model instance based on the model name and parameters.

    Args:
        model_name: Name of the model to create
        source: Source of the model - "custom" (STM-Graph's models) or "pytorch_geometric_temporal"
        task: "regression" or "classification"
        **model_params: Model-specific parameters

    Returns:
        PyTorch model instance
    """
    if source == "auto":
        try:
            return create_model(model_name, "custom", task, **model_params)
        except ValueError:
            return create_model(
                model_name, "pytorch_geometric_temporal", task, **model_params
            )

    elif source == "custom":
        model_name = model_name.lower()
        model_info = ModelRegistry.get_model_info(model_name)

        if not model_info:
            available_models = ModelRegistry.list_models()
            raise ValueError(
                f"Unknown model type: {model_name}. Available custom models: {', '.join(available_models)}"
            )

        module_path, class_name = model_info.class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        missing_params = []
        for param in model_info.parameters:
            if param.required and param.name not in model_params:
                missing_params.append(param.name)

        if missing_params:
            raise ValueError(
                f"Missing required parameters for {model_name}: {', '.join(missing_params)}"
            )

        try:
            valid_params = {}
            for param in model_info.parameters:
                if param.name in model_params:
                    valid_params[param.name] = model_params[param.name]
                elif not param.required:
                    valid_params[param.name] = param.default

            model = model_class(**valid_params)
        except Exception as e:
            raise ValueError(f"Error creating {model_name} model: {str(e)}")

        if not model_info.supports_batching:
            model = BatchHandlerWrapper(model)

    elif source == "pytorch_geometric_temporal":
        special_format_models = [
            "ASTGCN",
            "MSTGCNBlock",
            "MSTGCN",
            "BatchedDCRNN",
            "MTGNN",
            "GMAN",
            "AAGCN",
            "STConv",
        ]
        if model_name in special_format_models:
            raise ValueError(
                f"Model {model_name} has a special input format requirement that is not yet supported. "
                f"Support for this model will be added in a future release."
            )
        discovered_models = (
            ModelDiscoveryService.discover_pytorch_geometric_temporal_models()
        )

        if model_name not in discovered_models:
            available_models = list(discovered_models.keys())
            raise ValueError(
                f"Unknown model type: {model_name}. Available PyTorch Geometric Temporal models: {', '.join(available_models)}"
            )

        model_params_info = discovered_models[model_name]["params"]

        missing_params = []
        for param_name, param_info in model_params_info.items():
            if param_info["required"] and param_name not in model_params:
                missing_params.append(param_name)

        if missing_params:
            raise ValueError(
                f"Missing required parameters for {model_name}: {', '.join(missing_params)}"
            )

        batch_handling = discovered_models[model_name]["batch_handling"]

        try:
            model = create_pytorch_geometric_temporal_adapter(
                model_name, batch_handling, **model_params
            )
        except Exception as e:
            raise ValueError(f"Error creating {model_name} model: {str(e)}")

    else:
        raise ValueError(
            f"Invalid source: {source}. Must be 'auto', 'custom', or 'pytorch_geometric_temporal'"
        )

    if task == "classification":
        model = ClassificationWrapper(model)

    return model


def list_pytorch_geometric_temporal_models() -> None:
    """
    Print information about available models from PyTorch Geometric Temporal.
    """
    discovered_models = (
        ModelDiscoveryService.discover_pytorch_geometric_temporal_models()
    )

    unsupported_models = [
        "ASTGCN",
        "MSTGCNBlock",
        "MSTGCN",
        "BatchedDCRNN",
        "MTGNN",
        "GMAN",
        "AAGCN",
        "STConv",
    ]
    grouped_models = {}

    for name, info in discovered_models.items():
        model_type = info["type"]
        batch_handling = info["batch_handling"]

        if model_type not in grouped_models:
            grouped_models[model_type] = {}

        if batch_handling not in grouped_models[model_type]:
            grouped_models[model_type][batch_handling] = []

        grouped_models[model_type][batch_handling].append(name)

    print(f"Available PyTorch Geometric Temporal Models ({len(discovered_models)}):")

    batch_handling_display = {
        "standard": "Standard (no batch support)",
        "fixed_batch": "Fixed Batch Size",
        "native_batch": "Native Batch Support",
    }

    for model_type, batch_groups in grouped_models.items():
        print(f"\n{model_type.title()} Models:")

        for batch_handling, models in batch_groups.items():
            if batch_handling != "special_format":  # Skip special format models
                print(
                    f"  {batch_handling_display.get(batch_handling, batch_handling)}:"
                )
                for name in sorted(models):
                    print(f"    - {name}")

    print("\nModels Coming in Future Releases (Special Format Support):")
    for name in sorted(unsupported_models):
        print(f"  - {name}")

    print(
        "\nFor detailed information about a model, use get_pytorch_geometric_temporal_model_info(model_name)"
    )


def get_pytorch_geometric_temporal_model_info(model_name: str) -> None:
    """
    Print detailed information about a specific PyTorch Geometric Temporal model.

    Args:
        model_name: Name of the model to get information about
    """
    discovered_models = (
        ModelDiscoveryService.discover_pytorch_geometric_temporal_models()
    )

    if model_name not in discovered_models:
        available_models = list(discovered_models.keys())
        raise ValueError(
            f"Model {model_name} not found. Available models: {', '.join(available_models)}"
        )

    model_info = discovered_models[model_name]

    print(f"Model: {model_name}")
    print(f"Type: {model_info['type'].title()} Model")
    print(f"Module: {model_info['module']}")

    batch_handling = model_info["batch_handling"]
    batch_support_message = {
        "standard": "No batch support (processes one sample at a time)",
        "fixed_batch": "Fixed batch size (requires batch_size parameter at initialization)",
        "native_batch": "Native batch support",
        "special_format": "Special input format (see documentation)",
    }.get(batch_handling, "Unknown")

    print(f"Batch handling: {batch_handling}")
    print(f"Batch support: {batch_support_message}")

    print("\nParameters:")
    for param_name, param_info in model_info["params"].items():
        req = (
            "(Required)"
            if param_info["required"]
            else f"(Default: {param_info['default']})"
        )
        param_type = param_info["type"]
        type_name = (
            param_type.__name__
            if hasattr(param_type, "__name__")
            else str(param_type).replace("typing.", "")
        )
        print(f"  {param_name}: {type_name} {req}")

    if model_info["doc"]:
        print("\nDescription:")
        print(f"  {model_info['doc'].strip()}")


def get_pytorch_geometric_temporal_model_params(model_name: str) -> Dict[str, Any]:
    """
    Get default parameters for a specific PyTorch Geometric Temporal model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary of default parameter values
    """
    return ModelDiscoveryService.get_model_parameter_defaults(model_name)


def list_available_models(verbose: bool = False) -> None:
    """
    Print information about available custom models.

    Args:
        verbose: Whether to print detailed information for each model
    """
    if verbose:
        ModelRegistry.print_model_info()
    else:
        print("Available custom models:")
        for name in sorted(ModelRegistry.list_models()):
            info = ModelRegistry.get_model_info(name)
            print(f"- {name}: {info.description}")
        print(
            "\nFor detailed information about a model, use get_model_info(model_name)"
        )

    print(
        "\nTo list PyTorch Geometric Temporal models, use list_pytorch_geometric_temporal_models()"
    )


def get_model_info(model_name: str) -> None:
    """
    Print detailed information about a specific custom model.

    Args:
        model_name: Name of the model to get information about
    """
    ModelRegistry.print_model_info(model_name)


def get_model_params(model_name: str) -> Dict[str, Any]:
    """
    Get default parameters for a specific custom model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary of default parameter values
    """
    model_info = ModelRegistry.get_model_info(model_name)
    if not model_info:
        available_models = ModelRegistry.list_models()
        raise ValueError(
            f"Unknown model type: {model_name}. Available models: {', '.join(available_models)}"
        )

    params = {}
    for param in model_info.parameters:
        if not param.required:  # Only include parameters with defaults
            params[param.name] = param.default

    return params


class BatchHandlerWrapper(nn.Module):
    """
    Wrapper for models that don't support batch processing natively.

    This wrapper handles batch processing by sequentially processing each sample.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the wrapper.

        Args:
            model: The model to wrap
        """
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_weight=None, h=None):
        """
        Forward pass that handles batched inputs.

        Args:
            x: Input features [batch_size, num_nodes, in_channels] or [batch_size, in_channels]
            edge_index: Graph edge indices
            edge_weight: Edge weights
            h: Hidden state (optional)

        Returns:
            Updated hidden state and output
        """
        # Check if input is batched
        if x.dim() <= 2:  # Not batched
            return self.model(x, edge_index, edge_weight, h)

        batch_size = x.size(0)
        device = x.device

        all_h = []
        all_out = []

        for i in range(batch_size):
            sample_x = x[i]  # [num_nodes, features] or just [features]

            sample_h = None
            if h is not None:
                if h.dim() == 3:  # [batch, nodes, hidden]
                    sample_h = h[i : i + 1]
                else:
                    sample_h = h

            new_h, out = self.model(sample_x, edge_index, edge_weight, sample_h)

            all_h.append(new_h)
            all_out.append(out)

        if all_h[0].dim() >= 2:
            combined_h = torch.cat(all_h, dim=0)
        else:
            combined_h = torch.stack(all_h)

        if all_out[0].dim() >= 2:
            combined_out = torch.cat(all_out, dim=0)
        else:
            combined_out = torch.stack(all_out)

        return combined_h, combined_out


class ClassificationWrapper(nn.Module):
    """
    Wrapper for classification models.

    This wrapper enables models for binary classification tasks.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the wrapper.

        Args:
            model: The model to wrap
        """
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        """
        Forward pass for classification.

        Args:
            *args: Arguments to pass to the model
            **kwargs: Keyword arguments to pass to the model

        Returns:
            Model output without sigmoid (sigmoid is applied in the loss function)
        """
        return self.model(*args, **kwargs)
