"""
Service for discovering available models from PyTorch Geometric Temporal.
"""

import importlib
import inspect
from typing import Dict, List, Set, Type, Any
import torch.nn as nn


class ModelDiscoveryService:
    """
    Service for discovering available models from PyTorch Geometric Temporal.
    """

    @staticmethod
    def discover_pytorch_geometric_temporal_models() -> Dict[str, Dict[str, Any]]:
        """
        Discover all available models in PyTorch Geometric Temporal.

        Returns:
            Dictionary mapping model names to information about the model
        """
        discovered_models = {}

        modules = [
            "torch_geometric_temporal.nn.recurrent",
            "torch_geometric_temporal.nn.attention",
        ]

        excluded_models = {
            "MessagePassing",
            "GCNCheb",
            "GCLSTM",
            "GConvGRU",
            "GConvLSTM",
            "BaseGNN",
            "TimeEncoder",
            "Node2Vec",
            # Special format models that require custom handling - will be added in future release
            "ASTGCN",
            "MSTGCNBlock",
            "MSTGCN",
            "BatchedDCRNN",
            "MTGNN",
            "GMAN",
            "AAGCNBlock",
            "AAGCN",
            "STConv",
        }

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, nn.Module)
                        and name not in excluded_models
                    ):

                        sig = inspect.signature(obj.__init__)
                        params = {}

                        for param_name, param in sig.parameters.items():
                            if param_name == "self":
                                continue

                            default = (
                                param.default
                                if param.default != inspect.Parameter.empty
                                else None
                            )
                            required = param.default == inspect.Parameter.empty

                            param_type = (
                                param.annotation
                                if param.annotation != inspect.Parameter.empty
                                else Any
                            )

                            params[param_name] = {
                                "default": default,
                                "required": required,
                                "type": param_type,
                            }

                        model_type = (
                            "recurrent" if "recurrent" in module_name else "attention"
                        )
                        batch_handling = "standard"

                        if "batch_size" in params:
                            batch_handling = "fixed_batch"

                        elif name.endswith("2"):
                            batch_handling = "native_batch"

                        elif name.startswith("Batched"):
                            batch_handling = "native_batch"

                        docstring = obj.__doc__ or ""
                        if (
                            "batch_size" in docstring.lower()
                            or "batches" in docstring.lower()
                        ):
                            batch_handling = "native_batch"

                        discovered_models[name] = {
                            "module": module_name,
                            "class": obj,
                            "params": params,
                            "type": model_type,
                            "batch_handling": batch_handling,
                            "doc": obj.__doc__
                            or f"PyTorch Geometric Temporal {name} model",
                        }
            except ImportError:
                print(
                    f"Warning: Could not import module {module_name}. Make sure PyTorch Geometric Temporal is installed."
                )

        return discovered_models

    @staticmethod
    def get_model_parameter_defaults(model_name: str) -> Dict[str, Any]:
        """
        Get default parameter values for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of parameter names to default values
        """
        models = ModelDiscoveryService.discover_pytorch_geometric_temporal_models()

        if model_name not in models:
            raise ValueError(
                f"Model {model_name} not found in PyTorch Geometric Temporal. Available models: {', '.join(models.keys())}"
            )

        defaults = {}
        for param_name, param_info in models[model_name]["params"].items():
            if not param_info["required"]:  # Only include parameters with defaults
                defaults[param_name] = param_info["default"]

        return defaults

    @staticmethod
    def get_batch_handling(model_name: str) -> str:
        """
        Get the batch handling mode for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Batch handling mode: "standard", "fixed_batch", "native_batch", or "special_format"
        """
        models = ModelDiscoveryService.discover_pytorch_geometric_temporal_models()

        if model_name not in models:
            raise ValueError(
                f"Model {model_name} not found in PyTorch Geometric Temporal."
            )

        return models[model_name]["batch_handling"]
