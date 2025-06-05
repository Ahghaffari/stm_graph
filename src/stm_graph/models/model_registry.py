"""
Registry of available models with parameter information.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Type, Optional, Union
import inspect
import torch.nn as nn


@dataclass
class ModelParameter:
    """Information about a model parameter"""

    name: str
    type: Type
    default: Any
    description: str
    required: bool = False


@dataclass
class ModelInfo:
    """Information about a model"""

    name: str
    class_path: str
    description: str
    parameters: List[ModelParameter] = field(default_factory=list)
    supports_batching: bool = True
    example_config: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Registry for all available models in STM-Graph"""

    _registry: Dict[str, ModelInfo] = {}

    @classmethod
    def register(
        cls,
        name: str,
        model_class: Any,
        description: str,
        supports_batching: bool,
        example_config: Dict[str, Any] = None,
        param_descriptions: Dict[str, str] = None,
    ) -> None:
        """Register a model with the registry"""
        # Extract parameters from the model's __init__ method
        sig = inspect.signature(model_class.__init__)
        params = []

        if param_descriptions is None:
            param_descriptions = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )
            default_value = (
                param.default if param.default != inspect.Parameter.empty else None
            )
            required = param.default == inspect.Parameter.empty

            # Get description from provided dict or use default
            param_description = param_descriptions.get(
                param_name, f"Parameter {param_name} for {model_class.__name__}"
            )

            # Add parameter info
            params.append(
                ModelParameter(
                    name=param_name,
                    type=param_type,
                    default=default_value,
                    description=param_description,
                    required=required,
                )
            )

        # Create ModelInfo
        model_info = ModelInfo(
            name=name,
            class_path=f"{model_class.__module__}.{model_class.__name__}",
            description=description,
            parameters=params,
            supports_batching=supports_batching,
            example_config=example_config or {},
        )

        # Add to registry
        cls._registry[name.lower()] = model_info

    @classmethod
    def get_model_info(cls, name: str) -> Optional[ModelInfo]:
        """Get information about a registered model"""
        return cls._registry.get(name.lower())

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models"""
        return list(cls._registry.keys())

    @classmethod
    def print_model_info(cls, name: str = None) -> None:
        """Print information about a model or all models"""
        if name:
            info = cls.get_model_info(name)
            if not info:
                print(
                    f"Model '{name}' not found. Available models: {', '.join(cls.list_models())}"
                )
                return

            print(f"Model: {info.name}")
            print(f"Description: {info.description}")
            print(f"Supports batching: {info.supports_batching}")
            print("\nParameters:")
            for param in info.parameters:
                req = "(Required)" if param.required else f"(Default: {param.default})"
                try:
                    type_name = param.type.__name__
                except AttributeError:
                    type_name = str(param.type).replace("typing.", "")
                print(f"  {param.name}: {type_name} {req}")
                print(f"    {param.description}")

            print("\nExample configuration:")
            for k, v in info.example_config.items():
                print(f"  {k}: {v}")
        else:
            print("Available Models:")
            for name, info in sorted(cls._registry.items()):
                print(f"- {name}: {info.description}")
            print(
                "\nUse print_model_info(model_name) to see details about a specific model."
            )


# Register models with detailed information
def register_models():
    """Register all available models with the registry"""

    from .gcn import GCN
    from .tgcn import TGCN
    from .dcrnn import RecurrentGCN_DCRNN
    from .agcrn import RecurrentGCN_AGCRN
    from .stgcn import STGCN

    # Parameter descriptions
    gcn_params = {
        "in_channels": "Number of input features",
        "hidden_channels": "Number of hidden features",
        "out_channels": "Number of output features",
        "dropout": "Dropout probability",
    }

    tgcn_params = {
        "in_channels": "Number of input features",
        "out_channels": "Number of output features",
        "batch_size": "Batch size for TGCN2 implementation",
    }

    dcrnn_params = {
        "in_channels": "Number of input features",
        "out_channels": "Number of output features",
        "hidden_dim": "Size of hidden layers",
        "K": "Filter size of graph diffusion convolution",
        "dropout": "Dropout probability",
    }

    agcrn_params = {
        "num_nodes": "Number of nodes in the graph",
        "in_channels": "Number of input features per node",
        "hidden_dim": "Size of hidden layers",
        "out_channels": "Number of output features",
        "k": "Number of Chebyshev filter taps",
        "embedding_dimensions": "Size of node embedding",
    }

    stgcn_params = {
        "num_nodes": "Number of nodes in the graph",
        "in_channels": "Number of input features per node",
        "hidden_channels": "Number of hidden features in ST-Conv blocks",
        "out_channels": "Number of output features",
        "K": "Order of Chebyshev polynomials",
        "kernel_size": "Size of temporal convolution kernel",
        "dropout": "Dropout probability",
        "num_st_blocks": "Number of ST-Conv blocks",
    }

    ModelRegistry.register(
        name="GCN",
        model_class=GCN,
        description="Graph Convolutional Network for simple graph learning tasks",
        supports_batching=True,
        param_descriptions=gcn_params,
        example_config={
            "in_channels": 64,
            "hidden_channels": 32,
            "out_channels": 1,
            "dropout": 0.2,
        },
    )

    ModelRegistry.register(
        name="TGCN",
        model_class=TGCN,
        description="Temporal Graph Convolutional Network with recurrent architecture",
        supports_batching=True,
        param_descriptions=tgcn_params,
        example_config={"in_channels": 64, "out_channels": 32, "batch_size": 4},
    )

    ModelRegistry.register(
        name="DCRNN",
        model_class=RecurrentGCN_DCRNN,
        description="Diffusion Convolutional Recurrent Neural Network for traffic forecasting",
        supports_batching=True,
        param_descriptions=dcrnn_params,
        example_config={
            "in_channels": 64,
            "out_channels": 1,
            "hidden_dim": 32,
            "K": 3,
            "dropout": 0.2,
        },
    )

    ModelRegistry.register(
        name="AGCRN",
        model_class=RecurrentGCN_AGCRN,
        description="Adaptive Graph Convolutional Recurrent Network with node-specific parameters",
        supports_batching=True,
        param_descriptions=agcrn_params,
        example_config={
            "num_nodes": 100,
            "in_channels": 64,
            "out_channels": 1,
            "k": 3,
            "embedding_dimensions": 32,
        },
    )

    ModelRegistry.register(
        name="STGCN",
        model_class=STGCN,
        description="Spatio-Temporal Graph Convolutional Network with temporal gating",
        supports_batching=True,
        param_descriptions=stgcn_params,
        example_config={
            "num_nodes": 100,
            "in_channels": 64,
            "hidden_channels": 32,
            "out_channels": 1,
            "kernel_size": 3,
            "K": 3,
            "dropout": 0.2,
            "num_st_blocks": 2,
        },
    )
