"""
Adapters for PyTorch Geometric Temporal models.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
import importlib
import inspect


class BasePyTorchGeometricTemporalAdapter(nn.Module):
    """
    Base adapter class for PyTorch Geometric Temporal models.
    """

    def __init__(self, model_name: str, batch_handling: str, **kwargs):
        """
        Initialize the base adapter.

        Args:
            model_name: Name of the PyTorch Geometric Temporal model
            batch_handling: Type of batch handling ("standard", "fixed_batch", "native_batch", "special_format")
            **kwargs: Parameters to pass to the model constructor
        """
        super().__init__()
        self.model_name = model_name
        self.batch_handling = batch_handling
        self.kwargs = kwargs

        try:
            # First try recurrent models
            module = importlib.import_module("torch_geometric_temporal.nn.recurrent")
            if hasattr(module, model_name):
                model_class = getattr(module, model_name)
            else:
                # Then try attention models
                module = importlib.import_module(
                    "torch_geometric_temporal.nn.attention"
                )
                if hasattr(module, model_name):
                    model_class = getattr(module, model_name)
                else:
                    # Finally try embedding models
                    module = importlib.import_module(
                        "torch_geometric_temporal.nn.embedding"
                    )
                    model_class = getattr(module, model_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Could not find model {model_name} in PyTorch Geometric Temporal: {str(e)}"
            )

        self.model_class = model_class
        self.module = module

        self.model = None

        self.is_recurrent = "recurrent" in model_class.__module__

    def supports_batching(self) -> bool:
        """
        Check if this model supports batching.

        Returns:
            True if the model natively supports batch processing
        """
        return self.batch_handling in ["native_batch", "fixed_batch", "special_format"]

    def get_batch_size_constraint(self) -> Optional[int]:
        """
        Get the required batch size for this model, if any.

        Returns:
            Required batch size or None if flexible
        """
        if self.batch_handling == "fixed_batch":
            return self.kwargs.get("batch_size", 1)
        return None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Abstract forward method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward")


class StandardAdapter(BasePyTorchGeometricTemporalAdapter):
    """
    Adapter for standard PyTorch Geometric Temporal models that expect [num_nodes, in_channels].
    These models don't support batching natively and need to be processed one sample at a time.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, "standard", **kwargs)
        # Initialize model
        self.model = self.model_class(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process each batch item individually.

        Args:
            x: Input features [batch_size, num_nodes, in_channels] or [num_nodes, in_channels]
            edge_index: Graph edge indices
            edge_weight: Edge weights
            h: Hidden state for recurrent models

        Returns:
            Updated hidden state and output
        """
        if x.dim() == 3:  # [batch_size, num_nodes, in_channels]
            batch_size = x.size(0)
            outputs = []
            hidden_states = []

            for i in range(batch_size):
                batch_x = x[i]  # [num_nodes, in_channels]

                batch_h = None
                if h is not None:
                    if h.dim() == 3:  # [batch_size, num_nodes, hidden_dim]
                        batch_h = h[i : i + 1]  # Keep batch dimension
                    else:
                        batch_h = h

                if self.is_recurrent:
                    if "LSTM" in self.model_name or "GC-LSTM" in self.model_name:
                        # LSTM models return (h, c)
                        new_h, new_c = self.model(
                            batch_x, edge_index, edge_weight, batch_h
                        )
                        hidden_states.append(new_h)
                        outputs.append(new_h)  # Use hidden state as output
                    else:
                        # GRU-like models return h
                        new_h = self.model(batch_x, edge_index, edge_weight, batch_h)
                        hidden_states.append(new_h)
                        outputs.append(new_h)
                else:
                    # Attention models directly return output
                    output = self.model(batch_x, edge_index, edge_weight)
                    outputs.append(output)

            combined_output = torch.stack(outputs)

            if self.is_recurrent:
                combined_h = torch.stack(hidden_states)
                return combined_h, combined_output
            else:
                return None, combined_output
        else:
            # Single sample (no batch dimension)
            if self.is_recurrent:
                if "LSTM" in self.model_name or "GC-LSTM" in self.model_name:
                    new_h, new_c = self.model(x, edge_index, edge_weight, h)
                    return new_h, new_h
                else:
                    new_h = self.model(x, edge_index, edge_weight, h)
                    return new_h, new_h
            else:
                output = self.model(x, edge_index, edge_weight)
                return None, output


class FixedBatchAdapter(BasePyTorchGeometricTemporalAdapter):
    """
    Adapter for models like TGCN2 that require a fixed batch_size at initialization.
    This adapter will create a new model instance if the batch size changes.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, "fixed_batch", **kwargs)
        self.batch_size = kwargs.get("batch_size", 1)
        self.model = self.model_class(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input with fixed batch size.

        Args:
            x: Input features [batch_size, num_nodes, in_channels]
            edge_index: Graph edge indices
            edge_weight: Edge weights
            h: Hidden state for recurrent models

        Returns:
            Updated hidden state and output
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, num_nodes, in_channels]

        actual_batch_size = x.size(0)

        # If batch size doesn't match the initialized model, recreate the model
        if actual_batch_size != self.batch_size:
            print(
                f"Warning: Recreating {self.model_name} model with batch_size={actual_batch_size} (was {self.batch_size})"
            )

            new_kwargs = self.kwargs.copy()
            new_kwargs["batch_size"] = actual_batch_size

            self.model = self.model_class(**new_kwargs)
            self.batch_size = actual_batch_size
            self.model = self.model.to(x.device)

        if self.is_recurrent:
            out = self.model(x, edge_index, edge_weight, h)
            return out, out
        else:
            out = self.model(x, edge_index, edge_weight)
            return None, out


class NativeBatchAdapter(BasePyTorchGeometricTemporalAdapter):
    """
    Adapter for models that natively support batch processing without needing special handling.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, "native_batch", **kwargs)
        self.model = self.model_class(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process with native batch support.

        Args:
            x: Input features [batch_size, num_nodes, in_channels] or [num_nodes, in_channels]
            edge_index: Graph edge indices
            edge_weight: Edge weights
            h: Hidden state for recurrent models

        Returns:
            Updated hidden state and output
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, num_nodes, in_channels]

        if self.is_recurrent:
            if "LSTM" in self.model_name or "GC-LSTM" in self.model_name:
                new_h, new_c = self.model(x, edge_index, edge_weight, h)
                return new_h, new_h
            else:
                new_h = self.model(x, edge_index, edge_weight, h)
                return new_h, new_h
        else:
            output = self.model(x, edge_index, edge_weight)
            return None, output


def create_pytorch_geometric_temporal_adapter(
    model_name: str, batch_handling: str, **kwargs
) -> BasePyTorchGeometricTemporalAdapter:
    """
    Create an appropriate adapter for the given model.

    Args:
        model_name: PyTorch Geometric Temporal model name
        batch_handling: Type of batch handling
        **kwargs: Parameters for the model

    Returns:
        Model adapter instance
    """
    if batch_handling == "fixed_batch":
        return FixedBatchAdapter(model_name, **kwargs)
    elif batch_handling == "native_batch":
        return NativeBatchAdapter(model_name, **kwargs)
    else:  # "standard"
        return StandardAdapter(model_name, **kwargs)
