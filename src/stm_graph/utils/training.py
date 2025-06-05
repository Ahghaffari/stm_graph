"""Preprocessing functions for training and evaluation of models."""

import numpy as np
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


def preprocess_dataset(dataset):
    """Convert dataset components to PyTorch tensors once before training."""
    print("Preprocessing dataset...")

    processed_features = []
    processed_targets = []

    for i in range(len(dataset.features)):
        if isinstance(dataset.features[i], np.ndarray):
            processed_features.append(
                torch.tensor(dataset.features[i], dtype=torch.float32)
            )
        elif isinstance(dataset.features[i], torch.Tensor):
            processed_features.append(dataset.features[i])
        else:
            processed_features.append(
                torch.tensor(dataset.features[i], dtype=torch.float32)
            )

        if (
            hasattr(dataset, "targets")
            and dataset.targets is not None
            and dataset.targets[i] is not None
        ):
            if isinstance(dataset.targets[i], np.ndarray):
                processed_targets.append(
                    torch.tensor(dataset.targets[i], dtype=torch.float32)
                )
            elif isinstance(dataset.targets[i], torch.Tensor):
                processed_targets.append(dataset.targets[i])
            else:
                processed_targets.append(
                    torch.tensor(dataset.targets[i], dtype=torch.float32)
                )

    if hasattr(dataset, "edge_index"):
        if isinstance(dataset.edge_index, list):
            dataset.edge_index = torch.tensor(dataset.edge_index, dtype=torch.long)
        elif isinstance(dataset.edge_index, np.ndarray):
            dataset.edge_index = torch.tensor(dataset.edge_index, dtype=torch.long)

    if hasattr(dataset, "edge_weight"):
        if dataset.edge_weight is not None:
            if isinstance(dataset.edge_weight, list):
                dataset.edge_weight = torch.tensor(
                    dataset.edge_weight, dtype=torch.float32
                )
            elif isinstance(dataset.edge_weight, np.ndarray):
                dataset.edge_weight = torch.tensor(
                    dataset.edge_weight, dtype=torch.float32
                )

    dataset.processed_features = processed_features
    if len(processed_targets) > 0:
        dataset.processed_targets = processed_targets

    print(f"Preprocessing complete: {len(processed_features)} time steps processed")
    return dataset


def create_time_aware_batches(
    indices, batch_size, min_batch_size=1, fixed_batch_size=False
):
    """
    Create batches respecting temporal order of indices.

    Args:
        indices: List of indices to batch
        batch_size: Size of each batch
        min_batch_size: Minimum size requirement for a batch to be included
        fixed_batch_size: If True, all batches will have exactly batch_size elements
                         (last batch will use the last batch_size elements)

    Returns:
        List of batches where each batch is a list of indices
    """
    if fixed_batch_size:
        batches = []

        num_full_batches = len(indices) // batch_size
        for i in range(num_full_batches):
            start_idx = i * batch_size
            batches.append(indices[start_idx : start_idx + batch_size])

        remaining = len(indices) % batch_size
        if remaining > 0 and len(indices) >= batch_size:
            batches.append(indices[-batch_size:])

        return batches
    else:
        batches = []

        for i in range(0, len(indices), batch_size):
            batch = indices[i : i + batch_size]
            if len(batch) >= min_batch_size:
                batches.append(batch)

        return batches
