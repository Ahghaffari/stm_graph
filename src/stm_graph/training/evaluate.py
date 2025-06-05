"""
Evaluation utilities for spatial-temporal graph models.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    average_precision_score,
)
from ..utils.training import create_time_aware_batches


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset,
    loss_fn: Callable,
    indices: List[int],
    to_device: bool,
    device: torch.device,
    batch_size: int = 1,
    min_batch_size: int = 1,
    task: str = "regression",
    use_tqdm: bool = True,
    fixed_batch_size: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Evaluate a model on the given dataset indices.
    Handles both single sample and batch processing.

    Args:
        model: PyTorch model to evaluate
        dataset: PyTorch Geometric Temporal dataset
        loss_fn: Loss function
        indices: List of indices to use for evaluation
        to_device: Whether to move data to device
        device: PyTorch device
        batch_size: Number of samples per batch (default: 1)
        min_batch_size: Minimum acceptable batch size (default: 1)
        task: "regression" or "classification"
        use_tqdm: Whether to show progress bar
        fixed_batch_size: Whether to use fixed batch size for batching

    Returns:
        For regression: average loss
        For classification: tuple of (average loss, metrics dictionary)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    hidden = None

    if batch_size > 1:
        batches = create_time_aware_batches(
            indices, batch_size, min_batch_size, fixed_batch_size
        )
        iterator = (
            tqdm(batches, desc="Evaluating", leave=False) if use_tqdm else batches
        )

        for batch_indices in iterator:
            # Prepare batch data
            batch_x = [dataset.processed_features[idx] for idx in batch_indices]
            batch_y = (
                [dataset.processed_targets[idx] for idx in batch_indices]
                if hasattr(dataset, "processed_targets")
                else []
            )

            batch_x = torch.stack(batch_x)
            if len(batch_y) > 0:
                batch_y = torch.stack(batch_y)
            else:
                batch_y = None

            edge_index = dataset.edge_index if hasattr(dataset, "edge_index") else None
            edge_weight = (
                dataset.edge_weight if hasattr(dataset, "edge_weight") else None
            )

            if to_device:
                batch_x = batch_x.to(device)
                if batch_y is not None:
                    batch_y = batch_y.to(device)
                if edge_index is not None:
                    edge_index = edge_index.to(device)
                if edge_weight is not None:
                    edge_weight = edge_weight.to(device)

            hidden, y_hat = model(batch_x, edge_index, edge_weight, hidden)

            if hidden is not None:
                hidden = hidden.detach()
                if hidden.size(0) != batch_x.size(0):
                    hidden = None

            loss = loss_fn(y_hat, batch_y)
            batch_loss = loss.item() * len(batch_indices)
            total_loss += batch_loss

            if task == "classification":
                y_hat_probs = torch.sigmoid(y_hat).cpu().numpy()
                batch_y_numpy = batch_y.cpu().numpy()
                all_preds.extend(y_hat_probs.reshape(-1).tolist())
                all_labels.extend(batch_y_numpy.reshape(-1).tolist())

    else:
        iterator = (
            tqdm(indices, desc="Evaluating", leave=False) if use_tqdm else indices
        )

        for idx in iterator:
            x_features = dataset.processed_features[idx]
            y_target = (
                dataset.processed_targets[idx]
                if hasattr(dataset, "processed_targets")
                else None
            )

            edge_index = dataset.edge_index if hasattr(dataset, "edge_index") else None
            edge_weight = (
                dataset.edge_weight if hasattr(dataset, "edge_weight") else None
            )

            if to_device:
                x_features = x_features.to(device)
                if y_target is not None:
                    y_target = y_target.to(device)
                if edge_index is not None:
                    edge_index = edge_index.to(device)
                if edge_weight is not None:
                    edge_weight = edge_weight.to(device)

            hidden, y_hat = model(
                x_features.unsqueeze(0), edge_index, edge_weight, hidden
            )
            y_hat = y_hat.squeeze(0)

            if hidden is not None:
                hidden = hidden.detach()
                if hidden.size(0) != 1:
                    hidden = None

            loss = loss_fn(y_hat, y_target)
            total_loss += loss.item()

            if task == "classification":
                y_hat_prob = torch.sigmoid(y_hat).cpu().numpy()
                y_target_numpy = y_target.cpu().numpy()
                all_preds.extend(y_hat_prob.reshape(-1).tolist())
                all_labels.extend(y_target_numpy.reshape(-1).tolist())

    avg_loss = total_loss / len(indices)

    if task == "classification":
        metrics = calculate_classification_metrics(
            np.array(all_preds), np.array(all_labels)
        )
        return avg_loss, metrics

    return avg_loss


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    dataset,
    indices: List[int],
    to_device: bool,
    device: torch.device,
    batch_size: int = 1,
    min_batch_size: int = 1,
    task: str = "regression",
    fixed_batch_size: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from model for the given dataset indices.

    Args:
        model: PyTorch model to evaluate
        dataset: PyTorch Geometric Temporal dataset
        indices: List of indices to use
        to_device: Whether to move data to device
        device: PyTorch device
        batch_size: Number of samples per batch (default: 1)
        min_batch_size: Minimum acceptable batch size (default: 1)
        task: "classification" or "regression"
        fixed_batch_size: Whether to use fixed batch size for batching

    Returns:
        Tuple of (predictions, true labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    hidden = None

    if batch_size > 1:
        batches = create_time_aware_batches(
            indices, batch_size, min_batch_size, fixed_batch_size
        )

        for batch_indices in tqdm(batches, desc="Getting predictions"):
            batch_x = [dataset.processed_features[idx] for idx in batch_indices]
            batch_y = (
                [dataset.processed_targets[idx] for idx in batch_indices]
                if hasattr(dataset, "processed_targets")
                else []
            )

            batch_x = torch.stack(batch_x)
            if len(batch_y) > 0:
                batch_y = torch.stack(batch_y)
            else:
                batch_y = None

            edge_index = dataset.edge_index if hasattr(dataset, "edge_index") else None
            edge_weight = (
                dataset.edge_weight if hasattr(dataset, "edge_weight") else None
            )

            if to_device:
                batch_x = batch_x.to(device)
                if edge_index is not None:
                    edge_index = edge_index.to(device)
                if edge_weight is not None:
                    edge_weight = edge_weight.to(device)

            hidden, y_hat = model(batch_x, edge_index, edge_weight, hidden)

            if hidden is not None:
                hidden = hidden.detach()
                if hidden.size(0) != batch_x.size(0):
                    hidden = None

            if task == "classification":
                y_hat = torch.sigmoid(y_hat)

            all_preds.extend(y_hat.cpu().numpy().reshape(-1).tolist())
            if batch_y is not None:
                all_labels.extend(batch_y.cpu().numpy().reshape(-1).tolist())

    else:
        for idx in tqdm(indices, desc="Getting predictions"):
            x_features = dataset.processed_features[idx]
            y_target = (
                dataset.processed_targets[idx]
                if hasattr(dataset, "processed_targets")
                else None
            )

            edge_index = dataset.edge_index if hasattr(dataset, "edge_index") else None
            edge_weight = (
                dataset.edge_weight if hasattr(dataset, "edge_weight") else None
            )

            if to_device:
                x_features = x_features.to(device)
                if edge_index is not None:
                    edge_index = edge_index.to(device)
                if edge_weight is not None:
                    edge_weight = edge_weight.to(device)

            hidden, y_hat = model(
                x_features.unsqueeze(0), edge_index, edge_weight, hidden
            )
            y_hat = y_hat.squeeze(0)

            if hidden is not None:
                hidden = hidden.detach()
                if hidden.size(0) != 1:
                    hidden = None

            if task == "classification":
                y_hat = torch.sigmoid(y_hat)

            all_preds.extend(y_hat.cpu().numpy().reshape(-1).tolist())
            if y_target is not None:
                all_labels.extend(y_target.cpu().numpy().reshape(-1).tolist())

    return np.array(all_preds), np.array(all_labels)


def calculate_classification_metrics(predictions, labels):
    """
    Calculate classification metrics from predictions and labels.

    Args:
        predictions: NumPy array of predicted probabilities
        labels: NumPy array of ground truth labels

    Returns:
        Dictionary of classification metrics
    """
    binary_preds = (predictions > 0.5).astype(int)
    
    metrics = {}

    metrics["accuracy"] = accuracy_score(labels, binary_preds)
    metrics["balanced_accuracy"] = balanced_accuracy_score(labels, binary_preds)
    metrics["precision"] = precision_score(
        labels, binary_preds, average="binary", zero_division=0
    )
    metrics["recall"] = recall_score(
        labels, binary_preds, average="binary", zero_division=0
    )
    metrics["f1"] = f1_score(labels, binary_preds, average="binary", zero_division=0)
    metrics["mcc"] = matthews_corrcoef(labels, binary_preds)

    tn = np.sum((labels == 0) & (binary_preds == 0))
    fp = np.sum((labels == 0) & (binary_preds == 1))
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        metrics["auc"] = roc_auc_score(labels, predictions)
    except:
        metrics["auc"] = 0.0

    try:
        metrics["pr_auc"] = average_precision_score(labels, predictions)
    except:
        metrics["pr_auc"] = 0.0

    return metrics
