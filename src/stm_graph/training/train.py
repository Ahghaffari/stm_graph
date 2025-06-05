"""
Training utilities for spatial-temporal graph models.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Callable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import logging
from datetime import datetime

from ..utils.device import get_device, clear_gpu_memory
from ..utils.wandb_login import authenticate_wandb
from ..utils.training import create_time_aware_batches, preprocess_dataset
from ..data.loader import move_batch_to_device
from .evaluate import evaluate_model


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    dataset,
    indices: List[int],
    to_device: bool,
    device: torch.device,
    batch_size: int = 1,
    min_batch_size: int = 1,
    task: str = "regression",
    use_tqdm: bool = True,
    time_aware: bool = True,
    fixed_batch_size: bool = False,
) -> float:
    """Train model for one epoch using the provided dataset and indices."""
    model.train()
    total_loss = 0.0
    hidden = None

    training_indices = (
        indices if time_aware else np.random.permutation(indices).tolist()
    )

    if batch_size > 1:
        batches = create_time_aware_batches(
            training_indices, batch_size, min_batch_size, fixed_batch_size
        )
        iterator = (
            tqdm(batches, desc="Training batches", leave=False) if use_tqdm else batches
        )

        for batch_indices in iterator:
            optimizer.zero_grad()

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

            loss = loss_fn(y_hat, batch_y)

            loss.backward()
            optimizer.step()

            if hidden is not None:
                hidden = hidden.detach()
                if hidden.size(0) != batch_x.size(0):
                    hidden = None

            batch_loss = loss.item() * len(batch_indices)
            total_loss += batch_loss

            if use_tqdm:
                iterator.set_postfix({"loss": loss.item()})
    else:
        iterator = (
            tqdm(training_indices, desc="Training samples", leave=False)
            if use_tqdm
            else training_indices
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

            optimizer.zero_grad()
            hidden, y_hat = model(
                x_features.unsqueeze(0), edge_index, edge_weight, hidden
            )
            y_hat = y_hat.squeeze(0)

            loss = loss_fn(y_hat, y_target)

            loss.backward()
            optimizer.step()

            if hidden is not None:
                hidden = hidden.detach()
                if hidden.size(0) != 1:
                    hidden = None

            total_loss += loss.item()

            if use_tqdm:
                iterator.set_postfix({"loss": loss.item()})

    return total_loss / len(indices)


def train_model(
    model: nn.Module,
    dataset,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    scheduler_type: str = "step",
    lr_decay_epochs: int = 10,
    lr_decay_factor: float = 0.1,
    lr_patience: int = 5,
    task: str = "regression",
    test_size: float = 0.15,
    val_size: float = 0.15,
    num_epochs: int = 100,
    batch_to_device: bool = True,
    early_stopping: bool = True,
    patience: int = 10,
    batch_size: int = 1,
    time_aware: bool = True,
    min_batch_size: int = 1,
    use_nested_tqdm: bool = False,
    wandb_api_key: Optional[str] = None,
    wandb_project: str = "stm_graph",
    experiment_name: str = "stm_graph",
    log_dir: str = "logs",
    use_wandb: bool = False,
    device: Optional[torch.device] = None,
    fixed_batch_size: bool = False,
) -> Dict:
    """
    Train a spatial-temporal model with the provided parameters.

    Args:
        model: PyTorch model to train
        dataset: PyTorch Geometric Temporal dataset
        optimizer_name: Name of optimizer to use ('adam', 'sgd', 'rmsprop')
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay (L2 penalty)
        momentum: Momentum factor for SGD
        scheduler_type: Learning rate scheduler type ('step', 'plateau', None)
        lr_decay_epochs: Number of epochs between learning rate decay (for StepLR)
        lr_decay_factor: Multiplicative factor for learning rate decay
        lr_patience: Number of epochs without improvement before reducing LR (for ReduceLROnPlateau)
        task: Type of task ('regression' or 'classification')
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        num_epochs: Maximum number of training epochs
        batch_to_device: Whether to move batch data to device
        early_stopping: Whether to use early stopping
        patience: Number of epochs without improvement before early stopping
        batch_size: Number of samples per batch
        use_nested_tqdm: Whether to use progress bars for batches
        wandb_project: Name of Weights & Biases project
        wandb_api_key: Weights & Biases API key (if not logged in)
        experiment_name: Name for the experiment (used for saving models)
        log_dir: Directory to save logs and models
        use_wandb: Whether to log metrics to Weights & Biases
        device: Device to use for training (if None, will use GPU if available)

    Returns:
        Dictionary with training results
    """
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Configure logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log experiment start
    logger.info(f"{'='*20} EXPERIMENT: {experiment_name} {'='*20}")
    logger.info(f"Started training at: {timestamp}")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Task: {task}")
    logger.info(f"Batch size: {batch_size} (fixed: {fixed_batch_size})")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {num_epochs}")

    # Initialize Weights & Biases if requested
    if use_wandb:
        try:
            import wandb

            if wandb_api_key is not None:
                wandb_connected = authenticate_wandb(api_key=wandb_api_key)
                logger.info(f"WandB login result: {wandb_connected}")
                if wandb_connected != "OK":
                    logger.warning(
                        "WandB login failed. Metrics will not be logged online."
                    )
                    os.environ["WANDB_API_KEY"] = wandb_api_key
                    wandb.login()

                    use_wandb = False

            if wandb.run is None:
                # Create a config dictionary for wandb
                config = {
                    "optimizer": optimizer_name,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                    "scheduler": scheduler_type,
                    "lr_decay_epochs": lr_decay_epochs,
                    "lr_decay_factor": lr_decay_factor,
                    "lr_patience": lr_patience,
                    "task": task,
                    "test_size": test_size,
                    "val_size": val_size,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "early_stopping": early_stopping,
                    "patience": patience,
                }
                wandb.init(project=wandb_project, name=experiment_name, config=config)
                logger.info(f"Initialized WandB run: {wandb.run.name}")
        except Exception as e:
            logger.error(f"Error initializing WandB: {str(e)}")
            use_wandb = False
    try:
        if device is None:
            device = get_device()
        logger.info(f"Using device: {device}")

        model = model.to(device)
        dataset = preprocess_dataset(dataset)

        optimizer_name = optimizer_name.lower()

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            logger.error(f"Unsupported optimizer: {optimizer_name}")
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        logger.info(f"Using optimizer: {optimizer.__class__.__name__}")

        if scheduler_type == "step":
            scheduler = StepLR(
                optimizer, step_size=lr_decay_epochs, gamma=lr_decay_factor
            )
            logger.info(
                f"Using StepLR scheduler (step={lr_decay_epochs}, gamma={lr_decay_factor})"
            )

        elif scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=lr_decay_factor, patience=lr_patience
            )
            logger.info(
                f"Using ReduceLROnPlateau scheduler (factor={lr_decay_factor}, patience={lr_patience})"
            )

        else:
            scheduler = None
            logger.info("No learning rate scheduler")

        if task == "classification":
            if hasattr(dataset, "targets") and dataset.targets is not None:
                pos_samples = 0
                total_samples = 0

                for label in dataset.targets:
                    if isinstance(label, torch.Tensor):
                        pos_samples += label.sum().item()
                        total_samples += label.numel()
                    else:  # NumPy array
                        pos_samples += np.sum(label)
                        total_samples += label.size

                neg_samples = total_samples - pos_samples
                pos_weight = torch.tensor(
                    [neg_samples / pos_samples if pos_samples > 0 else 1.0]
                )
                pos_weight = pos_weight.to(device)
                logger.info(
                    f"Class weights for positive samples: {pos_weight.item():.4f}"
                )
            else:
                pos_weight = None

            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logger.info("Using BCEWithLogitsLoss for classification")
        else:
            loss_fn = nn.MSELoss()
            logger.info("Using MSELoss for regression")

        total_timesteps = len(dataset.features)

        # Time-based split (no random shuffling)
        train_end = int((1 - (val_size + test_size)) * total_timesteps)
        val_end = int((1 - test_size) * total_timesteps)

        train_idx = list(range(0, train_end))
        val_idx = list(range(train_end, val_end))
        test_idx = list(range(val_end, total_timesteps))

        logger.info(
            f"Data split: {len(train_idx)} train, {len(val_idx)} validation, {len(test_idx)} test "
            f"(total: {total_timesteps} timesteps)"
        )

        best_val_loss = float("inf")
        best_val_metrics = {}
        best_epoch = 0
        patience_counter = 0
        best_model_path = os.path.join(model_dir, f"{experiment_name}_best_model.pt")

        results = {
            "train_losses": [],
            "val_losses": [],
            "test_metrics": {},
            "best_epoch": 0,
            "best_val_loss": float("inf"),
            "best_val_metrics": {},
            "training_time": 0,
        }

        # Start timing
        start_time = time.time()

        # Main training loop with tqdm
        train_iterator = tqdm(range(1, num_epochs + 1), desc="Training epochs")
        for epoch in train_iterator:
            try:
                train_loss = train_one_epoch(
                    model=model,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    dataset=dataset,
                    indices=train_idx,
                    to_device=batch_to_device,
                    device=device,
                    batch_size=batch_size,
                    task=task,
                    use_tqdm=use_nested_tqdm,
                    time_aware=time_aware,
                    min_batch_size=min_batch_size,
                    fixed_batch_size=fixed_batch_size,
                )

                # Evaluate on validation set
                val_result = evaluate_model(
                    model=model,
                    dataset=dataset,
                    loss_fn=loss_fn,
                    indices=val_idx,
                    to_device=batch_to_device,
                    device=device,
                    fixed_batch_size=fixed_batch_size,
                    batch_size=batch_size,
                    task=task,
                    use_tqdm=use_nested_tqdm,
                )

                if task == "classification":
                    val_loss, val_metrics = val_result
                    val_acc = val_metrics["accuracy"]
                    val_f1 = val_metrics["f1"]

                    # Update progress bar
                    train_iterator.set_postfix(
                        {
                            "train_loss": f"{train_loss:.4f}",
                            "val_loss": f"{val_loss:.4f}",
                            "val_acc": f"{val_acc:.4f}",
                            "val_f1": f"{val_f1:.4f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                        }
                    )

                    logger.info(
                        f"Epoch {epoch}/{num_epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                        f"Val acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
                    )
                    logger.info(f"Validation metrics: {val_metrics}")

                    if use_wandb:
                        wandb.log(
                            {
                                "epoch": epoch,
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "val_accuracy": val_acc,
                                "val_f1": val_f1,
                                "val_precision": val_metrics["precision"],
                                "val_recall": val_metrics["recall"],
                                "val_auc": val_metrics["auc"],
                                "val_mcc": val_metrics["mcc"],
                                "learning_rate": optimizer.param_groups[0]["lr"],
                            },
                            step=epoch,
                        )
                else:
                    val_loss = val_result

                    # Update progress bar
                    train_iterator.set_postfix(
                        {
                            "train_loss": f"{train_loss:.4f}",
                            "val_loss": f"{val_loss:.4f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                        }
                    )

                    logger.info(
                        f"Epoch {epoch}/{num_epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}"
                    )

                    if use_wandb:
                        wandb.log(
                            {
                                "epoch": epoch,
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                            },
                            step=epoch,
                        )

                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                results["train_losses"].append(train_loss)
                results["val_losses"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    if task == "classification":
                        best_val_metrics = val_metrics

                    # Save best model
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(
                        f"Saved new best model at epoch {epoch} with val_loss={val_loss:.4f}"
                    )

                    # Reset patience counter
                    patience_counter = 0
                else:
                    # Increment patience counter
                    patience_counter += 1

                    # Check for early stopping
                    if early_stopping and patience_counter >= patience:
                        logger.info(
                            f"Early stopping at epoch {epoch} after {patience} epochs without improvement"
                        )
                        break
            except Exception as e:
                logger.error(f"Error during epoch {epoch}: {str(e)}")
                logger.exception("Traceback:")
                if use_wandb:
                    wandb.log({"error": str(e)}, step=epoch)
                break

        # Training complete
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Load best model for testing
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        logger.info(f"Loaded best model from epoch {best_epoch}")

        # Evaluate on test set
        logger.info("Evaluating best model on test set...")
        test_result = evaluate_model(
            model=model,
            dataset=dataset,
            loss_fn=loss_fn,
            indices=test_idx,
            to_device=batch_to_device,
            device=device,
            batch_size=batch_size,
            task=task,
            use_tqdm=True,
            fixed_batch_size=fixed_batch_size,
        )

        if task == "classification":
            test_loss, test_metrics = test_result

            logger.info("\nTest Results:")
            logger.info(f"Loss: {test_loss:.4f}")
            logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
            logger.info(f"F1 Score: {test_metrics['f1']:.4f}")
            logger.info(f"Precision: {test_metrics['precision']:.4f}")
            logger.info(f"Recall: {test_metrics['recall']:.4f}")
            logger.info(f"AUC: {test_metrics['auc']:.4f}")
            logger.info(f"MCC: {test_metrics['mcc']:.4f}")

            logger.info(
                f"Best epoch: {best_epoch}, Best validation loss: {best_val_loss:.4f}"
            )
            logger.info(f"Best validation metrics: {best_val_metrics}")

            if use_wandb:
                wandb.log(
                    {
                        "test_loss": test_loss,
                        "test_accuracy": test_metrics["accuracy"],
                        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                        "test_f1": test_metrics["f1"],
                        "test_precision": test_metrics["precision"],
                        "test_recall": test_metrics["recall"],
                        "test_auc": test_metrics["auc"],
                        "test_mcc": test_metrics["mcc"],
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                    },
                    step=epoch,
                )
        else:
            test_loss = test_result

            # Print test results
            logger.info("\nTest Results:")
            logger.info(f"Loss: {test_loss:.4f}")

            logger.info(
                f"Best epoch: {best_epoch}, Best validation loss: {best_val_loss:.4f}"
            )
            logger.info(f"Best validation metrics: {best_val_metrics}")

            # Log final metrics to wandb
            if use_wandb:
                wandb.log(
                    {
                        "test_loss": test_loss,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                    },
                    step=epoch,
                )

        # Update results dictionary
        results["test_loss"] = test_loss
        results["best_epoch"] = best_epoch
        results["best_val_loss"] = best_val_loss
        results["best_val_metrics"] = best_val_metrics
        results["training_time"] = training_time
        if task == "classification":
            results["test_metrics"] = test_metrics

        # Save final results summary to log
        logger.info(f"{'='*20} TRAINING COMPLETE {'='*20}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.exception("Traceback:")
        results["error"] = str(e)
        results["completed"] = False
        if use_wandb:
            wandb.log({"error": str(e)}, step=epoch)
    else:
        results["completed"] = True
    finally:
        # Close wandb if needed
        if use_wandb:
            wandb.finish()
            logger.info("WandB logging completed")

        # Close log handlers
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

        logger.info("Logging completed")

    return results
