"""
Device utilities for managing GPU memory and selecting devices.
"""

import torch
import gc


def get_device(device_id: int = None) -> torch.device:
    """
    Get the appropriate device for running models.

    Args:
        device_id: Optional GPU device ID to use

    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")


def clear_gpu_memory():
    """Clear GPU memory by releasing PyTorch allocated memory."""
    gc.collect()
    torch.cuda.empty_cache()
