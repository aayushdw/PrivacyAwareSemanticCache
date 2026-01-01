"""Device utilities for M4 MacBook Air (MPS) and other devices."""

import gc
import os
from typing import Optional

import torch


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Get the optimal device for computation.

    Priority: CUDA > MPS > CPU (unless specified)

    Args:
        preferred: Preferred device ('cuda', 'mps', 'cpu')

    Returns:
        torch.device instance
    """
    if preferred:
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name() -> str:
    """Get human-readable device name."""
    device = get_device()
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device.type == "mps":
        return "Apple Silicon (MPS)"
    else:
        return "CPU"


def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors between model evaluations."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if torch.backends.mps.is_available():
        # MPS doesn't have explicit cache clearing, but garbage collection helps
        # Force synchronization
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def estimate_memory_usage(
    dimension: int,
    batch_size: int = 64,
    seq_length: int = 128,
    include_model: bool = True,
) -> float:
    """
    Estimate memory usage for embedding generation.

    Args:
        dimension: Embedding dimension
        batch_size: Batch size for inference
        seq_length: Maximum sequence length
        include_model: Include estimated model weight memory

    Returns:
        Estimated memory in MB
    """
    # Embedding output: batch_size * dimension * 4 bytes (float32)
    embedding_memory = batch_size * dimension * 4 / (1024 * 1024)

    # Input tensors: batch_size * seq_length * 4 bytes (int32 for token ids)
    # Plus attention mask: batch_size * seq_length * 4 bytes
    input_memory = 2 * batch_size * seq_length * 4 / (1024 * 1024)

    # Intermediate activations (rough estimate: 10x embedding size)
    activation_memory = embedding_memory * 10

    total = embedding_memory + input_memory + activation_memory

    if include_model:
        # Add estimated model weights based on dimension
        # This is a rough approximation
        if dimension <= 384:
            model_memory = 100.0
        elif dimension <= 768:
            model_memory = 400.0
        elif dimension <= 1024:
            model_memory = 800.0
        else:
            model_memory = 1500.0
        total += model_memory

    return total


def get_available_memory_mb() -> float:
    """Get available GPU memory in MB (CUDA only, estimates for MPS)."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / (1024 * 1024)
    elif torch.backends.mps.is_available():
        # MPS doesn't expose memory info, estimate based on M4 MacBook Air
        # Typical unified memory available: ~6GB for ML
        return 6000.0
    else:
        # CPU - use system memory (rough estimate)
        import psutil

        return psutil.virtual_memory().available / (1024 * 1024)


def can_fit_model(estimated_memory_mb: float, safety_margin: float = 0.2) -> bool:
    """
    Check if a model can fit in available memory.

    Args:
        estimated_memory_mb: Estimated model memory requirement
        safety_margin: Safety margin (0.2 = 20% buffer)

    Returns:
        True if model should fit
    """
    available = get_available_memory_mb()
    required = estimated_memory_mb * (1 + safety_margin)
    return available >= required


def setup_mps_environment():
    """Set up environment variables for optimal MPS performance."""
    # Enable MPS fallback for unsupported ops
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Disable tokenizers parallelism to avoid fork warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_optimal_batch_size(
    model_memory_mb: float,
    dimension: int,
    max_batch_size: int = 128,
    target_utilization: float = 0.7,
) -> int:
    """
    Calculate optimal batch size based on available memory.

    Args:
        model_memory_mb: Model memory requirement
        dimension: Embedding dimension
        max_batch_size: Maximum allowed batch size
        target_utilization: Target memory utilization (0.7 = 70%)

    Returns:
        Recommended batch size
    """
    available = get_available_memory_mb()
    target_memory = available * target_utilization

    # Memory available for batches after loading model
    batch_memory = target_memory - model_memory_mb

    if batch_memory <= 0:
        return 1  # Minimum batch size

    # Estimate memory per sample (rough: dimension * 40 bytes including overhead)
    memory_per_sample = dimension * 40 / (1024 * 1024)

    optimal_batch = int(batch_memory / memory_per_sample)

    # Clamp to reasonable range
    return max(1, min(optimal_batch, max_batch_size))


class DeviceContext:
    """Context manager for device setup and cleanup."""

    def __init__(self, model_memory_mb: Optional[float] = None):
        """
        Initialize device context.

        Args:
            model_memory_mb: Expected model memory for pre-check
        """
        self.model_memory_mb = model_memory_mb
        self.device = None

    def __enter__(self) -> torch.device:
        """Set up device and environment."""
        setup_mps_environment()
        clear_gpu_memory()

        self.device = get_device()

        if self.model_memory_mb and not can_fit_model(self.model_memory_mb):
            import warnings

            warnings.warn(
                f"Model may not fit in memory. "
                f"Required: ~{self.model_memory_mb:.0f}MB, "
                f"Available: ~{get_available_memory_mb():.0f}MB"
            )

        return self.device

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up GPU memory."""
        clear_gpu_memory()
        return False
