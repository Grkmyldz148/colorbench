"""Precision and device routing for ColorBench.

Single source of truth for (device, dtype) pair selection. Read by run.py
based on CLI flags, then passed to every ColorSpace and metric.

Modes
-----
default      : CPU + float64. Bit-exact baseline. Snapshot regression.
fast         : auto GPU + float32. Iteration-grade, ~1e-6 drift, sub-JND.
explicit     : --device {cpu,cuda,mps} --precision {32,64}

Combinations
------------
- CPU float64        : bit-exact baseline (paper)
- CPU float32        : reference for fast mode
- CUDA float64       : workstation GPU bit-exact (A100/A6000)
- CUDA float32       : NVIDIA fast mode
- MPS float32        : Apple Silicon fast mode
- MPS float64        : NOT SUPPORTED (PyTorch MPS limitation)
"""
import torch


def select(mode: str = "default", device: str = None, precision: int = None):
    """Return (torch.device, torch.dtype, label).

    Args:
        mode: 'default' | 'fast' | 'explicit'
        device: 'cpu' | 'cuda' | 'mps' (only for mode='explicit')
        precision: 32 | 64 (only for mode='explicit')

    Returns:
        (device, dtype, label) — label is human-readable status string.
    """
    if mode == "default":
        return torch.device("cpu"), torch.float64, "CPU float64 (bit-exact baseline)"

    if mode == "fast":
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float32, "CUDA float32 (fast mode)"
        if torch.backends.mps.is_available():
            return torch.device("mps"), torch.float32, "MPS float32 (fast mode)"
        return torch.device("cpu"), torch.float32, "CPU float32 (fast mode, no GPU)"

    if mode == "explicit":
        if device is None or precision is None:
            raise ValueError("explicit mode requires both device and precision")
        if precision not in (32, 64):
            raise ValueError(f"precision must be 32 or 64, got {precision}")
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"device must be cpu/cuda/mps, got {device}")
        if device == "mps" and precision == 64:
            raise RuntimeError(
                "MPS does not support float64. Use --precision 32 with --device mps, "
                "or use --device cpu / --device cuda for float64."
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        dt = torch.float64 if precision == 64 else torch.float32
        return torch.device(device), dt, f"{device.upper()} float{precision}"

    raise ValueError(f"Unknown mode '{mode}'")


def is_bit_exact_capable(device: torch.device, dtype: torch.dtype) -> bool:
    """True iff (device, dtype) yields snapshot-grade bit-exact results.

    Used to decide whether snapshot regression should run, or whether to
    fall back to ranking-agreement comparison.
    """
    return dtype == torch.float64 and device.type in ("cpu", "cuda")
