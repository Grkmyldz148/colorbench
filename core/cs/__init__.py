"""ColorBench color spaces — public API.

Each color space is a small focused module; this file re-exports the public
classes for convenient import.

Usage:
    from core.spaces import OKLab, CIELab
    sp = OKLab(device=device, dtype=torch.float64)
"""
from .base import ColorSpace, signed_cbrt, signed_cube
from .oklab import OKLab, OKLab32

__all__ = [
    "ColorSpace",
    "signed_cbrt",
    "signed_cube",
    "OKLab",
    "OKLab32",
]
