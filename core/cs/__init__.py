"""ColorBench color spaces — public API.

Each color space is a small focused module; this file re-exports the public
classes for convenient import.

Usage:
    from core.cs import OKLab, CIELab
    sp = OKLab(device=device, dtype=torch.float64)
"""
from .base import ColorSpace, signed_cbrt, signed_cube
from .oklab import OKLab, OKLab32
from .cielab import CIELab
from . import transfer

__all__ = [
    "ColorSpace",
    "signed_cbrt",
    "signed_cube",
    "OKLab",
    "OKLab32",
    "CIELab",
    "transfer",
]
