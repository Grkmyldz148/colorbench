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
from .pwL import PiecewiseLinearL
from .enrichment import LGatedHueEnrichment, ChromaPreservingHueRotation
from .neutral import neutral_blend, NCLut
from .helmct import HelmCT
from . import transfer

__all__ = [
    "ColorSpace",
    "signed_cbrt",
    "signed_cube",
    "OKLab",
    "OKLab32",
    "CIELab",
    "PiecewiseLinearL",
    "LGatedHueEnrichment",
    "ChromaPreservingHueRotation",
    "neutral_blend",
    "NCLut",
    "HelmCT",
    "transfer",
]
