"""ColorBench color spaces — public API.

Each color space is a small focused module; this file re-exports the public
classes for convenient import.

Usage:
    from core.cs import OKLab, CIELab
    sp = OKLab(device=device, dtype=torch.float64)
"""
from .base import ColorSpace, signed_cbrt, signed_cube
from .constants import D65, M_SRGB
from .oklab import OKLab, OKLab32
from .cielab import CIELab
from .pwL import PiecewiseLinearL
from .enrichment import LGatedHueEnrichment, ChromaPreservingHueRotation
from .neutral import neutral_blend, NCLut
from .helmct import HelmCT
from .literature import IPT, JzAzBz, ICtCp, CAM16UCS, DIN99d
from .literature_canonical import (
    IPTCanonical, JzAzBzCanonical, CAM16UCSCanonical, DIN99dCanonical,
)
from .genspace_legacy import (
    GenSpaceAdapter, NakaRushtonEnriched, GenSpaceEnriched,
    GenSpaceBlueFix, NonlinearM1, CustomSpace,
)
from . import transfer

__all__ = [
    "ColorSpace",
    "signed_cbrt",
    "signed_cube",
    "D65",
    "M_SRGB",
    "OKLab",
    "OKLab32",
    "CIELab",
    "PiecewiseLinearL",
    "LGatedHueEnrichment",
    "ChromaPreservingHueRotation",
    "neutral_blend",
    "NCLut",
    "HelmCT",
    "IPT",
    "JzAzBz",
    "ICtCp",
    "CAM16UCS",
    "DIN99d",
    "IPTCanonical",
    "JzAzBzCanonical",
    "CAM16UCSCanonical",
    "DIN99dCanonical",
    "GenSpaceAdapter",
    "NakaRushtonEnriched",
    "GenSpaceEnriched",
    "GenSpaceBlueFix",
    "NonlinearM1",
    "CustomSpace",
    "transfer",
]
