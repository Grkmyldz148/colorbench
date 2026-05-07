"""Module-level color science constants used by spaces and metrics.

Kept on CPU at float64; cast to space.dtype/device at use site.
Re-exported from package root for `from core.cs import D65` shorthand.
"""
import torch

# CIE D65 illuminant (ASTM E308)
D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)

# sRGB (BT.709) → XYZ matrix, D65
M_SRGB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float64)
