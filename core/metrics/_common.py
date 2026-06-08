"""Shared utilities for metric modules.

Constants and transfer functions used across multiple measure_* functions.
All operations preserve dtype and device of input tensor.
"""
import math
import torch


# RGB→XYZ matrices (D65), float-list form for lazy device/dtype materialization
# cycle 36 — sRGB consolidated to 17-decimal canonical via cs/constants.py
from ..cs.constants import M_SRGB as _M_SRGB_TENSOR_CANONICAL
_M_SRGB_LIST = _M_SRGB_TENSOR_CANONICAL.tolist()
_M_P3_LIST = [
    [0.4865709486482162, 0.26566769316909306, 0.1982172852343625],
    [0.2289745640697488, 0.6917385218365064,  0.079286914093745],
    [0.0,                0.04511338185890264, 1.0439443689009757],
]
_M_REC2020_LIST = [
    [0.6369580483012914, 0.14461690358620832, 0.1688809751641721],
    [0.2627002120112671, 0.6779980715188708,  0.05930171646986196],
    [0.0,                0.028072693049087428, 1.0609850577107909],
]
# D65 illuminant (ASTM E308)
_D65_LIST = [0.95047, 1.0, 1.08883]


def matrix(values, device, dtype):
    """Materialize a list-of-lists as a (3,3) tensor on (device, dtype)."""
    return torch.tensor(values, device=device, dtype=dtype)


def vec(values, device, dtype):
    """Materialize a list as a (N,) tensor on (device, dtype)."""
    return torch.tensor(values, device=device, dtype=dtype)


def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    """sRGB transfer function: gamma decode. Preserves dtype/device of c."""
    threshold = 0.04045
    return torch.where(
        c <= threshold,
        c / 12.92,
        ((c + 0.055) / 1.055).pow(2.4),
    )


def linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    """sRGB transfer function: gamma encode."""
    return torch.where(
        c <= 0.0031308,
        c * 12.92,
        1.055 * c.clamp(min=1e-12).pow(1.0 / 2.4) - 0.055,
    )


def xyz_to_cielab(xyz: torch.Tensor, d65: torch.Tensor) -> torch.Tensor:
    """XYZ (N,3) → CIE L*a*b* (N,3). Standard CIE 1976 formula.

    Both inputs must share dtype and device. Preserved on output.
    """
    r = xyz / d65
    delta3 = (6.0 / 29.0) ** 3
    f = torch.where(
        r > delta3,
        r.pow(1.0 / 3.0),
        r / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0,
    )
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], dim=-1)
