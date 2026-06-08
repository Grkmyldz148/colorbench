"""OKLab — Björn Ottosson 2020.

64-bit clean matrices from facelessuser (ASTM E308 D65), zero noise near zero.
Pipeline: XYZ → M1 → cbrt → M2 → Lab.

Bit-exact bijective: round-trip ≤ 4e-15 in float64 across all 16.7M sRGB colors.
"""
import torch

from .base import ColorSpace, signed_cbrt, signed_cube


# 64-bit clean XYZ → LMS, ASTM E308 D65 (facelessuser)
_M1_LIST = [
    [ 0.81898988960052209,  0.36189146695672597, -0.12886851374185956],
    [ 0.03298391014473871,  0.92929408015501220,  0.03614494711728918],
    [ 0.04818410922430736,  0.26427748543637186,  0.63363882724502550],
]
# 64-bit clean LMS^(1/3) → OKLab
_M2_LIST = [
    [ 0.21045426830931396,  0.79361777470230530, -0.00407204301161926],
    [ 1.97799853243116860, -2.42859222042048580,  0.45059370961741140],
    [ 0.02590404246554773,  0.78277171245752970, -0.80867575492307740],
]

# Original 32-bit truncated matrices (Björn's original publication)
_M1_32_LIST = [
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
]
_M2_32_LIST = [
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
]
from .constants import M_SRGB as _M_SRGB_TENSOR
_M_SRGB = _M_SRGB_TENSOR.tolist()  # 17-decimal canonical (cycle 36 consolidation)


class OKLab(ColorSpace):
    """OKLab with 64-bit clean matrices."""

    name = "OKLab"

    def __init__(self, device: torch.device = None, dtype: torch.dtype = torch.float64):
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.M1 = torch.tensor(_M1_LIST, device=self.device, dtype=self.dtype)
        self.M2 = torch.tensor(_M2_LIST, device=self.device, dtype=self.dtype)
        self.M1_inv = torch.linalg.inv(self.M1)
        self.M2_inv = torch.linalg.inv(self.M2)

    def forward(self, xyz):
        return signed_cbrt(xyz @ self.M1.T) @ self.M2.T

    def inverse(self, lab):
        return signed_cube(lab @ self.M2_inv.T) @ self.M1_inv.T


class OKLab32(ColorSpace):
    """OKLab with original 32-bit truncated matrices (for comparison)."""

    name = "OKLab(32bit)"

    def __init__(self, device: torch.device = None, dtype: torch.dtype = torch.float64):
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        # Reconstruct M1 from sRGB-anchored matrix, exactly as Björn's paper
        M1_srgb = torch.tensor(_M1_32_LIST, device=self.device, dtype=self.dtype)
        M_srgb = torch.tensor(_M_SRGB, device=self.device, dtype=self.dtype)
        self.M1 = M1_srgb @ torch.linalg.inv(M_srgb)
        self.M2 = torch.tensor(_M2_32_LIST, device=self.device, dtype=self.dtype)
        self.M1_inv = torch.linalg.inv(self.M1)
        self.M2_inv = torch.linalg.inv(self.M2)

    def forward(self, xyz):
        return signed_cbrt(xyz @ self.M1.T) @ self.M2.T

    def inverse(self, lab):
        return signed_cube(lab @ self.M2_inv.T) @ self.M1_inv.T
