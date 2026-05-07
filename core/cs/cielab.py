"""CIE L*a*b* (1976) — classic perceptual color space.

Pipeline:
  Forward: XYZ → r = XYZ/D65 → f(r) → L = 116·f(Y)-16, a = 500·(f(X)-f(Y)), b = 200·(f(Y)-f(Z))
  Inverse: from L,a,b solve f(Y), f(X), f(Z) → r = f^-1(...) → XYZ = r·D65

Where f is the standard CIE piecewise:
  f(t) = t^(1/3)             if t > (6/29)³
       = t / (3·(6/29)²) + 4/29  otherwise
"""
import torch

from .base import ColorSpace


_D65_LIST = [0.95047, 1.0, 1.08883]
_DELTA = 6.0 / 29.0
_DELTA3 = _DELTA ** 3


class CIELab(ColorSpace):
    """CIE L*a*b* 1976."""

    name = "CIELab"

    def __init__(self, device: torch.device = None, dtype: torch.dtype = torch.float64):
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.d65 = torch.tensor(_D65_LIST, device=self.device, dtype=self.dtype)

    def forward(self, xyz):
        r = xyz / self.d65
        f = torch.where(
            r > _DELTA3,
            r.pow(1.0 / 3.0),
            r / (3.0 * _DELTA ** 2) + 4.0 / 29.0,
        )
        L = 116.0 * f[:, 1] - 16.0
        a = 500.0 * (f[:, 0] - f[:, 1])
        b = 200.0 * (f[:, 1] - f[:, 2])
        return torch.stack([L, a, b], dim=-1)

    def inverse(self, lab):
        fy = (lab[:, 0] + 16.0) / 116.0
        fx = lab[:, 1] / 500.0 + fy
        fz = fy - lab[:, 2] / 200.0
        f = torch.stack([fx, fy, fz], dim=-1)
        xyz = torch.where(
            f > _DELTA,
            f.pow(3.0),
            3.0 * _DELTA ** 2 * (f - 4.0 / 29.0),
        )
        return xyz * self.d65
