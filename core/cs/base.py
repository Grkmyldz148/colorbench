"""ColorSpace abstract base + shared utilities.

Every concrete color space subclasses ColorSpace and exposes:
  .name    str  — identifier
  .device  torch.device
  .dtype   torch.dtype
  .forward(xyz) → lab
  .inverse(lab) → xyz

Tensor creation rule
--------------------
Subclasses MUST create internal tensors with `device=self.device, dtype=self.dtype`.
No hardcoded `torch.float64` or `torch.device('cpu')`. This enables --fast mode
(MPS/CUDA float32) without code duplication.
"""
from abc import ABC, abstractmethod
import torch


class ColorSpace(ABC):
    """Abstract base for any color space under test."""

    name: str
    device: torch.device
    dtype: torch.dtype

    @abstractmethod
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """XYZ (N, 3) → Lab (N, 3). Both in self.dtype on self.device."""

    @abstractmethod
    def inverse(self, lab: torch.Tensor) -> torch.Tensor:
        """Lab (N, 3) → XYZ (N, 3). Both in self.dtype on self.device."""


def signed_cbrt(x: torch.Tensor) -> torch.Tensor:
    """Sign-preserving cube root: sign(x) · |x|^(1/3).

    Bit-exact bijective with signed_cube. Domain: all real numbers.
    """
    return x.sign() * x.abs().pow(1.0 / 3.0)


def signed_cube(x: torch.Tensor) -> torch.Tensor:
    """Signed cube: x³ (sign preserved naturally since odd power)."""
    return x.sign() * x.abs().pow(3.0)
