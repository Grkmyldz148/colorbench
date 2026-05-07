"""Canonical literature spaces — colour-science wrapper'lar.

ColorBench `spaces_literature.py`'daki implementations Helmlab paper için
tuned (4-decimal matrices, custom luminance scales, vs). Bu modül **AYNI
isimle** colour-science'a delegate eden bit-identical versiyonları sunar.

Kullanım:
  - Paper baseline doğrulama: bu modülü kullan (canonical/colour-science)
  - ColorBench-tuned numbers: orijinal `spaces_literature.py` kullan

Trade-off: Bu wrapper'lar tensor → numpy → colour-science → tensor
olduğu için PyTorch GPU avantajı yok; CPU'da çalışır. ColorBench
90-metric pipeline'da nadir kullanım için yeterli.
"""
from __future__ import annotations

import numpy as np
import torch
import colour

from .spaces import ColorSpace, D65

D65_XYZ = np.array([0.95047, 1.0, 1.08883])
D65_XY = D65_XYZ[:2] / D65_XYZ.sum()


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


def _t(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(arr, dtype=np.float64), device=device, dtype=torch.float64)


# ═════════════════════════════════════════════════════════════════════════════
# IPT (canonical) — Ebner & Fairchild 1998 reference
# ═════════════════════════════════════════════════════════════════════════════

class IPTCanonical(ColorSpace):
    """IPT via colour.XYZ_to_IPT / IPT_to_XYZ (Ebner-Fairchild reference)."""
    name = "IPT-canonical"

    def __init__(self, device: torch.device):
        self.device = device

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return _t(colour.XYZ_to_IPT(_np(xyz)), self.device)

    def inverse(self, ipt: torch.Tensor) -> torch.Tensor:
        return _t(colour.IPT_to_XYZ(_np(ipt)), self.device)


# ═════════════════════════════════════════════════════════════════════════════
# JzAzBz (canonical) — Safdar et al. 2017 reference
# ═════════════════════════════════════════════════════════════════════════════

class JzAzBzCanonical(ColorSpace):
    """JzAzBz via colour.XYZ_to_Jzazbz / Jzazbz_to_XYZ (Safdar reference)."""
    name = "JzAzBz-canonical"

    def __init__(self, device: torch.device):
        self.device = device

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return _t(colour.XYZ_to_Jzazbz(_np(xyz)), self.device)

    def inverse(self, jab: torch.Tensor) -> torch.Tensor:
        return _t(colour.Jzazbz_to_XYZ(_np(jab)), self.device)


# ═════════════════════════════════════════════════════════════════════════════
# CAM16-UCS (canonical) — Li et al. 2017 reference
# ═════════════════════════════════════════════════════════════════════════════

class CAM16UCSCanonical(ColorSpace):
    """CAM16-UCS via colour.XYZ_to_CAM16UCS (default viewing conditions)."""
    name = "CAM16-UCS-canonical"

    def __init__(self, device: torch.device):
        self.device = device

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return _t(colour.XYZ_to_CAM16UCS(_np(xyz)), self.device)

    def inverse(self, jab: torch.Tensor) -> torch.Tensor:
        # colour-science API: CAM16UCS_to_XYZ exists in 0.4.7
        return _t(colour.CAM16UCS_to_XYZ(_np(jab)), self.device)


# ═════════════════════════════════════════════════════════════════════════════
# DIN99d (canonical) — Cui et al. 2002 reference
# ═════════════════════════════════════════════════════════════════════════════

class DIN99dCanonical(ColorSpace):
    """DIN99d via colour.Lab_to_DIN99(method='DIN99d')."""
    name = "DIN99d-canonical"

    def __init__(self, device: torch.device):
        self.device = device

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        xyz_np = _np(xyz)
        lab = colour.XYZ_to_Lab(xyz_np, illuminant=D65_XY)
        din99 = colour.Lab_to_DIN99(lab, method="DIN99d")
        return _t(din99, self.device)

    def inverse(self, din99: torch.Tensor) -> torch.Tensor:
        din99_np = _np(din99)
        lab = colour.DIN99_to_Lab(din99_np, method="DIN99d")
        xyz = colour.Lab_to_XYZ(lab, illuminant=D65_XY)
        return _t(xyz, self.device)


# ═════════════════════════════════════════════════════════════════════════════
# ICtCp — already passing pin test, keep using original
# ═════════════════════════════════════════════════════════════════════════════
# (no canonical wrapper needed; orijinal `spaces_literature.ICtCp` doğru)
