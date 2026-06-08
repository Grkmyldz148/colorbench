"""HelmCT_M1M2 — strict Hedef 3 candidate (18 params = OKLab equality).

M1 (9) + M2 (9) + plain cube root (0 params) = 18 params total.
Replaces HelmCT's DepCubicTransfer (1 param) with CbrtTransfer (0 params).
Same parameter count as OKLab (M1 + M2 = 18).

Architecturally equivalent to "OKLab with HelmCT's M2 rotation φ=−28.2°".

Test: does M2 rotation alone (without depcubic_alpha tuning) preserve
HelmCT_minimal's 54-W performance? Or does it require the depcubic α tuning?
"""
from __future__ import annotations

import torch

from .helmct_minimal import HelmCT_minimal
from .transfer import CbrtTransfer


class HelmCT_M1M2(HelmCT_minimal):
    """HelmCT_minimal with depcubic replaced by plain cube root (18 params)."""

    def __init__(self, json_path: str, device: torch.device, dtype=torch.float64):
        super().__init__(json_path, device, dtype=dtype)
        # Replace transfer with plain cube root (drop depcubic alpha param)
        self._transfer = CbrtTransfer()
