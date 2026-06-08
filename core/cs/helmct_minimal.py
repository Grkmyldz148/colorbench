"""HelmCT_minimal — research-only minimal-arch HelmCT (Cycle 40 Hedef 3 candidate).

Strips L_corr_pw + enrichment + NC + chroma_power. Just M1 + M2 + depcubic_alpha.
Total parameters: 19 (vs OKLab ~18, vs full HelmCT 47).

Preserves M2 rotation φ=−28.2° → cusps 360/360 WIN.
Loses Munsell V uniformity (no L_corr_pw) and Blue G/R fix (no enrichment).

Hedef 3 ("OKLab'dan az parametre + daha iyi") test candidate.
"""
from __future__ import annotations

import torch

from .helmct import HelmCT


class HelmCT_minimal(HelmCT):
    """HelmCT v0.11.1 with non-essential stages disabled.

    Param count: 19 = M1(9) + M2(9) + depcubic_alpha(1).

    Disabled (set to None / False after super().__init__):
    - L_corr_pw (and lc7/5/3 backups)
    - enrichment (L-gated Hering)
    - NC LUT
    - chroma_power
    - hue_dep_L
    - ab axis scaling
    """

    def __init__(self, json_path: str, device: torch.device, dtype=torch.float64):
        super().__init__(json_path, device, dtype=dtype)
        # Disable all non-essential stages
        self._L_pwl = None
        self._lc7 = None
        self._lc5 = None
        self._lc3 = None
        self._enrichment = None
        if hasattr(self, "_has_nc"):
            self._has_nc = False
        if hasattr(self, "_has_chroma"):
            self._has_chroma = False
        if hasattr(self, "_has_hue_L"):
            self._has_hue_L = False
        if hasattr(self, "_has_ab"):
            self._has_ab = False
