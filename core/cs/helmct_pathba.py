"""HelmCT_PathBA — research-only HelmCT v0.11.1 + side-channel antisymmetric hue corr.

Cycle 25-28 prototype. NOT a deploy candidate yet — gatekeeper-pending.

δh(h) = Σᵢ αᵢ · uᵢ · exp(-½ uᵢ²)  where uᵢ = (h − h_iᵒ)/σᵢ

Centers h_iᵒ measured in HelmCT's (a, b) plane at sRGB primaries (R, Y, G, C, B, M).
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .helmct import HelmCT
from .constants import M_SRGB

PI = math.pi


def _measure_primary_centers_via_base(self_obj, device, dtype):
    """Re-measure primary hue angles using HelmCT's (a, b) plane (parent forward)."""
    primaries = {
        "R": [1.0, 0.0, 0.0], "Y": [1.0, 1.0, 0.0], "G": [0.0, 1.0, 0.0],
        "C": [0.0, 1.0, 1.0], "B": [0.0, 0.0, 1.0], "M": [1.0, 0.0, 1.0],
    }
    M = M_SRGB.to(device=device, dtype=dtype)
    centers = {}
    for name, rgb in primaries.items():
        rgb_t = torch.tensor([rgb], device=device, dtype=dtype)
        rgb_lin = torch.where(rgb_t > 0.04045, ((rgb_t + 0.055) / 1.055).pow(2.4), rgb_t / 12.92)
        xyz = rgb_lin @ M.T
        # Use HelmCT's forward (parent class) directly — bypass subclass override
        lab = HelmCT.forward(self_obj, xyz)
        a, b = lab[0, 1].item(), lab[0, 2].item()
        h_deg = (math.degrees(math.atan2(b, a))) % 360
        centers[name] = math.radians(h_deg)
    return centers


class HelmCT_PathBA(HelmCT):
    """HelmCT v0.11.1 + side-channel antisymmetric hue correction (cycle 25 default)."""

    def __init__(self, json_path: str, device: torch.device, dtype=torch.float64,
                 amps_deg=None, sigmas_deg=None, halley_iters=6):
        super().__init__(json_path, device, dtype=dtype)
        self.dtype = dtype
        self._halley_iters = halley_iters

        if amps_deg is None:
            amps_deg = [-3.0] * 6  # default cycle 25 winner: α=-3°
        if sigmas_deg is None:
            sigmas_deg = [10.0] * 6  # default cycle 25 winner: σ=10°

        # Measure primary centers (R, Y, G, C, B, M order) via parent forward
        centers_rad = _measure_primary_centers_via_base(self, device, dtype)
        order = ["R", "Y", "G", "C", "B", "M"]
        self._centers_rad = torch.tensor(
            [centers_rad[k] for k in order], device=device, dtype=dtype,
        )
        self._amps = torch.tensor([math.radians(x) for x in amps_deg], device=device, dtype=dtype)
        self._sigmas = torch.tensor([math.radians(x) for x in sigmas_deg], device=device, dtype=dtype)

    def _u_g(self, h):
        """(u, g) where u=(h-center)/σ wrapped, g=exp(-½ u²)."""
        h_unsq = h.unsqueeze(-1)
        dh = h_unsq - self._centers_rad
        dh = (dh + PI) % (2 * PI) - PI
        u = dh / self._sigmas
        g = torch.exp(-0.5 * u * u)
        return u, g

    def _delta_h(self, h):
        u, g = self._u_g(h)
        return (self._amps * u * g).sum(dim=-1)

    def _delta_h_p(self, h):
        u, g = self._u_g(h)
        return (self._amps / self._sigmas * g * (1.0 - u * u)).sum(dim=-1)

    def _delta_h_pp(self, h):
        u, g = self._u_g(h)
        return (self._amps / self._sigmas.pow(2) * g * u * (u * u - 3.0)).sum(dim=-1)

    def forward(self, xyz):
        lab = super().forward(xyz)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        C = torch.sqrt(a * a + b * b + 1e-30)
        h = torch.atan2(b, a)
        h_new = h + self._delta_h(h)
        return torch.stack([L, C * torch.cos(h_new), C * torch.sin(h_new)], dim=-1)

    def inverse(self, lab):
        L = lab[:, 0]
        a, b = lab[:, 1], lab[:, 2]
        C = torch.sqrt(a * a + b * b + 1e-30)
        h_t = torch.atan2(b, a)
        # Halley iteration on h + δh(h) = h_t
        h = h_t.clone()
        for _ in range(self._halley_iters):
            F = h + self._delta_h(h) - h_t
            Fp = 1.0 + self._delta_h_p(h)
            Fpp = self._delta_h_pp(h)
            denom = 2.0 * Fp * Fp - F * Fpp
            denom = torch.where(denom.abs() < 1e-30, torch.ones_like(denom), denom)
            h = h - 2.0 * F * Fp / denom
        a_n = C * torch.cos(h)
        b_n = C * torch.sin(h)
        lab_inner = torch.stack([L, a_n, b_n], dim=-1)
        return super().inverse(lab_inner)
