"""Gamut mapping smoothness — chroma-reduction trajectory quality.

For each (gamut, L_test, hue), sweep chroma 0.4 → 0.001 in test space's Lab,
inverse-map → quantize to gamut RGB → re-encode XYZ → CIE Lab. Measure:

  non_monotonic_hues : count of hues whose CIELab C* increases on the way
                       from high chroma to neutral (boundary fold artifact)
  max_de_jump        : largest CIE 2000 ΔE between consecutive chroma steps
  max_hue_jump       : largest hue swing in CIE Lab while reducing chroma

Tested at 8 lightness levels × 36 hues × 100 chroma steps × 3 gamuts.
"""
import math
import torch

from ._common import (
    _M_SRGB_LIST, _M_P3_LIST, _M_REC2020_LIST, _D65_LIST,
    matrix, vec, srgb_to_linear, linear_to_srgb, xyz_to_cielab,
)
from ..gpu_de import ciede2000

PI = math.pi


def _measure_one(space, gamut_mat, gamut_inv, d65, L_test):
    """Sweep all 36 hues × 100 chroma at fixed L. Returns aggregate stats."""
    dev, dt = space.device, space.dtype
    non_mono = 0
    max_de_jump = 0.0
    hue_jumps = []

    for h_deg in range(0, 360, 10):
        h_rad = h_deg * PI / 180
        ch, sh = math.cos(h_rad), math.sin(h_rad)
        Cs = torch.linspace(0.4, 0.001, 100, device=dev, dtype=dt)
        lab = torch.stack([
            torch.full_like(Cs, L_test),
            Cs * ch,
            Cs * sh,
        ], dim=-1)

        xyz = space.inverse(lab)
        lin = (xyz @ gamut_inv.T).clamp(0, 1)
        s8 = (linear_to_srgb(lin) * 255).round() / 255.0
        xyz_q = srgb_to_linear(s8) @ gamut_mat.T
        cielab = xyz_to_cielab(xyz_q.clamp(min=1e-10), d65)

        C_star = (cielab[:, 1] ** 2 + cielab[:, 2] ** 2).sqrt()
        diffs = C_star[1:] - C_star[:-1]
        if (diffs > 0.5).any():
            non_mono += 1

        de = ciede2000(cielab[:-1], cielab[1:])
        max_de_jump = max(max_de_jump, de.max().item())

        h_cl = torch.atan2(cielab[:, 2], cielab[:, 1])
        chromatic = C_star > 2
        if chromatic.sum() > 2:
            h_chrom = h_cl[chromatic]
            dh = h_chrom[1:] - h_chrom[:-1]
            dh = torch.atan2(torch.sin(dh), torch.cos(dh)).abs() * (180.0 / PI)
            if dh.numel() > 0:
                hue_jumps.append(dh.max().item())

    return {
        "non_monotonic_hues": non_mono,
        "max_de_jump": max_de_jump,
        "max_hue_jump": max(hue_jumps) if hue_jumps else 0,
    }


def measure_gamut_mapping(space, device=None) -> dict:
    """Chroma reduction smoothness for sRGB / P3 / Rec.2020 at 8 L levels."""
    dev, dt = space.device, space.dtype
    d65 = vec(_D65_LIST, dev, dt)
    gamuts = [
        ("sRGB",    matrix(_M_SRGB_LIST, dev, dt)),
        ("P3",      matrix(_M_P3_LIST, dev, dt)),
        ("Rec2020", matrix(_M_REC2020_LIST, dev, dt)),
    ]
    L_levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}
    for name, mat in gamuts:
        gamut_inv = torch.linalg.inv(mat)
        for L_test in L_levels:
            results[f"{name}_L{L_test}"] = _measure_one(
                space, mat, gamut_inv, d65, L_test
            )
    return results
