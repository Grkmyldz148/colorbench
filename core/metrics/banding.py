"""Perceptual banding — invisible step count in 256-step 8-bit gradients.

For 42 standard gradients (sRGB + P3), interpolate 256 steps in test space,
quantize to 8-bit, measure consecutive ΔE2000 in CIE Lab. Steps with ΔE < 1.0
are "invisible" — perceptual banding indicator. Also count duplicate
consecutive 8-bit triplets (posterization).
"""
import torch

from ._common import (
    _M_SRGB_LIST, _M_P3_LIST, _D65_LIST,
    matrix, vec, srgb_to_linear, linear_to_srgb, xyz_to_cielab,
)
from ..rulers import get_ruler as _get_ruler

_step_ruler = _get_ruler("spacing")  # uniformity -> spacing ruler (Perceptia-Spacing); notebook 2026-05-30


_SRGB_GRADIENTS = [
    ("R→W", [1, 0, 0], [1, 1, 1]),
    ("G→W", [0, 1, 0], [1, 1, 1]),
    ("B→W", [0, 0, 1], [1, 1, 1]),
    ("Y→W", [1, 1, 0], [1, 1, 1]),
    ("C→W", [0, 1, 1], [1, 1, 1]),
    ("M→W", [1, 0, 1], [1, 1, 1]),
    ("R→K", [1, 0, 0], [0, 0, 0]),
    ("G→K", [0, 1, 0], [0, 0, 0]),
    ("B→K", [0, 0, 1], [0, 0, 0]),
    ("Y→K", [1, 1, 0], [0, 0, 0]),
    ("C→K", [0, 1, 1], [0, 0, 0]),
    ("M→K", [1, 0, 1], [0, 0, 0]),
    ("K→W", [0, 0, 0], [1, 1, 1]),
    ("R→C", [1, 0, 0], [0, 1, 1]),
    ("G→M", [0, 1, 0], [1, 0, 1]),
    ("B→Y", [0, 0, 1], [1, 1, 0]),
    ("C→R", [0, 1, 1], [1, 0, 0]),
    ("M→G", [1, 0, 1], [0, 1, 0]),
    ("Y→B", [1, 1, 0], [0, 0, 1]),
    ("R→Y", [1, 0, 0], [1, 1, 0]),
    ("Y→G", [1, 1, 0], [0, 1, 0]),
    ("G→C", [0, 1, 0], [0, 1, 1]),
    ("C→B", [0, 1, 1], [0, 0, 1]),
    ("B→M", [0, 0, 1], [1, 0, 1]),
    ("M→R", [1, 0, 1], [1, 0, 0]),
    ("dR→dB", [0.3, 0, 0], [0, 0, 0.3]),
    ("dG→dM", [0, 0.3, 0], [0.3, 0, 0.3]),
    ("dY→dC", [0.3, 0.3, 0], [0, 0.3, 0.3]),
    ("dK→dR", [0.05, 0.05, 0.05], [0.3, 0.05, 0.05]),
    ("pR→pB", [1, 0.7, 0.7], [0.7, 0.7, 1]),
    ("pG→pM", [0.7, 1, 0.7], [1, 0.7, 1]),
    ("pY→pC", [1, 1, 0.7], [0.7, 1, 1]),
    ("pR→pG", [1, 0.7, 0.7], [0.7, 1, 0.7]),
]
_P3_GRADIENTS = [
    ("P3_R→W", [1, 0, 0], [1, 1, 1]),
    ("P3_G→W", [0, 1, 0], [1, 1, 1]),
    ("P3_B→W", [0, 0, 1], [1, 1, 1]),
    ("P3_R→K", [1, 0, 0], [0, 0, 0]),
    ("P3_G→K", [0, 1, 0], [0, 0, 0]),
    ("P3_B→K", [0, 0, 1], [0, 0, 0]),
    ("P3_R→C", [1, 0, 0], [0, 1, 1]),
    ("P3_G→M", [0, 1, 0], [1, 0, 1]),
    ("P3_B→Y", [0, 0, 1], [1, 1, 0]),
    ("P3_K→W", [0, 0, 0], [1, 1, 1]),
]


def measure_perceptual_banding(space, device=None) -> dict:
    """ΔE<1 invisible steps + duplicate 8-bit RGB across 42 gradients."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    mp3 = matrix(_M_P3_LIST, dev, dt)
    mp3i = torch.linalg.inv(mp3)
    d65 = vec(_D65_LIST, dev, dt)

    all_grads = (
        [(name, rgb1, rgb2, ms, msi) for name, rgb1, rgb2 in _SRGB_GRADIENTS]
        + [(name, rgb1, rgb2, mp3, mp3i) for name, rgb1, rgb2 in _P3_GRADIENTS]
    )

    per_gradient = {}
    for name, rgb1, rgb2, gamut_mat, gamut_inv in all_grads:
        xyz1 = srgb_to_linear(torch.tensor(rgb1, device=dev, dtype=dt)) @ gamut_mat.T
        xyz2 = srgb_to_linear(torch.tensor(rgb2, device=dev, dtype=dt)) @ gamut_mat.T
        lab1 = space.forward(xyz1.unsqueeze(0))[0]
        lab2 = space.forward(xyz2.unsqueeze(0))[0]

        t = torch.linspace(0, 1, 256, device=dev, dtype=dt)
        labs = lab1.unsqueeze(0) + t.unsqueeze(1) * (lab2 - lab1).unsqueeze(0)
        xyz_all = space.inverse(labs)
        s8 = (linear_to_srgb((xyz_all @ gamut_inv.T).clamp(0, 1)) * 255).round() / 255.0
        xyz_q = srgb_to_linear(s8) @ gamut_mat.T
        cielab = xyz_to_cielab(xyz_q.clamp(min=1e-10), d65)

        de = _step_ruler(cielab[:-1], cielab[1:])
        invisible = (de < 1.0).sum().item()
        duplicate = ((s8[1:] * 255).to(torch.int32) ==
                     (s8[:-1] * 255).to(torch.int32)).all(dim=1).sum().item()
        per_gradient[name] = {
            "invisible_steps": int(invisible),
            "invisible_pct": invisible / 255 * 100,
            "duplicate_rgb": int(duplicate),
            "de_min": de.min().item(),
            "de_max": de.max().item(),
        }

    total_steps = 255 * len(per_gradient)
    return {
        "per_gradient": per_gradient,
        "total_invisible_pct": sum(r["invisible_steps"] for r in per_gradient.values())
                               / total_steps * 100,
        "total_duplicate_pct": sum(r["duplicate_rgb"] for r in per_gradient.values())
                               / total_steps * 100,
    }
