"""Eased animation step uniformity.

For 5 color pairs (R-B, W-R, K-C, Y-M, G-W), interpolate 60 frames with
ease-in-out timing in the test space, quantize to 8-bit sRGB, measure CV
of consecutive ΔE2000 in CIE Lab. Lower = more uniform perceived motion
under non-linear timing.
"""
import torch

from ._common import (
    _M_SRGB_LIST, _D65_LIST,
    matrix, vec, srgb_to_linear, linear_to_srgb, xyz_to_cielab,
)
from ..gpu_de import ciede2000


_ANIM_PAIRS = [
    ("R-B", [1, 0, 0], [0, 0, 1]),
    ("W-R", [1, 1, 1], [1, 0, 0]),
    ("K-C", [0, 0, 0], [0, 1, 1]),
    ("Y-M", [1, 1, 0], [1, 0, 1]),
    ("G-W", [0, 1, 0], [1, 1, 1]),
]


def _ease_in_out(t):
    return torch.where(t < 0.5, 2 * t ** 2, 1 - (-2 * t + 2) ** 2 / 2)


def measure_eased_animation(space, device=None) -> dict:
    """CV of frame-to-frame ΔE2000 with ease-in-out timing."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    d65 = vec(_D65_LIST, dev, dt)

    cvs = {}
    for name, rgb1, rgb2 in _ANIM_PAIRS:
        xyz1 = ms @ srgb_to_linear(torch.tensor(rgb1, device=dev, dtype=dt))
        xyz2 = ms @ srgb_to_linear(torch.tensor(rgb2, device=dev, dtype=dt))
        lab1 = space.forward(xyz1.unsqueeze(0))
        lab2 = space.forward(xyz2.unsqueeze(0))

        t_lin = torch.linspace(0, 1, 60, device=dev, dtype=dt).unsqueeze(1)
        t_eased = _ease_in_out(t_lin)
        lab_eased = lab1 + t_eased * (lab2 - lab1)

        xyz_eased = space.inverse(lab_eased)
        rgb_eased = linear_to_srgb((xyz_eased @ msi.T).clamp(0, 1)).clamp(0, 1)
        rgb8 = (rgb_eased * 255).round() / 255.0
        xyz_q = srgb_to_linear(rgb8) @ ms.T
        cielab = xyz_to_cielab(xyz_q, d65)

        de = ciede2000(cielab[:-1], cielab[1:])
        mask = de > 0.001
        if mask.sum() > 1:
            cv = (de[mask].std() / (de[mask].mean() + 1e-10) * 100).item()
        else:
            cv = 0.0
        cvs[name] = cv

    cvs["mean_cv"] = sum(v for k, v in cvs.items() if k != "mean_cv") / len(_ANIM_PAIRS)
    return cvs
