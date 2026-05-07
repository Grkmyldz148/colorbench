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


def _hsv_to_rgb_scalar(h, s, v):
    if s == 0:
        return v, v, v
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]


def measure_animation(space, device=None) -> dict:
    """Linear (non-eased) 60fps animation smoothness — CV of frame-to-frame ΔE.

    50+ transitions: explicit primary→primary list + per-hue pastel/dark sweep
    + 20 random pairs. 120 frames (2s @ 60fps), linear interpolation in
    test space's Lab.
    """
    import random as _rnd
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    d65 = vec(_D65_LIST, dev, dt)

    transitions = [
        ("R→B", [1, 0, 0], [0, 0, 1]),
        ("Y→C", [1, 1, 0], [0, 1, 1]),
        ("G→M", [0, 1, 0], [1, 0, 1]),
        ("K→W", [0, 0, 0], [1, 1, 1]),
        ("R→W", [1, 0, 0], [1, 1, 1]),
        ("B→W", [0, 0, 1], [1, 1, 1]),
        ("G→W", [0, 1, 0], [1, 1, 1]),
        ("Y→W", [1, 1, 0], [1, 1, 1]),
        ("R→K", [1, 0, 0], [0, 0, 0]),
        ("B→K", [0, 0, 1], [0, 0, 0]),
        ("R→G", [1, 0, 0], [0, 1, 0]),
        ("R→C", [1, 0, 0], [0, 1, 1]),
    ]
    for h_deg in range(0, 360, 45):
        h = h_deg / 360
        r1, g1, b1 = _hsv_to_rgb_scalar(h, 0.2, 0.9)
        r2, g2, b2 = _hsv_to_rgb_scalar(h, 1.0, 1.0)
        transitions.append((f"p→v_h{h_deg}", [r1, g1, b1], [r2, g2, b2]))
        r1, g1, b1 = _hsv_to_rgb_scalar(h, 0.7, 0.15)
        r2, g2, b2 = _hsv_to_rgb_scalar((h + 0.25) % 1.0, 0.7, 0.15)
        transitions.append((f"dk_h{h_deg}", [r1, g1, b1], [r2, g2, b2]))

    _rnd.seed(55)
    for k in range(20):
        rgb1 = [_rnd.random(), _rnd.random(), _rnd.random()]
        rgb2 = [_rnd.random(), _rnd.random(), _rnd.random()]
        transitions.append((f"rnd{k}", rgb1, rgb2))

    n_frames = 120
    results = {}
    for name, rgb1, rgb2 in transitions:
        xyz1 = (srgb_to_linear(torch.tensor(rgb1, device=dev, dtype=dt)) @ ms.T)
        xyz2 = (srgb_to_linear(torch.tensor(rgb2, device=dev, dtype=dt)) @ ms.T)
        lab1 = space.forward(xyz1.unsqueeze(0))[0]
        lab2 = space.forward(xyz2.unsqueeze(0))[0]

        t = torch.linspace(0, 1, n_frames, device=dev, dtype=dt)
        labs = lab1.unsqueeze(0) + t.unsqueeze(1) * (lab2 - lab1).unsqueeze(0)

        xyz_frames = space.inverse(labs)
        lin = (xyz_frames @ msi.T).clamp(0, 1)
        s8 = (linear_to_srgb(lin) * 255).round() / 255.0
        xyz_q = srgb_to_linear(s8) @ ms.T
        cielab = xyz_to_cielab(xyz_q.clamp(min=1e-10), d65)

        de = ciede2000(cielab[:-1], cielab[1:])
        cv = (de.std() / de.mean()).item() if de.mean() > 0.001 else 0
        step_ratio = (de.max() / de.min()).item() if de.min() > 0.001 else float("inf")

        results[name] = {
            "cv": cv,
            "step_ratio": min(step_ratio, 999),
            "de_mean": de.mean().item(),
            "de_max": de.max().item(),
            "de_min": de.min().item(),
        }
    return results
