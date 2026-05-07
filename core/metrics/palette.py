"""Palette quality metrics — lightness uniformity + tint/shade hue stability.

measure_palette_uniformity
--------------------------
For 7 hues, build 10-shade palette by interpolating in test space (white→base→black).
Re-encode through CIE Lab and report CV of consecutive ΔL.

measure_tint_shade_hue
----------------------
For 12 hues, interpolate base→white and base→black in test space. Measure
max CIE Lab hue drift across the chromatic portion of the gradient.
"""
import math
import torch

from ._common import (
    _M_SRGB_LIST, _D65_LIST,
    matrix, vec, srgb_to_linear, xyz_to_cielab,
)

PI = math.pi


def _hsv_to_rgb_scalar(h, s, v):
    """HSV → RGB on Python scalars (preserved from legacy)."""
    if s == 0:
        return v, v, v
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]


def measure_palette_uniformity(space, device=None) -> dict:
    """CV of ΔL across 7-hue × 10-shade palette interpolated in test space."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    d65 = vec(_D65_LIST, dev, dt)
    test_hues = [0, 30, 60, 120, 200, 270, 330]

    cvs = []
    for h_deg in test_hues:
        r, g, b = _hsv_to_rgb_scalar(h_deg / 360, 0.9, 0.9)
        rgb_base = torch.tensor([r, g, b], device=dev, dtype=dt)
        xyz_base = ms @ srgb_to_linear(rgb_base)

        lab_base = space.forward(xyz_base.unsqueeze(0))
        lab_white = space.forward(d65.unsqueeze(0))
        lab_black = space.forward(torch.zeros(1, 3, device=dev, dtype=dt))

        # 5 tints + base + 4 shades = 10 points
        fracs_tint = [0.9, 0.7, 0.5, 0.3, 0.1]
        fracs_shade = [0.3, 0.5, 0.7, 0.9]

        shade_xyzs = []
        for frac in fracs_tint:
            lab_interp = lab_white + frac * (lab_base - lab_white)
            shade_xyzs.append(space.inverse(lab_interp)[0])
        shade_xyzs.append(xyz_base)
        for frac in fracs_shade:
            lab_interp = lab_base + frac * (lab_black - lab_base)
            shade_xyzs.append(space.inverse(lab_interp)[0])

        shade_xyzs = torch.stack(shade_xyzs)
        cielab = xyz_to_cielab(shade_xyzs, d65)
        L_vals = cielab[:, 0]
        dL = (L_vals[1:] - L_vals[:-1]).abs()
        cv = (dL.std() / (dL.mean() + 1e-10) * 100).item()
        cvs.append(cv)

    return {
        "mean_cv": sum(cvs) / len(cvs),
        "max_cv": max(cvs),
        "per_hue": {str(h): cv for h, cv in zip(test_hues, cvs)},
    }


def measure_tint_shade_hue(space, device=None) -> dict:
    """Max CIE Lab hue drift during tinting/shading across 12 hues."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    d65 = vec(_D65_LIST, dev, dt)

    max_drifts = []
    for h_deg in range(0, 360, 30):
        r, g, b = _hsv_to_rgb_scalar(h_deg / 360, 1.0, 1.0)
        rgb = torch.tensor([r, g, b], device=dev, dtype=dt)
        xyz_base = ms @ srgb_to_linear(rgb)
        cielab_base = xyz_to_cielab(xyz_base.unsqueeze(0), d65)[0]
        h_ref = torch.atan2(cielab_base[2], cielab_base[1])

        for xyz_ach in [d65, torch.zeros(3, device=dev, dtype=dt)]:
            lab_start = space.forward(xyz_ach.unsqueeze(0))
            lab_end = space.forward(xyz_base.unsqueeze(0))
            t = torch.linspace(0, 1, 11, device=dev, dtype=dt).unsqueeze(1)
            lab_interp = lab_start + t * (lab_end - lab_start)
            xyz_interp = space.inverse(lab_interp)
            cielab_interp = xyz_to_cielab(xyz_interp, d65)
            C = (cielab_interp[:, 1] ** 2 + cielab_interp[:, 2] ** 2).sqrt()
            h_interp = torch.atan2(cielab_interp[:, 2], cielab_interp[:, 1])

            mask = C > 5.0
            if mask.sum() > 0:
                dh = torch.atan2(torch.sin(h_interp[mask] - h_ref),
                                 torch.cos(h_interp[mask] - h_ref))
                max_drifts.append(dh.abs().max().item() * 180 / PI)

    return {
        "mean_max_drift_deg": sum(max_drifts) / len(max_drifts) if max_drifts else 0,
        "overall_max_drift_deg": max(max_drifts) if max_drifts else 0,
    }
