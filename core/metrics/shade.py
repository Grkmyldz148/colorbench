"""Shade palette quality + chroma preservation.

measure_shade_hue_consistency
-----------------------------
Designer's #1 complaint: "blue palette turns purple in the lighter shades."
For 12 base hues (incl. blue/cyan/red), generate 10-shade palettes by interpolating
toward white (tints) and black (shades) in the test space. Re-encode to CIE Lab,
measure max hue drift from the base.

measure_chroma_preservation
---------------------------
For 18 vivid color pairs, interpolate 32 steps in the test space, measure the
minimum CIE Lab C* along the path. Score = min_chroma / endpoint_chroma. Higher
= less "muddy midpoint." This is what OKLab famously fixed vs CIE Lab.
"""
import math
import torch

from ._common import (
    _M_SRGB_LIST, _D65_LIST,
    matrix, vec, srgb_to_linear, linear_to_srgb, xyz_to_cielab,
)

PI = math.pi


_BASE_COLORS = [
    ("Red", [1.0, 0.2, 0.2]),
    ("Orange", [1.0, 0.5, 0.0]),
    ("Yellow", [0.9, 0.9, 0.0]),
    ("Lime", [0.4, 0.8, 0.0]),
    ("Green", [0.0, 0.7, 0.2]),
    ("Teal", [0.0, 0.7, 0.7]),
    ("Blue", [0.2, 0.4, 1.0]),
    ("Indigo", [0.3, 0.0, 0.8]),
    ("Purple", [0.6, 0.2, 0.8]),
    ("Pink", [0.9, 0.3, 0.6]),
    ("Rose", [0.9, 0.2, 0.4]),
    ("Cyan", [0.0, 0.8, 0.9]),
]

_VIVID_PAIRS = [
    ("Red-Green", [1, 0, 0], [0, 1, 0]),
    ("Red-Blue", [1, 0, 0], [0, 0, 1]),
    ("Green-Blue", [0, 1, 0], [0, 0, 1]),
    ("Red-Cyan", [1, 0, 0], [0, 1, 1]),
    ("Green-Magenta", [0, 1, 0], [1, 0, 1]),
    ("Blue-Yellow", [0, 0, 1], [1, 1, 0]),
    ("Orange-Teal", [1, 0.5, 0], [0, 0.7, 0.7]),
    ("Pink-Mint", [1, 0.4, 0.6], [0.4, 1, 0.6]),
    ("Red-Yellow", [1, 0, 0], [1, 1, 0]),
    ("Yellow-Green", [1, 1, 0], [0, 1, 0]),
    ("Green-Cyan", [0, 1, 0], [0, 1, 1]),
    ("Cyan-Blue", [0, 1, 1], [0, 0, 1]),
    ("Blue-Magenta", [0, 0, 1], [1, 0, 1]),
    ("Magenta-Red", [1, 0, 1], [1, 0, 0]),
    ("Warm-Cool", [1, 0.6, 0.2], [0.2, 0.4, 1]),
    ("Coral-Teal", [1, 0.5, 0.5], [0.5, 1, 1]),
    ("Purple-Gold", [0.6, 0.2, 0.8], [0.9, 0.8, 0.2]),
    ("DarkRed-DarkBlue", [0.6, 0, 0], [0, 0, 0.6]),
]

_SHADE_FRACS = [0.1, 0.2, 0.35, 0.5, 0.7,   # tints toward white
                0.7, 0.5, 0.35, 0.2, 0.1]   # shades toward black


def measure_shade_hue_consistency(space, device=None) -> dict:
    """Max CIE Lab hue drift across 10-shade palette for each of 12 base hues."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    d65 = vec(_D65_LIST, dev, dt)

    results = {}
    max_drifts = []

    for name, rgb in _BASE_COLORS:
        xyz_base = ms @ srgb_to_linear(torch.tensor(rgb, device=dev, dtype=dt))
        xyz_white = d65.clone()
        xyz_black = torch.zeros(3, device=dev, dtype=dt)

        cielab_base = xyz_to_cielab(xyz_base.unsqueeze(0), d65)[0]
        C_base = (cielab_base[1] ** 2 + cielab_base[2] ** 2).sqrt()
        h_base = torch.atan2(cielab_base[2], cielab_base[1])

        if C_base < 1.0:
            results[name] = {"max_drift_deg": 0.0, "mean_drift_deg": 0.0}
            continue

        lab_base = space.forward(xyz_base.unsqueeze(0))
        lab_white = space.forward(xyz_white.unsqueeze(0))
        lab_black = space.forward(xyz_black.unsqueeze(0))

        drifts = []
        for i, frac in enumerate(_SHADE_FRACS):
            if i < 5:
                lab_shade = lab_base + frac * (lab_white - lab_base)
            else:
                lab_shade = lab_base + frac * (lab_black - lab_base)

            xyz_shade = space.inverse(lab_shade)
            rgb_shade = linear_to_srgb((xyz_shade @ msi.T).clamp(0, 1)).clamp(0, 1)
            rgb8 = (rgb_shade * 255).round() / 255.0
            xyz_q = srgb_to_linear(rgb8) @ ms.T
            cielab_shade = xyz_to_cielab(xyz_q, d65)[0]

            C_shade = (cielab_shade[1] ** 2 + cielab_shade[2] ** 2).sqrt()
            if C_shade < 3.0:
                continue

            h_shade = torch.atan2(cielab_shade[2], cielab_shade[1])
            dh = torch.atan2(torch.sin(h_shade - h_base), torch.cos(h_shade - h_base))
            drifts.append((dh.abs() * 180 / PI).item())

        max_drift = max(drifts) if drifts else 0
        mean_drift = sum(drifts) / len(drifts) if drifts else 0
        max_drifts.append(max_drift)
        results[name] = {"max_drift_deg": max_drift, "mean_drift_deg": mean_drift}

    results["overall_max_drift_deg"] = max(max_drifts) if max_drifts else 0
    results["overall_mean_max_drift_deg"] = (
        sum(max_drifts) / len(max_drifts) if max_drifts else 0
    )
    results["n_hues"] = len(_BASE_COLORS)
    return results


def measure_chroma_preservation(space, device=None) -> dict:
    """Min C* / endpoint C* across 18 vivid pair gradients (muddy detection)."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    d65 = vec(_D65_LIST, dev, dt)

    n_steps = 32
    results = {}
    ratios = []

    for name, rgb1, rgb2 in _VIVID_PAIRS:
        xyz1 = ms @ srgb_to_linear(torch.tensor(rgb1, device=dev, dtype=dt))
        xyz2 = ms @ srgb_to_linear(torch.tensor(rgb2, device=dev, dtype=dt))

        # Some spaces define interpolate() (e.g. PolarBlend); use it if available.
        if hasattr(space, "interpolate"):
            xyz_interp = space.interpolate(xyz1, xyz2, n_steps)
        else:
            lab1 = space.forward(xyz1.unsqueeze(0))
            lab2 = space.forward(xyz2.unsqueeze(0))
            t = torch.linspace(0, 1, n_steps, device=dev, dtype=dt).unsqueeze(1)
            lab_interp = lab1 + t * (lab2 - lab1)
            xyz_interp = space.inverse(lab_interp)

        rgb_interp = linear_to_srgb((xyz_interp @ msi.T).clamp(0, 1)).clamp(0, 1)
        rgb8 = (rgb_interp * 255).round() / 255.0
        xyz_q = srgb_to_linear(rgb8) @ ms.T
        cielab = xyz_to_cielab(xyz_q, d65)
        C_star = (cielab[:, 1] ** 2 + cielab[:, 2] ** 2).sqrt()

        C_endpoints = 0.5 * (C_star[0] + C_star[-1])
        C_min = C_star[1:-1].min().item()

        if C_endpoints > 1.0:
            ratio = C_min / C_endpoints.item()
        else:
            ratio = 1.0  # achromatic pair

        ratios.append(ratio)
        results[name] = {
            "min_chroma": C_min,
            "endpoint_chroma": C_endpoints.item(),
            "preservation_ratio": ratio,
        }

    results["mean_preservation"] = sum(ratios) / len(ratios) if ratios else 0
    results["n_muddy"] = sum(1 for r in ratios if r < 0.5)
    results["n_pairs"] = len(_VIVID_PAIRS)
    return results
