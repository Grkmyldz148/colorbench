"""Munsell uniformity metrics — value scale + hue spacing.

measure_munsell_value
---------------------
CV of consecutive ΔL for the Munsell value scale (V=1..9 neutrals).
Lower = more uniform lightness ladder.

measure_munsell_hue
-------------------
CV of hue-angle gaps between the 10 Munsell principal hue chips.
Ideal: 36° apart (CV=0%). Lower = more uniform hue distribution.
"""
import math
import torch

from ._common import _M_SRGB_LIST, _D65_LIST, matrix, vec, srgb_to_linear
from ..constants import MUNSELL_VALUE_Y, MUNSELL_HUE_CHIPS_RGB

PI = math.pi


def measure_munsell_value(space, device=None) -> dict:
    """CV of L spacing for Munsell V=1..9 neutral grays."""
    dev, dt = space.device, space.dtype
    d65 = vec(_D65_LIST, dev, dt)
    grays = torch.stack([d65 * MUNSELL_VALUE_Y[v] for v in range(1, 10)])  # (9, 3)

    lab = space.forward(grays)
    L = lab[:, 0]
    dL = L[1:] - L[:-1]

    mean_dL = dL.abs().mean()
    std_dL = dL.std()
    cv = (std_dL / (mean_dL + 1e-10) * 100).item()
    return {
        "dL_cv": cv,
        "L_values": L.tolist(),
        "dL_steps": dL.tolist(),
        "mean_dL": mean_dL.item(),
    }


def measure_munsell_hue(space, device=None) -> dict:
    """CV of hue-angle gaps for 10 Munsell principal hues."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)

    xyzs = []
    for name in MUNSELL_HUE_CHIPS_RGB:
        r, g, b = MUNSELL_HUE_CHIPS_RGB[name]
        rgb = torch.tensor([r / 255.0, g / 255.0, b / 255.0], device=dev, dtype=dt)
        xyzs.append(ms @ srgb_to_linear(rgb))
    xyzs = torch.stack(xyzs)

    lab = space.forward(xyzs)
    h = torch.atan2(lab[:, 2], lab[:, 1])
    h_sorted, _ = torch.sort(h)
    dh = h_sorted[1:] - h_sorted[:-1]
    last_gap = h_sorted[0] + 2 * PI - h_sorted[-1]
    dh = torch.cat([dh, last_gap.unsqueeze(0)])

    cv = (dh.std() / (dh.mean() + 1e-10) * 100).item()
    return {
        "spacing_cv": cv,
        "hue_angles_deg": (h * 180 / PI).tolist(),
        "gaps_deg": (dh * 180 / PI).tolist(),
        "min_gap_deg": (dh.min() * 180 / PI).item(),
        "max_gap_deg": (dh.max() * 180 / PI).item(),
    }
