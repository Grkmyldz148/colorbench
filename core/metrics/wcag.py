"""WCAG midpoint contrast preservation.

For each (hex1, hex2) pair, interpolate midpoint in the test space, decode back
to sRGB, measure WCAG contrast ratio of midpoint vs each endpoint. Reports the
worst (min) of the two contrast values per pair, and the mean across all pairs.
"""
import torch

from ._common import _M_SRGB_LIST, matrix, srgb_to_linear, linear_to_srgb
from ..constants import WCAG_CONTRAST_PAIRS


def _hex_to_xyz(hexstr, ms):
    h = hexstr.lstrip('#')
    r, g, b = int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
    rgb = torch.tensor([r, g, b], device=ms.device, dtype=ms.dtype)
    return ms @ srgb_to_linear(rgb)


def _relative_luminance(rgb):
    lin = srgb_to_linear(rgb.clamp(0, 1))
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast_ratio(lum1, lum2):
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


def measure_wcag_midpoint_contrast(space, device=None) -> dict:
    """WCAG midpoint contrast across 5 hex pairs."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)

    min_ratios = []
    for h1, h2 in WCAG_CONTRAST_PAIRS:
        xyz1 = _hex_to_xyz(h1, ms)
        xyz2 = _hex_to_xyz(h2, ms)

        lab1 = space.forward(xyz1.unsqueeze(0))
        lab2 = space.forward(xyz2.unsqueeze(0))
        lab_mid = 0.5 * (lab1 + lab2)
        xyz_mid = space.inverse(lab_mid)[0]

        rgb_mid = linear_to_srgb((xyz_mid @ msi.T).clamp(0, 1)).clamp(0, 1)
        rgb1 = linear_to_srgb((xyz1 @ msi.T).clamp(0, 1)).clamp(0, 1)
        rgb2 = linear_to_srgb((xyz2 @ msi.T).clamp(0, 1)).clamp(0, 1)

        lum_mid = _relative_luminance(rgb_mid).item()
        lum1 = _relative_luminance(rgb1).item()
        lum2 = _relative_luminance(rgb2).item()

        cr1 = _contrast_ratio(lum_mid, lum1)
        cr2 = _contrast_ratio(lum_mid, lum2)
        min_ratios.append(min(cr1, cr2))

    return {
        "mean_min_contrast": sum(min_ratios) / len(min_ratios) if min_ratios else 0,
        "worst_contrast": min(min_ratios) if min_ratios else 0,
        "per_pair": min_ratios,
    }
