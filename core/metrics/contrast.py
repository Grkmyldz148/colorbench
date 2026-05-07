"""WCAG L-pair contrast ratio scan.

For each (hue × L_pair) combination, build chromatic samples in test space at
fixed C=0.15, inverse to XYZ, derive Y → relative luminance → WCAG contrast.
36 hues × 10 L pairs = 360 measurements.

Note: this is distinct from measure_wcag_midpoint_contrast (which checks
midpoint preservation). This one checks luminance recoverability across
explicit dark/light L levels at varying chromatic content.
"""
import math
import torch

PI = math.pi


_L_PAIRS = [
    (0.1, 0.3), (0.1, 0.5), (0.1, 0.9), (0.2, 0.6),
    (0.2, 0.8), (0.3, 0.7), (0.3, 0.9), (0.4, 0.8),
    (0.5, 0.9), (0.6, 0.95),
]


def measure_contrast(space, device=None) -> dict:
    """WCAG contrast across 36 hues × 10 L pairs at C=0.15."""
    dev, dt = space.device, space.dtype

    results_per_hue = []
    for h_deg in range(0, 360, 10):
        for L_lo, L_hi in _L_PAIRS:
            h_rad = h_deg * PI / 180
            ch, sh = math.cos(h_rad), math.sin(h_rad)
            lab_dark = torch.tensor([[L_lo, 0.15 * ch, 0.15 * sh]], device=dev, dtype=dt)
            lab_light = torch.tensor([[L_hi, 0.15 * ch, 0.15 * sh]], device=dev, dtype=dt)

            xyz_dark = space.inverse(lab_dark)[0]
            xyz_light = space.inverse(lab_light)[0]
            Y_dark = max(xyz_dark[1].item(), 0.0001)
            Y_light = max(xyz_light[1].item(), 0.0001)
            L1 = max(Y_dark, Y_light) + 0.05
            L2 = min(Y_dark, Y_light) + 0.05
            cr = L1 / L2
            results_per_hue.append({
                "hue": h_deg, "L_lo": L_lo, "L_hi": L_hi,
                "contrast_ratio": cr,
            })

    crs = [r["contrast_ratio"] for r in results_per_hue]
    crs_t = torch.tensor(crs)
    return {
        "per_hue": results_per_hue,
        "cr_mean": sum(crs) / len(crs),
        "cr_min": min(crs),
        "cr_max": max(crs),
        "cr_cv": (crs_t.std() / crs_t.mean()).item(),
    }
