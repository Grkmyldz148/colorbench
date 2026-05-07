"""Color Vision Deficiency gradient distinguishability.

Generate ~100 color pairs (primary combos + pastels + darks + random),
interpolate 16 steps in test space, simulate Brettel-1997 CVD on each frame
(protan/deutan/tritan), measure min ΔE2000 between consecutive CVD-simulated
frames in CIE Lab. Lower min = adjacent steps merge under that CVD type.
"""
import torch

from ._common import (
    _M_SRGB_LIST, _D65_LIST,
    matrix, vec, srgb_to_linear, xyz_to_cielab,
)
from ..gpu_de import ciede2000


# Brettel 1997 simulation matrices in LMS domain
_CVD_LIST = {
    "protan": [
        [0.0, 1.05118294, -0.05116099],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    "deutan": [
        [1.0, 0.0, 0.0],
        [0.9513092, 0.0, 0.04866992],
        [0.0, 0.0, 1.0],
    ],
    "tritan": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-0.86744736, 1.86727089, 0.0],
    ],
}

# Hunt-Pointer-Estévez XYZ → LMS (cone fundamentals)
_M_HPE_LIST = [
    [0.4002, 0.7076, -0.0808],
    [-0.2263, 1.1653, 0.0457],
    [0.0, 0.0, 0.9182],
]


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


def _build_pairs(dev, dt):
    """Build the standard CVD test pair list. Order preserved for snapshot stability."""
    pairs = []
    names = []

    primaries = torch.tensor([
        [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 1, 1], [0, 0, 1], [1, 0, 1],
    ], device=dev, dtype=dt)
    p_names = ["R", "Y", "G", "C", "B", "M"]

    for i in range(6):
        for j in range(i + 1, 6):
            pairs.append((primaries[i], primaries[j]))
            names.append(f"{p_names[i]}-{p_names[j]}")

    for h in range(0, 360, 30):
        r1, g1, b1 = _hsv_to_rgb_scalar(h / 360, 0.3, 0.9)
        r2, g2, b2 = _hsv_to_rgb_scalar(((h + 60) % 360) / 360, 0.3, 0.9)
        pairs.append((torch.tensor([r1, g1, b1], device=dev, dtype=dt),
                      torch.tensor([r2, g2, b2], device=dev, dtype=dt)))
        names.append(f"pastel_h{h}")

    for h in range(0, 360, 30):
        r1, g1, b1 = _hsv_to_rgb_scalar(h / 360, 0.7, 0.25)
        r2, g2, b2 = _hsv_to_rgb_scalar(((h + 60) % 360) / 360, 0.7, 0.25)
        pairs.append((torch.tensor([r1, g1, b1], device=dev, dtype=dt),
                      torch.tensor([r2, g2, b2], device=dev, dtype=dt)))
        names.append(f"dark_h{h}")

    gen = torch.Generator(device="cpu").manual_seed(88)
    for k in range(50):
        r1 = torch.rand(3, generator=gen, dtype=torch.float64).to(device=dev, dtype=dt)
        r2 = torch.rand(3, generator=gen, dtype=torch.float64).to(device=dev, dtype=dt)
        pairs.append((r1, r2))
        names.append(f"rnd{k}")

    return pairs, names


def measure_cvd(space, device=None) -> dict:
    """CVD gradient distinguishability across protan/deutan/tritan."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    d65 = vec(_D65_LIST, dev, dt)
    M_hpe = matrix(_M_HPE_LIST, dev, dt)
    M_hpe_inv = torch.linalg.inv(M_hpe)

    pairs, names = _build_pairs(dev, dt)
    results = {}

    for cvd_type, cvd_mat_list in _CVD_LIST.items():
        cvd_m = matrix(cvd_mat_list, dev, dt)
        pair_results = []

        for (rgb1, rgb2), name in zip(pairs, names):
            xyz1 = (srgb_to_linear(rgb1) @ ms.T).unsqueeze(0)
            xyz2 = (srgb_to_linear(rgb2) @ ms.T).unsqueeze(0)
            lab1 = space.forward(xyz1)[0]
            lab2 = space.forward(xyz2)[0]

            t = torch.linspace(0, 1, 16, device=dev, dtype=dt)
            labs = lab1.unsqueeze(0) + t.unsqueeze(1) * (lab2 - lab1).unsqueeze(0)
            xyz_grad = space.inverse(labs)

            lms = xyz_grad @ M_hpe.T
            lms_cvd = lms @ cvd_m.T
            xyz_cvd = lms_cvd @ M_hpe_inv.T
            cielab_cvd = xyz_to_cielab(xyz_cvd.clamp(min=1e-10), d65)

            de = ciede2000(cielab_cvd[:-1], cielab_cvd[1:])
            pair_results.append({
                "pair": name,
                "min_de": de.min().item(),
                "mean_de": de.mean().item(),
            })

        results[cvd_type] = {
            "pairs": pair_results,
            "worst_min_de": min(r["min_de"] for r in pair_results),
            "mean_de": sum(r["mean_de"] for r in pair_results) / len(pair_results),
        }
    return results
