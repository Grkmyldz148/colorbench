"""Out-of-gamut excursion in Lab-interpolated gradients.

For in-gamut sRGB endpoint pairs, interpolate 256 steps in test space, decode
back to linear sRGB and check if intermediate steps fall outside [0, 1] (with
small tolerance). Catches the OKLab blue-white purple shift issue, where the
midpoint maps to negative-channel sRGB.
"""
import torch

from ._common import _M_SRGB_LIST, matrix, srgb_to_linear


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


def _build_pairs():
    """Standard endpoint set: primaries→W/K + complementary + adjacent + pastels + 100 random.
    Order preserved for snapshot stability.
    """
    import random as _rnd
    endpoints = []
    primaries = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                 [1, 1, 0], [0, 1, 1], [1, 0, 1]]
    p_names = ["R", "G", "B", "Y", "C", "M"]
    for rgb, name in zip(primaries, p_names):
        endpoints.append((rgb, [1, 1, 1], f"{name}->W"))
        endpoints.append((rgb, [0, 0, 0], f"{name}->K"))
    for i in range(3):
        endpoints.append((primaries[i], primaries[i + 3], f"{p_names[i]}->{p_names[i+3]}"))
    for i in range(6):
        j = (i + 1) % 6
        endpoints.append((primaries[i], primaries[j], f"{p_names[i]}->{p_names[j]}"))
    for h_deg in range(0, 360, 30):
        h = h_deg / 360.0
        r1, g1, b1 = _hsv_to_rgb_scalar(h, 0.3, 0.9)
        r2, g2, b2 = _hsv_to_rgb_scalar(((h_deg + 60) % 360) / 360.0, 0.3, 0.9)
        endpoints.append(([r1, g1, b1], [r2, g2, b2], f"pastel_h{h_deg}"))
    _rnd.seed(99)
    for k in range(100):
        rgb1 = [_rnd.random(), _rnd.random(), _rnd.random()]
        rgb2 = [_rnd.random(), _rnd.random(), _rnd.random()]
        endpoints.append((rgb1, rgb2, f"rnd{k}"))
    return endpoints


def measure_oog_excursion(space, device=None) -> dict:
    """For in-gamut endpoint pairs: how often does the interpolation excurse OOG?"""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    endpoints = _build_pairs()

    n_steps = 256
    excursion_pairs = 0
    max_oog_dist = 0.0
    pair_details = []

    for rgb1, rgb2, name in endpoints:
        t1 = torch.tensor(rgb1, device=dev, dtype=dt)
        t2 = torch.tensor(rgb2, device=dev, dtype=dt)
        xyz1 = (srgb_to_linear(t1) @ ms.T).unsqueeze(0)
        xyz2 = (srgb_to_linear(t2) @ ms.T).unsqueeze(0)
        lab1 = space.forward(xyz1)[0]
        lab2 = space.forward(xyz2)[0]

        t = torch.linspace(0, 1, n_steps, device=dev, dtype=dt)
        labs = lab1.unsqueeze(0) + t.unsqueeze(1) * (lab2 - lab1).unsqueeze(0)
        xyz_interp = space.inverse(labs)
        lin = xyz_interp @ msi.T

        oog = ((lin < -0.001).any(dim=1) | (lin > 1.001).any(dim=1))
        if oog.any():
            excursion_pairs += 1
            dist_low = (-lin).clamp(min=0).max().item()
            dist_high = (lin - 1.0).clamp(min=0).max().item()
            pair_max = max(dist_low, dist_high)
            max_oog_dist = max(max_oog_dist, pair_max)
            pair_details.append({
                "pair": name,
                "oog_steps": int(oog.sum().item()),
                "max_oog_dist": pair_max,
            })

    return {
        "total_pairs": len(endpoints),
        "excursion_pairs": excursion_pairs,
        "excursion_pct": excursion_pairs / len(endpoints) * 100,
        "max_oog_dist": max_oog_dist,
        "worst_pairs": sorted(pair_details, key=lambda x: -x["max_oog_dist"])[:10],
    }
