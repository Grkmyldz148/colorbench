"""Hue leaf constancy — does constant hue in test space stay constant in CIE Lab?

For each hue (every 10°), build a 40 × 30 (L × C) grid in test space's Lab,
inverse-map, filter to in-gamut sRGB and chromatic (C* > 2 in CIE Lab), then
measure max angular deviation from the circular mean of CIE Lab hue across
the leaf. Lower = better hue uniformity.
"""
import math
import torch

from ._common import (
    _M_SRGB_LIST, _D65_LIST,
    matrix, vec, xyz_to_cielab,
)

PI = math.pi


def measure_hue_leaf(space, device=None) -> dict:
    """For 36 hues × 1200 (L,C) points: max CIE Lab hue deviation from mean."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    d65 = vec(_D65_LIST, dev, dt)

    per_hue = {}
    for h_deg in range(0, 360, 10):
        h_rad = h_deg * PI / 180
        ch, sh = math.cos(h_rad), math.sin(h_rad)

        Ls = torch.linspace(0.05, 0.95, 40, device=dev, dtype=dt)
        Cs = torch.linspace(0.02, 0.4, 30, device=dev, dtype=dt)
        Le = Ls.view(-1, 1).expand(40, 30).reshape(-1)
        Ce = Cs.view(1, -1).expand(40, 30).reshape(-1)

        lab = torch.stack([Le, Ce * ch, Ce * sh], dim=-1)
        xyz = space.inverse(lab)
        lin = xyz @ msi.T
        in_gamut = ((lin >= -0.01) & (lin <= 1.01)).all(dim=1)
        xyz_valid = xyz[in_gamut]
        if xyz_valid.shape[0] < 5:
            continue

        cielab = xyz_to_cielab(xyz_valid.clamp(min=1e-10), d65)
        C_star = (cielab[:, 1] ** 2 + cielab[:, 2] ** 2).sqrt()
        chromatic = C_star > 2
        if chromatic.sum() < 3:
            continue

        h_cl_deg = torch.atan2(cielab[chromatic, 2], cielab[chromatic, 1]) * (180 / PI)
        h_rad_cl = cielab[chromatic, 2].atan2(cielab[chromatic, 1])
        mean_h = torch.atan2(h_rad_cl.sin().mean(), h_rad_cl.cos().mean()) * (180 / PI)
        dh = h_cl_deg - mean_h.item()
        dh = torch.where(dh > 180, dh - 360, dh)
        dh = torch.where(dh < -180, dh + 360, dh)

        per_hue[str(h_deg)] = {
            "mean_cielab_hue": mean_h.item() % 360,
            "max_deviation": dh.abs().max().item(),
            "std_deviation": dh.std().item(),
            "n_points": int(chromatic.sum().item()),
        }

    if per_hue:
        all_max = max(v["max_deviation"] for v in per_hue.values())
        all_std = sum(v["std_deviation"] for v in per_hue.values()) / len(per_hue)
    else:
        all_max = 0
        all_std = 0

    return {
        "per_hue": per_hue,
        "max_deviation": all_max,
        "mean_std": all_std,
    }
