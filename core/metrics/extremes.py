"""Dark/light extreme behavior + Jacobian conditioning.

measure_extremes
----------------
Three probes:
  - Near-black hue stability: 6 primaries scaled to dark (0.005-0.05)
    Reports circular variance of hue across the dark ramp.
  - Near-white L reversals: 20 grays in [0.95, 1.0]
  - Full L ordering: 256 sRGB grays (0..1) — count L decreases.

measure_jacobian
----------------
Numerical 3×3 Jacobian condition number at 5000 random colors (sRGB+P3+Rec2020).
Reports mean / max / p95 + per-region (dark / mid / bright).
"""
import math
import torch

from ._common import (
    _M_SRGB_LIST, _M_P3_LIST, _M_REC2020_LIST, _D65_LIST,
    matrix, vec, srgb_to_linear,
)

PI = math.pi


def measure_extremes(space, device=None) -> dict:
    """Near-black hue, near-white L reversals, full L ordering."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)

    hues_srgb = torch.tensor([
        [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 1, 1], [0, 0, 1], [1, 0, 1],
    ], device=dev, dtype=dt)
    h_names = ["R", "Y", "G", "C", "B", "M"]

    dark_hue_var = []
    for i, name in enumerate(h_names):
        scales = torch.linspace(0.005, 0.05, 20, device=dev, dtype=dt)
        dark_srgb = scales.unsqueeze(1) * hues_srgb[i].unsqueeze(0)
        dark_xyz = srgb_to_linear(dark_srgb) @ ms.T
        lab = space.forward(dark_xyz)
        h_rad = lab[:, 2].atan2(lab[:, 1])
        mean_sin = h_rad.sin().mean()
        mean_cos = h_rad.cos().mean()
        R_len = (mean_sin ** 2 + mean_cos ** 2).sqrt()
        circ_var = (1 - R_len).item()
        dark_hue_var.append({"primary": name, "circular_variance": circ_var})

    results = {
        "near_black_hue_stability": dark_hue_var,
        "near_black_max_variance": max(d["circular_variance"] for d in dark_hue_var),
    }

    bright_g = torch.linspace(0.95, 1.0, 20, device=dev, dtype=dt)
    bright_xyz = srgb_to_linear(bright_g.unsqueeze(1).expand(20, 3)) @ ms.T
    L_diffs = space.forward(bright_xyz)[1:, 0] - space.forward(bright_xyz)[:-1, 0]
    results["near_white_L_reversals"] = int((L_diffs < -1e-10).sum().item())

    g256 = torch.linspace(0, 1, 256, device=dev, dtype=dt)
    g256_xyz = srgb_to_linear(g256.unsqueeze(1).expand(256, 3)) @ ms.T
    g256_lab = space.forward(g256_xyz)
    L_diffs_full = g256_lab[1:, 0] - g256_lab[:-1, 0]
    results["full_L_reversals"] = int((L_diffs_full < -1e-10).sum().item())
    results["L_range"] = [g256_lab[0, 0].item(), g256_lab[-1, 0].item()]

    return results


def measure_jacobian(space, device=None) -> dict:
    """Numerical Jacobian condition number across 5000 random colors."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    mp3 = matrix(_M_P3_LIST, dev, dt)
    mr2020 = matrix(_M_REC2020_LIST, dev, dt)
    eps = 1e-7

    gen = torch.Generator(device=dev).manual_seed(77)
    srgb_c = torch.rand(2000, 3, generator=gen, device=dev, dtype=dt)
    p3_c = torch.rand(1500, 3, generator=gen, device=dev, dtype=dt)
    r2020_c = torch.rand(1500, 3, generator=gen, device=dev, dtype=dt)
    xyz = torch.cat([
        srgb_to_linear(srgb_c) @ ms.T,
        srgb_to_linear(p3_c) @ mp3.T,
        srgb_to_linear(r2020_c) @ mr2020.T,
    ], dim=0)

    conditions = []
    for k in range(xyz.shape[0]):
        x0 = xyz[k]
        lab0 = space.forward(x0.unsqueeze(0))[0]
        J = torch.zeros(3, 3, device=dev, dtype=dt)
        for j in range(3):
            dx = torch.zeros(3, device=dev, dtype=dt)
            dx[j] = eps
            lab_plus = space.forward((x0 + dx).unsqueeze(0))[0]
            J[:, j] = (lab_plus - lab0) / eps
        conditions.append(torch.linalg.cond(J).item())

    conditions = torch.tensor(conditions, device=dev, dtype=dt)
    L_vals = space.forward(xyz)[:, 0]

    dark_mask = L_vals < 0.2
    mid_mask = (L_vals >= 0.2) & (L_vals <= 0.8)
    bright_mask = L_vals > 0.8

    return {
        "mean": conditions.mean().item(),
        "max": conditions.max().item(),
        "p95": conditions.quantile(0.95).item(),
        "by_region": {
            "dark": conditions[dark_mask.cpu()].mean().item() if dark_mask.any() else 0,
            "mid": conditions[mid_mask.cpu()].mean().item() if mid_mask.any() else 0,
            "bright": conditions[bright_mask.cpu()].mean().item() if bright_mask.any() else 0,
        },
    }
