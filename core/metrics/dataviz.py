"""Data visualization palette metrics.

measure_dataviz_distinguishability
-----------------------------------
For palette sizes {5, 10, 20}, generate evenly-hue-spaced colors at moderate
L/C in the test space. After sRGB clipping, measure min pairwise CIEDE2000.
Higher = more distinguishable categories.

measure_multistop_gradient
--------------------------
For 4 multi-stop CSS gradients, interpolate per-segment in the test space,
quantize to 8-bit, measure CV of consecutive ΔE2000 in CIE Lab. Lower = more
uniform stepping across all stops.
"""
import math
import torch

from ._common import (
    _M_SRGB_LIST, _D65_LIST,
    matrix, vec, srgb_to_linear, linear_to_srgb, xyz_to_cielab,
)
from ..gpu_de import ciede2000
from ..constants import MULTI_STOP_GRADIENTS

PI = math.pi


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


def _hex_to_xyz(hexstr, ms):
    h = hexstr.lstrip('#')
    r, g, b = int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
    rgb = torch.tensor([r, g, b], device=ms.device, dtype=ms.dtype)
    return ms @ srgb_to_linear(rgb)


def measure_dataviz_distinguishability(space, device=None) -> dict:
    """Min pairwise CIEDE2000 across n-category hue palettes."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    d65 = vec(_D65_LIST, dev, dt)

    # Reference: moderate L, moderate C
    r, g, b = _hsv_to_rgb_scalar(0.0, 0.7, 0.85)
    rgb_ref = torch.tensor([r, g, b], device=dev, dtype=dt)
    xyz_ref = ms @ srgb_to_linear(rgb_ref)
    lab_ref = space.forward(xyz_ref.unsqueeze(0))[0]
    L_ref = lab_ref[0]
    C_ref = (lab_ref[1] ** 2 + lab_ref[2] ** 2).sqrt() * 0.6

    results = {}
    for n_cats in [5, 10, 20]:
        hue_angles = torch.linspace(0, 2 * PI, n_cats + 1, device=dev, dtype=dt)[:-1]
        labs = torch.stack([
            torch.tensor(
                [L_ref.item(), (C_ref * torch.cos(h)).item(), (C_ref * torch.sin(h)).item()],
                device=dev, dtype=dt,
            )
            for h in hue_angles
        ])
        xyzs = space.inverse(labs)
        rgbs = linear_to_srgb((xyzs @ msi.T).clamp(0, 1)).clamp(0, 1)
        xyzs_clipped = srgb_to_linear(rgbs) @ ms.T
        cielabs = xyz_to_cielab(xyzs_clipped, d65)

        # Pairwise CIEDE2000 — vectorized via outer broadcast
        ci_a = cielabs.unsqueeze(0).expand(n_cats, n_cats, 3)  # (i, j, 3)
        ci_b = cielabs.unsqueeze(1).expand(n_cats, n_cats, 3)
        de_mat = ciede2000(ci_a.reshape(-1, 3), ci_b.reshape(-1, 3)).reshape(n_cats, n_cats)
        # Mask diagonal + lower triangle (we only want i<j upper triangle)
        mask = torch.triu(torch.ones(n_cats, n_cats, device=dev, dtype=torch.bool), diagonal=1)
        min_de = de_mat[mask].min().item()
        results[f"n{n_cats}_min_de"] = min_de

    results["mean_min_de"] = sum(results.values()) / len(results)
    return results


def measure_multistop_gradient(space, device=None) -> dict:
    """CV of consecutive ΔE2000 across 4 multi-stop CSS gradients."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    d65 = vec(_D65_LIST, dev, dt)

    cvs = {}
    for gname, stops in MULTI_STOP_GRADIENTS.items():
        xyz_pts = torch.stack([_hex_to_xyz(h, ms) for h in stops])
        lab_pts = space.forward(xyz_pts)

        # Interpolate per-segment, dropping junction duplicates
        all_labs = []
        K = lab_pts.shape[0]
        for i in range(K - 1):
            n_steps = 25
            t = torch.linspace(0, 1, n_steps, device=dev, dtype=dt).unsqueeze(1)
            seg = lab_pts[i:i + 1] + t * (lab_pts[i + 1:i + 2] - lab_pts[i:i + 1])
            if i < K - 2:
                seg = seg[:-1]
            all_labs.append(seg)
        all_labs = torch.cat(all_labs, dim=0)

        xyz_interp = space.inverse(all_labs)
        rgb = linear_to_srgb((xyz_interp @ msi.T).clamp(0, 1))
        rgb8 = (rgb * 255).round() / 255.0
        xyz_q = srgb_to_linear(rgb8) @ ms.T
        cielab = xyz_to_cielab(xyz_q, d65)

        de = ciede2000(cielab[:-1], cielab[1:])
        mask = de > 0.001
        if mask.sum() > 1:
            de_valid = de[mask]
            cv = (de_valid.std() / (de_valid.mean() + 1e-10) * 100).item()
        else:
            cv = 0.0
        cvs[gname] = cv

    cvs["mean_cv"] = sum(cvs.values()) / len(cvs) if cvs else 0
    return cvs
