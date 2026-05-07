"""Hue reversal detection — does CIE Lab hue change direction during chroma reduction?

For each hue (every 1°), find cusp L via mini-scan, then sweep chroma 0.001 → 0.4
in test space at that L. Inverse, filter to chromatic in-gamut points, compute
CIE Lab hue along the path. Count direction reversals (sign changes in dh).
"""
import math
import torch

from ._common import (
    _M_SRGB_LIST, _D65_LIST,
    matrix, vec, xyz_to_cielab,
)

PI = math.pi


def _scan_cusp_L(space, hue_deg, msi, n_L=100, n_C=80):
    """Quick scan to find cusp L for a single hue."""
    dev, dt = space.device, space.dtype
    h_rad = hue_deg * 2 * PI / 360
    ch = math.cos(h_rad)
    sh = math.sin(h_rad)
    Ls = torch.linspace(0.05, 0.95, n_L, device=dev, dtype=dt)
    Cs = torch.linspace(0.001, 0.45, n_C, device=dev, dtype=dt)

    Le = Ls.view(n_L, 1).expand(n_L, n_C).reshape(-1)
    Ce = Cs.view(1, n_C).expand(n_L, n_C).reshape(-1)
    lab = torch.stack([Le, Ce * ch, Ce * sh], dim=-1)
    xyz = space.inverse(lab)
    lin = xyz @ msi.T
    ok = ((lin >= -0.002) & (lin <= 1.002)).all(dim=1).reshape(n_L, n_C)
    cv = Cs.view(1, n_C).expand(n_L, n_C)
    mc, _ = torch.where(ok, cv, torch.zeros_like(cv)).max(dim=1)
    return Ls[mc.argmax()].item()


def measure_hue_reversal(space, device=None) -> dict:
    """For 360 hues: count directional reversals of CIE Lab hue along chroma sweep."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    d65 = vec(_D65_LIST, dev, dt)

    n_C_steps = 100
    Cs = torch.linspace(0.001, 0.4, n_C_steps, device=dev, dtype=dt)

    reversal_count = 0
    max_reversal_angle = 0.0
    per_hue_results = []

    for h_deg in range(360):
        h_rad = h_deg * 2 * PI / 360
        ch, sh = math.cos(h_rad), math.sin(h_rad)
        L_val = _scan_cusp_L(space, h_deg, msi)

        lab = torch.stack([
            torch.full((n_C_steps,), L_val, device=dev, dtype=dt),
            Cs * ch,
            Cs * sh,
        ], dim=-1)
        xyz = space.inverse(lab)
        lin = xyz @ msi.T
        in_gamut = ((lin >= -0.01) & (lin <= 1.01)).all(dim=1)
        if in_gamut.sum() < 3:
            continue

        cielab = xyz_to_cielab(xyz[in_gamut].clamp(min=1e-10), d65)
        C_star = (cielab[:, 1] ** 2 + cielab[:, 2] ** 2).sqrt()
        chromatic = C_star > 1.0
        if chromatic.sum() < 3:
            continue

        h_cl = torch.atan2(cielab[chromatic, 2], cielab[chromatic, 1])
        dh = h_cl[1:] - h_cl[:-1]
        dh = torch.atan2(torch.sin(dh), torch.cos(dh))
        if dh.numel() < 2:
            continue

        signs = dh.sign()
        nonzero = signs != 0
        if nonzero.sum() < 2:
            continue
        signs_nz = signs[nonzero]
        sign_changes = (signs_nz[1:] * signs_nz[:-1] < 0).sum().item()

        if sign_changes > 0:
            reversal_count += 1
            max_rev = dh.abs().max().item() * (180 / PI)
            max_reversal_angle = max(max_reversal_angle, max_rev)
            per_hue_results.append({
                "hue": h_deg,
                "n_reversals": int(sign_changes),
                "max_angle": max_rev,
            })

    return {
        "hues_with_reversals": reversal_count,
        "max_reversal_angle": max_reversal_angle,
        "total_hues_tested": 360,
        "worst_hues": sorted(per_hue_results, key=lambda x: -x["max_angle"])[:10],
    }
