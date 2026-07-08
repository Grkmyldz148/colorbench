"""Gradient quality metrics — CV, hue drift, banding across pair set.

For each pair (xyz1, xyz2), interpolate S steps in the test color space's Lab,
inverse-map to XYZ, quantize through 8-bit sRGB, and measure:

  - CV (coefficient of variation): std(ΔE)/mean(ΔE) of consecutive ΔE values
    in CIE Lab. Lower = more uniform stepping. Achromatic-crossing gradients
    use ΔE thresholds to avoid divide-by-zero.

  - Hue drift: max angular hue change in CIE Lab between consecutive steps.
    Only counts where both endpoints have C* > 5 (chromatic).

  - Banding: count of duplicate consecutive 8-bit RGB triplets along the
    interpolated path. Indicates posterization.

Aggregations: per-category stats, overall percentiles, and subset CVs by
brightness / chroma / lightness range.
"""
import torch

from ._common import (
    _M_SRGB_LIST, _D65_LIST,
    matrix, vec,
    srgb_to_linear, linear_to_srgb,
    xyz_to_cielab,
)
# Bridge to existing CIEDE2000 implementation (gpu_de.py is shared infrastructure)
from ..rulers import get_ruler as _get_ruler, spacing_members as _spacing_members

_step_ruler = _get_ruler("spacing")  # uniformity -> spacing consensus ruler; notebook 2026-05-30
_RULER_MEMBERS = _spacing_members()  # individual rulers for the sensitivity check

CHUNK = 5000
PI = 3.141592653589793


def _gradient_chunk(space, pair_xyz_chunk, S, ms, msi, d65):
    """Process one chunk of pairs: interpolate, inverse, quantize, measure.

    Returns four tensors, all device-resident: (cvs, drift_max, is_crossing, banding).
    No .item() calls — synced at outer level.
    """
    nc = pair_xyz_chunk.shape[0]
    # Cast pair tensor to space's dtype/device — pairs.py creates float64 CPU.
    pair_xyz_chunk = pair_xyz_chunk.to(device=space.device, dtype=space.dtype)
    lab1 = space.forward(pair_xyz_chunk[:, 0])
    lab2 = space.forward(pair_xyz_chunk[:, 1])

    # Interpolate in the test space's Lab: (nc, S, 3)
    t = torch.linspace(0, 1, S, device=space.device, dtype=space.dtype).view(1, -1, 1)
    interp = lab1.unsqueeze(1) + t * (lab2 - lab1).unsqueeze(1)

    # Inverse → 8-bit sRGB quantize → re-encode XYZ → CIE Lab for ΔE
    flat = interp.reshape(nc * S, 3)
    xyz_flat = space.inverse(flat)
    lin = (xyz_flat @ msi.T).clamp(0, 1)
    s8 = (linear_to_srgb(lin) * 255).round() / 255.0
    xyz_q = srgb_to_linear(s8) @ ms.T
    cielab = xyz_to_cielab(xyz_q.clamp(min=1e-10), d65).reshape(nc, S, 3)

    # CV via consecutive spacing-ruler ΔE
    cl1, cl2 = cielab[:, :-1], cielab[:, 1:]
    de = _step_ruler(cl1, cl2)
    de = de.reshape(nc, S - 1)
    md = de.mean(dim=1)
    sd = de.std(dim=1)
    ok = md > 0.001
    cvs = torch.where(ok, sd / md, torch.zeros_like(md))

    # Same CV under EACH consensus member separately (ruler-sensitivity check)
    ruler_cvs = {}
    flat1 = cl1.reshape(-1, 3)
    flat2 = cl2.reshape(-1, 3)
    for rkey, rfn in _RULER_MEMBERS.items():
        de_r = torch.as_tensor(rfn(flat1, flat2), device=de.device,
                               dtype=de.dtype).reshape(nc, S - 1)
        md_r = de_r.mean(dim=1)
        sd_r = de_r.std(dim=1)
        ruler_cvs[rkey] = torch.where(md_r > 0.001, sd_r / md_r,
                                      torch.zeros_like(md_r))

    # Hue drift in CIE Lab (degrees, only where both endpoints chromatic)
    C_star = (cielab[..., 1] ** 2 + cielab[..., 2] ** 2).sqrt()
    h = torch.atan2(cielab[..., 2], cielab[..., 1])
    dh = h[:, 1:] - h[:, :-1]
    dh = torch.atan2(torch.sin(dh), torch.cos(dh)).abs() * (180.0 / PI)
    both_chrom = (C_star[:, :-1] > 5) & (C_star[:, 1:] > 5)
    drift_max = torch.where(both_chrom, dh, torch.zeros_like(dh)).max(dim=1).values
    is_crossing = (C_star < 3).any(dim=1)

    # Banding: duplicate 8-bit triplets
    s8_reshaped = s8.reshape(nc, S, 3)
    s8_int = (s8_reshaped * 255).to(torch.int32)
    dupes = (s8_int[:, 1:] == s8_int[:, :-1]).all(dim=2)
    banding = dupes.sum(dim=1).float()

    return cvs, drift_max, is_crossing, banding, ruler_cvs


def _category_stats(cvs, drift_max, banding, pair_labels, device):
    """Per-category aggregations."""
    cats = sorted(set(c for c, _ in pair_labels))
    out = {}
    for cat in cats:
        mask = torch.tensor([lbl[0] == cat for lbl in pair_labels], device=device)
        c_cvs = cvs[mask]
        c_drift = drift_max[mask]
        c_band = banding[mask]
        if c_cvs.numel() == 0:
            continue
        out[cat] = {
            "count": int(mask.sum().item()),
            "cv_mean": c_cvs.mean().item(),
            "cv_p95": c_cvs.quantile(0.95).item() if c_cvs.numel() >= 2 else c_cvs.max().item(),
            "cv_max": c_cvs.max().item(),
            "drift_mean": c_drift.mean().item(),
            "drift_max": c_drift.max().item(),
            "banding_mean": c_band.mean().item(),
        }
    return out


def _overall_stats(cvs, drift_max, is_crossing, banding, N):
    """Aggregate stats across all pairs."""
    valid_cv = cvs[cvs > 0]
    non_crossing = ~is_crossing
    out = {
        "n_total": N,
        "n_crossing": int(is_crossing.sum().item()),
        "cv_mean": valid_cv.mean().item() if valid_cv.numel() > 0 else 0,
        "cv_p50": valid_cv.quantile(0.5).item() if valid_cv.numel() >= 2 else 0,
        "cv_p95": valid_cv.quantile(0.95).item() if valid_cv.numel() >= 2 else 0,
        "cv_p99": valid_cv.quantile(0.99).item() if valid_cv.numel() >= 2 else 0,
        "cv_max": cvs.max().item(),
        "drift_mean": drift_max.mean().item(),
        "drift_p95": drift_max.quantile(0.95).item() if N >= 2 else 0,
        "drift_max_all": drift_max.max().item(),
        "drift_max_noncrossing": drift_max[non_crossing].max().item() if non_crossing.any() else 0,
        "drift_max_crossing": drift_max[is_crossing].max().item() if is_crossing.any() else 0,
        "banding_mean": banding.mean().item(),
        "banding_max": int(banding.max().item()),
    }
    return out


def _subset_masks(ref_lab1, ref_lab2):
    """Standard subset masks (bright/dark/high_chroma/etc.) in FIXED CIE Lab
    (D65), NOT the candidate's own coords: own-coordinate thresholds select a
    different pair population per space (with the old `L1 > 0.6` an 0-100-L
    space had nearly every pair counted as 'bright'), so subset CVs compared
    non-identical populations. CIE Lab here is only the bucketing frame —
    identical for every candidate; the CV itself is still measured by the
    spacing ruler.
    """
    L1, L2 = ref_lab1[:, 0], ref_lab2[:, 0]
    C1 = (ref_lab1[:, 1] ** 2 + ref_lab1[:, 2] ** 2).sqrt()
    C2 = (ref_lab2[:, 1] ** 2 + ref_lab2[:, 2] ** 2).sqrt()
    return {
        "bright": (L1 > 60.0) & (L2 > 60.0),
        "dark": (L1 < 40.0) & (L2 < 40.0),
        "high_chroma": (C1 > 50.0) & (C2 > 50.0),
        "cross_lightness": (L1 - L2).abs() > 50.0,
        "near_achromatic": (C1 < 10.0) | (C2 < 10.0),
    }


def _subset_cvs(cvs, subset_masks):
    """CV averages over the standard subsets (see _subset_masks)."""
    out = {}
    for name, mask in subset_masks.items():
        sub = cvs[mask]
        valid = sub[sub > 0]
        out[f"cv_{name}"] = valid.mean().item() if valid.numel() > 0 else 0
    return out


def measure_gradients(space, pairs_xyz, pair_labels, device=None, n_steps=26) -> dict:
    """Gradient quality metric.

    Args:
        space: ColorSpace
        pairs_xyz: (N, 2, 3) tensor of XYZ pair endpoints
        pair_labels: list of (category, description) tuples, length N
        device: legacy compatibility (ignored)
        n_steps: interpolation steps per gradient (default 26)

    Returns: dict with overall, by_category, pairs sub-dicts.
    """
    dev, dt = space.device, space.dtype
    d65 = vec(_D65_LIST, dev, dt)
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    N, S = pairs_xyz.shape[0], n_steps

    all_cvs, all_drift, all_cross, all_band = [], [], [], []
    all_ruler = {}
    for cs in range(0, N, CHUNK):
        ce = min(cs + CHUNK, N)
        cvs, drift_max, is_crossing, banding, ruler_cvs = _gradient_chunk(
            space, pairs_xyz[cs:ce], S, ms, msi, d65
        )
        all_cvs.append(cvs)
        all_drift.append(drift_max)
        all_cross.append(is_crossing)
        all_band.append(banding)
        for rkey, rc in ruler_cvs.items():
            all_ruler.setdefault(rkey, []).append(rc)

    cvs = torch.cat(all_cvs)
    drift_max = torch.cat(all_drift)
    is_crossing = torch.cat(all_cross)
    banding = torch.cat(all_band)
    ruler_cvs = {k: torch.cat(v) for k, v in all_ruler.items()}

    # Per-pair detail records (for JSON output)
    pair_results = []
    cvs_l = cvs.tolist()
    drift_l = drift_max.tolist()
    cross_l = is_crossing.tolist()
    band_l = banding.tolist()
    for i in range(N):
        cat, desc = pair_labels[i]
        pair_results.append({
            "category": cat,
            "description": desc,
            "cv": cvs_l[i],
            "drift_max": drift_l[i],
            "is_crossing": bool(cross_l[i]),
            "duplicate_steps": int(band_l[i]),
        })

    # Subset bucketing in FIXED CIE Lab (D65) — identical pair populations
    # for every candidate space (see _subset_cvs docstring)
    pairs_for_subset = pairs_xyz.to(device=space.device, dtype=space.dtype)
    ref_lab1 = xyz_to_cielab(pairs_for_subset[:, 0].clamp(min=1e-10), d65)
    ref_lab2 = xyz_to_cielab(pairs_for_subset[:, 1].clamp(min=1e-10), d65)

    overall = _overall_stats(cvs, drift_max, is_crossing, banding, N)
    subset_masks = _subset_masks(ref_lab1, ref_lab2)
    overall.update(_subset_cvs(cvs, subset_masks))

    # Per-item arrays for the paired-bootstrap tie decision (core/bootstrap.py).
    # Items are shared across spaces (same generated pairs), so resampling the
    # same indices for two spaces gives a valid paired difference CI.
    cvs_list = cvs.tolist()
    boot = {
        "overall.cv_mean": {"items": cvs_list, "stat": "mean_pos"},
        "overall.cv_p95": {"items": cvs_list, "stat": "p95_pos"},
        "overall.banding_mean": {"items": banding.tolist(), "stat": "mean"},
    }
    for sname, mask in subset_masks.items():
        boot[f"overall.cv_{sname}"] = {
            "items": cvs[mask].tolist(), "stat": "mean_pos"}

    # Same aggregates under each consensus-member ruler — the sensitivity
    # check downstream flags any verdict that flips with the ruler choice.
    def _mean_pos(t):
        v = t[t > 0]
        return v.mean().item() if v.numel() else 0.0

    ruler_sens = {}
    for rkey, rc in ruler_cvs.items():
        ruler_sens.setdefault("overall.cv_mean", {})[rkey] = _mean_pos(rc)
        vpos = rc[rc > 0]
        ruler_sens.setdefault("overall.cv_p95", {})[rkey] = (
            vpos.quantile(0.95).item() if vpos.numel() >= 2 else 0.0)
        for sname, mask in subset_masks.items():
            ruler_sens.setdefault(f"overall.cv_{sname}", {})[rkey] = \
                _mean_pos(rc[mask])

    return {
        "overall": overall,
        "by_category": _category_stats(cvs, drift_max, banding, pair_labels, dev),
        "pairs": pair_results,
        "_bootstrap": boot,
        "_ruler_sensitivity": ruler_sens,
    }
