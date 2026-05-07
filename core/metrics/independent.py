"""Third-party psychophysical dataset metrics.

These use published datasets that ColorBench-tuned spaces were not optimized
on, providing independent generalization signal:

  - Hung & Berns (1995): Constant hue loci (CL/VL)
  - Ebner & Fairchild (1998): Constant perceived-hue surfaces
  - Pointer (1980): Real surface color gamut (Illuminant C → D65 via Bradford)
"""
import json
import math
import os

import torch

from ._common import _D65_LIST, vec

PI = math.pi
_ILL_C_LIST = [0.98074, 1.0, 1.18232]


def _datasets_dir():
    """Path to datasets/ alongside colorbench/ (one extra `..` vs legacy
    because metrics/ is now a subpackage of core/, not a sibling)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "..", "datasets")


def _load_json(relpath):
    path = os.path.join(_datasets_dir(), relpath)
    with open(path) as f:
        return json.load(f)


def _circular_deviation_deg(hue_angles_deg):
    """Mean and max angular deviation from circular mean, in degrees."""
    rad = hue_angles_deg * PI / 180.0
    mean_sin = rad.sin().mean()
    mean_cos = rad.cos().mean()
    mean_hue = torch.atan2(mean_sin, mean_cos)
    diff = torch.atan2((rad - mean_hue).sin(), (rad - mean_hue).cos())
    abs_diff = diff.abs() * 180.0 / PI
    return abs_diff.mean().item(), abs_diff.max().item()


def _cielab_to_xyz(lab, white):
    """CIE Lab → XYZ helper for Pointer's Lab→XYZ step."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    delta = 6.0 / 29.0
    xyz = torch.stack([
        torch.where(fx > delta, fx ** 3, 3 * delta ** 2 * (fx - 4.0 / 29.0)),
        torch.where(fy > delta, fy ** 3, 3 * delta ** 2 * (fy - 4.0 / 29.0)),
        torch.where(fz > delta, fz ** 3, 3 * delta ** 2 * (fz - 4.0 / 29.0)),
    ], dim=-1)
    return xyz * white


def measure_hung_berns(space, device=None) -> dict:
    """Hung & Berns (1995) constant hue loci — angular deviation per locus."""
    dev, dt = space.device, space.dtype
    data = _load_json("hung_berns/hung_berns_1995.json")
    cl_loci = data["constant_hue_loci_CL"]
    vl_loci = data["constant_hue_loci_VL"]

    all_mad, all_max = [], []
    per_hue = {}
    total_samples = 0

    for hue_name in cl_loci:
        cl = cl_loci[hue_name]
        vl = vl_loci.get(hue_name, {})
        targets = [cl["XYZ_center_reference"]] + cl["XYZ_color_targets"]
        if vl:
            targets += vl["XYZ_color_targets"]
        xyz = torch.tensor(targets, device=dev, dtype=dt)
        total_samples += len(targets)

        lab = space.forward(xyz)
        hue_angles = torch.atan2(lab[:, 2], lab[:, 1]) * 180.0 / PI
        mad, maxd = _circular_deviation_deg(hue_angles)
        all_mad.append(mad)
        all_max.append(maxd)
        per_hue[hue_name] = {"mad_deg": round(mad, 2), "max_deg": round(maxd, 2),
                             "n_samples": len(targets)}

    return {
        "mean_mad_deg": sum(all_mad) / len(all_mad) if all_mad else 0,
        "max_deviation_deg": max(all_max) if all_max else 0,
        "n_hues": len(all_mad),
        "n_samples": total_samples,
        "per_hue": per_hue,
    }


def measure_ebner_fairchild(space, device=None) -> dict:
    """Ebner & Fairchild (1998) constant perceived-hue surfaces."""
    dev, dt = space.device, space.dtype
    data = _load_json("ebner_fairchild/ebner_fairchild_1998.json")
    all_mad, all_max = [], []
    per_hue = {}
    total_samples = 0

    for key in sorted(data.keys(), key=lambda k: int(k)):
        locus = data[key]
        targets = [locus["XYZ_center_reference"]] + locus["XYZ_color_targets"]
        xyz = torch.tensor(targets, device=dev, dtype=dt)
        total_samples += len(targets)

        lab = space.forward(xyz)
        hue_angles = torch.atan2(lab[:, 2], lab[:, 1]) * 180.0 / PI
        mad, maxd = _circular_deviation_deg(hue_angles)
        all_mad.append(mad)
        all_max.append(maxd)
        per_hue[f"h{locus['hue_angle']}"] = {
            "mad_deg": round(mad, 2),
            "max_deg": round(maxd, 2),
            "n_samples": len(targets),
        }

    return {
        "mean_mad_deg": sum(all_mad) / len(all_mad) if all_mad else 0,
        "max_deviation_deg": max(all_max) if all_max else 0,
        "n_hues": len(all_mad),
        "n_samples": total_samples,
        "per_hue": per_hue,
    }


def measure_pointer_gamut(space, device=None) -> dict:
    """Pointer (1980) real surface color gamut — chroma + boundary + hue uniformity."""
    dev, dt = space.device, space.dtype
    pg = _load_json("pointer_gamut/pointer_gamut_lch.json")
    points = pg["data"]

    d65 = vec(_D65_LIST, dev, dt)
    ill_c = vec(_ILL_C_LIST, dev, dt)

    # Bradford CAT: Illuminant C → D65
    M_brad = torch.tensor([
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ], device=dev, dtype=dt)
    src_cone = M_brad @ ill_c
    dst_cone = M_brad @ d65
    scale = dst_cone / src_cone
    cat_matrix = torch.linalg.inv(M_brad) @ torch.diag(scale) @ M_brad

    valid = [(L, C, H) for L, C, H in points if C > 0]
    labs = []
    for L, C, H in valid:
        h_rad = H * PI / 180.0
        labs.append([L, C * math.cos(h_rad), C * math.sin(h_rad)])

    lab_tensor = torch.tensor(labs, device=dev, dtype=dt)
    xyz_ill_c = _cielab_to_xyz(lab_tensor, ill_c)
    xyz_d65 = (cat_matrix @ xyz_ill_c.T).T

    space_lab = space.forward(xyz_d65)
    mapped_chroma = (space_lab[:, 1] ** 2 + space_lab[:, 2] ** 2).sqrt()
    mapped_hue = torch.atan2(space_lab[:, 2], space_lab[:, 1]) * 180.0 / PI

    # Metric 1: per-L chroma CV
    l_levels = sorted(set(L for L, C, H in valid))
    chroma_cvs = []
    for l_val in l_levels:
        idx = [i for i, (L, C, H) in enumerate(valid) if L == l_val]
        if len(idx) < 3:
            continue
        mc = mapped_chroma[idx]
        cv = (mc.std() / mc.mean()).item() if mc.mean() > 1e-10 else 0
        chroma_cvs.append(cv)
    chroma_cv = sum(chroma_cvs) / len(chroma_cvs) if chroma_cvs else 0

    # Metric 2: per-L boundary smoothness (hue-neighbor chroma jumps)
    smoothness_scores = []
    for l_val in l_levels:
        entries = [(i, H) for i, (L, C, H) in enumerate(valid) if L == l_val]
        if len(entries) < 3:
            continue
        entries.sort(key=lambda x: x[1])
        indices = [e[0] for e in entries]
        mc = mapped_chroma[indices]
        jumps = [abs(mc[j].item() - mc[(j + 1) % len(mc)].item()) for j in range(len(mc))]
        mean_jump = sum(jumps) / len(jumps) if jumps else 0
        mean_c = mc.mean().item()
        smoothness_scores.append(mean_jump / mean_c if mean_c > 1e-10 else 0)
    boundary_smoothness = sum(smoothness_scores) / len(smoothness_scores) if smoothness_scores else 0

    # Metric 3: hue uniformity at L=50
    mid_entries = [(i, H) for i, (L, C, H) in enumerate(valid) if L == 50]
    hue_cv = 0.0
    if len(mid_entries) >= 6:
        mid_entries.sort(key=lambda x: x[1])
        indices = [e[0] for e in mid_entries]
        mh = mapped_hue[indices]
        spacings = []
        for j in range(len(mh)):
            diff = mh[(j + 1) % len(mh)] - mh[j]
            diff_norm = ((diff + 180) % 360) - 180
            spacings.append(abs(diff_norm.item()))
        spacings_t = torch.tensor(spacings, dtype=dt)
        hue_cv = (spacings_t.std() / spacings_t.mean()).item() if spacings_t.mean() > 0 else 0

    return {
        "chroma_cv": chroma_cv,
        "boundary_smoothness": boundary_smoothness,
        "hue_uniformity_cv": hue_cv,
        "n_points": len(valid),
        "n_l_levels": len(l_levels),
    }
