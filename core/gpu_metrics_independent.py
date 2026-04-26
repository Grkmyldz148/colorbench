"""Independent third-party benchmark metrics.

Uses published psychophysical datasets that Helmlab was NOT optimized on:
  - Hung & Berns (1995): Constant hue loci — THE standard hue linearity test.
    OKLab was explicitly optimized on this dataset.
  - Ebner & Fairchild (1998): Constant perceived-hue surfaces.
    IPT was derived from this; OKLab partially based on it.
  - Pointer (1980): Gamut of real surface colors.
    Tests gamut distortion and boundary smoothness.

Sources:
  Hung & Berns: Color Res Appl 20(5), 285-295. Zenodo 3367463.
  Ebner & Fairchild: CIC 6 (1998). Zenodo 3362536.
  Pointer: Color Res Appl 5(3), 145-155 (1980).
"""

import json
import math
import os
import torch

PI = math.pi
_D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)

# Illuminant C (used by Pointer's Gamut)
_ILL_C = torch.tensor([0.98074, 1.0, 1.18232], dtype=torch.float64)


def _to(t, device):
    return t.to(device=device, dtype=torch.float64)


def _datasets_dir():
    """Return path to datasets/ relative to colorbench/."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "..", "datasets")


def _load_json(relpath):
    path = os.path.join(_datasets_dir(), relpath)
    with open(path) as f:
        return json.load(f)


def _angular_deviation(hue_angles):
    """Mean absolute angular deviation from circular mean (degrees)."""
    if len(hue_angles) < 2:
        return 0.0, 0.0
    h = torch.tensor(hue_angles, dtype=torch.float64)
    rad = h * PI / 180.0
    mean_sin = rad.sin().mean()
    mean_cos = rad.cos().mean()
    mean_angle = torch.atan2(mean_sin, mean_cos)
    # Angular distance from mean
    diff = torch.atan2((rad - mean_angle).sin(), (rad - mean_angle).cos())
    abs_diff = diff.abs() * 180.0 / PI
    return abs_diff.mean().item(), abs_diff.max().item()


def _cielab_to_xyz(lab, white):
    """CIE Lab → XYZ (for Pointer's Gamut LCH→XYZ conversion)."""
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


# ═══════════════════════════════════════════════════════════════
#  Hung & Berns (1995) — Constant Hue Loci
# ═══════════════════════════════════════════════════════════════

def measure_hung_berns(space, device):
    """Hue linearity using Hung & Berns (1995) constant hue loci.

    12 hues × (4 CL + 9 VL) = up to 156 color targets.
    For each hue, all targets should have the same hue angle in the test space.
    Reports mean/max angular deviation in degrees. Lower is better.

    Reference: Hung & Berns, Color Res Appl 20(5), 285-295 (1995).
    Note: OKLab was explicitly optimized on this dataset.
    """
    data = _load_json("hung_berns/hung_berns_1995.json")
    cl_loci = data["constant_hue_loci_CL"]
    vl_loci = data["constant_hue_loci_VL"]

    all_mad = []      # mean absolute deviation per locus
    all_max = []       # max deviation per locus
    per_hue = {}
    total_samples = 0

    for hue_name in cl_loci:
        cl = cl_loci[hue_name]
        vl = vl_loci.get(hue_name, {})

        # Collect all XYZ targets for this hue
        targets = []
        targets.append(cl["XYZ_center_reference"])
        targets.extend(cl["XYZ_color_targets"])
        if vl:
            targets.extend(vl["XYZ_color_targets"])

        xyz = torch.tensor(targets, dtype=torch.float64, device=device)
        total_samples += len(targets)

        # Forward through space
        lab = space.forward(xyz)
        a_vals = lab[:, 1]
        b_vals = lab[:, 2]

        # Compute hue angles (degrees)
        hue_angles = torch.atan2(b_vals, a_vals) * 180.0 / PI

        # Angular deviation from circular mean
        mean_sin = (hue_angles * PI / 180.0).sin().mean()
        mean_cos = (hue_angles * PI / 180.0).cos().mean()
        mean_hue = torch.atan2(mean_sin, mean_cos)
        diff = torch.atan2(
            (hue_angles * PI / 180.0 - mean_hue).sin(),
            (hue_angles * PI / 180.0 - mean_hue).cos(),
        )
        abs_diff_deg = diff.abs() * 180.0 / PI

        mad = abs_diff_deg.mean().item()
        maxd = abs_diff_deg.max().item()

        all_mad.append(mad)
        all_max.append(maxd)
        per_hue[hue_name] = {
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


# ═══════════════════════════════════════════════════════════════
#  Ebner & Fairchild (1998) — Constant Perceived-Hue Surfaces
# ═══════════════════════════════════════════════════════════════

def measure_ebner_fairchild(space, device):
    """Hue surface planarity using Ebner & Fairchild (1998).

    15 hues × ~20 targets = ~306 samples.
    Tests whether constant-hue surfaces are planar in the test space.
    Reports mean/max angular deviation. Lower is better.

    Reference: Ebner & Fairchild, CIC 6 (1998). Zenodo 3362536.
    Note: IPT was derived from this dataset; OKLab partially based on it.
    """
    data = _load_json("ebner_fairchild/ebner_fairchild_1998.json")

    all_mad = []
    all_max = []
    per_hue = {}
    total_samples = 0

    for key in sorted(data.keys(), key=lambda k: int(k)):
        locus = data[key]
        hue_angle = locus["hue_angle"]

        targets = []
        targets.append(locus["XYZ_center_reference"])
        targets.extend(locus["XYZ_color_targets"])

        xyz = torch.tensor(targets, dtype=torch.float64, device=device)
        total_samples += len(targets)

        lab = space.forward(xyz)
        a_vals = lab[:, 1]
        b_vals = lab[:, 2]

        hue_angles = torch.atan2(b_vals, a_vals) * 180.0 / PI

        # Circular mean
        mean_sin = (hue_angles * PI / 180.0).sin().mean()
        mean_cos = (hue_angles * PI / 180.0).cos().mean()
        mean_hue = torch.atan2(mean_sin, mean_cos)
        diff = torch.atan2(
            (hue_angles * PI / 180.0 - mean_hue).sin(),
            (hue_angles * PI / 180.0 - mean_hue).cos(),
        )
        abs_diff_deg = diff.abs() * 180.0 / PI

        mad = abs_diff_deg.mean().item()
        maxd = abs_diff_deg.max().item()

        all_mad.append(mad)
        all_max.append(maxd)
        per_hue[f"h{hue_angle}"] = {
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


# ═══════════════════════════════════════════════════════════════
#  Pointer (1980) — Gamut of Real Surface Colors
# ═══════════════════════════════════════════════════════════════

def measure_pointer_gamut(space, device):
    """Gamut distortion analysis using Pointer's Gamut (1980).

    576 boundary points (16 L* levels × 36 hue angles) defining the
    maximum chroma of real surface colors. Tests how uniformly the
    space represents this gamut.

    Metrics:
      - chroma_cv: CV of mapped chroma across hues (isotropy)
      - boundary_smoothness: mean of per-L hue-neighbor chroma jumps
      - hue_uniformity: CV of hue angle spacing after mapping

    Reference: Pointer, Color Res Appl 5(3), 145-155 (1980).
    Illuminant C data, converted to D65 via Bradford CAT.
    """
    pg = _load_json("pointer_gamut/pointer_gamut_lch.json")
    points = pg["data"]  # [L*, C*ab, hab]

    # Convert CIE Lab LCH → CIE Lab → XYZ (under Illuminant C)
    # Then Bradford adapt Ill.C → D65
    d65 = _to(_D65, device)
    ill_c = _to(_ILL_C, device)

    # Bradford CAT: Illuminant C → D65
    _M_BRAD = torch.tensor([
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ], dtype=torch.float64, device=device)
    src_cone = _M_BRAD @ ill_c
    dst_cone = _M_BRAD @ d65
    scale = dst_cone / src_cone
    cat_matrix = torch.linalg.inv(_M_BRAD) @ torch.diag(scale) @ _M_BRAD

    # Filter out zero-chroma points
    valid = [(L, C, H) for L, C, H in points if C > 0]

    # LCH → Lab → XYZ
    labs = []
    for L, C, H in valid:
        h_rad = H * PI / 180.0
        a = C * math.cos(h_rad)
        b = C * math.sin(h_rad)
        labs.append([L, a, b])

    lab_tensor = torch.tensor(labs, dtype=torch.float64, device=device)
    xyz_ill_c = _cielab_to_xyz(lab_tensor, ill_c)

    # Bradford adapt to D65
    xyz_d65 = (cat_matrix @ xyz_ill_c.T).T

    # Forward through space
    space_lab = space.forward(xyz_d65)
    a_vals = space_lab[:, 1]
    b_vals = space_lab[:, 2]
    mapped_chroma = (a_vals ** 2 + b_vals ** 2).sqrt()
    mapped_hue = torch.atan2(b_vals, a_vals) * 180.0 / PI

    # --- Metric 1: Chroma isotropy ---
    # Group by L level, compute CV of mapped chroma per level
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

    # --- Metric 2: Boundary smoothness ---
    # For each L level, sort by original hue, compute neighbor jumps in mapped chroma
    smoothness_scores = []
    for l_val in l_levels:
        entries = [(i, H) for i, (L, C, H) in enumerate(valid) if L == l_val]
        if len(entries) < 3:
            continue
        entries.sort(key=lambda x: x[1])
        indices = [e[0] for e in entries]
        mc = mapped_chroma[indices]
        # Circular neighbor differences
        jumps = []
        for j in range(len(mc)):
            j_next = (j + 1) % len(mc)
            jumps.append(abs(mc[j].item() - mc[j_next].item()))
        mean_jump = sum(jumps) / len(jumps) if jumps else 0
        mean_c = mc.mean().item()
        rel_jump = mean_jump / mean_c if mean_c > 1e-10 else 0
        smoothness_scores.append(rel_jump)

    boundary_smoothness = sum(smoothness_scores) / len(smoothness_scores) if smoothness_scores else 0

    # --- Metric 3: Hue uniformity ---
    # At L=50 (mid-lightness), check if hue angles are evenly spaced after mapping
    mid_entries = [(i, H) for i, (L, C, H) in enumerate(valid) if L == 50]
    hue_cv = 0.0
    if len(mid_entries) >= 6:
        mid_entries.sort(key=lambda x: x[1])
        indices = [e[0] for e in mid_entries]
        mh = mapped_hue[indices]
        # Circular spacing between consecutive mapped hues
        spacings = []
        for j in range(len(mh)):
            j_next = (j + 1) % len(mh)
            diff = mh[j_next] - mh[j]
            # Normalize to [0, 360)
            diff_norm = ((diff + 180) % 360) - 180
            spacings.append(abs(diff_norm.item()))
        spacings_t = torch.tensor(spacings, dtype=torch.float64)
        hue_cv = (spacings_t.std() / spacings_t.mean()).item() if spacings_t.mean() > 0 else 0

    return {
        "chroma_cv": chroma_cv,
        "boundary_smoothness": boundary_smoothness,
        "hue_uniformity_cv": hue_cv,
        "n_points": len(valid),
        "n_l_levels": len(l_levels),
    }
