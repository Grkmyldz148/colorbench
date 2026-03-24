"""Perceptual uniformity & application metrics — ported from scripts/benchmark.

13 measure_* functions, all torch-based, following gpu_metrics_advanced.py pattern.
"""

import math
import torch
from .constants import (
    MUNSELL_VALUE_Y, MUNSELL_HUE_CHIPS_RGB, MACADAM_CENTERS,
    MULTI_STOP_GRADIENTS, WCAG_CONTRAST_PAIRS,
)

PI = math.pi

_D65 = torch.tensor([0.95047, 1.0, 1.08883])
_M_SRGB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_M_SRGB_INV = torch.linalg.inv(_M_SRGB)
_M_P3 = torch.tensor([
    [0.4865709486482162, 0.26566769316909306, 0.1982172852343625],
    [0.2289745640697488, 0.6917385218365064, 0.079286914093745],
    [0.0, 0.04511338185890264, 1.0439443689009757],
])


def _to(t, device):
    return t.to(device=device, dtype=torch.float64)


def _srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))


def _linear_to_srgb(c):
    return torch.where(c <= 0.0031308, c * 12.92,
                       1.055 * c.clamp(min=1e-12).pow(1.0 / 2.4) - 0.055)


def _xyz_to_cielab(xyz, d65):
    r = xyz / d65
    delta3 = (6.0 / 29.0) ** 3
    f = torch.where(r > delta3, r.pow(1.0 / 3.0),
                    r / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def _ciede2000_simplified(cl1, cl2):
    dL = cl2[..., 0] - cl1[..., 0]
    C1 = (cl1[..., 1] ** 2 + cl1[..., 2] ** 2).sqrt()
    C2 = (cl2[..., 1] ** 2 + cl2[..., 2] ** 2).sqrt()
    dC = C2 - C1
    dH = ((cl2[..., 1] - cl1[..., 1]) ** 2 +
          (cl2[..., 2] - cl1[..., 2]) ** 2 - dC ** 2).clamp(min=0).sqrt()
    SL = 1 + 0.015 * (cl1[..., 0] - 50) ** 2 / (20 + (cl1[..., 0] - 50) ** 2).sqrt()
    SC = 1 + 0.045 * C1
    SH = 1 + 0.015 * C1
    return ((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2).sqrt()


def _hsv_to_rgb(h, s, v):
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
    """Convert #rrggbb hex to XYZ."""
    h = hexstr.lstrip('#')
    r, g, b = int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
    rgb = torch.tensor([r, g, b], dtype=torch.float64, device=ms.device)
    lin = _srgb_to_linear(rgb)
    return ms @ lin


# ═══════════════════════════════════════════════════════════════
#  21. Munsell Value Scale Uniformity
# ═══════════════════════════════════════════════════════════════

def measure_munsell_value(space, device):
    """CV of L spacing for Munsell V=1..9 neutral grays.

    Expected: OKLab ~25%, CIE Lab ~3% (CIE Lab was designed for this).
    Lower is better.
    """
    d65 = _to(_D65, device)
    grays = torch.stack([d65 * MUNSELL_VALUE_Y[v] for v in range(1, 10)])  # (9, 3)
    grays = grays.to(device=device, dtype=torch.float64)

    lab = space.forward(grays)  # (9, 3)
    L = lab[:, 0]
    dL = L[1:] - L[:-1]  # (8,)

    mean_dL = dL.abs().mean()
    std_dL = dL.std()
    cv = (std_dL / (mean_dL + 1e-10) * 100).item()

    return {
        "dL_cv": cv,
        "L_values": L.tolist(),
        "dL_steps": dL.tolist(),
        "mean_dL": mean_dL.item(),
    }


# ═══════════════════════════════════════════════════════════════
#  22. Munsell Hue Spacing Uniformity
# ═══════════════════════════════════════════════════════════════

def measure_munsell_hue(space, device):
    """CV of hue angle gaps for 10 Munsell principal hues.

    Ideal: 10 hues at 36 degree intervals (CV=0%).
    Expected: OKLab ~15-20%, CIE Lab ~10%.
    Lower is better.
    """
    ms = _to(_M_SRGB, device)

    xyzs = []
    for name in MUNSELL_HUE_CHIPS_RGB:
        r, g, b = MUNSELL_HUE_CHIPS_RGB[name]
        rgb = torch.tensor([r / 255.0, g / 255.0, b / 255.0],
                           dtype=torch.float64, device=device)
        xyzs.append(ms @ _srgb_to_linear(rgb))
    xyzs = torch.stack(xyzs)  # (10, 3)

    lab = space.forward(xyzs)  # (10, 3)
    h = torch.atan2(lab[:, 2], lab[:, 1])  # (10,) radians

    h_sorted, _ = torch.sort(h)
    dh = h_sorted[1:] - h_sorted[:-1]  # (9,)
    last_gap = h_sorted[0] + 2 * PI - h_sorted[-1]
    dh = torch.cat([dh, last_gap.unsqueeze(0)])  # (10,)

    mean_dh = dh.mean()
    std_dh = dh.std()
    cv = (std_dh / (mean_dh + 1e-10) * 100).item()

    return {
        "spacing_cv": cv,
        "hue_angles_deg": (h * 180 / PI).tolist(),
        "gaps_deg": (dh * 180 / PI).tolist(),
        "min_gap_deg": (dh.min() * 180 / PI).item(),
        "max_gap_deg": (dh.max() * 180 / PI).item(),
    }


# ═══════════════════════════════════════════════════════════════
#  23. MacAdam Ellipse Isotropy
# ═══════════════════════════════════════════════════════════════

def measure_macadam_isotropy(space, device):
    """Mean anisotropy ratio at 25 MacAdam centers.

    Perturb in 8 directions in CIE xy, measure Lab distance ratio.
    Ideal: 1.0 (isotropic). Expected: OKLab ~2.0, CIE Lab ~2.5.
    Lower is better.
    """
    eps = 0.002
    n_dirs = 8
    Y_val = 0.5

    ratios = []
    for cx, cy in MACADAM_CENTERS:
        if cy < 1e-10:
            continue
        # Center XYZ
        X = cx / cy * Y_val
        Z = (1 - cx - cy) / cy * Y_val
        xyz_c = torch.tensor([X, Y_val, Z], dtype=torch.float64, device=device)
        lab_c = space.forward(xyz_c.unsqueeze(0))[0]

        dists = []
        for k in range(n_dirs):
            angle = k * 2 * PI / n_dirs
            dx = eps * math.cos(angle)
            dy = eps * math.sin(angle)
            nx, ny = cx + dx, cy + dy
            if ny < 1e-10:
                continue
            nX = nx / ny * Y_val
            nZ = (1 - nx - ny) / ny * Y_val
            xyz_p = torch.tensor([nX, Y_val, nZ], dtype=torch.float64, device=device)
            lab_p = space.forward(xyz_p.unsqueeze(0))[0]
            dist = (lab_p - lab_c).pow(2).sum().sqrt().item()
            dists.append(dist)

        if len(dists) >= 4:
            ratio = max(dists) / (min(dists) + 1e-30)
            ratios.append(ratio)

    mean_ratio = sum(ratios) / len(ratios) if ratios else 0
    max_ratio = max(ratios) if ratios else 0

    return {
        "mean_ratio": mean_ratio,
        "max_ratio": max_ratio,
        "n_centers": len(ratios),
    }


# ═══════════════════════════════════════════════════════════════
#  24. Palette L* Spacing Uniformity
# ═══════════════════════════════════════════════════════════════

def measure_palette_uniformity(space, device):
    """CV of CIE Lab L* spacing across 10-shade palettes.

    7 test hues, each with 10 shades (white→base→black).
    Lower is better.
    """
    ms = _to(_M_SRGB, device)
    d65 = _to(_D65, device)
    test_hues = [0, 30, 60, 120, 200, 270, 330]

    cvs = []
    for h_deg in test_hues:
        r, g, b = _hsv_to_rgb(h_deg / 360, 0.9, 0.9)
        rgb_base = torch.tensor([r, g, b], dtype=torch.float64, device=device)
        xyz_base = ms @ _srgb_to_linear(rgb_base)
        xyz_white = d65.clone()
        xyz_black = torch.zeros(3, dtype=torch.float64, device=device)

        lab_base = space.forward(xyz_base.unsqueeze(0))
        lab_white = space.forward(xyz_white.unsqueeze(0))
        lab_black = space.forward(xyz_black.unsqueeze(0))

        # 5 tints + base + 4 shades = 10 points
        fracs_tint = [0.9, 0.7, 0.5, 0.3, 0.1]
        fracs_shade = [0.3, 0.5, 0.7, 0.9]

        shade_xyzs = []
        for frac in fracs_tint:
            lab_interp = lab_white + frac * (lab_base - lab_white)
            shade_xyzs.append(space.inverse(lab_interp)[0])
        shade_xyzs.append(xyz_base)
        for frac in fracs_shade:
            lab_interp = lab_base + frac * (lab_black - lab_base)
            shade_xyzs.append(space.inverse(lab_interp)[0])

        shade_xyzs = torch.stack(shade_xyzs)  # (10, 3)
        cielab = _xyz_to_cielab(shade_xyzs, d65)
        L_vals = cielab[:, 0]
        dL = (L_vals[1:] - L_vals[:-1]).abs()
        cv = (dL.std() / (dL.mean() + 1e-10) * 100).item()
        cvs.append(cv)

    return {
        "mean_cv": sum(cvs) / len(cvs),
        "max_cv": max(cvs),
        "per_hue": {str(h): cv for h, cv in zip(test_hues, cvs)},
    }


# ═══════════════════════════════════════════════════════════════
#  25. Tint/Shade Hue Preservation
# ═══════════════════════════════════════════════════════════════

def measure_tint_shade_hue(space, device):
    """Max CIE Lab hue drift during tinting/shading.

    12 hues, interpolate toward white and black, measure hue drift.
    Lower is better.
    """
    ms = _to(_M_SRGB, device)
    d65 = _to(_D65, device)

    max_drifts = []
    for h_deg in range(0, 360, 30):
        r, g, b = _hsv_to_rgb(h_deg / 360, 1.0, 1.0)
        rgb = torch.tensor([r, g, b], dtype=torch.float64, device=device)
        xyz_base = ms @ _srgb_to_linear(rgb)
        cielab_base = _xyz_to_cielab(xyz_base.unsqueeze(0), d65)[0]
        h_ref = torch.atan2(cielab_base[2], cielab_base[1])

        xyz_white = d65.clone()
        xyz_black = torch.zeros(3, dtype=torch.float64, device=device)

        for xyz_ach in [xyz_white, xyz_black]:
            lab_start = space.forward(xyz_ach.unsqueeze(0))
            lab_end = space.forward(xyz_base.unsqueeze(0))
            t = torch.linspace(0, 1, 11, dtype=torch.float64, device=device).unsqueeze(1)
            lab_interp = lab_start + t * (lab_end - lab_start)  # (11, 3)
            xyz_interp = space.inverse(lab_interp)
            cielab_interp = _xyz_to_cielab(xyz_interp, d65)
            C = (cielab_interp[:, 1] ** 2 + cielab_interp[:, 2] ** 2).sqrt()
            h_interp = torch.atan2(cielab_interp[:, 2], cielab_interp[:, 1])

            mask = C > 5.0
            if mask.sum() > 0:
                dh = torch.atan2(torch.sin(h_interp[mask] - h_ref),
                                 torch.cos(h_interp[mask] - h_ref))
                max_drift = dh.abs().max().item() * 180 / PI
                max_drifts.append(max_drift)

    mean_drift = sum(max_drifts) / len(max_drifts) if max_drifts else 0
    overall_max = max(max_drifts) if max_drifts else 0

    return {
        "mean_max_drift_deg": mean_drift,
        "overall_max_drift_deg": overall_max,
    }


# ═══════════════════════════════════════════════════════════════
#  26. Data Viz Palette Distinguishability
# ═══════════════════════════════════════════════════════════════

def measure_dataviz_distinguishability(space, device):
    """Min pairwise CIEDE2000 for evenly-spaced hue palettes.

    Generate 5, 10, 20 category palettes with uniform hue spacing in space.
    Higher min dE is better (more distinguishable).
    """
    ms = _to(_M_SRGB, device)
    msi = _to(_M_SRGB_INV, device)
    d65 = _to(_D65, device)

    # Reference: moderate L, moderate C
    r, g, b = _hsv_to_rgb(0.0, 0.7, 0.85)
    rgb_ref = torch.tensor([r, g, b], dtype=torch.float64, device=device)
    xyz_ref = ms @ _srgb_to_linear(rgb_ref)
    lab_ref = space.forward(xyz_ref.unsqueeze(0))[0]
    L_ref = lab_ref[0]
    C_ref = (lab_ref[1] ** 2 + lab_ref[2] ** 2).sqrt() * 0.6

    results = {}
    for n_cats in [5, 10, 20]:
        hue_angles = torch.linspace(0, 2 * PI, n_cats + 1, dtype=torch.float64, device=device)[:-1]
        labs = torch.stack([
            torch.tensor([L_ref, C_ref * torch.cos(h), C_ref * torch.sin(h)],
                         dtype=torch.float64, device=device)
            for h in hue_angles
        ])
        xyzs = space.inverse(labs)
        # Clip to sRGB
        rgbs = _linear_to_srgb((xyzs @ msi.T).clamp(0, 1)).clamp(0, 1)
        xyzs_clipped = _srgb_to_linear(rgbs) @ ms.T
        cielabs = _xyz_to_cielab(xyzs_clipped, d65)

        # Min pairwise CIEDE2000
        min_de = float('inf')
        for i in range(n_cats):
            for j in range(i + 1, n_cats):
                de = _ciede2000_simplified(cielabs[i:i + 1], cielabs[j:j + 1]).item()
                if de < min_de:
                    min_de = de
        results[f"n{n_cats}_min_de"] = min_de

    mean_min = sum(results.values()) / len(results)
    results["mean_min_de"] = mean_min
    return results


# ═══════════════════════════════════════════════════════════════
#  27. Multi-Stop Gradient CV
# ═══════════════════════════════════════════════════════════════

def measure_multistop_gradient(space, device):
    """CV of step-size dE for multi-stop CSS gradients.

    4 gradient patterns, 25 steps per segment. Lower is better.
    """
    ms = _to(_M_SRGB, device)
    msi = _to(_M_SRGB_INV, device)
    d65 = _to(_D65, device)

    cvs = {}
    for gname, stops in MULTI_STOP_GRADIENTS.items():
        xyz_pts = torch.stack([_hex_to_xyz(h, ms) for h in stops])  # (K, 3)
        lab_pts = space.forward(xyz_pts)  # (K, 3)

        # Interpolate segment by segment
        all_labs = []
        K = lab_pts.shape[0]
        for i in range(K - 1):
            n_steps = 25
            t = torch.linspace(0, 1, n_steps, dtype=torch.float64, device=device).unsqueeze(1)
            seg = lab_pts[i:i + 1] + t * (lab_pts[i + 1:i + 2] - lab_pts[i:i + 1])
            if i < K - 2:
                seg = seg[:-1]  # avoid duplicating junction points
            all_labs.append(seg)
        all_labs = torch.cat(all_labs, dim=0)  # (N, 3)

        # Inverse, quantize to 8-bit, convert to CIE Lab
        xyz_interp = space.inverse(all_labs)
        rgb = _linear_to_srgb((xyz_interp @ msi.T).clamp(0, 1))
        rgb8 = (rgb * 255).round() / 255.0
        xyz_q = _srgb_to_linear(rgb8) @ ms.T
        cielab = _xyz_to_cielab(xyz_q, d65)

        de = _ciede2000_simplified(cielab[:-1], cielab[1:])
        mask = de > 0.001
        if mask.sum() > 1:
            de_valid = de[mask]
            cv = (de_valid.std() / (de_valid.mean() + 1e-10) * 100).item()
        else:
            cv = 0.0
        cvs[gname] = cv

    mean_cv = sum(cvs.values()) / len(cvs) if cvs else 0
    cvs["mean_cv"] = mean_cv
    return cvs


# ═══════════════════════════════════════════════════════════════
#  28. WCAG Midpoint Contrast
# ═══════════════════════════════════════════════════════════════

def measure_wcag_midpoint_contrast(space, device):
    """Contrast ratio preservation at gradient midpoint.

    5 hex pairs, interpolate midpoint in space, measure WCAG contrast vs endpoints.
    Higher is better.
    """
    ms = _to(_M_SRGB, device)
    msi = _to(_M_SRGB_INV, device)

    def relative_luminance(rgb):
        lin = _srgb_to_linear(rgb.clamp(0, 1))
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    def contrast_ratio(lum1, lum2):
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        return (lighter + 0.05) / (darker + 0.05)

    min_ratios = []
    for h1, h2 in WCAG_CONTRAST_PAIRS:
        xyz1 = _hex_to_xyz(h1, ms)
        xyz2 = _hex_to_xyz(h2, ms)

        lab1 = space.forward(xyz1.unsqueeze(0))
        lab2 = space.forward(xyz2.unsqueeze(0))
        lab_mid = 0.5 * (lab1 + lab2)
        xyz_mid = space.inverse(lab_mid)[0]

        rgb_mid = _linear_to_srgb((xyz_mid @ msi.T).clamp(0, 1)).clamp(0, 1)
        rgb1 = _linear_to_srgb((xyz1 @ msi.T).clamp(0, 1)).clamp(0, 1)
        rgb2 = _linear_to_srgb((xyz2 @ msi.T).clamp(0, 1)).clamp(0, 1)

        lum_mid = relative_luminance(rgb_mid).item()
        lum1 = relative_luminance(rgb1).item()
        lum2 = relative_luminance(rgb2).item()

        cr1 = contrast_ratio(lum_mid, lum1)
        cr2 = contrast_ratio(lum_mid, lum2)
        min_ratios.append(min(cr1, cr2))

    mean_min = sum(min_ratios) / len(min_ratios) if min_ratios else 0
    return {
        "mean_min_contrast": mean_min,
        "worst_contrast": min(min_ratios) if min_ratios else 0,
        "per_pair": min_ratios,
    }


# ═══════════════════════════════════════════════════════════════
#  29. Palette Harmony Hue Accuracy
# ═══════════════════════════════════════════════════════════════

def measure_harmony_accuracy(space, device):
    """Accuracy of hue rotations (complementary, triadic, analogous).

    12 base hues, rotate in space, measure actual CIE Lab hue rotation vs intended.
    Lower is better.
    """
    ms = _to(_M_SRGB, device)
    d65 = _to(_D65, device)

    errors = []
    for base_h_deg in range(0, 360, 30):
        r, g, b = _hsv_to_rgb(base_h_deg / 360, 0.8, 0.8)
        rgb = torch.tensor([r, g, b], dtype=torch.float64, device=device)
        xyz_base = ms @ _srgb_to_linear(rgb)
        lab_base = space.forward(xyz_base.unsqueeze(0))[0]
        C_base = (lab_base[1] ** 2 + lab_base[2] ** 2).sqrt()
        h_base = torch.atan2(lab_base[2], lab_base[1])

        cielab_base = _xyz_to_cielab(xyz_base.unsqueeze(0), d65)[0]
        h_cielab_base = torch.atan2(cielab_base[2], cielab_base[1])

        for rot_deg in [180, 120, 30]:
            rot_rad = rot_deg * PI / 180
            h_new = h_base + rot_rad
            lab_new = torch.tensor([lab_base[0], C_base * torch.cos(h_new),
                                    C_base * torch.sin(h_new)],
                                   dtype=torch.float64, device=device)
            xyz_new = space.inverse(lab_new.unsqueeze(0))[0]
            cielab_new = _xyz_to_cielab(xyz_new.unsqueeze(0), d65)[0]
            h_cielab_new = torch.atan2(cielab_new[2], cielab_new[1])

            actual_rot = torch.atan2(torch.sin(h_cielab_new - h_cielab_base),
                                     torch.cos(h_cielab_new - h_cielab_base))
            actual_deg = actual_rot.item() * 180 / PI
            error = abs(actual_deg - rot_deg)
            if error > 180:
                error = 360 - error
            errors.append(error)

    mean_error = sum(errors) / len(errors) if errors else 0
    return {
        "mean_error_deg": mean_error,
        "max_error_deg": max(errors) if errors else 0,
    }


# ═══════════════════════════════════════════════════════════════
#  30. Photo Gamut Map Fidelity (P3 → sRGB)
# ═══════════════════════════════════════════════════════════════

def measure_photo_gamut_map(space, device):
    """Hue shift when chroma-reducing P3 colors to sRGB via the space.

    100 random P3 colors, binary-search chroma reduction. Lower is better.
    """
    ms = _to(_M_SRGB, device)
    msi = _to(_M_SRGB_INV, device)
    mp3 = _to(_M_P3, device)
    d65 = _to(_D65, device)

    gen = torch.Generator(device=device).manual_seed(99)
    p3_rgb = torch.rand(500, 3, generator=gen, dtype=torch.float64, device=device)
    p3_xyz = _srgb_to_linear(p3_rgb) @ mp3.T  # (100, 3)

    # Check which are out of sRGB
    srgb_lin = p3_xyz @ msi.T
    out_of_gamut = (srgb_lin < -0.001).any(dim=1) | (srgb_lin > 1.001).any(dim=1)

    if out_of_gamut.sum() == 0:
        return {"mean_hue_shift_deg": 0.0, "n_mapped": 0}

    xyz_oog = p3_xyz[out_of_gamut]  # (M, 3)
    cielab_orig = _xyz_to_cielab(xyz_oog, d65)
    h_orig = torch.atan2(cielab_orig[:, 2], cielab_orig[:, 1])

    # Chroma reduction via bisection in space
    lab_oog = space.forward(xyz_oog)  # (M, 3)
    L_oog = lab_oog[:, 0:1]
    a_oog = lab_oog[:, 1:2]
    b_oog = lab_oog[:, 2:3]
    C_oog = (a_oog ** 2 + b_oog ** 2 + 1e-30).sqrt()
    h_space = torch.atan2(b_oog, a_oog)

    # Bisection: find max t in [0,1] where C*t is in sRGB
    lo = torch.zeros_like(C_oog)
    hi = torch.ones_like(C_oog)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        C_test = C_oog * mid
        lab_test = torch.cat([L_oog, C_test * torch.cos(h_space), C_test * torch.sin(h_space)], dim=1)
        xyz_test = space.inverse(lab_test)
        linear_rgb = xyz_test @ msi.T  # Don't clamp — check raw linear RGB
        in_gamut = (linear_rgb >= -0.001).all(dim=1, keepdim=True) & (linear_rgb <= 1.001).all(dim=1, keepdim=True)
        lo = torch.where(in_gamut, mid, lo)
        hi = torch.where(in_gamut, hi, mid)

    # Use the found boundary point
    C_mapped = C_oog * lo
    lab_mapped = torch.cat([L_oog, C_mapped * torch.cos(h_space), C_mapped * torch.sin(h_space)], dim=1)
    xyz_mapped = space.inverse(lab_mapped)
    cielab_mapped = _xyz_to_cielab(xyz_mapped, d65)
    h_mapped = torch.atan2(cielab_mapped[:, 2], cielab_mapped[:, 1])

    dh = torch.atan2(torch.sin(h_mapped - h_orig), torch.cos(h_mapped - h_orig))
    hue_shifts = dh.abs() * 180 / PI

    return {
        "mean_hue_shift_deg": hue_shifts.mean().item(),
        "max_hue_shift_deg": hue_shifts.max().item(),
        "n_mapped": int(out_of_gamut.sum().item()),
    }


# ═══════════════════════════════════════════════════════════════
#  31. Eased Animation CV
# ═══════════════════════════════════════════════════════════════

def measure_eased_animation(space, device):
    """CV of frame-to-frame dE with ease-in-out timing.

    Tests whether the space maintains uniformity under non-linear timing.
    Lower is better.
    """
    ms = _to(_M_SRGB, device)
    msi = _to(_M_SRGB_INV, device)
    d65 = _to(_D65, device)

    anim_pairs = [
        ("R-B", [1, 0, 0], [0, 0, 1]),
        ("W-R", [1, 1, 1], [1, 0, 0]),
        ("K-C", [0, 0, 0], [0, 1, 1]),
        ("Y-M", [1, 1, 0], [1, 0, 1]),
        ("G-W", [0, 1, 0], [1, 1, 1]),
    ]

    def ease_in_out(t):
        return torch.where(t < 0.5, 2 * t ** 2, 1 - (-2 * t + 2) ** 2 / 2)

    cvs = {}
    for name, rgb1, rgb2 in anim_pairs:
        xyz1 = ms @ _srgb_to_linear(torch.tensor(rgb1, dtype=torch.float64, device=device))
        xyz2 = ms @ _srgb_to_linear(torch.tensor(rgb2, dtype=torch.float64, device=device))
        lab1 = space.forward(xyz1.unsqueeze(0))
        lab2 = space.forward(xyz2.unsqueeze(0))

        t_lin = torch.linspace(0, 1, 60, dtype=torch.float64, device=device).unsqueeze(1)
        t_eased = ease_in_out(t_lin)
        lab_eased = lab1 + t_eased * (lab2 - lab1)  # (60, 3)

        xyz_eased = space.inverse(lab_eased)
        rgb_eased = _linear_to_srgb((xyz_eased @ msi.T).clamp(0, 1)).clamp(0, 1)
        rgb8 = (rgb_eased * 255).round() / 255.0
        xyz_q = _srgb_to_linear(rgb8) @ ms.T
        cielab = _xyz_to_cielab(xyz_q, d65)

        de = _ciede2000_simplified(cielab[:-1], cielab[1:])
        mask = de > 0.001
        if mask.sum() > 1:
            cv = (de[mask].std() / (de[mask].mean() + 1e-10) * 100).item()
        else:
            cv = 0.0
        cvs[name] = cv

    mean_cv = sum(cvs.values()) / len(cvs) if cvs else 0
    cvs["mean_cv"] = mean_cv
    return cvs


# ═══════════════════════════════════════════════════════════════
#  32. Gradient Chroma Preservation (Muddy Midpoint Detection)
# ═══════════════════════════════════════════════════════════════

def measure_chroma_preservation(space, device):
    """Detect muddy/gray midpoints in gradients between vivid colors.

    For each pair of saturated colors, interpolate in the space and measure
    the minimum chroma (in CIE Lab C*) along the path. A "muddy" gradient
    has a chroma dip in the middle — the colors desaturate through gray/brown
    before reaching the other end.

    Score: mean of (min_chroma / endpoint_chroma) across pairs.
    Higher is better (1.0 = no chroma loss, 0.0 = goes through gray).

    This is the metric OKLab was famous for fixing vs CIE Lab.
    """
    ms = _to(_M_SRGB, device)
    msi = _to(_M_SRGB_INV, device)
    d65 = _to(_D65, device)

    # Pairs of vivid colors where muddy midpoints are most visible
    vivid_pairs = [
        ("Red-Green", [1, 0, 0], [0, 1, 0]),
        ("Red-Blue", [1, 0, 0], [0, 0, 1]),
        ("Green-Blue", [0, 1, 0], [0, 0, 1]),
        ("Red-Cyan", [1, 0, 0], [0, 1, 1]),
        ("Green-Magenta", [0, 1, 0], [1, 0, 1]),
        ("Blue-Yellow", [0, 0, 1], [1, 1, 0]),
        ("Orange-Teal", [1, 0.5, 0], [0, 0.7, 0.7]),
        ("Pink-Mint", [1, 0.4, 0.6], [0.4, 1, 0.6]),
        ("Red-Yellow", [1, 0, 0], [1, 1, 0]),
        ("Yellow-Green", [1, 1, 0], [0, 1, 0]),
        ("Green-Cyan", [0, 1, 0], [0, 1, 1]),
        ("Cyan-Blue", [0, 1, 1], [0, 0, 1]),
        ("Blue-Magenta", [0, 0, 1], [1, 0, 1]),
        ("Magenta-Red", [1, 0, 1], [1, 0, 0]),
        ("Warm-Cool", [1, 0.6, 0.2], [0.2, 0.4, 1]),
        ("Coral-Teal", [1, 0.5, 0.5], [0.5, 1, 1]),
        ("Purple-Gold", [0.6, 0.2, 0.8], [0.9, 0.8, 0.2]),
        ("DarkRed-DarkBlue", [0.6, 0, 0], [0, 0, 0.6]),
    ]

    n_steps = 32
    results = {}
    ratios = []

    for name, rgb1, rgb2 in vivid_pairs:
        xyz1 = ms @ _srgb_to_linear(torch.tensor(rgb1, dtype=torch.float64, device=device))
        xyz2 = ms @ _srgb_to_linear(torch.tensor(rgb2, dtype=torch.float64, device=device))

        lab1 = space.forward(xyz1.unsqueeze(0))
        lab2 = space.forward(xyz2.unsqueeze(0))

        t = torch.linspace(0, 1, n_steps, dtype=torch.float64, device=device).unsqueeze(1)
        lab_interp = lab1 + t * (lab2 - lab1)  # (32, 3)

        # Convert to XYZ, clip to sRGB, measure CIE Lab chroma
        xyz_interp = space.inverse(lab_interp)
        rgb_interp = _linear_to_srgb((xyz_interp @ msi.T).clamp(0, 1)).clamp(0, 1)
        rgb8 = (rgb_interp * 255).round() / 255.0
        xyz_q = _srgb_to_linear(rgb8) @ ms.T
        cielab = _xyz_to_cielab(xyz_q, d65)
        C_star = (cielab[:, 1] ** 2 + cielab[:, 2] ** 2).sqrt()

        # Endpoint chroma (average of start and end)
        C_endpoints = 0.5 * (C_star[0] + C_star[-1])

        # Minimum chroma along path (excluding first and last)
        C_mid = C_star[1:-1]
        C_min = C_mid.min().item()

        # Ratio: how much chroma is preserved (1.0 = no loss)
        if C_endpoints > 1.0:
            ratio = C_min / C_endpoints.item()
        else:
            ratio = 1.0  # achromatic pair, skip

        ratios.append(ratio)
        results[name] = {
            "min_chroma": C_min,
            "endpoint_chroma": C_endpoints.item(),
            "preservation_ratio": ratio,
        }

    mean_ratio = sum(ratios) / len(ratios) if ratios else 0
    # Also count "muddy" gradients (ratio < 0.5 = lost >50% chroma)
    n_muddy = sum(1 for r in ratios if r < 0.5)

    results["mean_preservation"] = mean_ratio
    results["n_muddy"] = n_muddy
    results["n_pairs"] = len(vivid_pairs)
    return results


# ═══════════════════════════════════════════════════════════════
#  33. Hue Angle Agreement with CIE Lab
# ═══════════════════════════════════════════════════════════════

def measure_hue_agreement(space, device):
    """Mean absolute hue angle difference from CIE Lab.

    36 test colors at 10-degree hue intervals (HSV h, s=0.8, v=0.8).
    CIE Lab self-referential: will score 0.
    Lower is better.
    """
    ms = _to(_M_SRGB, device)
    d65 = _to(_D65, device)

    xyzs = []
    for h_deg in range(0, 360, 10):
        r, g, b = _hsv_to_rgb(h_deg / 360, 0.8, 0.8)
        rgb = torch.tensor([r, g, b], dtype=torch.float64, device=device)
        xyzs.append(ms @ _srgb_to_linear(rgb))
    xyzs = torch.stack(xyzs)  # (36, 3)

    lab = space.forward(xyzs)
    h_space = torch.atan2(lab[:, 2], lab[:, 1])

    cielab = _xyz_to_cielab(xyzs, d65)
    h_ref = torch.atan2(cielab[:, 2], cielab[:, 1])

    dh = torch.atan2(torch.sin(h_space - h_ref), torch.cos(h_space - h_ref))
    mae = (dh.abs() * 180 / PI).mean().item()
    max_diff = (dh.abs() * 180 / PI).max().item()

    return {
        "mae_deg": mae,
        "max_diff_deg": max_diff,
    }
