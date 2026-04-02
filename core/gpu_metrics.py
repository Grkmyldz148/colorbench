"""GPU-batched metrics for color space evaluation.

Comprehensive: tests every sRGB color, 100K+ gradient pairs,
3 gamuts (sRGB, P3, Rec.2020), gamut mapping, banding.

Zero per-sample Python loops. All tensor operations.
"""

import math
import torch

PI = math.pi

# ── Color constants (moved to device lazily) ──
_D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)

_M_SRGB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

_M_P3 = torch.tensor([
    [0.4865709486482162, 0.26566769316909306, 0.1982172852343625],
    [0.2289745640697488, 0.6917385218365064, 0.079286914093745],
    [0.0, 0.04511338185890264, 1.0439443689009757],
])

_M_REC2020 = torch.tensor([
    [0.6369580483012914, 0.14461690358620832, 0.1688809751641721],
    [0.2627002120112671, 0.6779980715188708, 0.05930171646986196],
    [0.0, 0.028072693049087428, 1.0609850577107909],
])


def _to(t, device):
    return t.to(device=device, dtype=torch.float64)


def _srgb_to_linear(c):
    c = c.to(torch.float64)
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))


def _linear_to_srgb(c):
    return torch.where(c <= 0.0031308, c * 12.92,
                       1.055 * c.clamp(min=1e-12).pow(1.0 / 2.4) - 0.055)


def _xyz_to_cielab(xyz, d65):
    """(N,3) XYZ → (N,3) CIE L*a*b*."""
    r = xyz / d65
    delta3 = (6.0 / 29.0) ** 3
    f = torch.where(r > delta3, r.pow(1.0 / 3.0),
                    r / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], dim=-1)


def _ciede2000_simplified(cl1, cl2):
    """Simplified CIEDE2000. cl1, cl2: (..., 3). Returns (...)."""
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


# ═══════════════════════════════════════════════════════════════
#  1. ROUND-TRIP — every sRGB color + wide gamut
# ═══════════════════════════════════════════════════════════════

def measure_roundtrip(space, device):
    """Round-trip for ALL 17M sRGB colors + P3/Rec.2020 primaries."""
    d65 = _to(_D65, device)
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)
    mp3 = _to(_M_P3, device)
    mr2020 = _to(_M_REC2020, device)

    results = {}

    # ── All 16.7M sRGB 8-bit colors in chunks ──
    max_err = 0.0
    total_nan = 0
    total_inf = 0
    chunk = 200000  # ~5MB per chunk
    for start in range(0, 256 ** 3, chunk):
        end = min(start + chunk, 256 ** 3)
        n = end - start
        # Generate sRGB values
        idx = torch.arange(start, end, device=device, dtype=torch.int64)
        r = (idx // 65536).to(torch.float64) / 255.0
        g = ((idx % 65536) // 256).to(torch.float64) / 255.0
        b = (idx % 256).to(torch.float64) / 255.0
        srgb = torch.stack([r, g, b], dim=1)
        xyz = _srgb_to_linear(srgb) @ ms.T

        lab = space.forward(xyz)
        total_nan += lab.isnan().sum().item()
        total_inf += lab.isinf().sum().item()

        xyz_rt = space.inverse(lab)
        total_nan += xyz_rt.isnan().sum().item()
        total_inf += xyz_rt.isinf().sum().item()

        err = (xyz - xyz_rt).abs().max().item()
        max_err = max(max_err, err)

    results["srgb_full_16M"] = {
        "max_error": max_err,
        "nan_count": int(total_nan),
        "inf_count": int(total_inf),
    }

    # ── P3 full 8-bit grid (16.7M, same as sRGB) ──
    max_err_p3 = 0.0
    nan_p3 = 0
    for start in range(0, 256 ** 3, chunk):
        end = min(start + chunk, 256 ** 3)
        idx = torch.arange(start, end, device=device)
        r = (idx // 65536) / 255.0
        g = ((idx % 65536) // 256) / 255.0
        b = (idx % 256) / 255.0
        p3_srgb = torch.stack([r, g, b], dim=1)
        # P3 uses same transfer function as sRGB
        xyz = _srgb_to_linear(p3_srgb) @ mp3.T
        lab = space.forward(xyz)
        nan_p3 += lab.isnan().sum().item() + lab.isinf().sum().item()
        xyz_rt = space.inverse(lab)
        nan_p3 += xyz_rt.isnan().sum().item() + xyz_rt.isinf().sum().item()
        err = (xyz - xyz_rt).abs().max().item()
        max_err_p3 = max(max_err_p3, err)

    results["p3_full_16M"] = {
        "max_error": max_err_p3,
        "nan_inf_count": int(nan_p3),
    }

    # ── Rec.2020 grid (128³ = 2.1M uniform + 50K boundary) ──
    steps = 128
    vals = torch.linspace(0, 1, steps, device=device, dtype=torch.float64)
    rr = vals.view(steps, 1, 1).expand(steps, steps, steps).reshape(-1)
    gg = vals.view(1, steps, 1).expand(steps, steps, steps).reshape(-1)
    bb = vals.view(1, 1, steps).expand(steps, steps, steps).reshape(-1)
    r2020_rgb = torch.stack([rr, gg, bb], dim=1)
    # Rec.2020 transfer: use PQ or gamma 2.4 approximation
    r2020_lin = r2020_rgb.pow(2.4)  # simplified
    xyz_r = r2020_lin @ mr2020.T
    lab_r = space.forward(xyz_r)
    xyz_r_rt = space.inverse(lab_r)
    r2020_uniform_err = (xyz_r - xyz_r_rt).abs().max().item()

    # Boundary-focused: near-primary colors (high one channel, low others)
    gen = torch.Generator(device=device).manual_seed(42)
    boundary = torch.zeros(50000, 3, device=device, dtype=torch.float64)
    for i in range(50000):
        ch = i % 3  # which channel is dominant
        boundary[i, ch] = 0.8 + torch.rand(1, generator=gen, device=device, dtype=torch.float64).item() * 0.2
        for j in range(3):
            if j != ch:
                boundary[i, j] = torch.rand(1, generator=gen, device=device, dtype=torch.float64).item() * 0.3
    xyz_b = boundary.pow(2.4) @ mr2020.T
    lab_b = space.forward(xyz_b)
    xyz_b_rt = space.inverse(lab_b)
    r2020_boundary_err = (xyz_b - xyz_b_rt).abs().max().item()

    results["rec2020_2M_uniform"] = {"max_error": r2020_uniform_err}
    results["rec2020_50K_boundary"] = {"max_error": r2020_boundary_err}

    # ── sRGB gamut boundary stress (colors near sRGB edge) ──
    # High saturation at various lightness levels
    boundary_colors = []
    for h_i in range(72):  # every 5°
        h = h_i / 72
        for v in [0.2, 0.4, 0.6, 0.8, 1.0]:
            i = int(h * 6.0) % 6
            f = h * 6.0 - int(h * 6.0)
            p_ = v * 0.02  # s=0.98
            q_ = v * (1.0 - 0.98 * f)
            t_ = v * (1.0 - 0.98 * (1.0 - f))
            rgb = [(v, t_, p_), (q_, v, p_), (p_, v, t_),
                   (p_, q_, v), (t_, p_, v), (v, p_, q_)][i]
            boundary_colors.append(rgb)
    bnd = torch.tensor(boundary_colors, device=device, dtype=torch.float64)
    bnd_xyz = _srgb_to_linear(bnd) @ ms.T
    bnd_lab = space.forward(bnd_xyz)
    bnd_rt = space.inverse(bnd_lab)
    results["srgb_boundary_360"] = {
        "max_error": (bnd_xyz - bnd_rt).abs().max().item(),
        "mean_error": (bnd_xyz - bnd_rt).abs().mean().item(),
        "n_colors": len(boundary_colors),
    }

    return results


# ═══════════════════════════════════════════════════════════════
#  2. ACHROMATIC — full 8-bit gray ramp
# ═══════════════════════════════════════════════════════════════

def measure_achromatic(space, device):
    """Full 257-step sRGB gray ramp + white/black mapping."""
    d65 = _to(_D65, device)
    ms = _to(_M_SRGB, device)

    # 257 sRGB gray levels via sRGB pipeline (includes matrix rounding effects)
    g = torch.linspace(0, 1, 257, device=device, dtype=torch.float64)
    gray_srgb = g.unsqueeze(1).expand(257, 3)
    gray_xyz_srgb = _srgb_to_linear(gray_srgb) @ ms.T
    lab_srgb = space.forward(gray_xyz_srgb)
    chroma_srgb = (lab_srgb[:, 1] ** 2 + lab_srgb[:, 2] ** 2).sqrt()

    # 500 D65-proportional grays (pure achromatic, no matrix rounding)
    Y_pure = torch.cat([
        torch.linspace(0.0001, 0.01, 50, device=device),
        torch.linspace(0.01, 0.1, 50, device=device),
        torch.linspace(0.1, 1.0, 200, device=device),
        torch.linspace(1.0, 2.0, 50, device=device),
    ], dim=0)
    pure_xyz = Y_pure.unsqueeze(1) * d65.unsqueeze(0)
    lab_pure = space.forward(pure_xyz)
    chroma_pure = (lab_pure[:, 1] ** 2 + lab_pure[:, 2] ** 2).sqrt()

    # White + Black
    white_lab = space.forward(d65.unsqueeze(0))[0]
    black_lab = space.forward(torch.zeros(1, 3, device=device, dtype=torch.float64))[0]

    return {
        "gray_ramp_srgb": {
            "max_chroma": chroma_srgb.max().item(),
            "mean_chroma": chroma_srgb.mean().item(),
            "max_a": lab_srgb[:, 1].abs().max().item(),
            "max_b": lab_srgb[:, 2].abs().max().item(),
            "note": "includes sRGB matrix rounding (~1e-7 XYZ offset)",
        },
        "gray_ramp_pure": {
            "max_chroma": chroma_pure.max().item(),
            "mean_chroma": chroma_pure.mean().item(),
            "max_a": lab_pure[:, 1].abs().max().item(),
            "max_b": lab_pure[:, 2].abs().max().item(),
            "note": "D65-proportional grays, no matrix rounding",
            "n_samples": int(Y_pure.shape[0]),
        },
        "white": {
            "L": white_lab[0].item(),
            "a": white_lab[1].item(),
            "b": white_lab[2].item(),
            "L_error": abs(white_lab[0].item() - 1.0),
        },
        "black": {
            "L": black_lab[0].item(),
            "a": black_lab[1].item(),
            "b": black_lab[2].item(),
        },
    }


# ═══════════════════════════════════════════════════════════════
#  3. GRADIENTS — 100K pairs, stratified
# ═══════════════════════════════════════════════════════════════

def measure_gradients(space, pairs_xyz, pair_labels, device, n_steps=26):
    """CV, hue drift, banding for ALL gradient pairs. Fully batched.

    pairs_xyz: (N, 2, 3) XYZ tensor.  pair_labels: list[(cat, desc)].
    """
    d65 = _to(_D65, device)
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)
    N = pairs_xyz.shape[0]
    S = n_steps

    # Process in chunks to manage VRAM (each pair → S inverse calls)
    CHUNK = 5000
    all_cvs = []
    all_drift_max = []
    all_is_crossing = []
    all_banding = []

    for cs in range(0, N, CHUNK):
        ce = min(cs + CHUNK, N)
        nc = ce - cs
        p = pairs_xyz[cs:ce]

        # Forward endpoints
        lab1 = space.forward(p[:, 0])
        lab2 = space.forward(p[:, 1])

        # Interpolate: (nc, S, 3)
        t = torch.linspace(0, 1, S, device=device, dtype=torch.float64).view(1, -1, 1)
        interp = lab1.unsqueeze(1) + t * (lab2 - lab1).unsqueeze(1)

        # Inverse → 8-bit quantize → CIE Lab
        flat = interp.reshape(nc * S, 3)
        xyz_flat = space.inverse(flat)
        lin = (xyz_flat @ msi.T).clamp(0, 1)
        s8 = (_linear_to_srgb(lin) * 255).round() / 255.0
        xyz_q = _srgb_to_linear(s8) @ ms.T
        cielab = _xyz_to_cielab(xyz_q.clamp(min=1e-10), d65).reshape(nc, S, 3)

        # ── CV ──
        cl1, cl2 = cielab[:, :-1], cielab[:, 1:]
        de = _ciede2000_simplified(cl1, cl2)
        md = de.mean(dim=1)
        sd = de.std(dim=1)
        ok = md > 0.001
        cvs = torch.where(ok, sd / md, torch.zeros_like(md))
        all_cvs.append(cvs)

        # ── Hue drift ──
        C_star = (cielab[..., 1] ** 2 + cielab[..., 2] ** 2).sqrt()
        h = torch.atan2(cielab[..., 2], cielab[..., 1])
        dh = h[:, 1:] - h[:, :-1]
        dh = torch.atan2(torch.sin(dh), torch.cos(dh)).abs() * (180.0 / PI)
        both_chrom = (C_star[:, :-1] > 5) & (C_star[:, 1:] > 5)
        dh_valid = torch.where(both_chrom, dh, torch.zeros_like(dh))
        drift_max = dh_valid.max(dim=1).values
        is_crossing = (C_star < 3).any(dim=1)
        all_drift_max.append(drift_max)
        all_is_crossing.append(is_crossing)

        # ── Banding: duplicate consecutive 8-bit colors ──
        s8_reshaped = s8.reshape(nc, S, 3)
        s8_int = (s8_reshaped * 255).to(torch.int32)
        dupes = (s8_int[:, 1:] == s8_int[:, :-1]).all(dim=2)  # (nc, S-1)
        banding_count = dupes.sum(dim=1).float()  # per pair
        all_banding.append(banding_count)

    cvs = torch.cat(all_cvs)
    drift_max = torch.cat(all_drift_max)
    is_crossing = torch.cat(all_is_crossing)
    banding = torch.cat(all_banding)

    # Per-pair results (for JSON)
    pair_results = []
    for i in range(N):
        cat, desc = pair_labels[i]
        pair_results.append({
            "category": cat,
            "description": desc,
            "cv": cvs[i].item(),
            "drift_max": drift_max[i].item(),
            "is_crossing": is_crossing[i].item(),
            "duplicate_steps": int(banding[i].item()),
        })

    # Category stats
    categories = sorted(set(c for c, _ in pair_labels))
    cat_stats = {}
    for cat in categories:
        mask = torch.tensor([l[0] == cat for l in pair_labels], device=device)
        c_cvs = cvs[mask]
        c_drift = drift_max[mask]
        c_band = banding[mask]
        if c_cvs.numel() == 0:
            continue
        cat_stats[cat] = {
            "count": int(mask.sum().item()),
            "cv_mean": c_cvs.mean().item(),
            "cv_p95": c_cvs.quantile(0.95).item() if c_cvs.numel() >= 2 else c_cvs.max().item(),
            "cv_max": c_cvs.max().item(),
            "drift_mean": c_drift.mean().item(),
            "drift_max": c_drift.max().item(),
            "banding_mean": c_band.mean().item(),
        }

    # Overall
    non_crossing = ~is_crossing
    valid_cv = cvs[cvs > 0]
    overall = {
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

    # Subset gradient CVs — measure uniformity in specific pair categories
    # Uses Lab coordinates from the space being tested for filtering
    lab1_all = space.forward(pairs_xyz[:, 0])
    lab2_all = space.forward(pairs_xyz[:, 1])
    L1_all = lab1_all[:, 0]
    L2_all = lab2_all[:, 0]
    C1_all = (lab1_all[:, 1]**2 + lab1_all[:, 2]**2).sqrt()
    C2_all = (lab2_all[:, 1]**2 + lab2_all[:, 2]**2).sqrt()

    subsets = {
        "bright": (L1_all > 0.6) & (L2_all > 0.6),
        "dark": (L1_all < 0.4) & (L2_all < 0.4),
        "high_chroma": (C1_all > 0.15) & (C2_all > 0.15),
        "cross_lightness": (L1_all - L2_all).abs() > 0.5,
        "near_achromatic": (C1_all < 0.05) | (C2_all < 0.05),
    }
    for sname, smask in subsets.items():
        sub_cv = cvs[smask]
        sub_valid = sub_cv[sub_cv > 0]
        overall[f"cv_{sname}"] = sub_valid.mean().item() if sub_valid.numel() > 0 else 0

    return {"overall": overall, "by_category": cat_stats, "pairs": pair_results}


# ═══════════════════════════════════════════════════════════════
#  4. GAMUT GEOMETRY — 360° × 3 gamuts
# ═══════════════════════════════════════════════════════════════

def _scan_gamut(space, device, gamut_matrix, n_hues=360, n_L=150, n_C=120):
    """Cusp scan for one gamut. Returns cusp_L, cusp_C, max_c_per_L arrays."""
    msi = torch.linalg.inv(gamut_matrix)
    Ls = torch.linspace(0.02, 0.998, n_L, device=device, dtype=torch.float64)
    Cs = torch.linspace(0.001, 0.5, n_C, device=device, dtype=torch.float64)

    cusp_L = torch.zeros(n_hues, device=device)
    cusp_C = torch.zeros(n_hues, device=device)
    mc_all = torch.zeros(n_hues, n_L, device=device)

    HB = 6
    for hs in range(0, n_hues, HB):
        he = min(hs + HB, n_hues)
        nh = he - hs
        Le = Ls.view(1, n_L, 1).expand(nh, n_L, n_C)
        Ce = Cs.view(1, 1, n_C).expand(nh, n_L, n_C)
        angles = torch.arange(hs, he, device=device, dtype=torch.float64) * (2 * PI / n_hues)
        ch = torch.cos(angles).view(nh, 1, 1)
        sh = torch.sin(angles).view(nh, 1, 1)
        lab = torch.stack([Le, Ce * ch, Ce * sh], dim=-1).reshape(-1, 3)

        xyz = space.inverse(lab)
        lin = xyz @ msi.T
        ok = ((lin >= -0.002) & (lin <= 1.002)).all(dim=1).reshape(nh, n_L, n_C)
        cv = Cs.view(1, 1, n_C).expand(nh, n_L, n_C)
        mc, _ = torch.where(ok, cv, torch.zeros_like(cv)).max(dim=2)
        mc_all[hs:he] = mc

        ci = mc.argmax(dim=1)
        for i in range(nh):
            cusp_L[hs + i] = Ls[ci[i]]
            cusp_C[hs + i] = mc[i, ci[i]]

    return cusp_L, cusp_C, mc_all, Ls


def measure_gamut(space, device, n_hues=360, n_L=300, n_C=200):
    """Full gamut scan for sRGB, P3, Rec.2020."""
    ms = _to(_M_SRGB, device)
    mp3 = _to(_M_P3, device)
    mr2020 = _to(_M_REC2020, device)

    results = {}

    for gamut_name, mat in [("sRGB", ms), ("P3", mp3), ("Rec2020", mr2020)]:
        cL, cC, mc_all, Ls = _scan_gamut(space, device, mat, n_hues, n_L, n_C)
        cL_np = cL.cpu()
        cC_np = cC.cpu()

        # Cusp existence
        valid = ((cL > 0.05) & (cL < 0.99)).sum().item()

        # Monotonicity — vectorized
        ci_all = mc_all.argmax(dim=1)
        mc_diff = mc_all[:, 1:] - mc_all[:, :-1]
        idx = torch.arange(n_L - 1, device=device).unsqueeze(0)
        after_cusp = idx >= ci_all.unsqueeze(1)
        violations = ((mc_diff > 0.001) & after_cusp).any(dim=1).sum().item()

        # Smoothness
        jumps = (cL_np[1:] - cL_np[:-1]).abs()
        wrap_jump = abs(cL_np[-1] - cL_np[0])
        all_jumps = torch.cat([jumps, wrap_jump.unsqueeze(0)])

        # Cliff
        max_cliff = 0.0
        for hi in range(0, n_hues, 5):
            ci = ci_all[hi].item()
            cc = mc_all[hi, ci].item()
            if ci < n_L - 2 and cc > 0.01:
                post = mc_all[hi, min(ci + 2, n_L - 1)].item()
                cliff = (cc - post) / cc
                max_cliff = max(max_cliff, cliff)

        # Gamut volume (fraction of L×C grid that's in-gamut)
        total_in_gamut = (mc_all > 0.001).float().mean().item()

        # Boundary continuity — detect non-monotonic drops on the RISING edge
        # (before cusp). A drop on the rising edge means the gamut boundary folds
        # inward, creating spikes/holes visible in L-C gamut slices.
        # mc_all shape: [n_hues, n_L]
        mc_L_diff = mc_all[:, 1:] - mc_all[:, :-1]  # [n_hues, n_L-1] (signed)
        idx_range = torch.arange(n_L - 1, device=device).unsqueeze(0)  # [1, n_L-1]
        before_cusp = idx_range < ci_all.unsqueeze(1)  # [n_hues, n_L-1]
        # On the rising edge, chroma should increase. A NEGATIVE diff is a fold.
        rising_drops = (-mc_L_diff).clamp(min=0) * before_cusp.float()  # only drops before cusp
        # Per-hue max drop (in C units)
        boundary_drops_per_hue = rising_drops.max(dim=1).values  # [n_hues]
        boundary_max_abs_jump = boundary_drops_per_hue.max().item()
        boundary_mean_rel_jump = boundary_drops_per_hue.mean().item()
        # Relative: normalize by cusp chroma of that hue
        cusp_c_safe = cC.clamp(min=0.01)
        boundary_rel_per_hue = boundary_drops_per_hue / cusp_c_safe
        boundary_max_rel_jump = boundary_rel_per_hue.max().item()
        # Count hues with >5% boundary fold relative to cusp chroma
        boundary_bad_hues = (boundary_rel_per_hue > 0.05).sum().item()
        # Worst hue
        worst_hue_idx = boundary_rel_per_hue.argmax().item()
        worst_hue_jump = boundary_rel_per_hue[worst_hue_idx].item()

        # Cusp anomaly detection — find cusp-drop zones
        anomalies = []
        cL_list = [cL_np[i].item() for i in range(n_hues)]
        for i in range(n_hues):
            j = (i + 1) % n_hues
            jump = abs(cL_list[j] - cL_list[i])
            if jump > 0.2:
                anomalies.append({
                    "hue_from": i, "hue_to": j,
                    "L_from": cL_list[i], "L_to": cL_list[j],
                    "jump": jump,
                })

        # Cusp-dead zones — consecutive hues with cusp_L < 0.05
        dead_zones = []
        in_dead = False
        dead_start = 0
        for i in range(n_hues):
            if cL_list[i] < 0.05:
                if not in_dead:
                    dead_start = i
                    in_dead = True
            else:
                if in_dead:
                    dead_zones.append({"start": dead_start, "end": i - 1,
                                       "span": i - dead_start})
                    in_dead = False
        if in_dead:
            dead_zones.append({"start": dead_start, "end": n_hues - 1,
                               "span": n_hues - dead_start})

        results[gamut_name] = {
            "valid_cusps": int(valid),
            "invalid_cusps": n_hues - int(valid),
            "monotonicity_violations": int(violations),
            "smoothness_max_jump": all_jumps.max().item(),
            "smoothness_mean_jump": all_jumps.mean().item(),
            "cliff_max": max_cliff,
            "volume_fraction": total_in_gamut,
            "boundary_max_rel_jump": boundary_max_rel_jump,
            "boundary_mean_rel_jump": boundary_mean_rel_jump,
            "boundary_max_abs_jump": boundary_max_abs_jump,
            "boundary_bad_hues": int(boundary_bad_hues),
            "boundary_worst_hue": int(worst_hue_idx),
            "boundary_worst_jump": worst_hue_jump,
            "anomalies": anomalies,
            "dead_zones": dead_zones,
            "cusps": [{"hue": i, "L": cL_np[i].item(), "C": cC_np[i].item()}
                      for i in range(n_hues)],
        }

    return results


# ═══════════════════════════════════════════════════════════════
#  5. GAMUT MAPPING — chroma reduction smoothness
# ═══════════════════════════════════════════════════════════════

def measure_gamut_mapping(space, device):
    """Chroma reduction smoothness for sRGB, P3→sRGB, Rec.2020→P3 boundaries."""
    ms = _to(_M_SRGB, device)
    mp3 = _to(_M_P3, device)
    mr2020 = _to(_M_REC2020, device)
    d65 = _to(_D65, device)

    results = {}

    # Test 3 gamut boundaries
    gamut_tests = [
        ("sRGB", ms),
        ("P3", mp3),
        ("Rec2020", mr2020),
    ]

    for gamut_name, gamut_mat in gamut_tests:
        gamut_inv = torch.linalg.inv(gamut_mat)
        for L_test in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            non_mono_hues = 0
            max_delta_e_jump = 0.0
            hue_jumps = []

            for h_deg in range(0, 360, 10):
                h_rad = h_deg * PI / 180
                ch, sh = math.cos(h_rad), math.sin(h_rad)

                # Chroma steps from 0.4 down to 0.001
                Cs = torch.linspace(0.4, 0.001, 100, device=device, dtype=torch.float64)
                lab = torch.stack([
                    torch.full_like(Cs, L_test),
                    Cs * ch,
                    Cs * sh,
                ], dim=-1)

                xyz = space.inverse(lab)
                # Check against THIS gamut's boundary
                lin = (xyz @ gamut_inv.T).clamp(0, 1)
                s8 = (_linear_to_srgb(lin) * 255).round() / 255.0
                xyz_q = _srgb_to_linear(s8) @ gamut_mat.T
                cielab = _xyz_to_cielab(xyz_q.clamp(min=1e-10), d65)

                C_star = (cielab[:, 1] ** 2 + cielab[:, 2] ** 2).sqrt()
                diffs = C_star[1:] - C_star[:-1]
                if (diffs > 0.5).any():
                    non_mono_hues += 1

                de = _ciede2000_simplified(cielab[:-1], cielab[1:])
                max_delta_e_jump = max(max_delta_e_jump, de.max().item())

                # Hue jump during chroma reduction
                h_cl = torch.atan2(cielab[:, 2], cielab[:, 1])
                chromatic = C_star > 2
                if chromatic.sum() > 2:
                    h_chrom = h_cl[chromatic]
                    dh = (h_chrom[1:] - h_chrom[:-1])
                    dh = torch.atan2(torch.sin(dh), torch.cos(dh)).abs() * (180 / PI)
                    if dh.numel() > 0:
                        hue_jumps.append(dh.max().item())

            results[f"{gamut_name}_L{L_test}"] = {
                "non_monotonic_hues": non_mono_hues,
                "max_de_jump": max_delta_e_jump,
                "max_hue_jump": max(hue_jumps) if hue_jumps else 0,
            }

    return results


# ═══════════════════════════════════════════════════════════════
#  6. HUE PROPERTIES
# ═══════════════════════════════════════════════════════════════

def measure_hue(space, device):
    """Primary hue ordering, linearity, yellow accuracy."""
    ms = _to(_M_SRGB, device)

    primaries_srgb = torch.tensor([
        [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 1, 1], [0, 0, 1], [1, 0, 1],
    ], dtype=torch.float64, device=device)
    expected_hue = torch.tensor([0, 60, 120, 180, 240, 300],
                                dtype=torch.float64, device=device)
    names = ["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"]

    xyz = _srgb_to_linear(primaries_srgb) @ ms.T
    lab = space.forward(xyz)

    h = torch.atan2(lab[:, 2], lab[:, 1]) * (180.0 / PI) % 360
    dh = h - expected_hue
    dh = torch.where(dh > 180, dh - 360, dh)
    dh = torch.where(dh < -180, dh + 360, dh)
    rms = (dh ** 2).mean().sqrt().item()

    per_primary = {}
    for i, name in enumerate(names):
        per_primary[name] = {
            "hue": h[i].item(),
            "expected": expected_hue[i].item(),
            "error": dh[i].item(),
            "L": lab[i, 0].item(),
            "C": (lab[i, 1] ** 2 + lab[i, 2] ** 2).sqrt().item(),
        }

    L_vals = lab[:, 0]
    # Check hue ordering with proper 360° wrap handling
    # Red may be near 360° (e.g. 357°), which is correct and should wrap to Yellow (e.g. 73°)
    h_list = h.tolist()
    hue_ordered = True
    for i in range(5):
        diff = h_list[i + 1] - h_list[i]
        # Normalize to (-180, 180]
        if diff < -180:
            diff += 360
        if diff > 180:
            diff -= 360
        if diff <= 0:
            hue_ordered = False
            break

    return {
        "hue_rms": rms,
        "hue_ordered": hue_ordered,
        "primary_L_range": (L_vals.max() - L_vals.min()).item(),
        "per_primary": per_primary,
    }


# ═══════════════════════════════════════════════════════════════
#  7. SPECIAL GRADIENTS
# ═══════════════════════════════════════════════════════════════

def measure_special_gradients(space, device):
    """Blue→White, Red→White midpoints + yellow chroma."""
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)

    def srgb_xyz(rgb):
        t = torch.tensor(rgb, device=device, dtype=torch.float64)
        return (_srgb_to_linear(t) @ ms.T).unsqueeze(0)

    bl = space.forward(srgb_xyz([0, 0, 1]))[0]
    wl = space.forward(srgb_xyz([1, 1, 1]))[0]
    rl = space.forward(srgb_xyz([1, 0, 0]))[0]

    mid_bw = space.inverse(((bl + wl) / 2).unsqueeze(0))[0]
    mid_bw_srgb = _linear_to_srgb((mid_bw @ msi.T).clamp(0, 1))

    mid_rw = space.inverse(((rl + wl) / 2).unsqueeze(0))[0]
    mid_rw_srgb = _linear_to_srgb((mid_rw @ msi.T).clamp(0, 1))

    yl = space.forward(srgb_xyz([1, 1, 0]))[0]

    return {
        "blue_white_midpoint": {
            "G_over_R": mid_bw_srgb[1].item() / max(mid_bw_srgb[0].item(), 1e-10),
            "srgb": [mid_bw_srgb[i].item() for i in range(3)],
        },
        "red_white_midpoint": {
            "G_minus_B": mid_rw_srgb[1].item() - mid_rw_srgb[2].item(),
            "srgb": [mid_rw_srgb[i].item() for i in range(3)],
        },
        "yellow_chroma": (yl[1] ** 2 + yl[2] ** 2).sqrt().item(),
    }


# ═══════════════════════════════════════════════════════════════
#  8. NUMERICAL STABILITY
# ═══════════════════════════════════════════════════════════════

def measure_stability(space, device):
    """Perturbation sensitivity + near-boundary behavior."""
    ms = _to(_M_SRGB, device)

    gen = torch.Generator(device=device).manual_seed(99)
    srgb = torch.rand(5000, 3, generator=gen, device=device, dtype=torch.float64)
    xyz = _srgb_to_linear(srgb) @ ms.T
    lab = space.forward(xyz)

    # 1e-8 XYZ perturbation
    perturb = torch.randn_like(xyz) * 1e-8
    lab2 = space.forward(xyz + perturb)
    diff = (lab - lab2).abs()

    # Near-black stability (sRGB < 0.01)
    dark = torch.rand(1000, 3, generator=gen, device=device, dtype=torch.float64) * 0.01
    dark_xyz = _srgb_to_linear(dark) @ ms.T
    dark_lab = space.forward(dark_xyz)
    dark_nan = dark_lab.isnan().sum().item()
    dark_inf = dark_lab.isinf().sum().item()

    # Near-white stability (sRGB > 0.99)
    bright = 0.99 + torch.rand(1000, 3, generator=gen, device=device, dtype=torch.float64) * 0.01
    bright_xyz = _srgb_to_linear(bright) @ ms.T
    bright_lab = space.forward(bright_xyz)
    bright_nan = bright_lab.isnan().sum().item()
    bright_inf = bright_lab.isinf().sum().item()

    return {
        "perturbation_1e8": {
            "max_lab_change": diff.max().item(),
            "mean_lab_change": diff.mean().item(),
        },
        "near_black": {"nan": int(dark_nan), "inf": int(dark_inf)},
        "near_white": {"nan": int(bright_nan), "inf": int(bright_inf)},
    }
