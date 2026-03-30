"""Advanced GPU-batched metrics — CVD, animation, Munsell, extremes, Jacobian, contrast.

These complement the base metrics in gpu_metrics.py.
"""

import math
import torch

PI = math.pi


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
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
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


# ═══════════════════════════════════════════════════════════════
#  9. CVD — Color Vision Deficiency
# ═══════════════════════════════════════════════════════════════

# Brettel 1997 simulation matrices (LMS domain)
# Protan: L-cone missing, Deutan: M-cone missing, Tritan: S-cone missing
_CVD_MATRICES = {
    "protan": torch.tensor([
        [0.0, 1.05118294, -0.05116099],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]),
    "deutan": torch.tensor([
        [1.0, 0.0, 0.0],
        [0.9513092, 0.0, 0.04866992],
        [0.0, 0.0, 1.0],
    ]),
    "tritan": torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-0.86744736, 1.86727089, 0.0],
    ]),
}

# Hunt-Pointer-Estévez XYZ→LMS
_M_HPE = torch.tensor([
    [0.4002, 0.7076, -0.0808],
    [-0.2263, 1.1653, 0.0457],
    [0.0, 0.0, 0.9182],
])


def measure_cvd(space, device):
    """CVD gradient distinguishability — 100+ pairs per CVD type."""
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)
    d65 = _to(_D65, device)
    M_hpe = _to(_M_HPE, device)
    M_hpe_inv = torch.linalg.inv(M_hpe)

    # Build comprehensive pair set: primaries + pastels + darks + random
    pairs = []
    names = []

    primaries = torch.tensor([
        [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 1, 1], [0, 0, 1], [1, 0, 1],
    ], dtype=torch.float64, device=device)
    p_names = ["R", "Y", "G", "C", "B", "M"]

    # All primary combos (15)
    for i in range(6):
        for j in range(i + 1, 6):
            pairs.append((primaries[i], primaries[j]))
            names.append(f"{p_names[i]}-{p_names[j]}")

    # Pastels at various hues (24)
    for h in range(0, 360, 30):
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.3, 0.9)
        r2, g2, b2 = _hsv_to_rgb(((h + 60) % 360) / 360, 0.3, 0.9)
        pairs.append((torch.tensor([r1, g1, b1], device=device, dtype=torch.float64),
                       torch.tensor([r2, g2, b2], device=device, dtype=torch.float64)))
        names.append(f"pastel_h{h}")

    # Dark colors (12)
    for h in range(0, 360, 30):
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.7, 0.25)
        r2, g2, b2 = _hsv_to_rgb(((h + 60) % 360) / 360, 0.7, 0.25)
        pairs.append((torch.tensor([r1, g1, b1], device=device, dtype=torch.float64),
                       torch.tensor([r2, g2, b2], device=device, dtype=torch.float64)))
        names.append(f"dark_h{h}")

    # Random (50)
    gen = torch.Generator(device=device).manual_seed(88)
    for k in range(50):
        r1 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        r2 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        pairs.append((r1, r2))
        names.append(f"rnd{k}")

    results = {}

    for cvd_type, cvd_mat in _CVD_MATRICES.items():
        cvd_m = _to(cvd_mat, device)
        pair_results = []

        for (rgb1, rgb2), name in zip(pairs, names):
            xyz1 = (_srgb_to_linear(rgb1) @ ms.T).unsqueeze(0)
            xyz2 = (_srgb_to_linear(rgb2) @ ms.T).unsqueeze(0)
            lab1 = space.forward(xyz1)[0]
            lab2 = space.forward(xyz2)[0]

            # 16-step gradient
            n_steps = 16
            t = torch.linspace(0, 1, n_steps, device=device, dtype=torch.float64)
            labs = lab1.unsqueeze(0) + t.unsqueeze(1) * (lab2 - lab1).unsqueeze(0)
            xyz_grad = space.inverse(labs)

            # Simulate CVD: XYZ → LMS → CVD → LMS → XYZ → CIE Lab
            lms = xyz_grad @ M_hpe.T
            lms_cvd = lms @ cvd_m.T
            xyz_cvd = lms_cvd @ M_hpe_inv.T
            cielab_cvd = _xyz_to_cielab(xyz_cvd.clamp(min=1e-10), d65)

            # Min ΔE between consecutive steps under CVD
            de = _ciede2000_simplified(cielab_cvd[:-1], cielab_cvd[1:])
            min_de = de.min().item()
            mean_de = de.mean().item()

            pair_results.append({
                "pair": name,
                "min_de": min_de,
                "mean_de": mean_de,
            })

        all_min = min(r["min_de"] for r in pair_results)
        all_mean = sum(r["mean_de"] for r in pair_results) / len(pair_results)
        results[cvd_type] = {
            "pairs": pair_results,
            "worst_min_de": all_min,
            "mean_de": all_mean,
        }

    return results


# ═══════════════════════════════════════════════════════════════
#  10. ANIMATION SMOOTHNESS
# ═══════════════════════════════════════════════════════════════

def measure_animation(space, device):
    """60fps animation smoothness — CV of frame-to-frame ΔE."""
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)
    d65 = _to(_D65, device)

    # 50 transitions: primaries, pastels, darks, neutrals
    transitions = [
        ("R→B", [1, 0, 0], [0, 0, 1]),
        ("Y→C", [1, 1, 0], [0, 1, 1]),
        ("G→M", [0, 1, 0], [1, 0, 1]),
        ("K→W", [0, 0, 0], [1, 1, 1]),
        ("R→W", [1, 0, 0], [1, 1, 1]),
        ("B→W", [0, 0, 1], [1, 1, 1]),
        ("G→W", [0, 1, 0], [1, 1, 1]),
        ("Y→W", [1, 1, 0], [1, 1, 1]),
        ("R→K", [1, 0, 0], [0, 0, 0]),
        ("B→K", [0, 0, 1], [0, 0, 0]),
        ("R→G", [1, 0, 0], [0, 1, 0]),
        ("R→C", [1, 0, 0], [0, 1, 1]),
    ]
    # Add pastel→vivid, dark→dark at various hues
    for h_deg in range(0, 360, 45):
        h = h_deg / 360
        r1, g1, b1 = _hsv_to_rgb(h, 0.2, 0.9)
        r2, g2, b2 = _hsv_to_rgb(h, 1.0, 1.0)
        transitions.append((f"p→v_h{h_deg}", [r1, g1, b1], [r2, g2, b2]))
        r1, g1, b1 = _hsv_to_rgb(h, 0.7, 0.15)
        r2, g2, b2 = _hsv_to_rgb((h + 0.25) % 1.0, 0.7, 0.15)
        transitions.append((f"dk_h{h_deg}", [r1, g1, b1], [r2, g2, b2]))
    # Random transitions
    import random as _rnd
    _rnd.seed(55)
    for k in range(20):
        rgb1 = [_rnd.random(), _rnd.random(), _rnd.random()]
        rgb2 = [_rnd.random(), _rnd.random(), _rnd.random()]
        transitions.append((f"rnd{k}", rgb1, rgb2))

    n_frames = 120  # 2 seconds at 60fps
    results = {}

    for name, rgb1, rgb2 in transitions:
        xyz1 = (_srgb_to_linear(torch.tensor(rgb1, device=device, dtype=torch.float64)) @ ms.T)
        xyz2 = (_srgb_to_linear(torch.tensor(rgb2, device=device, dtype=torch.float64)) @ ms.T)
        lab1 = space.forward(xyz1.unsqueeze(0))[0]
        lab2 = space.forward(xyz2.unsqueeze(0))[0]

        # Linear interpolation at 120 frames
        t = torch.linspace(0, 1, n_frames, device=device, dtype=torch.float64)
        labs = lab1.unsqueeze(0) + t.unsqueeze(1) * (lab2 - lab1).unsqueeze(0)

        # Inverse → quantize → CIE Lab
        xyz_frames = space.inverse(labs)
        lin = (xyz_frames @ msi.T).clamp(0, 1)
        s8 = (_linear_to_srgb(lin) * 255).round() / 255.0
        xyz_q = _srgb_to_linear(s8) @ ms.T
        cielab = _xyz_to_cielab(xyz_q.clamp(min=1e-10), d65)

        # Frame-to-frame ΔE
        de = _ciede2000_simplified(cielab[:-1], cielab[1:])
        cv = (de.std() / de.mean()).item() if de.mean() > 0.001 else 0
        step_ratio = (de.max() / de.min()).item() if de.min() > 0.001 else float("inf")

        results[name] = {
            "cv": cv,
            "step_ratio": min(step_ratio, 999),
            "de_mean": de.mean().item(),
            "de_max": de.max().item(),
            "de_min": de.min().item(),
        }

    return results


# ═══════════════════════════════════════════════════════════════
#  11. DARK/LIGHT EXTREMES
# ═══════════════════════════════════════════════════════════════

def measure_extremes(space, device):
    """Near-black hue stability, near-white chroma collapse, L ordering."""
    ms = _to(_M_SRGB, device)
    d65 = _to(_D65, device)

    results = {}

    # Near-black hue stability: 6 hues at very low L
    hues_srgb = torch.tensor([
        [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 1, 1], [0, 0, 1], [1, 0, 1],
    ], dtype=torch.float64, device=device)
    h_names = ["R", "Y", "G", "C", "B", "M"]

    dark_hue_var = []
    for i, name in enumerate(h_names):
        # Scale sRGB to very dark: 0.005 to 0.05
        scales = torch.linspace(0.005, 0.05, 20, device=device, dtype=torch.float64)
        dark_srgb = scales.unsqueeze(1) * hues_srgb[i].unsqueeze(0)
        dark_xyz = _srgb_to_linear(dark_srgb) @ ms.T
        lab = space.forward(dark_xyz)
        h = torch.atan2(lab[:, 2], lab[:, 1]) * (180 / PI)
        # Circular variance
        h_rad = lab[:, 2].atan2(lab[:, 1])
        mean_sin = h_rad.sin().mean()
        mean_cos = h_rad.cos().mean()
        R_len = (mean_sin ** 2 + mean_cos ** 2).sqrt()
        circ_var = (1 - R_len).item()
        dark_hue_var.append({"primary": name, "circular_variance": circ_var})

    results["near_black_hue_stability"] = dark_hue_var
    results["near_black_max_variance"] = max(d["circular_variance"] for d in dark_hue_var)

    # Near-white chroma collapse: should L still increase smoothly?
    bright_g = torch.linspace(0.95, 1.0, 20, device=device, dtype=torch.float64)
    bright_srgb = bright_g.unsqueeze(1).expand(20, 3)
    bright_xyz = _srgb_to_linear(bright_srgb) @ ms.T
    bright_lab = space.forward(bright_xyz)
    L_diffs = bright_lab[1:, 0] - bright_lab[:-1, 0]
    L_reversals = (L_diffs < -1e-10).sum().item()
    results["near_white_L_reversals"] = int(L_reversals)

    # Overall L ordering: 256 sRGB grays should have strictly increasing L
    g256 = torch.linspace(0, 1, 256, device=device, dtype=torch.float64)
    g256_srgb = g256.unsqueeze(1).expand(256, 3)
    g256_xyz = _srgb_to_linear(g256_srgb) @ ms.T
    g256_lab = space.forward(g256_xyz)
    L_diffs_full = g256_lab[1:, 0] - g256_lab[:-1, 0]
    results["full_L_reversals"] = int((L_diffs_full < -1e-10).sum().item())
    results["L_range"] = [g256_lab[0, 0].item(), g256_lab[-1, 0].item()]

    return results


# ═══════════════════════════════════════════════════════════════
#  12. JACOBIAN CONDITION — where is the space ill-conditioned?
# ═══════════════════════════════════════════════════════════════

def measure_jacobian(space, device):
    """Numerical Jacobian condition number across the gamut."""
    ms = _to(_M_SRGB, device)
    eps = 1e-7

    mp3 = _to(_M_P3, device)
    mr2020 = _to(torch.tensor([
        [0.6369580483012914, 0.14461690358620832, 0.1688809751641721],
        [0.2627002120112671, 0.6779980715188708, 0.05930171646986196],
        [0.0, 0.028072693049087428, 1.0609850577107909],
    ]), device)

    # Sample 5000 colors across sRGB + P3 + Rec.2020
    gen = torch.Generator(device=device).manual_seed(77)
    srgb_c = torch.rand(2000, 3, generator=gen, device=device, dtype=torch.float64)
    p3_c = torch.rand(1500, 3, generator=gen, device=device, dtype=torch.float64)
    r2020_c = torch.rand(1500, 3, generator=gen, device=device, dtype=torch.float64)
    xyz = torch.cat([
        _srgb_to_linear(srgb_c) @ ms.T,
        _srgb_to_linear(p3_c) @ mp3.T,
        _srgb_to_linear(r2020_c) @ mr2020.T,
    ], dim=0)

    conditions = []
    for k in range(xyz.shape[0]):
        x0 = xyz[k]
        lab0 = space.forward(x0.unsqueeze(0))[0]
        # Numerical Jacobian: 3x3
        J = torch.zeros(3, 3, device=device, dtype=torch.float64)
        for j in range(3):
            dx = torch.zeros(3, device=device, dtype=torch.float64)
            dx[j] = eps
            lab_plus = space.forward((x0 + dx).unsqueeze(0))[0]
            J[:, j] = (lab_plus - lab0) / eps
        cond = torch.linalg.cond(J).item()
        conditions.append(cond)

    conditions = torch.tensor(conditions, device=device, dtype=torch.float64)
    # Also measure by lightness region
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


# ═══════════════════════════════════════════════════════════════
#  13. CONTRAST RATIO (WCAG)
# ═══════════════════════════════════════════════════════════════

def measure_contrast(space, device):
    """WCAG contrast ratio preservation test.

    Generate L-based contrast pairs in the test space,
    check if the actual luminance contrast matches.
    """
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)
    d65 = _to(_D65, device)

    # 36 hues × 10 L-level pairs = 360 contrast measurements
    results_per_hue = []
    L_pairs = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.9), (0.2, 0.6),
               (0.2, 0.8), (0.3, 0.7), (0.3, 0.9), (0.4, 0.8),
               (0.5, 0.9), (0.6, 0.95)]
    for h_deg in range(0, 360, 10):
        for L_lo, L_hi in L_pairs:
            h_rad = h_deg * PI / 180
            ch, sh = math.cos(h_rad), math.sin(h_rad)
            lab_dark = torch.tensor([[L_lo, 0.15 * ch, 0.15 * sh]],
                                    device=device, dtype=torch.float64)
            lab_light = torch.tensor([[L_hi, 0.15 * ch, 0.15 * sh]],
                                     device=device, dtype=torch.float64)

            xyz_dark = space.inverse(lab_dark)[0]
            xyz_light = space.inverse(lab_light)[0]
            Y_dark = max(xyz_dark[1].item(), 0.0001)
            Y_light = max(xyz_light[1].item(), 0.0001)
            L1 = max(Y_dark, Y_light) + 0.05
            L2 = min(Y_dark, Y_light) + 0.05
            cr = L1 / L2
            results_per_hue.append({
                "hue": h_deg, "L_lo": L_lo, "L_hi": L_hi,
                "contrast_ratio": cr,
            })

    crs = [r["contrast_ratio"] for r in results_per_hue]
    return {
        "per_hue": results_per_hue,
        "cr_mean": sum(crs) / len(crs),
        "cr_min": min(crs),
        "cr_max": max(crs),
        "cr_cv": (torch.tensor(crs).std() / torch.tensor(crs).mean()).item(),
    }


# ═══════════════════════════════════════════════════════════════
#  14. HUE LEAF CONSTANCY
# ═══════════════════════════════════════════════════════════════

def measure_hue_leaf(space, device):
    """For constant hue in the test space, measure CIE Lab hue deviation."""
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)
    d65 = _to(_D65, device)

    results = {}

    for h_deg in range(0, 360, 10):
        h_rad = h_deg * PI / 180
        ch, sh = math.cos(h_rad), math.sin(h_rad)

        # Denser grid: 40 L × 30 C = 1200 points per hue
        Ls = torch.linspace(0.05, 0.95, 40, device=device, dtype=torch.float64)
        Cs = torch.linspace(0.02, 0.4, 30, device=device, dtype=torch.float64)
        Le = Ls.view(-1, 1).expand(40, 30).reshape(-1)
        Ce = Cs.view(1, -1).expand(40, 30).reshape(-1)

        lab = torch.stack([Le, Ce * ch, Ce * sh], dim=-1)
        xyz = space.inverse(lab)

        # Only keep in-gamut points
        lin = xyz @ msi.T
        in_gamut = ((lin >= -0.01) & (lin <= 1.01)).all(dim=1)
        xyz_valid = xyz[in_gamut]

        if xyz_valid.shape[0] < 5:
            continue

        # CIE Lab hue of these points
        cielab = _xyz_to_cielab(xyz_valid.clamp(min=1e-10), d65)
        C_star = (cielab[:, 1] ** 2 + cielab[:, 2] ** 2).sqrt()
        chromatic = C_star > 2
        if chromatic.sum() < 3:
            continue

        h_cielab = torch.atan2(cielab[chromatic, 2], cielab[chromatic, 1]) * (180 / PI)
        # Circular mean and spread
        h_rad_cl = cielab[chromatic, 2].atan2(cielab[chromatic, 1])
        mean_h = torch.atan2(h_rad_cl.sin().mean(), h_rad_cl.cos().mean()) * (180 / PI)
        # Max deviation from mean
        dh = h_cielab - mean_h.item()
        dh = torch.where(dh > 180, dh - 360, dh)
        dh = torch.where(dh < -180, dh + 360, dh)

        results[str(h_deg)] = {
            "mean_cielab_hue": mean_h.item() % 360,
            "max_deviation": dh.abs().max().item(),
            "std_deviation": dh.std().item(),
            "n_points": int(chromatic.sum().item()),
        }

    # Overall stats
    if results:
        all_max = max(v["max_deviation"] for v in results.values())
        all_std = sum(v["std_deviation"] for v in results.values()) / len(results)
    else:
        all_max = 0
        all_std = 0

    return {
        "per_hue": results,
        "max_deviation": all_max,
        "mean_std": all_std,
    }


# ═══════════════════════════════════════════════════════════════
#  15. 3-COLOR GRADIENTS
# ═══════════════════════════════════════════════════════════════

def measure_3color_gradients(space, device):
    """R→G→B, C→M→Y, etc. — multi-stop gradient quality."""
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)
    d65 = _to(_D65, device)

    paths = [
        ("R→G→B", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ("C→M→Y", [[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        ("R→W→B", [[1, 0, 0], [1, 1, 1], [0, 0, 1]]),
        ("K→R→W", [[0, 0, 0], [1, 0, 0], [1, 1, 1]]),
        ("B→G→Y", [[0, 0, 1], [0, 1, 0], [1, 1, 0]]),
        ("K→G→W", [[0, 0, 0], [0, 1, 0], [1, 1, 1]]),
        ("K→B→W", [[0, 0, 0], [0, 0, 1], [1, 1, 1]]),
        ("R→Y→G", [[1, 0, 0], [1, 1, 0], [0, 1, 0]]),
        ("G→C→B", [[0, 1, 0], [0, 1, 1], [0, 0, 1]]),
        ("B→M→R", [[0, 0, 1], [1, 0, 1], [1, 0, 0]]),
        # 4-color rainbow
        ("R→Y→G→B", [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]]),
        ("R→G→B→W", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]),
        # Pastel paths
        ("pR→pG→pB", [[1, 0.7, 0.7], [0.7, 1, 0.7], [0.7, 0.7, 1]]),
        # Dark paths
        ("dR→dG→dB", [[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]]),
        # Monochrome
        ("K→g50→W", [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]]),
    ]

    results = {}
    for name, stops in paths:
        # Forward all stops
        stop_xyz = []
        for rgb in stops:
            t = torch.tensor(rgb, device=device, dtype=torch.float64)
            stop_xyz.append(_srgb_to_linear(t) @ ms.T)
        stop_labs = [space.forward(x.unsqueeze(0))[0] for x in stop_xyz]

        # 2-segment gradient: 13 steps each
        all_labs = []
        for seg in range(len(stops) - 1):
            t = torch.linspace(0, 1, 13, device=device, dtype=torch.float64)
            seg_labs = stop_labs[seg].unsqueeze(0) + t.unsqueeze(1) * (
                stop_labs[seg + 1] - stop_labs[seg]).unsqueeze(0)
            all_labs.append(seg_labs if seg == 0 else seg_labs[1:])  # avoid duplicate midpoint

        labs = torch.cat(all_labs, dim=0)  # (25, 3)
        xyz_path = space.inverse(labs)
        lin = (xyz_path @ msi.T).clamp(0, 1)
        s8 = (_linear_to_srgb(lin) * 255).round() / 255.0
        xyz_q = _srgb_to_linear(s8) @ ms.T
        cielab = _xyz_to_cielab(xyz_q.clamp(min=1e-10), d65)

        de = _ciede2000_simplified(cielab[:-1], cielab[1:])
        md = de.mean()
        cv = (de.std() / md).item() if md > 0.001 else 0

        results[name] = {
            "cv": cv,
            "de_mean": md.item(),
            "de_max": de.max().item(),
            "n_steps": labs.shape[0],
        }

    return results


# ═══════════════════════════════════════════════════════════════
#  16. PERCEPTUAL BANDING (dE < 1.0 invisible steps)
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  17. DOUBLE ROUND-TRIP (error accumulation)
# ═══════════════════════════════════════════════════════════════

def measure_double_roundtrip(space, device):
    """XYZ → Lab → XYZ repeated N times. Does error accumulate?"""
    ms = _to(_M_SRGB, device)

    gen = torch.Generator(device=device).manual_seed(33)
    srgb = torch.rand(10000, 3, generator=gen, device=device, dtype=torch.float64)
    xyz_orig = _srgb_to_linear(srgb) @ ms.T

    results = {}
    xyz = xyz_orig.clone()
    for n_trips in [1, 10, 100, 1000]:
        # Do trips from current state to target
        while True:
            # Count how many we've done
            current = int(results.get(f"trips_{n_trips}", {}).get("_done", 0))
            remaining = n_trips - current
            if remaining <= 0:
                break
            batch = min(remaining, 100)
            for _ in range(batch):
                lab = space.forward(xyz)
                xyz = space.inverse(lab)
            if n_trips not in [k for k in results if k.startswith("trips_")]:
                pass
            break

        # Just do it cleanly
        xyz_test = xyz_orig.clone()
        for _ in range(n_trips):
            xyz_test = space.inverse(space.forward(xyz_test))
        err = (xyz_orig - xyz_test).abs()
        results[f"trips_{n_trips}"] = {
            "max_error": err.max().item(),
            "mean_error": err.mean().item(),
        }

    return results


# ═══════════════════════════════════════════════════════════════
#  18. sRGB↔P3 CONSISTENCY
# ═══════════════════════════════════════════════════════════════

def measure_cross_gamut_consistency(space, device):
    """Colors in sRGB ⊂ P3: same XYZ should give same Lab regardless of path."""
    ms = _to(_M_SRGB, device)
    mp3 = _to(_M_P3, device)
    mp3i = torch.linalg.inv(mp3)

    # 1000 random sRGB colors
    gen = torch.Generator(device=device).manual_seed(66)
    srgb = torch.rand(1000, 3, generator=gen, device=device, dtype=torch.float64)
    xyz = _srgb_to_linear(srgb) @ ms.T

    # Path 1: XYZ → Lab directly
    lab1 = space.forward(xyz)

    # Path 2: XYZ → P3 linear → P3 sRGB → P3 linear → XYZ → Lab
    p3_lin = xyz @ mp3i.T
    # These should be in P3 gamut since sRGB ⊂ P3
    p3_srgb = _linear_to_srgb(p3_lin.clamp(0, 1))
    p3_lin_back = _srgb_to_linear(p3_srgb)
    xyz_via_p3 = p3_lin_back @ mp3.T
    lab2 = space.forward(xyz_via_p3)

    # The difference is from P3 quantization, not from the space itself
    # But we measure if the space amplifies or dampens the quantization error
    xyz_diff = (xyz - xyz_via_p3).abs()
    lab_diff = (lab1 - lab2).abs()

    # Amplification ratio: lab_diff / xyz_diff
    # High ratio means the space amplifies quantization noise
    xyz_norm = xyz_diff.norm(dim=1).clamp(min=1e-15)
    lab_norm = lab_diff.norm(dim=1).clamp(min=1e-15)
    ratio = lab_norm / xyz_norm

    return {
        "max_lab_diff": lab_diff.max().item(),
        "mean_lab_diff": lab_diff.mean().item(),
        "amplification_mean": ratio.mean().item(),
        "amplification_max": ratio.max().item(),
    }


# ═══════════════════════════════════════════════════════════════
#  19. 8-BIT QUANTIZATION SYMMETRY
# ═══════════════════════════════════════════════════════════════

def measure_quantization_symmetry(space, device):
    """sRGB 8-bit → Lab → sRGB 8-bit. How many colors survive exactly?"""
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)

    # All 256 grays
    g = torch.arange(256, device=device, dtype=torch.float64) / 255.0
    gray_srgb = g.unsqueeze(1).expand(256, 3)
    gray_xyz = _srgb_to_linear(gray_srgb) @ ms.T
    gray_lab = space.forward(gray_xyz)
    gray_xyz_rt = space.inverse(gray_lab)
    gray_srgb_rt = _linear_to_srgb((gray_xyz_rt @ msi.T).clamp(0, 1))
    gray_8bit_rt = (gray_srgb_rt * 255).round()
    gray_8bit_orig = (gray_srgb * 255).round()
    gray_exact = (gray_8bit_rt == gray_8bit_orig).all(dim=1).sum().item()

    # 216 web-safe colors
    vals_ws = torch.tensor([0, 51, 102, 153, 204, 255],
                           device=device, dtype=torch.float64) / 255.0
    ws = torch.stack(torch.meshgrid(vals_ws, vals_ws, vals_ws, indexing='ij'),
                     dim=-1).reshape(-1, 3)
    ws_xyz = _srgb_to_linear(ws) @ ms.T
    ws_lab = space.forward(ws_xyz)
    ws_xyz_rt = space.inverse(ws_lab)
    ws_srgb_rt = _linear_to_srgb((ws_xyz_rt @ msi.T).clamp(0, 1))
    ws_8bit_rt = (ws_srgb_rt * 255).round()
    ws_8bit_orig = (ws * 255).round()
    ws_exact = (ws_8bit_rt == ws_8bit_orig).all(dim=1).sum().item()

    # Random 10K — max channel error
    gen = torch.Generator(device=device).manual_seed(55)
    rnd_8bit = torch.randint(0, 256, (10000, 3), generator=gen, device=device).to(torch.float64)
    rnd_srgb = rnd_8bit / 255.0
    rnd_xyz = _srgb_to_linear(rnd_srgb) @ ms.T
    rnd_lab = space.forward(rnd_xyz)
    rnd_xyz_rt = space.inverse(rnd_lab)
    rnd_srgb_rt = _linear_to_srgb((rnd_xyz_rt @ msi.T).clamp(0, 1))
    rnd_8bit_rt = (rnd_srgb_rt * 255).round()
    channel_err = (rnd_8bit - rnd_8bit_rt).abs()
    exact_10k = (channel_err == 0).all(dim=1).sum().item()

    return {
        "grays_exact": f"{gray_exact}/256",
        "grays_exact_count": gray_exact,
        "websafe_exact": f"{ws_exact}/216",
        "websafe_exact_count": ws_exact,
        "random_10k_exact": f"{exact_10k}/10000",
        "random_10k_exact_count": exact_10k,
        "max_channel_error": int(channel_err.max().item()),
        "mean_channel_error": channel_err.float().mean().item(),
    }


# ═══════════════════════════════════════════════════════════════
#  20. GRADIENT CHANNEL MONOTONICITY
# ═══════════════════════════════════════════════════════════════

def measure_channel_monotonicity(space, device):
    """In R→W gradient, does sRGB R decrease monotonically? G,B increase?"""
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)

    gradients = [
        # (name, rgb1, rgb2, expected_increase[R,G,B], gamut_matrix)
        ("R→W", [1, 0, 0], [1, 1, 1], [False, True, True], "srgb"),
        ("G→W", [0, 1, 0], [1, 1, 1], [True, False, True], "srgb"),
        ("B→W", [0, 0, 1], [1, 1, 1], [True, True, False], "srgb"),
        ("R→K", [1, 0, 0], [0, 0, 0], [False, False, False], "srgb"),
        ("K→W", [0, 0, 0], [1, 1, 1], [True, True, True], "srgb"),
        ("R→C", [1, 0, 0], [0, 1, 1], [False, True, True], "srgb"),
        ("Y→B", [1, 1, 0], [0, 0, 1], [False, False, True], "srgb"),
        # P3
        ("P3_R→W", [1, 0, 0], [1, 1, 1], [False, True, True], "p3"),
        ("P3_G→W", [0, 1, 0], [1, 1, 1], [True, False, True], "p3"),
        ("P3_B→W", [0, 0, 1], [1, 1, 1], [True, True, False], "p3"),
        ("P3_K→W", [0, 0, 0], [1, 1, 1], [True, True, True], "p3"),
        ("P3_R→C", [1, 0, 0], [0, 1, 1], [False, True, True], "p3"),
    ]

    mp3 = _to(_M_P3, device)
    mp3i = torch.linalg.inv(mp3)
    gamut_mats = {"srgb": (ms, msi), "p3": (mp3, mp3i)}

    results = {}
    for name, rgb1, rgb2, expected_increase, gamut_key in gradients:
        gmat, ginv = gamut_mats[gamut_key]
        xyz1 = _srgb_to_linear(torch.tensor(rgb1, device=device, dtype=torch.float64)) @ gmat.T
        xyz2 = _srgb_to_linear(torch.tensor(rgb2, device=device, dtype=torch.float64)) @ gmat.T
        lab1 = space.forward(xyz1.unsqueeze(0))[0]
        lab2 = space.forward(xyz2.unsqueeze(0))[0]

        t = torch.linspace(0, 1, 256, device=device, dtype=torch.float64)
        labs = lab1.unsqueeze(0) + t.unsqueeze(1) * (lab2 - lab1).unsqueeze(0)
        xyz_path = space.inverse(labs)
        srgb_path = _linear_to_srgb((xyz_path @ ginv.T).clamp(0, 1))

        violations = {}
        ch_names = ["R", "G", "B"]
        for ch in range(3):
            diffs = srgb_path[1:, ch] - srgb_path[:-1, ch]
            if expected_increase[ch]:
                # Should increase: count decreases > threshold
                n_viol = (diffs < -0.005).sum().item()
            else:
                # Should decrease: count increases > threshold
                n_viol = (diffs > 0.005).sum().item()
            violations[ch_names[ch]] = int(n_viol)

        results[name] = {
            "violations": violations,
            "total_violations": sum(violations.values()),
        }

    return results


# ═══════════════════════════════════════════════════════════════

def measure_perceptual_banding(space, device):
    """For 256-step 8-bit gradients, measure invisible steps (ΔE < 1.0)."""
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)
    d65 = _to(_D65, device)

    ms = _to(_M_SRGB, device)
    mp3 = _to(_M_P3, device)

    # 50+ key gradients at full 256 steps — sRGB + P3 + darks + pastels
    gradients = [
        # Primary→White (6)
        ("R→W", [1, 0, 0], [1, 1, 1]),
        ("G→W", [0, 1, 0], [1, 1, 1]),
        ("B→W", [0, 0, 1], [1, 1, 1]),
        ("Y→W", [1, 1, 0], [1, 1, 1]),
        ("C→W", [0, 1, 1], [1, 1, 1]),
        ("M→W", [1, 0, 1], [1, 1, 1]),
        # Primary→Black (6)
        ("R→K", [1, 0, 0], [0, 0, 0]),
        ("G→K", [0, 1, 0], [0, 0, 0]),
        ("B→K", [0, 0, 1], [0, 0, 0]),
        ("Y→K", [1, 1, 0], [0, 0, 0]),
        ("C→K", [0, 1, 1], [0, 0, 0]),
        ("M→K", [1, 0, 1], [0, 0, 0]),
        # Neutral
        ("K→W", [0, 0, 0], [1, 1, 1]),
        # Complementary (6)
        ("R→C", [1, 0, 0], [0, 1, 1]),
        ("G→M", [0, 1, 0], [1, 0, 1]),
        ("B→Y", [0, 0, 1], [1, 1, 0]),
        ("C→R", [0, 1, 1], [1, 0, 0]),
        ("M→G", [1, 0, 1], [0, 1, 0]),
        ("Y→B", [1, 1, 0], [0, 0, 1]),
        # Adjacent hues (6)
        ("R→Y", [1, 0, 0], [1, 1, 0]),
        ("Y→G", [1, 1, 0], [0, 1, 0]),
        ("G→C", [0, 1, 0], [0, 1, 1]),
        ("C→B", [0, 1, 1], [0, 0, 1]),
        ("B→M", [0, 0, 1], [1, 0, 1]),
        ("M→R", [1, 0, 1], [1, 0, 0]),
        # Dark→Dark (4)
        ("dR→dB", [0.3, 0, 0], [0, 0, 0.3]),
        ("dG→dM", [0, 0.3, 0], [0.3, 0, 0.3]),
        ("dY→dC", [0.3, 0.3, 0], [0, 0.3, 0.3]),
        ("dK→dR", [0.05, 0.05, 0.05], [0.3, 0.05, 0.05]),
        # Pastel (4)
        ("pR→pB", [1, 0.7, 0.7], [0.7, 0.7, 1]),
        ("pG→pM", [0.7, 1, 0.7], [1, 0.7, 1]),
        ("pY→pC", [1, 1, 0.7], [0.7, 1, 1]),
        ("pR→pG", [1, 0.7, 0.7], [0.7, 1, 0.7]),
        # P3 gradients (use P3 matrix below)
    ]
    # P3-specific gradients
    p3_gradients = [
        ("P3_R→W", [1, 0, 0], [1, 1, 1]),
        ("P3_G→W", [0, 1, 0], [1, 1, 1]),
        ("P3_B→W", [0, 0, 1], [1, 1, 1]),
        ("P3_R→K", [1, 0, 0], [0, 0, 0]),
        ("P3_G→K", [0, 1, 0], [0, 0, 0]),
        ("P3_B→K", [0, 0, 1], [0, 0, 0]),
        ("P3_R→C", [1, 0, 0], [0, 1, 1]),
        ("P3_G→M", [0, 1, 0], [1, 0, 1]),
        ("P3_B→Y", [0, 0, 1], [1, 1, 0]),
        ("P3_K→W", [0, 0, 0], [1, 1, 1]),
    ]

    results = {}

    # Process sRGB gradients
    all_grads = [(name, rgb1, rgb2, ms, msi) for name, rgb1, rgb2 in gradients]
    # Process P3 gradients
    mp3i = torch.linalg.inv(mp3)
    all_grads += [(name, rgb1, rgb2, mp3, mp3i) for name, rgb1, rgb2 in p3_gradients]

    for name, rgb1, rgb2, gamut_mat, gamut_inv in all_grads:
        xyz1 = _srgb_to_linear(torch.tensor(rgb1, device=device, dtype=torch.float64)) @ gamut_mat.T
        xyz2 = _srgb_to_linear(torch.tensor(rgb2, device=device, dtype=torch.float64)) @ gamut_mat.T
        lab1 = space.forward(xyz1.unsqueeze(0))[0]
        lab2 = space.forward(xyz2.unsqueeze(0))[0]

        # 256 steps
        t = torch.linspace(0, 1, 256, device=device, dtype=torch.float64)
        labs = lab1.unsqueeze(0) + t.unsqueeze(1) * (lab2 - lab1).unsqueeze(0)
        xyz_all = space.inverse(labs)
        lin = (xyz_all @ gamut_inv.T).clamp(0, 1)
        s8 = (_linear_to_srgb(lin) * 255).round() / 255.0
        xyz_q = _srgb_to_linear(s8) @ gamut_mat.T
        cielab = _xyz_to_cielab(xyz_q.clamp(min=1e-10), d65)

        de = _ciede2000_simplified(cielab[:-1], cielab[1:])
        invisible = (de < 1.0).sum().item()
        duplicate = ((s8[1:] * 255).to(torch.int32) ==
                     (s8[:-1] * 255).to(torch.int32)).all(dim=1).sum().item()

        results[name] = {
            "invisible_steps": int(invisible),
            "invisible_pct": invisible / 255 * 100,
            "duplicate_rgb": int(duplicate),
            "de_min": de.min().item(),
            "de_max": de.max().item(),
        }

    total_invisible = sum(r["invisible_steps"] for r in results.values())
    total_steps = 255 * len(results)
    return {
        "per_gradient": results,
        "total_invisible_pct": total_invisible / total_steps * 100,
        "total_duplicate_pct": sum(r["duplicate_rgb"] for r in results.values()) / total_steps * 100,
    }


# ═══════════════════════════════════════════════════════════════
#  21. INTERPOLATION OUT-OF-GAMUT EXCURSION
# ═══════════════════════════════════════════════════════════════

_M_REC2020 = torch.tensor([
    [0.6369580483012914, 0.14461690358620832, 0.1688809751641721],
    [0.2627002120112671, 0.6779980715188708, 0.05930171646986196],
    [0.0, 0.028072693049087428, 1.0609850577107909],
])


def measure_oog_excursion(space, device):
    """For in-gamut sRGB pairs, interpolate in Lab at 256 steps.
    Check if any intermediate step maps to out-of-gamut sRGB.
    Catches the OKLab blue-white purple shift issue.
    """
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)

    # Build pair set: all 6 primaries + white + black + pastels + random
    endpoints = []
    # Primary→White (6)
    primaries = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                 [1, 1, 0], [0, 1, 1], [1, 0, 1]]
    p_names = ["R", "G", "B", "Y", "C", "M"]
    for i, (rgb, name) in enumerate(zip(primaries, p_names)):
        endpoints.append((rgb, [1, 1, 1], f"{name}->W"))
        endpoints.append((rgb, [0, 0, 0], f"{name}->K"))
    # Complementary pairs
    for i in range(3):
        endpoints.append((primaries[i], primaries[i + 3], f"{p_names[i]}->{p_names[i+3]}"))
    # Adjacent pairs
    for i in range(6):
        j = (i + 1) % 6
        endpoints.append((primaries[i], primaries[j], f"{p_names[i]}->{p_names[j]}"))
    # Pastels
    for h_deg in range(0, 360, 30):
        h = h_deg / 360.0
        r1, g1, b1 = _hsv_to_rgb(h, 0.3, 0.9)
        r2, g2, b2 = _hsv_to_rgb(((h_deg + 60) % 360) / 360.0, 0.3, 0.9)
        endpoints.append(([r1, g1, b1], [r2, g2, b2], f"pastel_h{h_deg}"))
    # Random (100)
    import random as _rnd
    _rnd.seed(99)
    for k in range(100):
        rgb1 = [_rnd.random(), _rnd.random(), _rnd.random()]
        rgb2 = [_rnd.random(), _rnd.random(), _rnd.random()]
        endpoints.append((rgb1, rgb2, f"rnd{k}"))

    n_steps = 256
    excursion_pairs = 0
    max_oog_dist = 0.0
    total_pairs = len(endpoints)
    pair_details = []

    for rgb1, rgb2, name in endpoints:
        t1 = torch.tensor(rgb1, device=device, dtype=torch.float64)
        t2 = torch.tensor(rgb2, device=device, dtype=torch.float64)
        xyz1 = (_srgb_to_linear(t1) @ ms.T).unsqueeze(0)
        xyz2 = (_srgb_to_linear(t2) @ ms.T).unsqueeze(0)
        lab1 = space.forward(xyz1)[0]
        lab2 = space.forward(xyz2)[0]

        t = torch.linspace(0, 1, n_steps, device=device, dtype=torch.float64)
        labs = lab1.unsqueeze(0) + t.unsqueeze(1) * (lab2 - lab1).unsqueeze(0)

        xyz_interp = space.inverse(labs)
        lin = xyz_interp @ msi.T  # linear sRGB, not clamped

        # Check out-of-gamut: any channel < -0.001 or > 1.001
        oog_low = (lin < -0.001).any(dim=1)
        oog_high = (lin > 1.001).any(dim=1)
        oog = oog_low | oog_high

        if oog.any():
            excursion_pairs += 1
            # Max distance outside gamut
            dist_low = (-lin).clamp(min=0).max().item()  # how far below 0
            dist_high = (lin - 1.0).clamp(min=0).max().item()  # how far above 1
            pair_max_dist = max(dist_low, dist_high)
            max_oog_dist = max(max_oog_dist, pair_max_dist)
            pair_details.append({
                "pair": name,
                "oog_steps": int(oog.sum().item()),
                "max_oog_dist": pair_max_dist,
            })

    return {
        "total_pairs": total_pairs,
        "excursion_pairs": excursion_pairs,
        "excursion_pct": excursion_pairs / total_pairs * 100,
        "max_oog_dist": max_oog_dist,
        "worst_pairs": sorted(pair_details, key=lambda x: -x["max_oog_dist"])[:10],
    }


# ═══════════════════════════════════════════════════════════════
#  22. HUE REVERSAL DETECTION
# ═══════════════════════════════════════════════════════════════

def measure_hue_reversal(space, device):
    """For each hue (0-360), start from cusp and reduce chroma to 0.
    At each step compute the output hue angle. If hue changes direction
    (reversal), count it. Reports number of hues with reversals and max
    reversal angle.
    """
    ms = _to(_M_SRGB, device)
    msi = torch.linalg.inv(ms)
    d65 = _to(_D65, device)

    n_hues = 360
    n_C_steps = 100
    Cs = torch.linspace(0.001, 0.4, n_C_steps, device=device, dtype=torch.float64)

    # First, find cusp L for each hue via a quick scan
    n_L_scan = 100
    n_C_scan = 80
    Ls_scan = torch.linspace(0.05, 0.95, n_L_scan, device=device, dtype=torch.float64)
    Cs_scan = torch.linspace(0.001, 0.45, n_C_scan, device=device, dtype=torch.float64)

    cusp_L = torch.zeros(n_hues, device=device, dtype=torch.float64)

    # Scan for cusp L in batches
    for h_deg in range(n_hues):
        h_rad = h_deg * 2 * PI / 360
        ch = math.cos(h_rad)
        sh = math.sin(h_rad)
        Le = Ls_scan.view(n_L_scan, 1).expand(n_L_scan, n_C_scan).reshape(-1)
        Ce = Cs_scan.view(1, n_C_scan).expand(n_L_scan, n_C_scan).reshape(-1)
        lab = torch.stack([Le, Ce * ch, Ce * sh], dim=-1)
        xyz = space.inverse(lab)
        lin = xyz @ msi.T
        ok = ((lin >= -0.002) & (lin <= 1.002)).all(dim=1).reshape(n_L_scan, n_C_scan)
        cv = Cs_scan.view(1, n_C_scan).expand(n_L_scan, n_C_scan)
        mc, _ = torch.where(ok, cv, torch.zeros_like(cv)).max(dim=1)
        ci = mc.argmax()
        cusp_L[h_deg] = Ls_scan[ci]

    # Now check hue reversals: for each hue, reduce chroma from cusp C to 0
    reversal_count = 0
    max_reversal_angle = 0.0
    per_hue_results = []

    for h_deg in range(n_hues):
        h_rad = h_deg * 2 * PI / 360
        ch = math.cos(h_rad)
        sh = math.sin(h_rad)
        L_val = cusp_L[h_deg].item()

        lab = torch.stack([
            torch.full((n_C_steps,), L_val, device=device, dtype=torch.float64),
            Cs * ch,
            Cs * sh,
        ], dim=-1)

        xyz = space.inverse(lab)
        # Filter to in-gamut points
        lin = xyz @ msi.T
        in_gamut = ((lin >= -0.01) & (lin <= 1.01)).all(dim=1)

        if in_gamut.sum() < 3:
            continue

        xyz_valid = xyz[in_gamut].clamp(min=1e-10)
        cielab = _xyz_to_cielab(xyz_valid, d65)

        # Compute CIE Lab hue angles
        C_star = (cielab[:, 1] ** 2 + cielab[:, 2] ** 2).sqrt()
        chromatic = C_star > 1.0
        if chromatic.sum() < 3:
            continue

        h_cielab = torch.atan2(cielab[chromatic, 2], cielab[chromatic, 1])

        # Check for hue direction changes (reversals)
        dh = h_cielab[1:] - h_cielab[:-1]
        dh = torch.atan2(torch.sin(dh), torch.cos(dh))  # wrap to [-pi, pi]

        if dh.numel() < 2:
            continue

        # Detect sign changes in dh (reversal)
        signs = dh.sign()
        # Remove zero signs
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
        "total_hues_tested": n_hues,
        "worst_hues": sorted(per_hue_results, key=lambda x: -x["max_angle"])[:10],
    }


# ═══════════════════════════════════════════════════════════════
#  23. NEAR-PRIMARY HUE DISCONTINUITY
# ═══════════════════════════════════════════════════════════════

def measure_primary_hue_discontinuity(space, device):
    """For each sRGB/P3 primary, compute hue angle and compare to
    nearby colors (+-0.01 in each RGB channel). Large hue jumps
    indicate singularities near primaries.
    """
    ms = _to(_M_SRGB, device)
    mp3 = _to(_M_P3, device)
    d65 = _to(_D65, device)

    # sRGB primaries + secondaries
    srgb_primaries = {
        "R": [1, 0, 0], "G": [0, 1, 0], "B": [0, 0, 1],
        "C": [0, 1, 1], "M": [1, 0, 1], "Y": [1, 1, 0],
    }
    # P3 primaries
    p3_primaries = {
        "P3_R": [1, 0, 0], "P3_G": [0, 1, 0], "P3_B": [0, 0, 1],
        "P3_C": [0, 1, 1], "P3_M": [1, 0, 1], "P3_Y": [1, 1, 0],
    }

    delta = 0.01
    results = {}

    for gamut_name, primaries, gamut_mat in [
        ("sRGB", srgb_primaries, ms),
        ("P3", p3_primaries, mp3),
    ]:
        for name, rgb in primaries.items():
            rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
            xyz_center = (_srgb_to_linear(rgb_t) @ gamut_mat.T).unsqueeze(0)
            lab_center = space.forward(xyz_center)[0]
            h_center = math.atan2(lab_center[2].item(), lab_center[1].item())

            # Perturbations: +- delta in each channel, clamped to [0,1]
            max_jump = 0.0
            for ch in range(3):
                for sign in [-1, 1]:
                    perturbed = rgb_t.clone()
                    perturbed[ch] = (perturbed[ch] + sign * delta).clamp(0.0, 1.0)
                    # Skip if perturbation didn't change anything (at boundary)
                    if (perturbed == rgb_t).all():
                        continue
                    xyz_p = (_srgb_to_linear(perturbed) @ gamut_mat.T).unsqueeze(0)
                    lab_p = space.forward(xyz_p)[0]
                    C_p = (lab_p[1] ** 2 + lab_p[2] ** 2).sqrt().item()
                    C_c = (lab_center[1] ** 2 + lab_center[2] ** 2).sqrt().item()
                    # Only compare hue if both are chromatic
                    if C_p > 0.01 and C_c > 0.01:
                        h_p = math.atan2(lab_p[2].item(), lab_p[1].item())
                        dh = abs(h_p - h_center)
                        if dh > PI:
                            dh = 2 * PI - dh
                        dh_deg = dh * (180 / PI)
                        max_jump = max(max_jump, dh_deg)

            results[name] = {
                "max_hue_jump_deg": max_jump,
                "lab": [lab_center[0].item(), lab_center[1].item(), lab_center[2].item()],
            }

    # Summary stats
    srgb_jumps = [v["max_hue_jump_deg"] for k, v in results.items() if not k.startswith("P3_")]
    p3_jumps = [v["max_hue_jump_deg"] for k, v in results.items() if k.startswith("P3_")]

    return {
        "per_primary": results,
        "srgb_max_jump": max(srgb_jumps) if srgb_jumps else 0.0,
        "srgb_mean_jump": sum(srgb_jumps) / len(srgb_jumps) if srgb_jumps else 0.0,
        "p3_max_jump": max(p3_jumps) if p3_jumps else 0.0,
        "p3_mean_jump": sum(p3_jumps) / len(p3_jumps) if p3_jumps else 0.0,
    }


# ═══════════════════════════════════════════════════════════════
#  24. NEGATIVE LMS DETECTION
# ═══════════════════════════════════════════════════════════════

def measure_negative_lms(space, device):
    """Convert 10,000 random sRGB colors through the space. Check if any
    intermediate LMS values (after M1, before transfer function) are
    negative. Spaces with proper LMS handling should have 0 negatives.
    """
    ms = _to(_M_SRGB, device)

    # 10000 random sRGB colors
    gen = torch.Generator(device=device).manual_seed(42)
    srgb = torch.rand(10000, 3, generator=gen, device=device, dtype=torch.float64)
    xyz = _srgb_to_linear(srgb) @ ms.T

    # We need M1 from the space. Access it if available.
    has_M1 = hasattr(space, 'M1') or hasattr(space, '_M1') or hasattr(space, '_M1_mod')

    if not has_M1:
        # Can't measure LMS for spaces without explicit M1 (e.g. CIELab)
        return {
            "n_negative": 0,
            "max_negative": 0.0,
            "pct_negative": 0.0,
            "note": "Space has no M1 matrix (LMS check not applicable)",
        }

    # Get M1 matrix
    if hasattr(space, '_M1_mod'):
        M1 = space._M1_mod
    elif hasattr(space, '_M1'):
        M1 = space._M1
    else:
        M1 = space.M1

    lms = xyz @ M1.T

    # Count colors with any negative LMS
    neg_mask = lms < 0
    n_colors_with_neg = neg_mask.any(dim=1).sum().item()
    max_neg = 0.0
    if neg_mask.any():
        max_neg = (-lms[neg_mask]).max().item()

    # Per-channel stats
    per_channel = {}
    for ch, ch_name in enumerate(["L", "M", "S"]):
        ch_neg = (lms[:, ch] < 0).sum().item()
        ch_min = lms[:, ch].min().item()
        per_channel[ch_name] = {
            "n_negative": int(ch_neg),
            "min_value": ch_min,
        }

    return {
        "n_negative": int(n_colors_with_neg),
        "max_negative": max_neg,
        "pct_negative": n_colors_with_neg / 10000 * 100,
        "per_channel": per_channel,
    }


# ═══════════════════════════════════════════════════════════════
#  25. EXTREME CHROMA STABILITY
# ═══════════════════════════════════════════════════════════════

def measure_extreme_chroma_stability(space, device):
    """For P3 and Rec.2020 primaries, convert to Lab, add small
    perturbations (+-0.001 in Lab), convert back to XYZ. Check if
    round-trip is stable (no NaN, no huge jumps). Reports max
    perturbation amplification factor.
    """
    ms = _to(_M_SRGB, device)
    mp3 = _to(_M_P3, device)
    mr2020 = _to(_M_REC2020, device)

    # P3 and Rec.2020 primaries (encoded as linear RGB in their gamut)
    test_colors = {}
    for gname, gmat in [("P3", mp3), ("Rec2020", mr2020)]:
        primaries = torch.eye(3, device=device, dtype=torch.float64)
        # Linear primaries → XYZ
        for i, pname in enumerate(["R", "G", "B"]):
            xyz = (primaries[i] @ gmat.T).unsqueeze(0)
            test_colors[f"{gname}_{pname}"] = xyz
        # Also secondaries
        secondaries = torch.tensor([
            [1, 1, 0], [0, 1, 1], [1, 0, 1],
        ], device=device, dtype=torch.float64)
        for i, sname in enumerate(["Y", "C", "M"]):
            xyz = (secondaries[i] @ gmat.T).unsqueeze(0)
            test_colors[f"{gname}_{sname}"] = xyz

    eps = 0.001  # perturbation size in Lab
    max_amplification = 0.0
    nan_count = 0
    inf_count = 0
    per_color = {}

    for name, xyz_orig in test_colors.items():
        lab = space.forward(xyz_orig)
        lab_val = lab[0]

        # 6 perturbation directions: +/- in each of L, a, b
        perturbations = torch.zeros(6, 3, device=device, dtype=torch.float64)
        perturbations[0, 0] = eps   # +L
        perturbations[1, 0] = -eps  # -L
        perturbations[2, 1] = eps   # +a
        perturbations[3, 1] = -eps  # -a
        perturbations[4, 2] = eps   # +b
        perturbations[5, 2] = -eps  # -b

        lab_perturbed = lab_val.unsqueeze(0) + perturbations  # (6, 3)

        # Inverse perturbed → XYZ
        xyz_perturbed = space.inverse(lab_perturbed)

        # Check for NaN/Inf
        n_nan = xyz_perturbed.isnan().sum().item()
        n_inf = xyz_perturbed.isinf().sum().item()
        nan_count += n_nan
        inf_count += n_inf

        if n_nan > 0 or n_inf > 0:
            per_color[name] = {
                "amplification": float("inf"),
                "nan": int(n_nan),
                "inf": int(n_inf),
            }
            continue

        # Amplification: ||XYZ_perturbed - XYZ_orig|| / ||Lab_perturbed - Lab_orig||
        # Measures sensitivity of the inverse at extreme chroma points
        xyz_diff = (xyz_perturbed - xyz_orig).norm(dim=1)
        lab_diff = perturbations.norm(dim=1)  # all eps
        amp = xyz_diff / lab_diff.clamp(min=1e-15)
        max_amp = amp.max().item()
        max_amplification = max(max_amplification, max_amp)

        per_color[name] = {
            "amplification": max_amp,
            "max_xyz_diff": xyz_diff.max().item(),
        }

    return {
        "max_amplification": max_amplification,
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "per_color": per_color,
    }
