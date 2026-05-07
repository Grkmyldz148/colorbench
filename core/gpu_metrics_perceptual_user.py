"""ColorBench end-user perceptual metric'leri (yeni Phase 8).

Mevcut gpu_metrics_perceptual.py "perceptual_internal" tarafında
(Munsell, MacAdam, hue leaf — akademik benchmark'lar).

Bu modül **end-user görsel** perceptual test ekliyor:
  1. measure_image_synthetic_gradient — synthetic image üzerinde uzayda
     gradient mapping, output ΔE2000 ortalama (image quality proxy)
  2. measure_design_palette_quality — 30 popüler tasarım rengi (Tailwind/
     Material/Brand) × tone scale generation × CV ortalaması
  3. measure_skin_tone_fitzpatrick — 6 Fitzpatrick skin tone × 11-shade
     tint/shade ladder × hue stability (fotoğrafçılık + UI critical)
  4. measure_brand_color_preservation — gerçek brand renkler (Coca-Cola
     red, Tiffany blue, IKEA blue, vs) × roundtrip + tint/shade
"""
import torch
import numpy as np

from .spaces import D65
from .gpu_de import ciede2000

PI = np.pi
_M_SRGB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float64)


def _to(t, device):
    return t.to(device=device, dtype=torch.float64)


def _srgb_to_linear(c):
    return torch.where(c <= 0.04045, c/12.92, ((c+0.055)/1.055).pow(2.4))


def _xyz_to_cielab(xyz, d65):
    r = xyz / d65
    delta3 = (6.0 / 29.0) ** 3
    f = torch.where(r > delta3, r.pow(1.0/3.0),
                    r / (3 * (6.0/29.0)**2) + 4.0/29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def _hex_to_xyz(hex_str, ms, device):
    h = hex_str.lstrip("#")
    rgb = torch.tensor([int(h[i:i+2], 16)/255 for i in (0,2,4)], device=device, dtype=torch.float64)
    return ms @ _srgb_to_linear(rgb)


# ═══════════════════════════════════════════════════════════════════════
#  E1. Synthetic image gradient — perceptual quality proxy
# ═══════════════════════════════════════════════════════════════════════

def measure_image_synthetic_gradient(space, device):
    """Synthetic image quality: 256-stop gradient mapping rendered as image.

    Renders a horizontal gradient through the test space and measures
    perceptual quality (banding count, smoothness CV, hue drift).
    Stresses the same metric set ColorBench's `gradients` family does, but
    on visually distinct endpoints designers actually use:
      - sky blue → sunset orange (cross-hue, 180°)
      - dark forest → bright sky (L-jump + hue cross)
      - photo-grade twilight (#1A237E → #E91E63)
      - skin-blend (light skin → dark skin, FT I → FT VI)

    Returns dict with per-gradient banding and overall CV.
    """
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    PAIRS = [
        ("sky→sunset",     "#0EA5E9", "#FF8C00"),
        ("forest→sky",     "#166534", "#0EA5E9"),
        ("twilight",       "#1A237E", "#E91E63"),
        ("skin FT I → VI", "#F4D9C0", "#4A2D1B"),
        ("brand red→teal", "#DC2626", "#0D9488"),
    ]
    N = 256
    results = {}
    all_bandings, all_cvs, all_drifts = [], [], []
    for name, h1, h2 in PAIRS:
        xyz1 = _hex_to_xyz(h1, ms, device)
        xyz2 = _hex_to_xyz(h2, ms, device)
        lab1 = space.forward(xyz1.unsqueeze(0))
        lab2 = space.forward(xyz2.unsqueeze(0))
        ts = torch.linspace(0, 1, N, device=device, dtype=torch.float64).view(-1, 1)
        interp = lab1 + ts * (lab2 - lab1)
        # Forward to sRGB → quantize 8-bit → back to CIE Lab
        xyz_path = space.inverse(interp)
        # CIE Lab over D65 for ground-truth perceptual measurement
        cl = _xyz_to_cielab(xyz_path, d65)
        de_adj = ciede2000(cl[:-1], cl[1:])
        banding = (de_adj < 1.0).sum().item()
        cv = (de_adj.std() / (de_adj.mean() + 1e-9)).item()
        # Hue drift from start
        h_start = torch.atan2(cl[0, 2], cl[0, 1])
        h_path = torch.atan2(cl[:, 2], cl[:, 1])
        dh = (h_path - h_start) * 180.0 / PI
        # Wrap to [-180, 180]
        dh_wrapped = torch.atan2(torch.sin(dh*PI/180), torch.cos(dh*PI/180)) * 180/PI
        drift_max = dh_wrapped.abs().max().item()

        results[name] = {
            "banding": banding,
            "cv": cv,
            "drift_max_deg": drift_max,
            "n_steps": N,
        }
        all_bandings.append(banding); all_cvs.append(cv); all_drifts.append(drift_max)
    results["aggregate"] = {
        "mean_banding": float(np.mean(all_bandings)),
        "mean_cv": float(np.mean(all_cvs)),
        "mean_drift_max": float(np.mean(all_drifts)),
    }
    return results


# ═══════════════════════════════════════════════════════════════════════
#  E2. Design palette quality — 30 popüler renk × tone scale step CV
# ═══════════════════════════════════════════════════════════════════════

def measure_design_palette_quality(space, device):
    """30 popüler tasarım rengi × 11-shade tone scale × step CV mean.

    Endpoint set = Tailwind 500 + Material 500 + brand colors (curated).
    For each: generate 11-shade tone scale (constant a,b, sweep L) → measure
    CV of consecutive ΔE2000 (CIE Lab ground truth). Lower CV = more uniform.
    """
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    SEEDS = [
        # Tailwind 500
        "#3B82F6", "#EF4444", "#22C55E", "#F59E0B", "#A855F7",
        "#14B8A6", "#F43F5E", "#0EA5E9", "#8B5CF6", "#06B6D4",
        # Material 500
        "#F44336", "#9C27B0", "#3F51B5", "#009688", "#FF9800",
        "#795548", "#607D8B", "#E91E63", "#673AB7", "#FFC107",
        # Brand
        "#FF0000",  # Coca-Cola red
        "#0BCED9",  # Tiffany blue
        "#0051BA",  # IKEA blue
        "#FF6F00",  # Hermes orange
        "#1DA1F2",  # Twitter blue
        "#25D366",  # WhatsApp green
        "#9b59b6",  # Designer purple
        "#FF6B6B",  # Coral
        "#1ABC9C",  # Mint
        "#FF7F50",  # Coral
    ]
    N_LEVELS = 11
    L_light, L_dark = 0.97, 0.05  # in test space's L-coord; works since most spaces normalize
    cvs = []
    drift_maxes = []
    for hx in SEEDS:
        xyz_seed = _hex_to_xyz(hx, ms, device)
        lab_seed = space.forward(xyz_seed.unsqueeze(0))[0]
        seed_a, seed_b = lab_seed[1], lab_seed[2]
        base_L = lab_seed[0]
        # 11 L levels: 0=light, 1=dark
        levels_t = torch.linspace(0, 1, N_LEVELS, device=device, dtype=torch.float64)
        Ls = []
        for t in levels_t:
            if t.item() <= 0.5:
                tt = t.item() * 2
                L = L_light + tt*(base_L.item() - L_light)
            else:
                tt = (t.item() - 0.5) * 2
                L = base_L.item() + tt*(L_dark - base_L.item())
            Ls.append(L)
        Ls_t = torch.tensor(Ls, device=device, dtype=torch.float64)
        a_arr = seed_a.expand(N_LEVELS); b_arr = seed_b.expand(N_LEVELS)
        scale_lab = torch.stack([Ls_t, a_arr, b_arr], dim=-1)
        xyz_path = space.inverse(scale_lab)
        cl = _xyz_to_cielab(xyz_path, d65)
        de_adj = ciede2000(cl[:-1], cl[1:])
        cv = (de_adj.std() / (de_adj.mean() + 1e-9)).item()
        cvs.append(cv)
        # Hue drift across scale
        h_seed = torch.atan2(seed_b.unsqueeze(0), seed_a.unsqueeze(0))
        h_path_cl = torch.atan2(cl[:, 2], cl[:, 1])
        dh = (h_path_cl - h_path_cl[N_LEVELS//2]) * 180.0 / PI
        dh_wrapped = torch.atan2(torch.sin(dh*PI/180), torch.cos(dh*PI/180)) * 180/PI
        drift_maxes.append(dh_wrapped.abs().max().item())
    return {
        "n_seeds": len(SEEDS),
        "mean_step_cv": float(np.mean(cvs)),
        "max_step_cv": float(np.max(cvs)),
        "mean_hue_drift_max_deg": float(np.mean(drift_maxes)),
        "max_hue_drift_max_deg": float(np.max(drift_maxes)),
    }


# ═══════════════════════════════════════════════════════════════════════
#  E3. Skin tone Fitzpatrick — 6 ton × 11-shade × hue stability
# ═══════════════════════════════════════════════════════════════════════

def measure_skin_tone_fitzpatrick(space, device):
    """6 Fitzpatrick skin tone × 11-shade tint/shade × hue circular std.

    Fotoğrafçılık + UI design için kritik. Skin tone tint/shade'de
    hue ne kadar sabit kalıyor.
    """
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    SKIN = [
        ("FT_I",   "#F4D9C0"),
        ("FT_II",  "#E5BB9F"),
        ("FT_III", "#C99A78"),
        ("FT_IV",  "#A47553"),
        ("FT_V",   "#7C5337"),
        ("FT_VI",  "#4A2D1B"),
    ]
    N_LEVELS = 11
    L_light, L_dark = 0.97, 0.05
    results = {}
    all_hue_stds, all_step_cvs = [], []
    for name, hx in SKIN:
        xyz_seed = _hex_to_xyz(hx, ms, device)
        lab_seed = space.forward(xyz_seed.unsqueeze(0))[0]
        seed_a, seed_b = lab_seed[1], lab_seed[2]
        base_L = lab_seed[0]
        levels_t = torch.linspace(0, 1, N_LEVELS, device=device, dtype=torch.float64)
        Ls = []
        for t in levels_t:
            if t.item() <= 0.5:
                tt = t.item() * 2
                L = L_light + tt*(base_L.item() - L_light)
            else:
                tt = (t.item() - 0.5) * 2
                L = base_L.item() + tt*(L_dark - base_L.item())
            Ls.append(L)
        Ls_t = torch.tensor(Ls, device=device, dtype=torch.float64)
        scale_lab = torch.stack([Ls_t, seed_a.expand(N_LEVELS), seed_b.expand(N_LEVELS)], dim=-1)
        xyz_path = space.inverse(scale_lab)
        cl = _xyz_to_cielab(xyz_path, d65)
        # Hue circular std (in CIELab degrees)
        h_path = torch.atan2(cl[:, 2], cl[:, 1])
        sin_mean = h_path.sin().mean()
        cos_mean = h_path.cos().mean()
        R = torch.hypot(sin_mean, cos_mean).item()
        hue_std = float(np.sqrt(-2*np.log(max(R, 1e-12))) * 180.0/PI)
        de_adj = ciede2000(cl[:-1], cl[1:])
        step_cv = (de_adj.std() / (de_adj.mean() + 1e-9)).item()
        results[name] = {"hue_cstd_deg": hue_std, "step_cv": step_cv}
        all_hue_stds.append(hue_std); all_step_cvs.append(step_cv)
    results["mean_hue_cstd_deg"] = float(np.mean(all_hue_stds))
    results["max_hue_cstd_deg"] = float(np.max(all_hue_stds))
    results["mean_step_cv"] = float(np.mean(all_step_cvs))
    return results


# ═══════════════════════════════════════════════════════════════════════
#  E4. Brand color preservation — popüler brand renkleri × roundtrip dE
# ═══════════════════════════════════════════════════════════════════════

def measure_brand_color_preservation(space, device):
    """Popüler brand colors × forward∘inverse round-trip ΔE2000.

    Marka rengi reproduce'sinin uzayın her yönünde correct olduğunu
    doğrular (real-world brand fidelity test).
    """
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    BRANDS = [
        ("CocaCola",   "#FF0000"),
        ("TiffanyBlue","#0BCED9"),
        ("IKEABlue",   "#0051BA"),
        ("Hermes",     "#FF6F00"),
        ("Twitter",    "#1DA1F2"),
        ("WhatsApp",   "#25D366"),
        ("Spotify",    "#1DB954"),
        ("YouTube",    "#FF0000"),
        ("Slack",      "#4A154B"),
        ("Netflix",    "#E50914"),
    ]
    results = {}
    all_de = []
    for name, hx in BRANDS:
        xyz0 = _hex_to_xyz(hx, ms, device)
        lab = space.forward(xyz0.unsqueeze(0))
        xyz_back = space.inverse(lab)[0]
        # Compare in CIE Lab (ground truth)
        cl_orig = _xyz_to_cielab(xyz0.unsqueeze(0), d65)
        cl_back = _xyz_to_cielab(xyz_back.unsqueeze(0), d65)
        de = ciede2000(cl_orig, cl_back).item()
        results[name] = {"roundtrip_de": de, "hex": hx}
        all_de.append(de)
    results["mean_roundtrip_de"] = float(np.mean(all_de))
    results["max_roundtrip_de"] = float(np.max(all_de))
    return results
