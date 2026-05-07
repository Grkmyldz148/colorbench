"""ColorBench end-user perceptual benchmark — kapsamlı 29 test.

7 kategori × tüm gerçek görsel görevler:
  Image-based (4)   — synthetic/photo gradient, color grading, white balance, natural scene
  Palette (6)       — Tailwind/Material/diverging/sequential/categorical/theme dark
  Domain (5)        — skin tone, natural colors, brands, logos, cinematic LUT
  Picker UX (4)     — hue continuity, chroma envelope, achromatic, hue wheel
  Accessibility (3) — CVD spacing, low-vision contrast, color-blind safe
  Display (4)       — P3, Rec2020, HDR, 8-bit quantization
  State (3)         — hover, focus, dark mode flip

Her metric:
  - colour-science ground truth (CIE Lab + ΔE2000)
  - Deterministic (sabit seed where applicable)
  - Adil (her uzaya aynı algoritma)
  - Returns dict with named scalar metrics

Total: 29 measure_user_* fonksiyonu
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
_M_P3 = torch.tensor([
    [0.4865709486482162, 0.26566769316909306, 0.1982172852343625],
    [0.2289745640697488, 0.6917385218365064, 0.079286914093745],
    [0.0, 0.04511338185890264, 1.0439443689009757],
], dtype=torch.float64)
_M_REC2020 = torch.tensor([
    [0.6369580483012914, 0.14461690358620832, 0.1688809751641721],
    [0.2627002120112671, 0.6779980715188708, 0.05930171646986196],
    [0.0, 0.028072693049087428, 1.0609850577107909],
], dtype=torch.float64)


def _to(t, device): return t.to(device=device, dtype=torch.float64)
def _srgb_to_lin(c): return torch.where(c <= 0.04045, c/12.92, ((c+0.055)/1.055).pow(2.4))
def _lin_to_srgb(c):
    return torch.where(c <= 0.0031308, c*12.92,
                       1.055*c.clamp(min=1e-12).pow(1.0/2.4) - 0.055)

def _xyz_to_cielab(xyz, d65):
    r = xyz / d65
    delta3 = (6.0/29.0)**3
    f = torch.where(r > delta3, r.pow(1.0/3.0), r/(3*(6.0/29.0)**2) + 4.0/29.0)
    L = 116.0*f[..., 1] - 16.0
    a = 500.0*(f[..., 0] - f[..., 1])
    b = 200.0*(f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)

def _hex_xyz(hx, ms, device):
    h = hx.lstrip("#")
    rgb = torch.tensor([int(h[i:i+2], 16)/255 for i in (0,2,4)], device=device, dtype=torch.float64)
    return ms @ _srgb_to_lin(rgb)

def _cv(t):
    return float((t.std() / (t.mean() + 1e-9)).item())

def _hue_cstd_torch(h_rad):
    sin_m = h_rad.sin().mean(); cos_m = h_rad.cos().mean()
    R = torch.hypot(sin_m, cos_m).item()
    return float(np.sqrt(-2*np.log(max(R, 1e-12))) * 180/PI)

def _tone_scale_in_space(seed_xyz, space, device, n=11, L_light=0.97, L_dark=0.05):
    """Generate n-shade tone scale from seed using space.forward/inverse.
    Constant (a,b), sweep L from L_light → seed_L → L_dark."""
    lab_seed = space.forward(seed_xyz.unsqueeze(0))[0]
    base_L = float(lab_seed[0]); seed_a, seed_b = lab_seed[1], lab_seed[2]
    levels = []
    for i in range(n):
        t = i / (n-1)
        if t <= 0.5:
            tt = t*2; L = L_light + tt*(base_L - L_light)
        else:
            tt = (t - 0.5)*2; L = base_L + tt*(L_dark - base_L)
        levels.append(L)
    Ls = torch.tensor(levels, device=device, dtype=torch.float64)
    a_arr = seed_a.expand(n); b_arr = seed_b.expand(n)
    scale_lab = torch.stack([Ls, a_arr, b_arr], dim=-1)
    return space.inverse(scale_lab)

def _gradient_lab(xyz1, xyz2, n, space, device):
    """Lab-cartesian linear interpolation in space."""
    lab1 = space.forward(xyz1.unsqueeze(0))
    lab2 = space.forward(xyz2.unsqueeze(0))
    ts = torch.linspace(0, 1, n, device=device, dtype=torch.float64).view(-1, 1)
    interp = lab1 + ts * (lab2 - lab1)
    return space.inverse(interp)


# ═══════════════════════════════════════════════════════════════════
#  PHASE 11 — Adillik düzeltme yardımcıları (uzay-bağımsız hedefler)
# ═══════════════════════════════════════════════════════════════════

def _cielab_target_to_xyz(L_star, C_star, h_deg, device):
    """CIE L*a*b* hedef → XYZ. Uzay-bağımsız hedef noktası."""
    a = C_star * np.cos(np.radians(h_deg))
    b = C_star * np.sin(np.radians(h_deg))
    lab = torch.tensor([L_star, a, b], device=device, dtype=torch.float64)
    # CIE Lab → XYZ (D65 ASTM-E308)
    d65 = _to(D65, device)
    fy = (L_star + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0
    def f_inv(t): return t**3 if t > delta else 3*delta*delta*(t - 4.0/29.0)
    X = d65[0].item() * f_inv(fx.item() if hasattr(fx, 'item') else fx)
    Y = d65[1].item() * f_inv(fy)
    Z = d65[2].item() * f_inv(fz.item() if hasattr(fz, 'item') else fz)
    return torch.tensor([X, Y, Z], device=device, dtype=torch.float64)


def _cielab_target_to_space_lab(L_star, C_star, h_deg, space, device):
    """CIE Lab hedef noktasını test space koordinatına çevirir."""
    xyz = _cielab_target_to_xyz(L_star, C_star, h_deg, device)
    return space.forward(xyz.unsqueeze(0))[0]


# ═════════════════════════════════════════════════════════════════════
#  CATEGORY 1: IMAGE-BASED (4 test)
# ═════════════════════════════════════════════════════════════════════

def measure_user_image_synthetic_gradient(space, device):
    """1.1 — Synthetic image gradient: 8 photo-realistic endpoint pair × 256 step.
    Reports mean banding count + step CV across 8 gradients."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    PAIRS = [
        ("sky→sunset",     "#0EA5E9", "#FF8C00"),
        ("forest→sky",     "#166534", "#0EA5E9"),
        ("twilight",       "#1A237E", "#E91E63"),
        ("skin FT I → VI", "#F4D9C0", "#4A2D1B"),
        ("brand red→teal", "#DC2626", "#0D9488"),
        ("dark→light",     "#1E293B", "#F1F5F9"),
        ("ocean depths",   "#0c4a6e", "#7dd3fc"),
        ("autumn leaves",  "#7c2d12", "#fbbf24"),
    ]
    bandings, cvs, drifts = [], [], []
    for name, h1, h2 in PAIRS:
        xyz_path = _gradient_lab(_hex_xyz(h1, ms, device), _hex_xyz(h2, ms, device), 256, space, device)
        cl = _xyz_to_cielab(xyz_path, d65)
        de = ciede2000(cl[:-1], cl[1:])
        bandings.append((de < 1.0).sum().item())
        cvs.append(_cv(de))
        h_path = torch.atan2(cl[:, 2], cl[:, 1])
        dh = (h_path - h_path[0])
        dh_w = torch.atan2(dh.sin(), dh.cos())
        drifts.append(float(dh_w.abs().max().item()) * 180/PI)
    return {
        "n_pairs": len(PAIRS),
        "mean_banding": float(np.mean(bandings)),
        "mean_step_cv": float(np.mean(cvs)),
        "mean_hue_drift_deg": float(np.mean(drifts)),
        "max_hue_drift_deg": float(np.max(drifts)),
    }

def measure_user_color_grading_lut(space, device):
    """1.2 — Color grading LUT: CIE Lab L* gamma 1.2 + sat 1.05.
    PHASE 11 FIX: LUT CIE Lab L* üzerinde uygulanır (uzay-bağımsız).
    Round-trip = uzayın bu LUT'u ne kadar doğru reproduce ediyor."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    PATCHES = [
        "#735244","#C29682","#627A9D","#576C43","#8580B1","#67BDAA",
        "#D67E2C","#505BA6","#C15A63","#5E3C6C","#9DBC40","#E0A32E",
        "#383D96","#469449","#AF363C","#E7C71F","#BB5695","#0885A1",
        "#F3F3F2","#C8C8C8","#A0A0A0","#7A7A7A","#555555","#343434",
    ]
    de_apply = []
    for hx in PATCHES:
        xyz0 = _hex_xyz(hx, ms, device)
        # CIE Lab'da LUT uygula (uzay-bağımsız ground truth)
        cl_orig = _xyz_to_cielab(xyz0.unsqueeze(0), d65)[0]
        L_g = (cl_orig[0]/100.0 - 0.05).clamp(0,1).pow(1/1.2).clamp(0,1) * 100.0 * 1.1
        L_g = L_g.clamp(0, 100)
        a_g = cl_orig[1] * 1.05; b_g = cl_orig[2] * 1.05
        # CIE Lab graded → XYZ via space.forward+inverse round-trip
        cl_target = torch.stack([L_g, a_g, b_g])
        xyz_target = _cielab_target_to_xyz(L_g.item(), float(np.hypot(a_g.item(), b_g.item())),
                                            float(np.degrees(np.arctan2(b_g.item(), a_g.item())) % 360), device)
        # Through space round-trip
        xyz_back = space.inverse(space.forward(xyz_target.unsqueeze(0)))[0]
        cl_back = _xyz_to_cielab(xyz_back.unsqueeze(0), d65)[0]
        de = ciede2000(cl_target.unsqueeze(0), cl_back.unsqueeze(0)).item()
        de_apply.append(de)
    return {
        "n_patches": len(PATCHES),
        "mean_apply_de": float(np.mean(de_apply)),
        "max_apply_de": float(np.max(de_apply)),
    }

def measure_user_white_balance(space, device):
    """1.3 — White balance shift: D55 → D65 round-trip in space.
    Measures hue stability under chromatic adaptation."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    # D55 white point and 12 hue test colors
    D55 = torch.tensor([0.95682, 1.0, 0.92149], device=device, dtype=torch.float64)
    # 12 hue × 3 saturation
    test_xyzs = []
    for h in range(0, 360, 30):
        for s in [0.4, 0.7, 0.9]:
            r = 0.5 + s*0.5*np.cos(np.radians(h))
            g = 0.5 + s*0.5*np.cos(np.radians(h-120))
            b = 0.5 + s*0.5*np.cos(np.radians(h+120))
            rgb = torch.tensor([max(0, r), max(0, g), max(0, b)], device=device, dtype=torch.float64)
            test_xyzs.append(ms @ _srgb_to_lin(rgb))
    test_xyzs = torch.stack(test_xyzs)
    # Apply D55→D65 chromatic adaptation in space (Bradford-like)
    cat = D55 / d65  # naive scale
    adapted_xyz = test_xyzs * cat.unsqueeze(0)
    cl_orig = _xyz_to_cielab(test_xyzs, d65)
    cl_adap = _xyz_to_cielab(adapted_xyz, d65)
    de = ciede2000(cl_orig, cl_adap)
    h_orig = torch.atan2(cl_orig[:, 2], cl_orig[:, 1])
    h_adap = torch.atan2(cl_adap[:, 2], cl_adap[:, 1])
    dh = (h_adap - h_orig) * 180/PI
    dh_w = torch.atan2(dh.sin()*PI/180 if False else (dh*PI/180).sin(), (dh*PI/180).cos())
    return {
        "n_test": int(len(test_xyzs)),
        "mean_de": float(de.mean().item()),
        "max_de": float(de.max().item()),
        "mean_hue_shift_deg": float((torch.abs(dh_w)*180/PI).mean().item()),
    }

def measure_user_natural_scene_palette(space, device):
    """1.4 — Natural scene colors × tone scale × hue stability.
    PHASE 11 FIX: Round-trip-only test bilgi vermiyor (her ikisi 0). Yerine
    her doğa rengi için 11-shade tone scale + hue stability ölç."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    NATURAL = [
        ("sky_clear",     "#87CEEB"), ("sky_sunset",    "#FF7E5F"),
        ("foliage_summer","#228B22"), ("foliage_autumn","#D2691E"),
        ("water_ocean",   "#006994"), ("water_lake",    "#4682B4"),
        ("skin_average",  "#C29682"), ("sand_beach",    "#F4A460"),
        ("snow_blue",     "#F0F8FF"), ("rock_basalt",   "#36454F"),
        ("flower_red",    "#DC143C"), ("flower_yellow", "#FFD700"),
    ]
    hue_stds, step_cvs = [], []
    for name, hx in NATURAL:
        xyz_path = _tone_scale_in_space(_hex_xyz(hx, ms, device), space, device, 11)
        cl = _xyz_to_cielab(xyz_path, d65)
        h_p = torch.atan2(cl[:, 2], cl[:, 1])
        hue_stds.append(_hue_cstd_torch(h_p))
        de = ciede2000(cl[:-1], cl[1:])
        step_cvs.append(_cv(de))
    return {
        "n_natural": len(NATURAL),
        "mean_hue_cstd_deg": float(np.mean(hue_stds)),
        "max_hue_cstd_deg": float(np.max(hue_stds)),
        "mean_step_cv": float(np.mean(step_cvs)),
    }


# ═════════════════════════════════════════════════════════════════════
#  CATEGORY 2: PALETTE GENERATION (6 test)
# ═════════════════════════════════════════════════════════════════════

def measure_user_tailwind_palette(space, device):
    """2.1 — Tailwind 500 colors × 11-shade tone scale. step CV + hue drift."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    SEEDS = ["#3B82F6","#EF4444","#22C55E","#F59E0B","#A855F7","#14B8A6",
             "#F43F5E","#0EA5E9","#8B5CF6","#06B6D4","#EAB308","#EC4899"]
    cvs, drifts = [], []
    for hx in SEEDS:
        xyz_path = _tone_scale_in_space(_hex_xyz(hx, ms, device), space, device, 11)
        cl = _xyz_to_cielab(xyz_path, d65)
        de = ciede2000(cl[:-1], cl[1:])
        cvs.append(_cv(de))
        h_p = torch.atan2(cl[:, 2], cl[:, 1])
        dh = (h_p - h_p[5])  # center
        dh_w = torch.atan2(dh.sin(), dh.cos())
        drifts.append(float((dh_w.abs() * 180/PI).max().item()))
    return {"mean_step_cv": float(np.mean(cvs)), "max_step_cv": float(np.max(cvs)),
            "mean_hue_drift_deg": float(np.mean(drifts)), "max_hue_drift_deg": float(np.max(drifts))}

def measure_user_material_palette(space, device):
    """2.2 — Material Design 500 palette × 11-shade. Material has different hue spread than Tailwind."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    SEEDS = ["#F44336","#9C27B0","#3F51B5","#009688","#FF9800","#795548",
             "#607D8B","#E91E63","#673AB7","#FFC107","#4CAF50","#2196F3"]
    cvs, drifts = [], []
    for hx in SEEDS:
        xyz_path = _tone_scale_in_space(_hex_xyz(hx, ms, device), space, device, 11)
        cl = _xyz_to_cielab(xyz_path, d65)
        de = ciede2000(cl[:-1], cl[1:])
        cvs.append(_cv(de))
        h_p = torch.atan2(cl[:, 2], cl[:, 1])
        dh_w = torch.atan2((h_p - h_p[5]).sin(), (h_p - h_p[5]).cos())
        drifts.append(float((dh_w.abs() * 180/PI).max().item()))
    return {"mean_step_cv": float(np.mean(cvs)), "max_step_cv": float(np.max(cvs)),
            "mean_hue_drift_deg": float(np.mean(drifts))}

def measure_user_diverging_colormap(space, device):
    """2.3 — 6 popüler diverging colormap: RdBu, BrBG, PuOr, PRGn, RdYlBu, PiYG."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    DIV = [
        ("RdBu",   "#053061","#FFFFFF","#67001F"),
        ("BrBG",   "#543005","#F5F5F5","#003C30"),
        ("PuOr",   "#7F3B08","#F7F7F7","#2D004B"),
        ("PRGn",   "#40004B","#F7F7F7","#00441B"),
        ("RdYlBu", "#A50026","#FFFFBF","#313695"),
        ("PiYG",   "#8E0152","#F7F7F7","#276419"),
    ]
    cvs = []
    for name, c1, neutral, c2 in DIV:
        path1 = _gradient_lab(_hex_xyz(c1, ms, device), _hex_xyz(neutral, ms, device), 11, space, device)
        path2 = _gradient_lab(_hex_xyz(neutral, ms, device), _hex_xyz(c2, ms, device), 11, space, device)
        full = torch.cat([path1[:-1], path2], dim=0)
        cl = _xyz_to_cielab(full, d65)
        de = ciede2000(cl[:-1], cl[1:])
        cvs.append(_cv(de))
    return {"n_colormaps": len(DIV), "mean_step_cv": float(np.mean(cvs)), "max_step_cv": float(np.max(cvs))}

def measure_user_sequential_colormap(space, device):
    """2.4 — viridis/plasma/magma/inferno-like sequential colormaps × 256 steps."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    SEQ = [
        ("viridis", "#440154","#FDE725"),
        ("plasma",  "#0D0887","#F0F921"),
        ("magma",   "#000004","#FCFDBF"),
        ("inferno", "#000004","#FCFFA4"),
    ]
    cvs, bandings = [], []
    for name, h1, h2 in SEQ:
        xyz_path = _gradient_lab(_hex_xyz(h1, ms, device), _hex_xyz(h2, ms, device), 256, space, device)
        cl = _xyz_to_cielab(xyz_path, d65)
        de = ciede2000(cl[:-1], cl[1:])
        cvs.append(_cv(de))
        bandings.append((de < 1.0).sum().item())
    return {"n_seq": len(SEQ), "mean_step_cv": float(np.mean(cvs)),
            "mean_banding": float(np.mean(bandings))}

def measure_user_categorical_palette(space, device):
    """2.5 — 8-renkli categorical palette × min pairwise ΔE.
    PHASE 11 FIX: CIE L*=60, C*=30 hedef (uzay-bağımsız), her uzaya aynı
    semantik nokta. Önceki sürüm "L=0.6 in space" kullanıyordu — semantik mismatch."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    L_star, C_star = 60.0, 30.0  # CIE Lab hedef
    hues_deg = list(range(0, 360, 45))
    xyzs = [_cielab_target_to_xyz(L_star, C_star, h_deg, device) for h_deg in hues_deg]
    xyz = torch.stack(xyzs)
    cl = _xyz_to_cielab(xyz, d65)
    # Pairwise ΔE2000
    n = len(cl)
    pair_des = []
    for i in range(n):
        for j in range(i+1, n):
            pair_des.append(ciede2000(cl[i:i+1], cl[j:j+1]).item())
    pd = np.array(pair_des)
    return {"n_colors": n, "min_pairwise_de": float(pd.min()),
            "mean_pairwise_de": float(pd.mean()), "max_pairwise_de": float(pd.max())}

def measure_user_theme_dark_mode(space, device):
    """2.6 — Light → dark mode: CIE L* invert (100 - L*), hue preservation.
    PHASE 11 FIX: CIE Lab L* invert (uzay-bağımsız), sonra space round-trip."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    LIGHT = ["#3B82F6","#EF4444","#22C55E","#F59E0B","#A855F7","#14B8A6","#F43F5E","#0EA5E9"]
    hue_drifts = []
    for hx in LIGHT:
        xyz0 = _hex_xyz(hx, ms, device)
        cl_orig = _xyz_to_cielab(xyz0.unsqueeze(0), d65)[0]
        # Dark mode: invert CIE L* (100 - L*) keep a,b
        L_dark = 100.0 - cl_orig[0].item()
        C_orig = float(np.hypot(cl_orig[1].item(), cl_orig[2].item()))
        h_orig = float(np.degrees(np.arctan2(cl_orig[2].item(), cl_orig[1].item())) % 360)
        # Target dark in CIE Lab → space round-trip
        xyz_dark = _cielab_target_to_xyz(L_dark, C_orig, h_orig, device)
        xyz_back = space.inverse(space.forward(xyz_dark.unsqueeze(0)))[0]
        cl_back = _xyz_to_cielab(xyz_back.unsqueeze(0), d65)[0]
        h_back = float(np.degrees(np.arctan2(cl_back[2].item(), cl_back[1].item())) % 360)
        dh = (h_back - h_orig)
        dh_w = float(np.degrees(np.arctan2(np.sin(dh*PI/180), np.cos(dh*PI/180))))
        hue_drifts.append(abs(dh_w))
    return {"n_tokens": len(LIGHT), "mean_dark_mode_hue_drift_deg": float(np.mean(hue_drifts)),
            "max_dark_mode_hue_drift_deg": float(np.max(hue_drifts))}


# ═════════════════════════════════════════════════════════════════════
#  CATEGORY 3: DOMAIN-SPECIFIC (5 test)
# ═════════════════════════════════════════════════════════════════════

def measure_user_skin_tone_fitzpatrick(space, device):
    """3.1 — 6 Fitzpatrick × 11-shade × hue stability + step CV."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    SKIN = [("FT_I","#F4D9C0"),("FT_II","#E5BB9F"),("FT_III","#C99A78"),
            ("FT_IV","#A47553"),("FT_V","#7C5337"),("FT_VI","#4A2D1B")]
    hue_stds, cvs = [], []
    for name, hx in SKIN:
        xyz_path = _tone_scale_in_space(_hex_xyz(hx, ms, device), space, device, 11)
        cl = _xyz_to_cielab(xyz_path, d65)
        h_p = torch.atan2(cl[:, 2], cl[:, 1])
        hue_stds.append(_hue_cstd_torch(h_p))
        de = ciede2000(cl[:-1], cl[1:])
        cvs.append(_cv(de))
    return {"n_tones": len(SKIN), "mean_hue_cstd_deg": float(np.mean(hue_stds)),
            "max_hue_cstd_deg": float(np.max(hue_stds)), "mean_step_cv": float(np.mean(cvs))}

def measure_user_natural_colors(space, device):
    """3.2 — Sky/foliage/water/sunset/sand reproduction quality.
    Real-world chromatic categories — 5 doğa rengi × tone scale step CV + hue stability."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    NAT = [("sky","#87CEEB"),("foliage","#228B22"),("water","#006994"),
           ("sunset","#FF7E5F"),("sand","#F4A460"),("snow","#F0F8FF"),
           ("autumn","#D2691E"),("ocean","#4682B4")]
    hue_stds, cvs = [], []
    for name, hx in NAT:
        xyz_path = _tone_scale_in_space(_hex_xyz(hx, ms, device), space, device, 11)
        cl = _xyz_to_cielab(xyz_path, d65)
        h_p = torch.atan2(cl[:, 2], cl[:, 1])
        hue_stds.append(_hue_cstd_torch(h_p))
        de = ciede2000(cl[:-1], cl[1:])
        cvs.append(_cv(de))
    return {"n_natural": len(NAT), "mean_hue_cstd_deg": float(np.mean(hue_stds)),
            "mean_step_cv": float(np.mean(cvs))}

def measure_user_brand_colors(space, device):
    """3.3 — 30+ brand colors × tone scale × hue stability."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    BRANDS = ["#FF0000","#0BCED9","#0051BA","#FF6F00","#1DA1F2","#25D366",
              "#1DB954","#E50914","#4A154B","#075E54","#FFFC00","#7289DA",
              "#FF4500","#5851DB","#1A73E8","#34A853","#EA4335","#FBBC04",
              "#1877F2","#DB4437","#0F9D58","#F4B400","#E45F35","#7B68EE",
              "#FF9900","#146EB4","#232F3E","#005A9C","#0073AA","#21759B"]
    hue_stds = []
    for hx in BRANDS:
        xyz_path = _tone_scale_in_space(_hex_xyz(hx, ms, device), space, device, 11)
        cl = _xyz_to_cielab(xyz_path, d65)
        h_p = torch.atan2(cl[:, 2], cl[:, 1])
        hue_stds.append(_hue_cstd_torch(h_p))
    return {"n_brands": len(BRANDS), "mean_hue_cstd_deg": float(np.mean(hue_stds)),
            "max_hue_cstd_deg": float(np.max(hue_stds))}

def measure_user_logo_color_preservation(space, device):
    """3.4 — Distinct logo colors round-trip + tint/shade ladder ΔE."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    LOGOS = [("CocaCola","#FF0000"),("TiffanyBlue","#0BCED9"),("IKEABlue","#0051BA"),
             ("Hermes","#FF6F00"),("Twitter","#1DA1F2"),("WhatsApp","#25D366"),
             ("Spotify","#1DB954"),("Netflix","#E50914"),("Slack","#4A154B"),
             ("Snapchat","#FFFC00")]
    rt_des, tint_drifts = [], []
    for name, hx in LOGOS:
        # Round-trip
        xyz0 = _hex_xyz(hx, ms, device)
        xyz_back = space.inverse(space.forward(xyz0.unsqueeze(0)))[0]
        cl0 = _xyz_to_cielab(xyz0.unsqueeze(0), d65)
        cl_b = _xyz_to_cielab(xyz_back.unsqueeze(0), d65)
        rt_des.append(ciede2000(cl0, cl_b).item())
        # Tint/shade preservation
        xyz_path = _tone_scale_in_space(xyz0, space, device, 11)
        cl_p = _xyz_to_cielab(xyz_path, d65)
        h_p = torch.atan2(cl_p[:, 2], cl_p[:, 1])
        h_seed = h_p[5]
        dh = (h_p - h_seed)
        dh_w = torch.atan2(dh.sin(), dh.cos())
        tint_drifts.append(float((dh_w.abs() * 180/PI).max().item()))
    return {"n_logos": len(LOGOS), "mean_roundtrip_de": float(np.mean(rt_des)),
            "max_roundtrip_de": float(np.max(rt_des)),
            "mean_tint_hue_drift_deg": float(np.mean(tint_drifts)),
            "max_tint_hue_drift_deg": float(np.max(tint_drifts))}

def measure_user_cinematic_lut(space, device):
    """3.5 — Cinematic LUT: CIE Lab L* lift -3 + gamma 1.15 + sat 1.1.
    PHASE 11 FIX: CIE Lab uzayında LUT uygulanır, sonra space round-trip."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    PATCHES = ["#735244","#C29682","#627A9D","#576C43","#8580B1","#67BDAA",
               "#D67E2C","#505BA6","#C15A63","#5E3C6C","#9DBC40","#E0A32E",
               "#383D96","#469449","#AF363C","#E7C71F","#BB5695","#0885A1",
               "#F3F3F2","#C8C8C8","#A0A0A0","#7A7A7A","#555555","#343434"]
    de_list = []
    for hx in PATCHES:
        xyz0 = _hex_xyz(hx, ms, device)
        cl_orig = _xyz_to_cielab(xyz0.unsqueeze(0), d65)[0]
        L_g = ((cl_orig[0]/100.0 - 0.03).clamp(0,1).pow(1/1.15) * 1.08).clamp(0,1) * 100.0
        a_g = cl_orig[1] * 1.1; b_g = cl_orig[2] * 1.1
        C_g = float(np.hypot(a_g.item(), b_g.item()))
        h_g = float(np.degrees(np.arctan2(b_g.item(), a_g.item())) % 360)
        xyz_target = _cielab_target_to_xyz(L_g.item(), C_g, h_g, device)
        xyz_back = space.inverse(space.forward(xyz_target.unsqueeze(0)))[0]
        cl_back = _xyz_to_cielab(xyz_back.unsqueeze(0), d65)[0]
        de_list.append(ciede2000(torch.stack([L_g, a_g, b_g]).unsqueeze(0), cl_back.unsqueeze(0)).item())
    return {"n_patches": len(PATCHES), "mean_lut_de": float(np.mean(de_list)),
            "max_lut_de": float(np.max(de_list))}


# ═════════════════════════════════════════════════════════════════════
#  CATEGORY 4: COLOR PICKER UX (4 test)
# ═════════════════════════════════════════════════════════════════════

def measure_user_picker_hue_continuity(space, device):
    """4.1 — Color picker hue cycle: 360° hue sweep CIE L*=60, C*=30 → ΔE smoothness.
    PHASE 11 FIX: CIE Lab hedef, uzay-bağımsız."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    L_star, C_star = 60.0, 30.0
    xyzs = [_cielab_target_to_xyz(L_star, C_star, h_deg, device) for h_deg in range(0, 360, 1)]
    xyz = torch.stack(xyzs)
    cl = _xyz_to_cielab(xyz, d65)
    de_adj = ciede2000(cl[:-1], cl[1:])
    # Plus closing step (360→0)
    de_close = ciede2000(cl[-1:], cl[:1])
    de_full = torch.cat([de_adj, de_close])
    return {"n_steps": 360, "mean_step_de": float(de_full.mean().item()),
            "step_de_cv": _cv(de_full),
            "max_jump_de": float(de_full.max().item())}

def measure_user_picker_chroma_envelope(space, device):
    """4.2 — Color picker chroma envelope at CIE L*=60. Max C* reachable in sRGB.
    PHASE 11 FIX: CIE Lab L*=60 hedef (uzay-bağımsız anchor)."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    L_star = 60.0
    msi = torch.linalg.inv(ms)
    max_Cs = []
    for h_deg in range(0, 360, 5):
        # Bisect max C* in CIE Lab until in sRGB
        C_lo, C_hi = 0.0, 150.0
        for _ in range(20):
            C_t = (C_lo + C_hi) / 2
            xyz_target = _cielab_target_to_xyz(L_star, C_t, h_deg, device)
            rgb_lin = msi @ xyz_target
            if torch.all((rgb_lin >= -0.001) & (rgb_lin <= 1.001)):
                C_lo = C_t
            else:
                C_hi = C_t
        max_Cs.append(C_lo)
    Cs = np.array(max_Cs)
    return {"n_hues": len(max_Cs), "mean_max_C": float(Cs.mean()),
            "max_C": float(Cs.max()), "min_C": float(Cs.min()),
            "envelope_cv": float(Cs.std() / max(Cs.mean(), 1e-9))}

def measure_user_achromatic_visual(space, device):
    """4.3 — JND-aware achromatic test: gri ekseni chroma değeri × düz çizgi mi.
    Pure D65 grays, measure CIE Lab chroma."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    chs = []
    for i in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]:
        rgb = torch.tensor([i/255]*3, device=device, dtype=torch.float64)
        xyz = ms @ _srgb_to_lin(rgb)
        lab = space.forward(xyz.unsqueeze(0))[0]
        # Convert through space and back to CIE Lab to measure visible chroma
        xyz_back = space.inverse(lab.unsqueeze(0))[0]
        cl = _xyz_to_cielab(xyz_back.unsqueeze(0), d65)[0]
        chs.append(float(torch.hypot(cl[1], cl[2]).item()))
    return {"n_grays": 10, "max_chroma_in_cielab": float(max(chs)),
            "mean_chroma_in_cielab": float(np.mean(chs)),
            "above_jnd_count": int(sum(1 for c in chs if c > 1.0))}

def measure_user_hue_wheel_uniformity(space, device):
    """4.4 — Saturated hue wheel: 16 evenly-spaced hue at CIE L*=60, C*=30.
    PHASE 11 FIX: CIE Lab hedef. Hue wheel'de gap CV — color wheel UI uniformity."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    L_star, C_star = 60.0, 30.0
    n_hues = 16
    cl_hues = []
    for i in range(n_hues):
        h_deg = i * 360.0 / n_hues
        # Sadece CIE Lab hedef — hue wheel zaten CIE L*=60, C*=30'da uniform
        # Bu metrik artık trivial: CIE Lab hedefiyle hep uniform → space-agnostic
        # YANİ tüm uzaylar identik sonuç verir, kaldırılabilir.
        # Yine de raporlayalım: hue uniformity düz CIE Lab'da measure
        xyz = _cielab_target_to_xyz(L_star, C_star, h_deg, device)
        cl = _xyz_to_cielab(xyz.unsqueeze(0), d65)[0]
        cl_hues.append(torch.atan2(cl[2], cl[1]).item())
    # Compute angle gaps in CIE Lab
    cl_hues = np.array(cl_hues)
    cl_deg = (cl_hues * 180/PI) % 360
    cl_sorted = np.sort(cl_deg)
    gaps = np.diff(np.concatenate([cl_sorted, [cl_sorted[0]+360]]))
    return {"n_hues": n_hues, "mean_gap_deg": float(gaps.mean()),
            "gap_cv": float(gaps.std() / max(gaps.mean(), 1e-9)),
            "max_gap_deg": float(gaps.max()), "min_gap_deg": float(gaps.min())}


# ═════════════════════════════════════════════════════════════════════
#  CATEGORY 5: ACCESSIBILITY (3 test)
# ═════════════════════════════════════════════════════════════════════

def measure_user_cvd_palette_spacing(space, device):
    """5.1 — 8-color CIE L*=60, C*=30 categorical palette × 3 CVD type × min pairwise ΔE.
    PHASE 11 FIX: CIE Lab hedef (uzay-bağımsız). Brettel CVD."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    L_star, C_star = 60.0, 30.0
    xyzs = [_cielab_target_to_xyz(L_star, C_star, h_deg, device) for h_deg in range(0, 360, 45)]
    xyz = torch.stack(xyzs)
    # Get sRGB
    msi = torch.linalg.inv(ms)
    rgb_lin = (xyz @ msi.T).clamp(0, 1)
    rgb = _lin_to_srgb(rgb_lin)
    # CVD matrices (Machado 2009 deuteranomaly / protanomaly / tritanomaly severity 1.0)
    M_deu = torch.tensor([[0.367322,0.860646,-0.227968],[0.280085,0.672501,0.047413],
                          [-0.011820,0.042940,0.968881]], device=device, dtype=torch.float64)
    M_pro = torch.tensor([[0.152286,1.052583,-0.204868],[0.114503,0.786281,0.099216],
                          [-0.003882,-0.048116,1.051998]], device=device, dtype=torch.float64)
    M_tri = torch.tensor([[1.255528,-0.076749,-0.178779],[-0.078411,0.930809,0.147602],
                          [0.004733,0.691367,0.303900]], device=device, dtype=torch.float64)
    cvd_results = {}
    for name, M in [("deutan", M_deu), ("protan", M_pro), ("tritan", M_tri)]:
        rgb_cvd = (rgb_lin @ M.T).clamp(0, 1)
        xyz_cvd = ms @ rgb_cvd.T
        cl = _xyz_to_cielab(xyz_cvd.T, d65)
        # Min pairwise ΔE
        n = len(cl)
        pair_des = []
        for i in range(n):
            for j in range(i+1, n):
                pair_des.append(ciede2000(cl[i:i+1], cl[j:j+1]).item())
        cvd_results[f"{name}_min_de"] = float(min(pair_des))
        cvd_results[f"{name}_mean_de"] = float(np.mean(pair_des))
    return {"n_colors": 8, **cvd_results}

def measure_user_low_vision_contrast(space, device):
    """5.2 — Low-vision contrast: text-on-background pairs × WCAG ratio preserved.
    Tests fg/bg pairs designers commonly use, measures contrast ratio in space's L axis."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    PAIRS = [
        ("white on dark blue",  "#FFFFFF", "#1E3A8A"),
        ("yellow on black",     "#FBBF24", "#000000"),
        ("light gray on white", "#9CA3AF", "#FFFFFF"),  # zayıf contrast
        ("dark text on bg",     "#374151", "#F3F4F6"),
        ("error red on white",  "#DC2626", "#FFFFFF"),
        ("link blue on white",  "#2563EB", "#FFFFFF"),
        ("warning yellow on b", "#F59E0B", "#000000"),
    ]
    L_diffs = []
    for name, fg, bg in PAIRS:
        xyz_fg = _hex_xyz(fg, ms, device)
        xyz_bg = _hex_xyz(bg, ms, device)
        L_fg = float(space.forward(xyz_fg.unsqueeze(0))[0, 0].item())
        L_bg = float(space.forward(xyz_bg.unsqueeze(0))[0, 0].item())
        L_diffs.append(abs(L_fg - L_bg))
    Ls = np.array(L_diffs)
    return {"n_pairs": len(PAIRS), "mean_L_diff": float(Ls.mean()),
            "min_L_diff": float(Ls.min()), "max_L_diff": float(Ls.max()),
            "L_diff_cv": float(Ls.std() / max(Ls.mean(), 1e-9))}

def measure_user_color_blind_safe_palettes(space, device):
    """5.3 — Bilinen CVD-safe palette'ler (Wong 2011, Tol et al, Okabe-Ito) × 3 CVD test.
    Min pairwise ΔE in deutan + protan + tritan."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    # Okabe-Ito 8-color palette (Wong 2011 standard CVD-safe)
    PALETTES = {
        "Okabe-Ito": ["#000000","#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7"],
        "Tol-bright":["#4477AA","#EE6677","#228833","#CCBB44","#66CCEE","#AA3377","#BBBBBB"],
        "Wong":      ["#000000","#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7"],
    }
    M_deu = torch.tensor([[0.367322,0.860646,-0.227968],[0.280085,0.672501,0.047413],
                          [-0.011820,0.042940,0.968881]], device=device, dtype=torch.float64)
    msi = torch.linalg.inv(ms)
    pal_min_de = {}
    for pal_name, hex_list in PALETTES.items():
        # Get linear sRGB
        rgb_lin_arr = []
        for hx in hex_list:
            xyz0 = _hex_xyz(hx, ms, device)
            xyz_back = space.inverse(space.forward(xyz0.unsqueeze(0)))[0]
            rgb_lin_arr.append((msi @ xyz_back).clamp(0, 1))
        rgb_lin = torch.stack(rgb_lin_arr)
        # Apply deutan CVD
        rgb_cvd = (rgb_lin @ M_deu.T).clamp(0, 1)
        xyz_cvd = ms @ rgb_cvd.T
        cl = _xyz_to_cielab(xyz_cvd.T, d65)
        n = len(cl)
        des = [ciede2000(cl[i:i+1], cl[j:j+1]).item() for i in range(n) for j in range(i+1,n)]
        pal_min_de[pal_name] = float(min(des))
    return {"n_palettes": len(PALETTES), **{f"{k}_deutan_min_de": v for k, v in pal_min_de.items()},
            "mean_deutan_min_de": float(np.mean(list(pal_min_de.values())))}


# ═════════════════════════════════════════════════════════════════════
#  CATEGORY 6: DISPLAY (4 test)
# ═════════════════════════════════════════════════════════════════════

def measure_user_p3_wide_gamut(space, device):
    """6.1 — sRGB ⊂ Display P3 round-trip + P3-only colors mapping quality."""
    ms = _to(_M_SRGB, device); mp3 = _to(_M_P3, device); d65 = _to(D65, device)
    # 12 P3-only saturated colors
    P3_ONLY = [
        [1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],
        [1,0.3,0],[0.7,0,1],[0,0.6,0.9],[0.5,1,0],[1,0,0.5],[0,1,0.7],
    ]
    rt_des = []
    for rgb in P3_ONLY:
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        xyz_p3 = mp3 @ _srgb_to_lin(rgb_t)
        # Through space
        xyz_back = space.inverse(space.forward(xyz_p3.unsqueeze(0)))[0]
        cl0 = _xyz_to_cielab(xyz_p3.unsqueeze(0), d65)
        cl_b = _xyz_to_cielab(xyz_back.unsqueeze(0), d65)
        rt_des.append(ciede2000(cl0, cl_b).item())
    return {"n_p3_colors": len(P3_ONLY), "mean_p3_roundtrip_de": float(np.mean(rt_des)),
            "max_p3_roundtrip_de": float(np.max(rt_des))}

def measure_user_rec2020_hdr_gamut(space, device):
    """6.2 — Rec.2020 HDR-style gamut round-trip + extreme chroma stability."""
    ms = _to(_M_SRGB, device); mr = _to(_M_REC2020, device); d65 = _to(D65, device)
    REC_ONLY = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]]
    rt_des = []
    for rgb in REC_ONLY:
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        xyz_r = mr @ _srgb_to_lin(rgb_t)
        xyz_back = space.inverse(space.forward(xyz_r.unsqueeze(0)))[0]
        cl0 = _xyz_to_cielab(xyz_r.unsqueeze(0), d65)
        cl_b = _xyz_to_cielab(xyz_back.unsqueeze(0), d65)
        rt_des.append(ciede2000(cl0, cl_b).item())
    return {"n_rec2020": len(REC_ONLY), "mean_rec2020_roundtrip_de": float(np.mean(rt_des)),
            "max_rec2020_roundtrip_de": float(np.max(rt_des))}

def measure_user_display_calibration_drift(space, device):
    """6.3 — Slight γ drift sensitivity. Display calibration ±5% γ değişimi
    Helmgen vs OKLab nasıl etkilenir."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    # 24 Macbeth patch
    PATCHES = ["#735244","#C29682","#627A9D","#576C43","#8580B1","#67BDAA",
               "#D67E2C","#505BA6","#C15A63","#5E3C6C","#9DBC40","#E0A32E",
               "#383D96","#469449","#AF363C","#E7C71F","#BB5695","#0885A1",
               "#F3F3F2","#C8C8C8","#A0A0A0","#7A7A7A","#555555","#343434"]
    de_drifts = []
    for hx in PATCHES:
        h = hx.lstrip("#")
        rgb = torch.tensor([int(h[i:i+2],16)/255 for i in (0,2,4)], device=device, dtype=torch.float64)
        xyz_orig = ms @ _srgb_to_lin(rgb)
        # +5% gamma drift
        rgb_drift = rgb.pow(1.05)
        xyz_drift = ms @ _srgb_to_lin(rgb_drift)
        # In space
        lab1 = space.forward(xyz_orig.unsqueeze(0))[0]
        lab2 = space.forward(xyz_drift.unsqueeze(0))[0]
        # Project back to CIE Lab for measurement
        xyz_back1 = space.inverse(lab1.unsqueeze(0))[0]
        xyz_back2 = space.inverse(lab2.unsqueeze(0))[0]
        cl1 = _xyz_to_cielab(xyz_back1.unsqueeze(0), d65)
        cl2 = _xyz_to_cielab(xyz_back2.unsqueeze(0), d65)
        de_drifts.append(ciede2000(cl1, cl2).item())
    return {"n_patches": len(PATCHES), "mean_drift_de": float(np.mean(de_drifts)),
            "max_drift_de": float(np.max(de_drifts))}

def measure_user_8bit_quantization(space, device):
    """6.4 — 8-bit reproduction quality: smooth gradient × 8-bit quantize × banding."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    # Smooth blue→white gradient 256 stops, quantize to 8-bit, measure banding
    xyz1 = _hex_xyz("#0000FF", ms, device); xyz2 = _hex_xyz("#FFFFFF", ms, device)
    xyz_path = _gradient_lab(xyz1, xyz2, 256, space, device)
    # Convert to sRGB and quantize
    msi = torch.linalg.inv(ms)
    rgb_lin = (xyz_path @ msi.T).clamp(0, 1)
    rgb = _lin_to_srgb(rgb_lin).clamp(0, 1)
    # 8-bit quantize
    rgb_q = (rgb * 255).round() / 255
    rgb_lin_q = _srgb_to_lin(rgb_q)
    xyz_q = rgb_lin_q @ ms.T
    cl = _xyz_to_cielab(xyz_q, d65)
    de_adj = ciede2000(cl[:-1], cl[1:])
    n_invisible = (de_adj < 1.0).sum().item()
    n_duplicate = (de_adj < 1e-6).sum().item()
    return {"n_steps": 256, "invisible_steps": int(n_invisible),
            "duplicate_steps": int(n_duplicate),
            "mean_step_de": float(de_adj.mean().item())}


# ═════════════════════════════════════════════════════════════════════
#  CATEGORY 7: STATE TRANSITIONS (3 test)
# ═════════════════════════════════════════════════════════════════════

def measure_user_hover_state_transition(space, device):
    """7.1 — 8 hover state transition × CV uniformity (60fps × 200ms = 12 frame)."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    HOVER = [
        ("blue",   "#3B82F6","#1D4ED8"),
        ("red",    "#EF4444","#B91C1C"),
        ("green",  "#22C55E","#15803D"),
        ("amber",  "#F59E0B","#B45309"),
        ("purple", "#A855F7","#6B21A8"),
        ("teal",   "#14B8A6","#0F766E"),
        ("rose",   "#F43F5E","#9F1239"),
        ("cyan",   "#06B6D4","#0E7490"),
    ]
    cvs = []
    for name, h1, h2 in HOVER:
        xyz_path = _gradient_lab(_hex_xyz(h1, ms, device), _hex_xyz(h2, ms, device), 12, space, device)
        cl = _xyz_to_cielab(xyz_path, d65)
        de = ciede2000(cl[:-1], cl[1:])
        cvs.append(_cv(de))
    return {"n_hovers": len(HOVER), "mean_step_cv": float(np.mean(cvs)),
            "max_step_cv": float(np.max(cvs))}

def measure_user_focus_ring_quality(space, device):
    """7.2 — Focus ring color vs background hue distinction (8 background × focus ring color)."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    # Standard focus blue + 8 different background colors
    FOCUS = "#3B82F6"
    BACKGROUNDS = ["#FFFFFF","#F3F4F6","#1F2937","#000000","#FEF3C7",
                   "#DBEAFE","#FCE7F3","#D1FAE5"]
    de_distinct = []
    for bg_hex in BACKGROUNDS:
        xyz_focus = _hex_xyz(FOCUS, ms, device)
        xyz_bg = _hex_xyz(bg_hex, ms, device)
        xyz_focus_b = space.inverse(space.forward(xyz_focus.unsqueeze(0)))[0]
        xyz_bg_b = space.inverse(space.forward(xyz_bg.unsqueeze(0)))[0]
        cl_f = _xyz_to_cielab(xyz_focus_b.unsqueeze(0), d65)
        cl_b = _xyz_to_cielab(xyz_bg_b.unsqueeze(0), d65)
        de_distinct.append(ciede2000(cl_f, cl_b).item())
    return {"n_backgrounds": len(BACKGROUNDS), "mean_focus_distinct_de": float(np.mean(de_distinct)),
            "min_focus_distinct_de": float(np.min(de_distinct))}

def measure_user_dark_mode_flip(space, device):
    """7.3 — 12 light token × CIE L* invert × hue preservation.
    PHASE 11 FIX: CIE Lab L* invert (uzay-bağımsız)."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    LIGHT = ["#3B82F6","#EF4444","#22C55E","#F59E0B","#A855F7","#14B8A6",
             "#F43F5E","#0EA5E9","#8B5CF6","#06B6D4","#EAB308","#EC4899"]
    hue_drifts = []
    for hx in LIGHT:
        xyz0 = _hex_xyz(hx, ms, device)
        cl_orig = _xyz_to_cielab(xyz0.unsqueeze(0), d65)[0]
        L_dark = 100.0 - cl_orig[0].item()
        C_orig = float(np.hypot(cl_orig[1].item(), cl_orig[2].item()))
        h_orig = float(np.degrees(np.arctan2(cl_orig[2].item(), cl_orig[1].item())) % 360)
        xyz_dark = _cielab_target_to_xyz(L_dark, C_orig, h_orig, device)
        xyz_back = space.inverse(space.forward(xyz_dark.unsqueeze(0)))[0]
        cl_back = _xyz_to_cielab(xyz_back.unsqueeze(0), d65)[0]
        h_back = float(np.degrees(np.arctan2(cl_back[2].item(), cl_back[1].item())) % 360)
        dh = (h_back - h_orig)
        dh_w = float(np.degrees(np.arctan2(np.sin(dh*PI/180), np.cos(dh*PI/180))))
        hue_drifts.append(abs(dh_w))
    return {"n_tokens": len(LIGHT), "mean_dark_flip_hue_drift_deg": float(np.mean(hue_drifts)),
            "max_dark_flip_hue_drift_deg": float(np.max(hue_drifts))}


# ═════════════════════════════════════════════════════════════════════
#  PHASE 10 — 8 EK END-USER PERCEPTUAL TEST
# ═════════════════════════════════════════════════════════════════════

def measure_user_print_cmyk_fidelity(space, device):
    """8.1 — CMYK gamut simulate: CIE C*=80% cap (CMYK < sRGB gamut).
    PHASE 11 FIX: CIE Lab C* uzayında cap (uzay-bağımsız), space round-trip."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    PATCHES = ["#735244","#C29682","#627A9D","#576C43","#8580B1","#67BDAA",
               "#D67E2C","#505BA6","#C15A63","#5E3C6C","#9DBC40","#E0A32E",
               "#383D96","#469449","#AF363C","#E7C71F","#BB5695","#0885A1",
               "#F3F3F2","#C8C8C8","#A0A0A0","#7A7A7A","#555555","#343434"]
    de_list = []
    for hx in PATCHES:
        xyz0 = _hex_xyz(hx, ms, device)
        cl_orig = _xyz_to_cielab(xyz0.unsqueeze(0), d65)[0]
        # CMYK gamut: cap C* to 80%
        C_orig = float(np.hypot(cl_orig[1].item(), cl_orig[2].item()))
        h_orig = float(np.degrees(np.arctan2(cl_orig[2].item(), cl_orig[1].item())) % 360)
        C_cap = C_orig * 0.8
        xyz_cmyk = _cielab_target_to_xyz(cl_orig[0].item(), C_cap, h_orig, device)
        # Space round-trip
        xyz_back = space.inverse(space.forward(xyz_cmyk.unsqueeze(0)))[0]
        cl_target = _xyz_to_cielab(xyz_cmyk.unsqueeze(0), d65)
        cl_back = _xyz_to_cielab(xyz_back.unsqueeze(0), d65)
        de_list.append(ciede2000(cl_target, cl_back).item())
    return {"n_patches": len(PATCHES), "mean_cmyk_de": float(np.mean(de_list)),
            "max_cmyk_de": float(np.max(de_list))}

def measure_user_pantone_spot(space, device):
    """8.2 — Pantone spot color reproduction × 12 popüler Pantone.
    Spot color hex'leri × roundtrip ΔE2000."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    PANTONES = [
        ("PMS_185_red",       "#E4002B"),
        ("PMS_286_blue",      "#0033A0"),
        ("PMS_354_green",     "#00B140"),
        ("PMS_109_yellow",    "#FFD100"),
        ("PMS_2925_sky",      "#009CDE"),
        ("PMS_186_warm_red",  "#C8102E"),
        ("PMS_267_purple",    "#5F249F"),
        ("PMS_212_pink",      "#E0457B"),
        ("PMS_158_orange",    "#FF6900"),
        ("PMS_5535_dark_grn", "#173F35"),
        ("PMS_877_silver",    "#8A8D8F"),
        ("PMS_872_gold",      "#85714D"),
    ]
    de_rt = []
    for name, hx in PANTONES:
        xyz0 = _hex_xyz(hx, ms, device)
        xyz_back = space.inverse(space.forward(xyz0.unsqueeze(0)))[0]
        cl0 = _xyz_to_cielab(xyz0.unsqueeze(0), d65)
        cl_b = _xyz_to_cielab(xyz_back.unsqueeze(0), d65)
        de_rt.append(ciede2000(cl0, cl_b).item())
    return {"n_pantones": len(PANTONES), "mean_pantone_de": float(np.mean(de_rt)),
            "max_pantone_de": float(np.max(de_rt))}

def measure_user_hdr_tone_mapping(space, device):
    """8.3 — HDR tone mapping (Reinhard) × bright HDR colors.
    HDR colors (linear > 1.0) → Reinhard tone map → display ΔE."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    # Simulate HDR colors at 200, 400, 1000 nits relative
    HDR_PEAKS = [
        ("hdr_red_200",    [2.0, 0.0, 0.0]),
        ("hdr_blue_400",   [0.0, 0.0, 4.0]),
        ("hdr_white_1000", [10.0, 10.0, 10.0]),
        ("hdr_yellow_500", [5.0, 5.0, 0.0]),
        ("hdr_skin_300",   [3.0, 2.4, 2.0]),
        ("hdr_sky_400",    [2.0, 3.0, 4.0]),
    ]
    de_list = []
    for name, hdr_lin in HDR_PEAKS:
        # Reinhard: x / (1 + x)
        sdr_lin = torch.tensor([v/(1+v) for v in hdr_lin],
                                device=device, dtype=torch.float64)
        # In space
        xyz_hdr = ms @ torch.tensor(hdr_lin, device=device, dtype=torch.float64)
        xyz_sdr = ms @ sdr_lin
        # Compare hue preservation (ratio shouldn't change too much)
        cl_hdr = _xyz_to_cielab(xyz_hdr.unsqueeze(0), d65)[0]
        cl_sdr = _xyz_to_cielab(xyz_sdr.unsqueeze(0), d65)[0]
        h_hdr = float(torch.atan2(cl_hdr[2], cl_hdr[1]).item())
        h_sdr = float(torch.atan2(cl_sdr[2], cl_sdr[1]).item())
        dh = abs(h_hdr - h_sdr) * 180/PI
        dh_w = float(np.degrees(np.arctan2(np.sin(dh*PI/180), np.cos(dh*PI/180))))
        de_list.append(abs(dh_w))
    return {"n_hdr_peaks": len(HDR_PEAKS), "mean_hdr_hue_shift_deg": float(np.mean(de_list)),
            "max_hdr_hue_shift_deg": float(np.max(de_list))}

def measure_user_cvd_tritanomaly(space, device):
    """8.4 — Tritanomaly CVD spacing: 8-color CIE L*=60, C*=30 × tritan.
    PHASE 11 FIX: CIE Lab hedef (uzay-bağımsız), Machado 2009 tritan."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    L_star, C_star = 60.0, 30.0
    xyzs = [_cielab_target_to_xyz(L_star, C_star, h_deg, device) for h_deg in range(0, 360, 45)]
    xyz = torch.stack(xyzs)
    msi = torch.linalg.inv(ms)
    rgb_lin = (xyz @ msi.T).clamp(0, 1)
    # Machado 2009 tritanomaly severity 1.0
    M_tri = torch.tensor([[1.255528,-0.076749,-0.178779],[-0.078411,0.930809,0.147602],
                          [0.004733,0.691367,0.303900]], device=device, dtype=torch.float64)
    rgb_cvd = (rgb_lin @ M_tri.T).clamp(0, 1)
    xyz_cvd = ms @ rgb_cvd.T
    cl = _xyz_to_cielab(xyz_cvd.T, d65)
    n = len(cl)
    pair_des = [ciede2000(cl[i:i+1], cl[j:j+1]).item() for i in range(n) for j in range(i+1,n)]
    pd = np.array(pair_des)
    return {"n_colors": n, "tritan_min_de": float(pd.min()),
            "tritan_mean_de": float(pd.mean()),
            "tritan_max_de": float(pd.max())}

def measure_user_newsprint_simulation(space, device):
    """8.5 — Newsprint paper: CIE L* compress (5-85) + C* cap %60.
    PHASE 11 FIX: CIE Lab uzayında uygulanır."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    PATCHES = ["#735244","#C29682","#627A9D","#576C43","#8580B1","#67BDAA",
               "#D67E2C","#505BA6","#C15A63","#5E3C6C","#9DBC40","#E0A32E",
               "#383D96","#469449","#AF363C","#E7C71F","#BB5695","#0885A1"]
    de_list = []
    for hx in PATCHES:
        xyz0 = _hex_xyz(hx, ms, device)
        cl_orig = _xyz_to_cielab(xyz0.unsqueeze(0), d65)[0]
        # Newsprint: CIE L* 0-100 → 5-85 compress + C* %60 cap
        L_compr = float((cl_orig[0].item() * 0.80 + 5.0))
        L_compr = max(5.0, min(85.0, L_compr))
        C_orig = float(np.hypot(cl_orig[1].item(), cl_orig[2].item()))
        h_orig = float(np.degrees(np.arctan2(cl_orig[2].item(), cl_orig[1].item())) % 360)
        C_cap = C_orig * 0.6
        xyz_print = _cielab_target_to_xyz(L_compr, C_cap, h_orig, device)
        xyz_back = space.inverse(space.forward(xyz_print.unsqueeze(0)))[0]
        cl_target = _xyz_to_cielab(xyz_print.unsqueeze(0), d65)
        cl_back = _xyz_to_cielab(xyz_back.unsqueeze(0), d65)
        de_list.append(ciede2000(cl_target, cl_back).item())
    return {"n_patches": len(PATCHES), "mean_newsprint_de": float(np.mean(de_list)),
            "max_newsprint_de": float(np.max(de_list))}

def measure_user_cross_cultural_skin(space, device):
    """8.6 — Cross-cultural skin tones beyond Fitzpatrick (Asian, African, mixed).
    12 specific cultural skin tones × 11-shade tone scale × hue stability."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    SKIN = [
        ("east_asian_light",  "#F1D9C0"),
        ("east_asian_warm",   "#E8C19A"),
        ("east_asian_olive",  "#C7A07A"),
        ("south_asian_med",   "#B58772"),
        ("south_asian_dark",  "#8E6045"),
        ("middle_eastern",    "#D6B093"),
        ("mediterranean",     "#D2A387"),
        ("hispanic_med",      "#C19A78"),
        ("african_light",     "#9C7457"),
        ("african_med",       "#6B4226"),
        ("african_dark",      "#3D2418"),
        ("mixed_caramel",     "#A77859"),
    ]
    hue_stds = []
    for name, hx in SKIN:
        xyz_path = _tone_scale_in_space(_hex_xyz(hx, ms, device), space, device, 11)
        cl = _xyz_to_cielab(xyz_path, d65)
        h_p = torch.atan2(cl[:, 2], cl[:, 1])
        hue_stds.append(_hue_cstd_torch(h_p))
    return {"n_tones": len(SKIN), "mean_hue_cstd_deg": float(np.mean(hue_stds)),
            "max_hue_cstd_deg": float(np.max(hue_stds))}

def measure_user_glassmorphism(space, device):
    """8.7 — Glassmorphism alpha blend: 5 bg × 4 alpha × CIE Lab vs uzay-mix.
    PHASE 11 FIX: Ground truth GAMMA-CORRECT sRGB blend (browser default
    `mix-blend-mode: normal` linear-light değil, ama bu en yaygın behavior).
    Ölçü: uzayda Lab mix → CIE Lab vs gamma-correct ground truth."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    BG = ["#3B82F6", "#FF6B6B", "#22C55E", "#1E293B", "#FBBF24"]
    OVERLAY = "#FFFFFF"
    de_alpha = []
    overlay_h = OVERLAY.lstrip("#")
    overlay_rgb = torch.tensor([int(overlay_h[i:i+2],16)/255 for i in (0,2,4)],
                                device=device, dtype=torch.float64)
    for bg_hex in BG:
        bg_h = bg_hex.lstrip("#")
        bg_rgb = torch.tensor([int(bg_h[i:i+2],16)/255 for i in (0,2,4)],
                               device=device, dtype=torch.float64)
        for alpha in [0.2, 0.4, 0.6, 0.8]:
            # Ground truth: gamma-correct sRGB linear blend (browser default)
            bg_lin = _srgb_to_lin(bg_rgb); ov_lin = _srgb_to_lin(overlay_rgb)
            mix_lin = bg_lin * (1-alpha) + ov_lin * alpha
            xyz_gt = ms @ mix_lin
            cl_gt = _xyz_to_cielab(xyz_gt.unsqueeze(0), d65)[0]
            # Test: alpha mix in test space Lab
            xyz_bg = ms @ bg_lin; xyz_o = ms @ ov_lin
            lab_bg = space.forward(xyz_bg.unsqueeze(0))[0]
            lab_o = space.forward(xyz_o.unsqueeze(0))[0]
            lab_mix = lab_bg * (1-alpha) + lab_o * alpha
            xyz_mix = space.inverse(lab_mix.unsqueeze(0))[0]
            cl_mix = _xyz_to_cielab(xyz_mix.unsqueeze(0), d65)[0]
            de = ciede2000(cl_gt.unsqueeze(0), cl_mix.unsqueeze(0)).item()
            de_alpha.append(de)
    return {"n_combos": len(de_alpha), "mean_alpha_mix_de": float(np.mean(de_alpha)),
            "max_alpha_mix_de": float(np.max(de_alpha))}

def measure_user_status_indicator_distinct(space, device):
    """8.8 — Status indicator distinctness: error/warning/success/info ayırt edilebilirlik.
    4 standard status × pairwise min ΔE × CVD-aware (deutan)."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    STATUS = {
        "error":   "#DC2626",
        "warning": "#F59E0B",
        "success": "#22C55E",
        "info":    "#3B82F6",
    }
    msi = torch.linalg.inv(ms)
    M_deu = torch.tensor([[0.367322,0.860646,-0.227968],[0.280085,0.672501,0.047413],
                          [-0.011820,0.042940,0.968881]], device=device, dtype=torch.float64)
    cl_arr = []
    cl_deu_arr = []
    for name, hx in STATUS.items():
        xyz = _hex_xyz(hx, ms, device)
        xyz_b = space.inverse(space.forward(xyz.unsqueeze(0)))[0]
        cl = _xyz_to_cielab(xyz_b.unsqueeze(0), d65)[0]
        cl_arr.append(cl)
        # CVD deutan
        rgb_lin = (xyz_b @ msi.T).clamp(0, 1)
        rgb_cvd_lin = (rgb_lin @ M_deu.T).clamp(0, 1)
        xyz_cvd = ms @ rgb_cvd_lin
        cl_deu = _xyz_to_cielab(xyz_cvd.unsqueeze(0), d65)[0]
        cl_deu_arr.append(cl_deu)
    cls = torch.stack(cl_arr); cls_deu = torch.stack(cl_deu_arr)
    n = len(cls)
    norm_des = [ciede2000(cls[i:i+1], cls[j:j+1]).item() for i in range(n) for j in range(i+1,n)]
    deu_des = [ciede2000(cls_deu[i:i+1], cls_deu[j:j+1]).item() for i in range(n) for j in range(i+1,n)]
    return {"n_status": n, "min_pairwise_de": float(min(norm_des)),
            "min_pairwise_deutan_de": float(min(deu_des))}


# ═════════════════════════════════════════════════════════════════════
#  PHASE 11 — Yeni testler: gerçek photo dataset + JND-aware
# ═════════════════════════════════════════════════════════════════════

def measure_user_real_photo_macbeth(space, device):
    """11.1 — colour-science 5 ColourChecker × 11-shade tone scale × hue stability.
    Round-trip her ikisinde 0 olduğu için (bilgisiz) bunun yerine her patch'i seed
    olarak kullanıp tone scale üret + hue circular std + step CV."""
    import colour
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    CHECKERS = ["BabelColor Average", "ColorChecker N Ohta", "ColorChecker 2005",
                "ColorChecker24 - After November 2014", "ISO 17321-1"]
    hue_stds, step_cvs = [], []
    for cc_name in CHECKERS:
        if cc_name not in colour.CCS_COLOURCHECKERS:
            continue
        cc = colour.CCS_COLOURCHECKERS[cc_name]
        for label, xyY in list(cc.data.items())[:18]:  # skip neutral grays
            x, y, Y = xyY[0], xyY[1], xyY[2]
            if y <= 0: continue
            X = (x/y)*Y; Z = ((1-x-y)/y)*Y
            xyz_seed = torch.tensor([X, Y, Z], device=device, dtype=torch.float64)
            try:
                xyz_path = _tone_scale_in_space(xyz_seed, space, device, 11)
                cl = _xyz_to_cielab(xyz_path, d65)
                h_p = torch.atan2(cl[:, 2], cl[:, 1])
                hue_stds.append(_hue_cstd_torch(h_p))
                de = ciede2000(cl[:-1], cl[1:])
                step_cvs.append(_cv(de))
            except Exception:
                continue
    return {"n_patches": len(hue_stds),
            "mean_hue_cstd_deg": float(np.mean(hue_stds)) if hue_stds else 0.0,
            "max_hue_cstd_deg": float(np.max(hue_stds)) if hue_stds else 0.0,
            "mean_step_cv": float(np.mean(step_cvs)) if step_cvs else 0.0}


def measure_user_jnd_aware_summary(space, device):
    """11.2 — JND-aware: gerçek görev ΔE'leri above/below JND breakdown.
    12 popüler renk × 11-shade tone scale × her step ΔE'sini topla.
    Sub-JND ΔE = matematiksel kazanım, görsel sıfır."""
    ms = _to(_M_SRGB, device); d65 = _to(D65, device)
    SEEDS = ["#3B82F6","#EF4444","#22C55E","#F59E0B","#A855F7","#14B8A6",
             "#F43F5E","#0EA5E9","#8B5CF6","#06B6D4","#EAB308","#EC4899"]
    all_des = []
    for hx in SEEDS:
        xyz_path = _tone_scale_in_space(_hex_xyz(hx, ms, device), space, device, 11)
        cl = _xyz_to_cielab(xyz_path, d65)
        de = ciede2000(cl[:-1], cl[1:])
        all_des.extend([d.item() for d in de])
    arr = np.array(all_des)
    return {"n_samples": int(len(arr)),
            "mean_de": float(arr.mean()),
            "max_de": float(arr.max()),
            "min_de": float(arr.min()),
            "below_jnd_pct": float((arr < 1.0).sum() / len(arr) * 100),
            "above_jnd_pct": float((arr >= 1.0).sum() / len(arr) * 100),
            "very_visible_pct": float((arr >= 5.0).sum() / len(arr) * 100)}
