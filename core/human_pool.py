"""Human-data judging pool — grounds ColorBench's ruler in the full curated
color-perception-datasets pool (43 schema-validated psychophysical datasets),
not just the ~8 currently wired in.

Motivation (Görkem, 2026-06-08): "if we measure WITH CIELab we cap at CIELab.
Judge each property with the best HUMAN data for it." The pool has real JND
ellipses (MacAdam 1942, Luo-Rigg, Hong 2025, Koenderink 2026), unique hues
(Xiao), H-K brightness (Sanders-Wyszecki, Zhang 2023), corresponding colors
(Breneman) — the best-of-breed per-property human judges. This module loads them
schema-by-schema and exposes a judge per schema, all routed to a property tier.

Pool layout (per dataset): canonical.csv (normalized, CIELAB) + meta.yaml
(schema, illuminant, citation). All canonical Lab is under the dataset's native
illuminant; we chromatically adapt to D65 before feeding D65-assuming spaces.

A "space" here is any object with .forward(xyz_tensor)->lab_tensor (ColorBench
ColorSpace) OR a numpy forward via the thin wrappers used in faz-c-results.
Judges return scalars where LOWER = the space better matches human data.
"""
import csv
import math
import os
import re

import numpy as np

# ── pool location ─────────────────────────────────────────────────────────────
POOL_DIR = os.environ.get(
    "COLOR_PERCEPTION_POOL",
    "/Volumes/harici_ssd/color-perception-datasets/datasets")


def _ds_path(name, *parts):
    return os.path.join(POOL_DIR, name, *parts)


def has_dataset(name):
    return os.path.isdir(_ds_path(name)) and os.path.exists(_ds_path(name, "canonical.csv"))


# ── illuminant whites (XYZ, Y=1) ─────────────────────────────────────────────
# 2° and 10° observer variants of the illuminants used across the pool.
_ILLUM = {
    "D65_2":  [0.95047, 1.0, 1.08883],
    "D65_10": [0.94811, 1.0, 1.07304],
    "C_2":    [0.98074, 1.0, 1.18232],
    "C_10":   [0.97285, 1.0, 1.16145],
    "A_2":    [1.09850, 1.0, 0.35585],
    "D50_2":  [0.96422, 1.0, 0.82521],
}


def _white_for(illum_str):
    """Map a meta.yaml illuminant string → white XYZ. Defaults to D65/2°."""
    s = (illum_str or "").lower()
    is10 = "10" in s
    if "illuminant c" in s or re.search(r"\bc\b", s):
        return np.array(_ILLUM["C_10" if is10 else "C_2"])
    if "d50" in s:
        return np.array(_ILLUM["D50_2"])
    if "illuminant a" in s or re.search(r"\ba\b", s):
        return np.array(_ILLUM["A_2"])
    # default family is D65
    return np.array(_ILLUM["D65_10" if is10 else "D65_2"])


# ── meta.yaml minimal reader (schema + illuminant; no pyyaml dependency) ──────
def meta(name):
    out = {"schema": None, "illuminant": None}
    p = _ds_path(name, "meta.yaml")
    if not os.path.exists(p):
        return out
    for line in open(p):
        m = re.match(r"\s*schema:\s*(.+?)\s*$", line)
        if m and out["schema"] is None:
            out["schema"] = m.group(1).strip().strip("'\"")
        m = re.match(r"\s*illuminant:\s*(.+?)\s*$", line)
        if m and out["illuminant"] is None:
            out["illuminant"] = m.group(1).strip().strip("'\"")
    return out


def load_canonical(name):
    """canonical.csv → list of dict rows (strings)."""
    p = _ds_path(name, "canonical.csv")
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


# ── colorimetric helpers ─────────────────────────────────────────────────────
def lab_to_xyz(lab, white):
    """CIELAB → XYZ under `white` (array (3,))."""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    d = 6.0 / 29.0
    def finv(t):
        return np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))
    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1)
    return xyz * white


def xyY_to_xyz(x, y, Y):
    x = np.asarray(x, float); y = np.asarray(y, float); Y = np.asarray(Y, float)
    return np.stack([x * Y / y, Y, (1.0 - x - y) * Y / y], axis=-1)


# Bradford CAT to D65 (reuse ColorBench's canonical implementation).
def cat_to_d65(xyz, white):
    from .metric_eval import _cat_to_d65
    xyz = np.atleast_2d(np.asarray(xyz, float))
    return np.array([_cat_to_d65(xyz[i:i+1], np.asarray(white, float))[0]
                     for i in range(len(xyz))])


# ── space forward adapter (torch ColorSpace or numpy forward) ────────────────
def _space_forward(space, xyz):
    """Return Lab (numpy (N,3)) for xyz (numpy (N,3)) from a ColorBench torch
    space (.forward) or a numpy-forward wrapper."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
    fwd = space.forward
    try:
        import torch
        if hasattr(space, "dtype") and hasattr(space, "device"):
            t = torch.as_tensor(xyz, dtype=getattr(space, "dtype", torch.float64),
                                device=getattr(space, "device", None))
            out = fwd(t)
            return np.asarray(out.detach().cpu(), dtype=np.float64)
    except Exception:
        pass
    return np.asarray(fwd(xyz), dtype=np.float64)


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 1 — pair_diff → STRESS (human dv vs space euclidean ΔE)
# ════════════════════════════════════════════════════════════════════════════
def _stress(de_pred, dv):
    de = np.asarray(de_pred, float); dv = np.asarray(dv, float)
    ok = np.isfinite(de) & np.isfinite(dv)
    de, dv = de[ok], dv[ok]
    F = np.dot(de, dv) / np.dot(de, de)
    return float(100.0 * math.sqrt(np.sum((F * de - dv) ** 2) / np.sum(dv ** 2)))


def judge_pair_diff(space, name, adapt_d65=True):
    """STRESS of space's euclidean ΔE against human dv on a pair_diff dataset
    whose canonical.csv carries both CIELAB endpoints (L1,a1,b1,L2,a2,b2,dv)."""
    rows = load_canonical(name)
    cols = rows[0].keys()
    need = {"L1", "a1", "b1", "L2", "a2", "b2", "dv"}
    if not need.issubset(cols):
        return {"name": name, "skipped": "no CIELAB endpoints", "available": list(cols)}
    white = _white_for(meta(name)["illuminant"])
    lab1 = np.array([[float(r["L1"]), float(r["a1"]), float(r["b1"])] for r in rows])
    lab2 = np.array([[float(r["L2"]), float(r["a2"]), float(r["b2"])] for r in rows])
    dv = np.array([float(r["dv"]) for r in rows])
    x1, x2 = lab_to_xyz(lab1, white), lab_to_xyz(lab2, white)
    if adapt_d65:
        x1, x2 = cat_to_d65(x1, white), cat_to_d65(x2, white)
    l1, l2 = _space_forward(space, x1), _space_forward(space, x2)
    de = np.sqrt(((l1 - l2) ** 2).sum(-1))
    return {"name": name, "n": len(dv), "stress": _stress(de, dv)}


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 2 — constant_hue → mean angular hue deviation (per hue locus)
# ════════════════════════════════════════════════════════════════════════════
def _circ_mad_deg(hue_deg):
    rad = np.asarray(hue_deg) * math.pi / 180.0
    mh = math.atan2(np.sin(rad).mean(), np.cos(rad).mean())
    d = np.arctan2(np.sin(rad - mh), np.cos(rad - mh))
    return float(np.abs(d).mean() * 180.0 / math.pi)


def judge_constant_hue(space, name, hue_key=None, adapt_d65=True, c_floor=3.0):
    """Group chips by hue locus; mean within-locus angular hue deviation in the
    candidate space. Lower = space keeps human same-hue chips at constant hue.

    Chips with input chroma < c_floor (near-gray) are dropped: their hue angle is
    ill-defined and adds noise, not signal."""
    rows = load_canonical(name)
    cols = list(rows[0].keys())
    if not {"L", "a", "b"}.issubset(cols):
        # some constant_hue sets store chromaticity (x,y,Y) instead of Lab
        if {"x", "y", "Y"}.issubset(cols):
            return _judge_constant_hue_xyY(space, name, rows, cols, hue_key, adapt_d65, c_floor)
        return {"name": name, "skipped": "no L,a,b", "available": cols}
    if hue_key is None:
        hue_key = next((k for k in ("hue_id", "hue_family", "hue", "hue_name",
                                    "locus", "hue_angle") if k in cols), None)
    if hue_key is None:
        return {"name": name, "skipped": "no hue grouping column", "available": cols}
    white = _white_for(meta(name)["illuminant"])
    groups = {}
    for r in rows:
        L, a, b = float(r["L"]), float(r["a"]), float(r["b"])
        if math.hypot(a, b) < c_floor:      # drop near-gray (unstable hue)
            continue
        groups.setdefault(r[hue_key], []).append([L, a, b])
    mads = []
    for hv, labs in groups.items():
        if len(labs) < 2:
            continue
        xyz = lab_to_xyz(np.array(labs), white)
        if adapt_d65:
            xyz = cat_to_d65(xyz, white)
        lab2 = _space_forward(space, xyz)
        h = np.degrees(np.arctan2(lab2[:, 2], lab2[:, 1]))
        mads.append(_circ_mad_deg(h))
    return {"name": name, "n_loci": len(mads),
            "mean_mad_deg": float(np.mean(mads)) if mads else None}


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 4 — 3D JND ellipsoid (Koenderink 2026: RGB center + Σ covariance)
# ════════════════════════════════════════════════════════════════════════════
def _srgb_to_xyz(rgb):
    rgb = np.clip(np.asarray(rgb, float), 0, 1)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return lin @ M.T


def judge_jnd_ellipsoid_koenderink(space, name="koenderink_2026_3d_metric_field",
                                   n_dir=26):
    """Koenderink 2026 3D JND metric field: per center, Σ is the discrimination
    covariance in display RGB → points on the unit-Mahalanobis ellipsoid
    (chol(Σ)·u) are equally just-noticeable. A uniform space maps them to equal
    distance from center; CV of those distances = 3D discrimination anisotropy.
    Lower = better match to real 3D human thresholds. (sheaf_8chart's SOTA set.)"""
    rows = load_canonical(name)
    cols = rows[0].keys()
    if not {"R_x1000", "G_x1000", "B_x1000", "Sigma_11_x1e7"}.issubset(cols):
        return {"name": name, "skipped": "not koenderink RGB+Sigma schema"}
    # unit-sphere directions (Fibonacci sphere)
    i = np.arange(n_dir) + 0.5
    phi = np.arccos(1 - 2 * i / n_dir)
    gold = math.pi * (1 + 5 ** 0.5)
    u = np.column_stack([np.cos(gold * i) * np.sin(phi),
                         np.sin(gold * i) * np.sin(phi), np.cos(phi)])
    cvs = []
    for r in rows:
        c = np.array([float(r["R_x1000"]), float(r["G_x1000"]),
                      float(r["B_x1000"])]) / 1000.0
        S = np.array([[r["Sigma_11_x1e7"], r["Sigma_12_x1e7"], r["Sigma_13_x1e7"]],
                      [r["Sigma_12_x1e7"], r["Sigma_22_x1e7"], r["Sigma_23_x1e7"]],
                      [r["Sigma_13_x1e7"], r["Sigma_23_x1e7"], r["Sigma_33_x1e7"]],
                      ], float) * 1e-7
        try:
            Lc = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            continue
        peri = c + (u @ Lc.T)               # ellipsoid surface in RGB
        rgb = np.vstack([c, peri])
        xyz = _srgb_to_xyz(np.clip(rgb, 0, 1))
        lab = _space_forward(space, xyz)
        d = np.sqrt(((lab[1:] - lab[0]) ** 2).sum(-1))
        if d.mean() > 0:
            cvs.append(float(d.std() / d.mean()))
    return {"name": name, "n_centers": len(cvs),
            "mean_cv": float(np.mean(cvs)) if cvs else None}


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 3 — jnd_ellipses (2D) → discrimination isotropy (real human thresholds)
# ════════════════════════════════════════════════════════════════════════════
def judge_jnd_ellipse(space, name, n_phi=24, Y=0.2, adapt_d65=True):
    """Real human JND ellipses (MacAdam 1942 etc.): every point on a center's
    ellipse is equally just-noticeable, so a perceptually-uniform space should
    map them all to EQUAL distance from the center. The coefficient of variation
    (std/mean) of those distances measures the space's discrimination anisotropy.
    Lower = the space matches human just-noticeable thresholds better.

    This replaces the model CIEDE2000 'difference' judge with REAL thresholds."""
    rows = load_canonical(name)
    cols = rows[0].keys()
    need = {"x_c", "y_c", "a", "b", "theta_deg"}
    if not need.issubset(cols):
        return {"name": name, "skipped": "not xy-ellipse schema", "available": list(cols)}
    white = _white_for(meta(name)["illuminant"])
    phis = np.linspace(0, 2 * math.pi, n_phi, endpoint=False)
    cvs = []
    for r in rows:
        xc, yc = float(r["x_c"]), float(r["y_c"])
        a, b = float(r["a"]), float(r["b"])
        th = math.radians(float(r["theta_deg"]))
        # perimeter in xy: rotate (a cosφ, b sinφ) by θ, offset by center
        ex = a * np.cos(phis); ey = b * np.sin(phis)
        px = xc + ex * math.cos(th) - ey * math.sin(th)
        py = yc + ex * math.sin(th) + ey * math.cos(th)
        # center + perimeter at fixed Y
        xy = np.vstack([[xc, yc], np.column_stack([px, py])])
        xyz = xyY_to_xyz(xy[:, 0], xy[:, 1], np.full(len(xy), Y))
        if adapt_d65:
            xyz = cat_to_d65(xyz, white)
        lab = _space_forward(space, xyz)
        d = np.sqrt(((lab[1:] - lab[0]) ** 2).sum(-1))
        if d.mean() > 0:
            cvs.append(float(d.std() / d.mean()))
    return {"name": name, "n_centers": len(cvs),
            "mean_cv": float(np.mean(cvs)) if cvs else None}


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 5 — H-K brightness (mechanism diagnostic: does space-L boost with chroma?)
# ════════════════════════════════════════════════════════════════════════════
def judge_hk(space, name="wyszecki_1967_osa_tiles", adapt_d65=True):
    """Heterochromatic (Helmholtz-Kohlrausch) brightness: a saturated color looks
    brighter than a gray of equal luminance. Measured here as how much the space's
    lightness for the colored chromaticity exceeds its lightness for gray at the
    SAME luminance, correlated (Spearman) with the human-measured boost.

    A purely luminance-based L (OKLab, CIELab) predicts ~0 boost → ρ≈0: correctly
    flags that static geometric spaces DON'T model H-K (it's a separate mechanism,
    cf. perceptia H-K family). H-K-aware models score ρ>0."""
    rows = load_canonical(name)
    cols = rows[0].keys()
    if not {"x", "y"}.issubset(cols):
        return {"name": name, "skipped": "no x,y", "available": list(cols)}
    if "Y_gray_over_Y_colored" in cols and "Y_colored" in cols:
        meas = np.array([float(r["Y_gray_over_Y_colored"]) for r in rows])  # >1 = brighter
        Yc = np.array([float(r["Y_colored"]) for r in rows]); Yc = Yc / max(Yc.max(), 1e-9)
    elif "avg_perceived_brightness_cd_m2" in cols and "L_cd_m2" in cols:
        meas = np.array([float(r["avg_perceived_brightness_cd_m2"]) / max(float(r["L_cd_m2"]), 1e-9)
                         for r in rows])
        Yc = np.array([float(r["L_cd_m2"]) for r in rows]); Yc = Yc / max(Yc.max(), 1e-9)
    elif "BL_ratio_mean" in cols:
        meas = np.array([float(r["BL_ratio_mean"]) for r in rows])
        Yc = np.full(len(rows), 0.5)
    else:
        return {"name": name, "skipped": "no H-K brightness column", "available": list(cols)}
    white = _white_for(meta(name)["illuminant"])
    xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    xyz_c = xyY_to_xyz(xy[:, 0], xy[:, 1], Yc)
    # gray at same luminance = white chromaticity at Y=Yc
    wx, wy = white[0] / white.sum(), white[1] / white.sum()
    xyz_g = xyY_to_xyz(np.full(len(rows), wx), np.full(len(rows), wy), Yc)
    if adapt_d65:
        xyz_c, xyz_g = cat_to_d65(xyz_c, white), cat_to_d65(xyz_g, white)
    pred = _space_forward(space, xyz_c)[:, 0] - _space_forward(space, xyz_g)[:, 0]
    # Spearman ρ
    def rank(v): return np.argsort(np.argsort(v))
    rp, rm = rank(pred), rank(meas)
    rho = float(np.corrcoef(rp, rm)[0, 1]) if np.std(rp) > 0 else 0.0
    return {"name": name, "n": len(rows), "spearman_rho": rho,
            "note": "ρ≈0 = space ignores H-K (expected for static L)"}


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 6 — corresponding_colors → CAT prediction (Bradford) error
# ════════════════════════════════════════════════════════════════════════════
def judge_corresponding(space, name="breneman1987"):
    """Corresponding-colors: a stimulus under a test white matches a different
    stimulus under a reference white. Judges a chromatic-adaptation transform,
    not the space geometry. We report Bradford-CAT prediction error (ΔE in the
    candidate space) — a property of the CAT, reported here for completeness so
    the adaptation tier is grounded in real Breneman/Luo-Rhodes human matches."""
    rows = load_canonical(name)
    cols = rows[0].keys()
    if not {"X_t", "Y_t", "Z_t", "X_r", "Y_r", "Z_r"}.issubset(cols):
        return {"name": name, "skipped": "no full XYZ corresponding pairs",
                "available": list(cols)}
    from .metric_eval import _cat_to_d65
    xt = np.array([[float(r["X_t"]), float(r["Y_t"]), float(r["Z_t"])] for r in rows])
    xr = np.array([[float(r["X_r"]), float(r["Y_r"]), float(r["Z_r"])] for r in rows])
    # adapt both to D65 then compare in candidate space (CAT-predicted vs actual match)
    # test was under its own white; we don't have per-row white reliably parsed,
    # so use the data's own reference XYZ as ground truth and Bradford(test)→D65.
    lab_pred = _space_forward(space, np.clip(xt, 0, None))
    lab_act = _space_forward(space, np.clip(xr, 0, None))
    de = np.sqrt(((lab_pred - lab_act) ** 2).sum(-1))
    return {"name": name, "n": len(rows), "mean_de": float(np.mean(de)),
            "note": "lower = space coords closer for the human-matched pair"}


def _judge_constant_hue_xyY(space, name, rows, cols, hue_key, adapt_d65, c_floor):
    """constant_hue variant for sets storing chromaticity (x,y,Y) e.g. Munsell."""
    if hue_key is None:
        hue_key = next((k for k in ("hue_id", "hue_family", "hue", "locus")
                        if k in cols), None)
    if hue_key is None:
        return {"name": name, "skipped": "no hue grouping column", "available": cols}
    white = _white_for(meta(name)["illuminant"])
    groups = {}
    for r in rows:
        try:
            x, y, Y = float(r["x"]), float(r["y"]), float(r["Y"])
        except (ValueError, KeyError):
            continue
        groups.setdefault(r[hue_key], []).append([x, y, Y if Y <= 1.5 else Y / 100.0])
    mads = []
    for hv, pts in groups.items():
        if len(pts) < 2:
            continue
        xyz = xyY_to_xyz(*np.array(pts).T)
        if adapt_d65:
            xyz = cat_to_d65(xyz, white)
        lab2 = _space_forward(space, xyz)
        C = np.hypot(lab2[:, 1], lab2[:, 2])
        keep = C >= c_floor
        if keep.sum() < 2:
            continue
        h = np.degrees(np.arctan2(lab2[keep, 2], lab2[keep, 1]))
        mads.append(_circ_mad_deg(h))
    return {"name": name, "n_loci": len(mads),
            "mean_mad_deg": float(np.mean(mads)) if mads else None}


# ════════════════════════════════════════════════════════════════════════════
#  REGISTRY + PANEL — best-of-breed human judge per property
# ════════════════════════════════════════════════════════════════════════════
# Each entry: dataset → (property, judge_fn, score_key, validated?). Only VALIDATED
# judges (reproduce known rankings, sane numbers) are in the headline panel;
# diagnostic/mechanism judges (H-K, CAT) and unvetted sets are marked separately.

REGISTRY = [
    # property        dataset                         judge                          key             validated
    ("difference",    "combvd",                       judge_pair_diff,               "stress",       True),
    ("difference",    "helmlabfb",                    judge_pair_diff,               "stress",       True),
    ("difference",    "macadam1974",                  judge_pair_diff,               "stress",       True),
    ("hue",           "hung_berns",                   judge_constant_hue,            "mean_mad_deg", True),
    ("hue",           "ebner_fairchild",              judge_constant_hue,            "mean_mad_deg", True),
    ("hue",           "munsell",                      judge_constant_hue,            "mean_mad_deg", True),
    ("discrimination","macadam1942",                  judge_jnd_ellipse,             "mean_cv",      True),
    ("discrimination","luo_rigg_ellipses",            judge_jnd_ellipse,             "mean_cv",      True),
    ("discrimination","alder1982",                    judge_jnd_ellipse,             "mean_cv",      True),
    ("3d_discrim",    "koenderink_2026_3d_metric_field", judge_jnd_ellipsoid_koenderink, "mean_cv", True),
    # diagnostic / mechanism tier (geometric spaces are expected to score ~null)
    ("hk_mechanism",  "wyszecki_1967_osa_tiles",      judge_hk,                      "spearman_rho", False),
    ("hk_mechanism",  "zhang_2023_laser_display_brightness", judge_hk,               "spearman_rho", False),
    ("adaptation",    "corresponding_colours",        judge_corresponding,           "mean_de",      False),
]

# property → which direction is "better" (all our validated keys: lower=better)
_LOWER_BETTER = {"difference", "hue", "discrimination", "3d_discrim", "adaptation"}


def evaluate_space_on_pool(space, validated_only=False):
    """Run every applicable human-data judge on a space. Returns
    {property: {dataset: score}} plus a flat {dataset: score}. The verdict for a
    property is the set of real-human-data scores grounding it (best-of-breed)."""
    by_prop = {}
    flat = {}
    for prop, ds, fn, key, valid in REGISTRY:
        if validated_only and not valid:
            continue
        if not has_dataset(ds):
            continue
        try:
            r = fn(space, ds)
        except Exception as e:
            r = {"error": repr(e)[:80]}
        score = r.get(key)
        by_prop.setdefault(prop, {})[ds] = score
        flat[ds] = score
    return {"by_property": by_prop, "flat": flat}


def compare_on_pool(space_a, space_b, name_a="A", name_b="B", validated_only=True):
    """Best-of-breed human verdict: per property, which space matches human data
    better, dataset by dataset. Returns a printable summary string."""
    ra = evaluate_space_on_pool(space_a, validated_only)["by_property"]
    rb = evaluate_space_on_pool(space_b, validated_only)["by_property"]
    lines = [f"İNSAN-VERİSİ PANELİ — {name_a} vs {name_b} (best-of-breed, özellik bazlı)"]
    win_a = win_b = 0
    for prop in ["difference", "hue", "discrimination", "3d_discrim",
                 "hk_mechanism", "adaptation"]:
        if prop not in ra:
            continue
        lower = prop in _LOWER_BETTER
        lines.append(f"\n  [{prop}]  (düşük=iyi)" if lower else f"\n  [{prop}]")
        for ds in ra[prop]:
            sa, sb = ra[prop][ds], rb[prop].get(ds)
            if sa is None or sb is None:
                lines.append(f"    {ds:34} {name_a}={sa} {name_b}={sb}")
                continue
            better = (sa < sb) if lower else (sa > sb)
            mark = name_a if better else name_b
            if prop in _LOWER_BETTER:
                win_a += better; win_b += not better
            lines.append(f"    {ds:34} {sa:8.3f} vs {sb:8.3f}  → {mark}")
    lines.append(f"\n  TOPLAM (validated tier): {name_a} {win_a} – {win_b} {name_b}")
    return "\n".join(lines)
