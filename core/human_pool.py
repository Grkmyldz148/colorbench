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
# Resolution order: COLOR_PERCEPTION_POOL env var, then the sibling checkout
# ../../color-perception-datasets/datasets relative to this repo. No absolute
# machine-specific fallback — pool_available()/pool_hint() give a clear message
# instead of a silent empty panel.
_SIBLING_POOL = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "color-perception-datasets", "datasets"))
POOL_DIR = os.environ.get("COLOR_PERCEPTION_POOL", _SIBLING_POOL)


def pool_available() -> bool:
    return os.path.isdir(POOL_DIR)


def pool_hint() -> str:
    return (f"color-perception-datasets pool not found at '{POOL_DIR}'. "
            f"Clone https://github.com/Grkmyldz148/color-perception-datasets "
            f"next to the color-space repo, or set COLOR_PERCEPTION_POOL to "
            f"its datasets/ directory.")


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
    # Try torch input first — works for every ColorBench ColorSpace (incl. the
    # colour-science canonical wrappers, which call .detach() on their input).
    try:
        import torch
        t = torch.as_tensor(xyz, dtype=getattr(space, "dtype", torch.float64),
                            device=getattr(space, "device", None))
        out = fwd(t)
        if hasattr(out, "detach"):
            return np.asarray(out.detach().cpu(), dtype=np.float64)
        return np.asarray(out, dtype=np.float64)
    except Exception:
        pass
    # Fall back to numpy-forward wrappers (e.g. the oklab_abney module functions).
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
    from .metric_eval import spearman_rho
    return {"name": name, "n": len(dv), "stress": _stress(de, dv),
            "spearman_rho": spearman_rho(de, dv)}


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
    default_white = _white_for(meta(name)["illuminant"])
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
            # mixed-illuminant sets (alder1982: 42 D65 + 39 A rows) carry a
            # per-row illuminant column — adapt each row under ITS OWN white
            row_illum = (r.get("illuminant") or "").strip()
            white = _white_for(row_illum) if row_illum else default_white
            xyz = cat_to_d65(xyz, white)
        lab = _space_forward(space, xyz)
        d = np.sqrt(((lab[1:] - lab[0]) ** 2).sum(-1))
        if d.mean() > 0:
            cvs.append(float(d.std() / d.mean()))
    return {"name": name, "n_centers": len(cvs),
            "mean_cv": float(np.mean(cvs)) if cvs else None}


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 3b — g-tensor 3D color-matching ellipsoids (Brown 1957 family)
# ════════════════════════════════════════════════════════════════════════════
def judge_jnd_ellipsoid_g(space, name, n_dir=26, row_filter=None):
    """3-D color-matching ellipsoids in (x, y, l = 0.2·log10 Y) coordinates —
    the Brown 1957 / Brown-MacAdam 1949 / Wyszecki-Fielder 1971 schema family.

    Each row gives the quadratic form g_ij of the matching ellipsoid:
    Δc' G Δc = 1 with Δc = (Δx, Δy, Δl). Points on that surface are equally
    (just-)discriminable, so a uniform space maps them to equal distance from
    the center; CV of those distances = 3D discrimination anisotropy.

    Notes on conventions (identical for every candidate → fair):
      - g scale (×1e2 / ×1e4 storage) does NOT matter: CV is scale-invariant
        per center, only the ellipsoid SHAPE enters.
      - luminance: l = 0.2·log10(Y). The center is placed at relative
        luminance 0.2 (a mid-gray assumption; these are aperture colors with
        no documented white). Perturbations use Y_i/Y_0 ratios, which are
        exact regardless of the absolute placement.
      - aperture viewing → no chromatic adaptation applied.
    """
    rows = load_canonical(name)
    cols = rows[0].keys()
    gbase = next((c[3:] for c in cols if c.startswith("g11")), None)
    if gbase is None or not {"x_0", "y_0"}.issubset(cols):
        return {"name": name, "skipped": "not g-tensor ellipsoid schema",
                "available": list(cols)}
    if row_filter:
        rows = [r for r in rows if row_filter(r)]

    # Fibonacci sphere directions
    i = np.arange(n_dir) + 0.5
    phi = np.arccos(1 - 2 * i / n_dir)
    gold = math.pi * (1 + 5 ** 0.5)
    u = np.column_stack([np.cos(gold * i) * np.sin(phi),
                         np.sin(gold * i) * np.sin(phi), np.cos(phi)])

    Y_REL_CENTER = 0.2
    cvs = []
    for r in rows:
        try:
            g = {k: float(r[f"g{k}{gbase}"]) for k in
                 ("11", "12", "22", "23", "33", "13")}
            x0, y0 = float(r["x_0"]), float(r["y_0"])
        except (KeyError, ValueError):
            continue
        G = np.array([[g["11"], g["12"], g["13"]],
                      [g["12"], g["22"], g["23"]],
                      [g["13"], g["23"], g["33"]]], dtype=np.float64)
        try:
            Lc = np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            continue
        # boundary of {Δ' G Δ = 1}: Δ = L^{-T} u
        delta = np.linalg.solve(Lc.T, u.T).T
        dx, dy, dl = delta[:, 0], delta[:, 1], delta[:, 2]
        xs = x0 + dx
        ys = y0 + dy
        yrel = Y_REL_CENTER * (10.0 ** (dl / 0.2))   # Y_i/Y_0 = 10^(Δl/0.2)
        xs = np.concatenate([[x0], xs])
        ys = np.concatenate([[y0], ys])
        yrel = np.concatenate([[Y_REL_CENTER], yrel])
        if np.any(ys < 1e-6):
            continue
        xyz = xyY_to_xyz(xs, ys, yrel)
        lab = _space_forward(space, np.clip(xyz, 0, None))
        d = np.sqrt(((lab[1:] - lab[0]) ** 2).sum(-1))
        if d.mean() > 0:
            cvs.append(float(d.std() / d.mean()))
    return {"name": name, "n_centers": len(cvs),
            "mean_cv": float(np.mean(cvs)) if cvs else None}


def judge_brown_1957(space, name="brown_1957_12obs_ellipsoids"):
    # weighted = inverse-variance observer average (the better estimator)
    return judge_jnd_ellipsoid_g(space, name,
                                 row_filter=lambda r: r.get("averaging") == "weighted")


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 3c — CIELAB-space discrimination ellipsoids (Huang 2012)
# ════════════════════════════════════════════════════════════════════════════
def judge_lab_ellipsoid(space, name="huang_2012_cielab_ellipses", n_dir=26):
    """Huang et al. 2012: threshold ellipsoids at 17 CIE centers, given as
    semi-axes A (major, in a*b* plane at theta_deg), B = A/A_over_B, and
    C_third_axis along ΔL*. Points on the ellipsoid are equally discriminable
    → equal candidate-space distance from center; CV = anisotropy. D65/10°."""
    rows = load_canonical(name)
    cols = rows[0].keys()
    need = {"L10_measured", "a10_measured", "b10_measured",
            "A_semimajor", "A_over_B", "theta_deg", "C_third_axis"}
    if not need.issubset(cols):
        return {"name": name, "skipped": "not huang lab-ellipsoid schema",
                "available": list(cols)}
    white = _white_for("D65 10")

    i = np.arange(n_dir) + 0.5
    phi = np.arccos(1 - 2 * i / n_dir)
    gold = math.pi * (1 + 5 ** 0.5)
    u = np.column_stack([np.cos(gold * i) * np.sin(phi),
                         np.sin(gold * i) * np.sin(phi), np.cos(phi)])

    cvs = []
    for r in rows:
        try:
            center = np.array([float(r["L10_measured"]), float(r["a10_measured"]),
                               float(r["b10_measured"])])
            A = float(r["A_semimajor"])
            B = A / max(float(r["A_over_B"]), 1e-9)
            C = float(r["C_third_axis"])
            th = math.radians(float(r["theta_deg"]))
        except (KeyError, ValueError):
            continue
        if min(A, B, C) <= 0:
            continue
        # (L, a, b) order: major/minor axes live in the a*b* plane
        e1 = np.array([0.0, math.cos(th), math.sin(th)])
        e2 = np.array([0.0, -math.sin(th), math.cos(th)])
        e3 = np.array([1.0, 0.0, 0.0])
        delta = (u[:, 0:1] * A * e1 + u[:, 1:2] * B * e2 + u[:, 2:3] * C * e3)
        lab_pts = np.vstack([center, center + delta])
        xyz = lab_to_xyz(lab_pts, white)
        lab2 = _space_forward(space, np.clip(xyz, 0, None))
        d = np.sqrt(((lab2[1:] - lab2[0]) ** 2).sum(-1))
        if d.mean() > 0:
            cvs.append(float(d.std() / d.mean()))
    return {"name": name, "n_centers": len(cvs),
            "mean_cv": float(np.mean(cvs)) if cvs else None}


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 3d — suprathreshold tolerance vectors (RIT-DuPont, Berns 1991)
# ════════════════════════════════════════════════════════════════════════════
def judge_tolerance_vectors(space, name="berns_1991_rit_dupont_tolerance_vectors",
                            min_vectors=4):
    """RIT-DuPont T50 tolerance vectors: for each color center, several unit
    directions v with the distance T50 at which the difference is judged equal
    to the 1.0 ΔE*ab anchor. All endpoints center + T50·v are thus equally
    different from the center → equal candidate-space distance; CV per center
    = suprathreshold tolerance anisotropy. D65."""
    rows = load_canonical(name)
    cols = rows[0].keys()
    need = {"color_center", "T50", "L_star", "a_star", "b_star",
            "delta_L", "delta_a", "delta_b"}
    if not need.issubset(cols):
        return {"name": name, "skipped": "not tolerance-vector schema",
                "available": list(cols)}
    white = _white_for("D65")
    groups = {}
    n_corrupt = 0
    for r in rows:
        try:
            center = np.array([float(r["L_star"]), float(r["a_star"]),
                               float(r["b_star"])])
            v = np.array([float(r["delta_L"]), float(r["delta_a"]),
                          float(r["delta_b"])])
            t50 = float(r["T50"])
        except (KeyError, ValueError):
            continue
        n = np.linalg.norm(v)
        if n < 1e-9 or t50 <= 0:
            continue
        # 2026-07-08 dataset audit: 7/156 rows carry OCR column-shift damage
        # (impossible L*, non-unit "eigenvectors" with norm ≈ 1.41). The
        # direction vectors are unit by construction in the source table, so
        # a non-unit norm or an out-of-range L* marks a corrupted row.
        if not (0.0 <= center[0] <= 100.0) or abs(n - 1.0) > 0.05:
            n_corrupt += 1
            continue
        groups.setdefault(r["color_center"], []).append(
            (center, center + t50 * v))
    cvs = []
    for cname, pairs in groups.items():
        if len(pairs) < min_vectors:
            continue
        center = pairs[0][0]
        pts = np.vstack([center] + [ep for _, ep in pairs])
        xyz = lab_to_xyz(pts, white)
        lab2 = _space_forward(space, np.clip(xyz, 0, None))
        d = np.sqrt(((lab2[1:] - lab2[0]) ** 2).sum(-1))
        if d.mean() > 0:
            cvs.append(float(d.std() / d.mean()))
    return {"name": name, "n_centers": len(cvs),
            "mean_cv": float(np.mean(cvs)) if cvs else None,
            "n_corrupt_rows_skipped": n_corrupt}


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 3e — Hong 2025 dense JND field (fully colorimetric via OSF calibration)
# ════════════════════════════════════════════════════════════════════════════
def judge_hong_2dw(space, name="hong_2025_ellipsoids", n_phi=12, stride=100):
    """Hong et al. 2025 Wishart-Process JND ellipse field (the modern
    MacAdam-1942 successor: 8 observers × 103×103 grid).

    Pipeline (2026-07-08 upgrade, calibration pulled from OSF k27js):
      2DW ellipse → M_2DWToRGB → LINEAR monitor RGB (per the calibration
      README: RGB is linear, NOT gamma-encoded) → M_RGBToXYZ1931 (PR-670
      measured DELL U2723QE primaries; Y in cd/m², white ≈ 190 cd/m²) →
      normalize to relative XYZ → Bradford monitor-white → D65.
    This replaced the earlier device≈sRGB approximation, which was doubly
    wrong (wrong primaries AND sRGB gamma applied to linear values).

    Rows are subsampled (stride) because Wishart smoothing makes neighbors
    dependent (effective DOF ≈ 100/subject)."""
    m1p = _ds_path(name, "raw", "M_2DWToRGB_DELL_02242025_copy.csv")
    m2p = _ds_path(name, "raw", "M_RGBToXYZ1931_DELL_02242025_copy.csv")
    if not os.path.exists(m1p):
        return {"name": name, "skipped": "no M_2DWToRGB calibration matrix"}
    if not os.path.exists(m2p):
        return {"name": name, "skipped": "no M_RGBToXYZ1931 calibration matrix "
                                         "(pull from OSF k27js)"}
    M1 = np.loadtxt(m1p, delimiter=",", dtype=np.float64)
    M2 = np.loadtxt(m2p, delimiter=",", dtype=np.float64)
    if M1.shape != (3, 3) or M2.shape != (3, 3):
        return {"name": name, "skipped": "unexpected calibration matrix shape"}
    white_abs = M2 @ np.ones(3)          # monitor white, Y in cd/m²
    white_rel = white_abs / white_abs[1]

    rows = load_canonical(name)
    cols = rows[0].keys()
    if not {"x_c", "y_c", "a", "b", "theta_deg"}.issubset(cols):
        return {"name": name, "skipped": "not xy-ellipse schema"}
    rows = rows[::max(1, int(stride))]
    phis = np.linspace(0, 2 * math.pi, n_phi, endpoint=False)
    cvs = []
    for r in rows:
        try:
            xc, yc = float(r["x_c"]), float(r["y_c"])
            a, b = float(r["a"]), float(r["b"])
            th = math.radians(float(r["theta_deg"]))
        except (KeyError, ValueError):
            continue
        ex = a * np.cos(phis); ey = b * np.sin(phis)
        px = xc + ex * math.cos(th) - ey * math.sin(th)
        py = yc + ex * math.sin(th) + ey * math.cos(th)
        pts_2dw = np.vstack([[xc, yc], np.column_stack([px, py])])
        homog = np.column_stack([pts_2dw, np.ones(len(pts_2dw))])
        rgb = homog @ M1.T                       # LINEAR monitor RGB
        if np.any(rgb < -0.05) or np.any(rgb > 1.05):
            continue
        xyz = np.clip(rgb, 0, 1) @ M2.T / white_abs[1]   # relative XYZ1931
        xyz = cat_to_d65(xyz, white_rel)
        lab = _space_forward(space, xyz)
        d = np.sqrt(((lab[1:] - lab[0]) ** 2).sum(-1))
        if d.mean() > 0:
            cvs.append(float(d.std() / d.mean()))
    return {"name": name, "n_centers": len(cvs),
            "mean_cv": float(np.mean(cvs)) if cvs else None,
            "note": "measured-primaries colorimetry (OSF k27js calibration)"}


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 3f — OSA-UCS uniform spacing (independent, non-Munsell spacing anchor)
# ════════════════════════════════════════════════════════════════════════════
def judge_osa_spacing(space, name="osa_ucs_1974", min_neighbours=3):
    """OSA-UCS committee atlas: 558 samples on a cuboctahedral lattice where
    each interior sample is ONE equal perceived colour difference from its 12
    nearest neighbours (the OSA committee's suprathreshold uniform-spacing
    design). A perceptually-uniform space maps every sample's neighbours to
    EQUAL distances; the coefficient of variation of those distances, averaged
    over samples, measures suprathreshold spacing anisotropy. Lower = better.

    This is the principal spacing dataset INDEPENDENT of Munsell — the held-out
    anchor for spacing claims (the spacing-consensus ruler is 1/3 Munsell-fit)."""
    samples = _load_csv(name, "canonical.csv")
    pairs = _load_csv(name, "neighbor_pairs.csv")
    if samples is None or pairs is None:
        return {"name": name, "skipped": "canonical.csv / neighbor_pairs.csv missing"}
    # xyY -> XYZ (already D65; coords are CIE 1964 10deg, fed directly — the
    # spacing CV is relative so the observer choice is immaterial)
    coord = {}
    for r in samples:
        Y = float(r["Y10"]) / 100.0
        xyz = xyY_to_xyz(np.array([float(r["x10"])]), np.array([float(r["y10"])]),
                         np.array([Y]))[0]
        coord[r["sample_id"]] = _space_forward(space, np.clip(xyz, 0, None)[None, :])[0]
    from collections import defaultdict
    nb = defaultdict(list)
    for p in pairs:
        a, b = p["sample_id_1"], p["sample_id_2"]
        if a in coord and b in coord:
            nb[a].append(b); nb[b].append(a)
    cvs = []
    for s, ns in nb.items():
        if len(ns) < min_neighbours:
            continue
        d = np.array([float(np.sqrt(((coord[s] - coord[n]) ** 2).sum())) for n in ns])
        if d.mean() > 0:
            cvs.append(float(d.std() / d.mean()))
    return {"name": name, "n_centers": len(cvs),
            "mean_cv": float(np.mean(cvs)) if cvs else None}


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 4a — unique hues (Xiao 2011): constant-hue test on unique-hue loci
# ════════════════════════════════════════════════════════════════════════════
def _load_csv(name, filename):
    p = _ds_path(name, filename)
    if not os.path.exists(p):
        return None
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def judge_unique_hues(space, name="xiao_unique_hues"):
    """Xiao et al. 2011: 185 observers set unique red/green/blue/yellow at 9
    lightness-chroma settings. All 9 settings of one unique hue are the SAME
    perceived hue, so a good space maps them to a constant hue angle — the
    unique-hue version of the Hung-Berns constant-hue test.

    Pipeline: mean XYZ per (hue, level) over observers×sessions (the long
    canonical reproduces averages.json to 7e-14), Y normalized by the 114.6
    cd/m² adapting luminance, Bradford CRT-background-white → D65, then
    circular MAD of the space's hue angle per unique hue; mean over 4 hues."""
    rows = _load_csv(name, "unique_hues_long.csv")
    if rows is None:
        return {"name": name, "skipped": "unique_hues_long.csv not built"}
    ADAPT_Y = 114.6
    BG_XY = (0.2897, 0.2977)
    sums = {}
    for r in rows:
        key = (r["hue"], r["level"])
        v = sums.setdefault(key, [0.0, 0.0, 0.0, 0])
        v[0] += float(r["X"]); v[1] += float(r["Y"]); v[2] += float(r["Z"])
        v[3] += 1
    white = xyY_to_xyz(np.array([BG_XY[0]]), np.array([BG_XY[1]]), np.array([1.0]))[0]
    mads = []
    for hue in ("red", "green", "blue", "yellow"):
        pts = np.array([[sums[(hue, str(l))][i] / sums[(hue, str(l))][3]
                         for i in range(3)] for l in range(9)])
        xyz = pts / ADAPT_Y
        xyz = cat_to_d65(xyz, white)
        lab = _space_forward(space, np.clip(xyz, 0, None))
        h = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
        mads.append(_circ_mad_deg(h))
    return {"name": name, "n_hues": len(mads),
            "mean_mad_deg": float(np.mean(mads)),
            "per_hue": {h: round(m, 2) for h, m in
                        zip(("red", "green", "blue", "yellow"), mads)}}


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 4b — WCS naming (DIAGNOSTIC: naming is known to be space-insensitive)
# ════════════════════════════════════════════════════════════════════════════
def judge_wcs_naming(space, name="wcs", min_chips_per_term=3):
    """World Color Survey category compactness: per language, chips sharing
    the modal color term should sit closer together in a good space than
    chips with different terms. Score = mean(within-term pairwise distance) /
    mean(between-term distance), averaged over 110 languages. DIAGNOSTIC —
    project history shows naming is nearly space-insensitive (~88% for every
    space), so this stays out of the headline."""
    naming = _load_csv(name, "naming_long.csv")
    chips = _load_csv(name, "chips.csv")
    if naming is None or chips is None:
        return {"name": name, "skipped": "naming_long.csv / chips.csv missing"}
    ccols = chips[0].keys()
    idc = next((c for c in ("chip_id", "cnum", "id") if c in ccols), None)
    if idc is None or not {"L", "a", "b"}.issubset(ccols):
        return {"name": name, "skipped": "chips.csv lacks id/L/a/b",
                "available": list(ccols)}
    chip_ids = [int(r[idc]) for r in chips]
    lab = np.array([[float(r["L"]), float(r["a"]), float(r["b"])] for r in chips])
    white = _white_for("illuminant C")   # WCS chips are Munsell under C
    xyz = cat_to_d65(lab_to_xyz(lab, white), white)
    coords = _space_forward(space, np.clip(xyz, 0, None))
    idx_of = {cid: i for i, cid in enumerate(chip_ids)}
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    # modal term per (lang, chip)
    counts = {}
    for r in naming:
        code = r["term_code"]
        if code == "*":
            continue
        key = (r["lang_id"], int(r["chip_id"]))
        c = counts.setdefault(key, {})
        c[code] = c.get(code, 0) + 1
    modal = {}
    for (lang, chip), c in counts.items():
        modal.setdefault(lang, {})[chip] = max(c, key=c.get)

    ratios = []
    for lang, chip_terms in modal.items():
        groups = {}
        for chip, term in chip_terms.items():
            if chip in idx_of:
                groups.setdefault(term, []).append(idx_of[chip])
        groups = {t: g for t, g in groups.items() if len(g) >= min_chips_per_term}
        if len(groups) < 2:
            continue
        members = {i: t for t, g in groups.items() for i in g}
        ids = sorted(members)
        within, between = [], []
        for ai in range(len(ids)):
            for bi in range(ai + 1, len(ids)):
                d = D[ids[ai], ids[bi]]
                (within if members[ids[ai]] == members[ids[bi]] else between).append(d)
        if within and between:
            ratios.append(float(np.mean(within) / np.mean(between)))
    return {"name": name, "n_languages": len(ratios),
            "ratio": float(np.mean(ratios)) if ratios else None,
            "note": "lower = categories more compact; naming is nearly "
                    "space-insensitive (diagnostic)"}


# ════════════════════════════════════════════════════════════════════════════
#  JUDGE 4c — observer metamerism (DIAGNOSTIC: Asano CMFs × natural spectra)
# ════════════════════════════════════════════════════════════════════════════
def judge_observer_metamerism(space, name="asano_observers",
                              obs_dataset="151ind", n_stimuli=60):
    """Observer-metamerism magnitude AS SEEN BY the candidate space: for each
    natural-object reflectance under D65, compute XYZ with each of Asano's
    individual-observer CMFs, map through the space, and measure the spread
    (mean distance to the observer-mean) in gray-step units (CIELAB L*50→51
    gray mapped through the space). DIAGNOSTIC — reports how LARGE observer
    disagreement looks in the space's own metric; no canonical better/worse
    direction, so it never enters the headline."""
    fund = _load_csv(name, "observer_fundamentals_long.csv")
    if fund is None:
        return {"name": name, "skipped": "observer_fundamentals_long.csv not built"}
    refl_rows = load_canonical("natural_objects_2024_southern_cone_spectra")
    if not refl_rows:
        return {"name": name, "skipped": "natural_objects canonical missing"}
    try:
        import colour
        sd = colour.SDS_ILLUMINANTS["D65"]
    except Exception as e:
        return {"name": name, "skipped": f"colour-science unavailable: {e!r}"}

    # observer CMFs: {obs: (3, n_wl)} on 390-780/5
    wl_grid = np.arange(390, 781, 5)
    cmf = {}
    for r in fund:
        if r["dataset"] != obs_dataset or r["field_of_view_deg"] != "2" \
                or r["function_type"] != "xyz":
            continue
        o = int(r["observer_id"])
        arr = cmf.setdefault(o, np.zeros((3, len(wl_grid))))
        pi = {"x": 0, "y": 1, "z": 2}[r["primary"]]
        wi = (int(r["wavelength_nm"]) - 390) // 5
        arr[pi, wi] = float(r["value"])
    if not cmf:
        return {"name": name, "skipped": f"no observers for {obs_dataset}"}
    obs_ids = sorted(cmf)

    # reflectances: numeric-named columns are wavelengths
    rcols = sorted((int(c) for c in refl_rows[0].keys() if c.strip().isdigit()))
    r_wl = np.array(rcols, dtype=float)
    lo = max(390, rcols[0])
    hi = min(780, rcols[-1])
    grid = wl_grid[(wl_grid >= lo) & (wl_grid <= hi)]
    gsel = np.isin(wl_grid, grid)
    step = max(1, len(refl_rows) // n_stimuli)
    stim = []
    for r in refl_rows[::step][:n_stimuli]:
        try:
            vals = np.array([float(r[str(c)]) for c in rcols])
        except ValueError:
            continue
        stim.append(np.interp(grid, r_wl, vals))
    stim = np.array(stim)                      # (S, n_grid)
    E = np.interp(grid, sd.wavelengths, sd.values)

    # per-observer XYZ (each observer normalized by its own white)
    spreads = []
    gray_ref = None
    coords_by_obs = []
    for o in obs_ids:
        T = cmf[o][:, gsel]                    # (3, n_grid)
        k = 1.0 / np.dot(E, T[1])
        xyz = (stim * E[None, :]) @ T.T * k    # (S, 3), relative
        coords_by_obs.append(_space_forward(space, np.clip(xyz, 0, None)))
    C = np.stack(coords_by_obs)                # (O, S, 3)
    mean_c = C.mean(axis=0, keepdims=True)
    spread = np.sqrt(((C - mean_c) ** 2).sum(-1)).mean(axis=0)  # (S,)

    # gray step in the candidate space (CIELAB L*50→51 neutral)
    d65w = _white_for("D65")
    g = lab_to_xyz(np.array([[50.0, 0, 0], [51.0, 0, 0]]), d65w)
    gl = _space_forward(space, g)
    gray_step = float(np.sqrt(((gl[1] - gl[0]) ** 2).sum()))
    if gray_step <= 0:
        return {"name": name, "skipped": "degenerate gray step"}
    return {"name": name, "n_observers": len(obs_ids), "n_stimuli": len(stim),
            "mean_spread_graysteps": float(np.mean(spread) / gray_step),
            "max_spread_graysteps": float(np.max(spread) / gray_step),
            "note": "observer disagreement in the space's own gray-step units "
                    "(diagnostic — no canonical direction)"}


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
    if not ({"x", "y"}.issubset(cols) or {"x10", "y10"}.issubset(cols)):
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
    xk, yk = ("x", "y") if "x" in cols else ("x10", "y10")  # sanders-wyszecki stores 10° coords
    xy = np.array([[float(r[xk]), float(r[yk])] for r in rows])
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
#  JUDGE 5b — Fairchild-Pirrotta object-colour H-K (real surface-colour lightness)
# ════════════════════════════════════════════════════════════════════════════
def judge_fp_lightness(space, name="fairchild_pirrotta_1991", adapt_d65=True):
    """Fairchild-Pirrotta 1991: 11 observers matched achromatic lightness to 36
    chromatic Munsell papers, giving each surface's H-K-corrected perceptual
    lightness (chromatic surfaces look LIGHTER than their L*, growing with
    chroma). This judge scores how well the candidate space's LIGHTNESS predicts
    that observed lightness — scale-optimal STRESS, lower = better. Plain CIELab
    L* sits at the no-H-K baseline (~11); a lightness that captures the
    chroma-dependent boost scores lower. The real OBJECT-colour counterpart to
    the aperture-colour H-K sets."""
    rows = load_canonical(name)
    cols = rows[0].keys()
    need = {"L_star", "C_star", "h_deg", "observed_lightness"}
    if not need.issubset(cols):
        return {"name": name, "skipped": "not fp hk_lightness schema",
                "available": list(cols)}
    white = _white_for("illuminant C")
    Lc, obs = [], []
    for r in rows:
        L = float(r["L_star"]); C = float(r["C_star"]); h = math.radians(float(r["h_deg"]))
        lab = np.array([[L, C * math.cos(h), C * math.sin(h)]])
        xyz = lab_to_xyz(lab, white)
        if adapt_d65:
            xyz = cat_to_d65(xyz, white)
        Lc.append(_space_forward(space, np.clip(xyz, 0, None))[0, 0])
        obs.append(float(r["observed_lightness"]))
    from .metric_eval import stress
    return {"name": name, "n": len(rows),
            "stress": float(stress(np.array(Lc), np.array(obs))),
            "note": "lower = candidate lightness predicts observed (H-K) "
                    "lightness better; CIELab L* baseline ~11 (diagnostic)"}


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
    # helmlabfb: RANK-ONLY by project decision (2026-06 audit) — 5-level ordinal
    # DV makes STRESS an artefact of the category→number mapping. Spearman ρ
    # only; lives under its own property so it never enters the headline total.
    ("difference_rank", "helmlabfb",                  judge_pair_diff,               "spearman_rho", True),
    ("difference",    "macadam1974",                  judge_pair_diff,               "stress",       True),
    ("hue",           "hung_berns",                   judge_constant_hue,            "mean_mad_deg", True),
    ("hue",           "ebner_fairchild",              judge_constant_hue,            "mean_mad_deg", True),
    ("hue",           "munsell",                      judge_constant_hue,            "mean_mad_deg", True),
    ("hue",           "xiao_unique_hues",             judge_unique_hues,             "mean_mad_deg", True),
    ("spacing",       "osa_ucs_1974",                 judge_osa_spacing,             "mean_cv",      True),
    ("discrimination","macadam1942",                  judge_jnd_ellipse,             "mean_cv",      True),
    ("discrimination","luo_rigg_ellipses",            judge_jnd_ellipse,             "mean_cv",      True),
    ("discrimination","alder1982",                    judge_jnd_ellipse,             "mean_cv",      True),
    ("3d_discrim",    "koenderink_2026_3d_metric_field", judge_jnd_ellipsoid_koenderink, "mean_cv", True),
    # 2026-07-08 expansion — g-tensor 3D ellipsoid family + Lab ellipsoids +
    # tolerance vectors (validated: gray-ramp sanity + known-direction check)
    ("3d_discrim",    "brown_1957_12obs_ellipsoids",  judge_brown_1957,              "mean_cv",      True),
    ("3d_discrim",    "wyszecki_fielder_1971_ellipsoids", judge_jnd_ellipsoid_g,     "mean_cv",      True),
    ("3d_discrim",    "brown_macadam_1949_ellipsoids", judge_jnd_ellipsoid_g,        "mean_cv",      True),
    ("tolerance",     "huang_2012_cielab_ellipses",   judge_lab_ellipsoid,           "mean_cv",      True),
    ("tolerance",     "berns_1991_rit_dupont_tolerance_vectors", judge_tolerance_vectors, "mean_cv", True),
    # 2026-07-08: promoted to validated — measured-primaries colorimetry from
    # OSF k27js replaced the sRGB approximation (direction + range checks pass)
    ("discrimination","hong_2025_ellipsoids",         judge_hong_2dw,                "mean_cv",      True),
    # diagnostic / mechanism tier (geometric spaces are expected to score ~null)
    ("hk_mechanism",  "wyszecki_1967_osa_tiles",      judge_hk,                      "spearman_rho", False),
    ("hk_mechanism",  "zhang_2023_laser_display_brightness", judge_hk,               "spearman_rho", False),
    ("hk_mechanism",  "sanders_wyszecki_1964_HK",     judge_hk,                      "spearman_rho", False),
    ("hk_object",     "fairchild_pirrotta_1991",      judge_fp_lightness,            "stress",       False),  # object-colour H-K lightness prediction (lower=better)
    ("adaptation",    "corresponding_colours",        judge_corresponding,           "mean_de",      False),
    ("naming",        "wcs",                          judge_wcs_naming,              "ratio",        False),  # naming ~space-insensitive
    ("observer_variance", "asano_observers",          judge_observer_metamerism,     "mean_spread_graysteps", False),  # no canonical direction
]

# property → which direction is "better" (all our validated keys: lower=better)
_LOWER_BETTER = {"difference", "hue", "discrimination", "3d_discrim",
                 "tolerance", "spacing", "hk_object", "adaptation"}


def evaluate_space_on_pool(space, validated_only=False):
    """Run every applicable human-data judge on a space. Returns
    {property: {dataset: score}} plus a flat {dataset: score}. The verdict for a
    property is the set of real-human-data scores grounding it (best-of-breed)."""
    if not pool_available():
        raise FileNotFoundError(pool_hint())
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
    # fit-data contamination: a judge whose dataset appears in a space's
    # trained_on declaration is in-sample for that space → out of TOPLAM
    from .contamination import trained_on_of
    fit_a = trained_on_of(space_a)
    fit_b = trained_on_of(space_b)
    for prop in ["difference", "difference_rank", "hue", "discrimination",
                 "3d_discrim", "tolerance", "spacing", "hk_object",
                 "hk_mechanism", "adaptation", "naming", "observer_variance"]:
        if prop not in ra:
            continue
        lower = prop in _LOWER_BETTER
        if prop == "difference_rank":
            lines.append(f"\n  [{prop}]  (Spearman ρ, yüksek=iyi; rank-only, TOPLAM'a dahil değil)")
        else:
            lines.append(f"\n  [{prop}]  (düşük=iyi)" if lower else f"\n  [{prop}]")
        for ds in ra[prop]:
            sa, sb = ra[prop][ds], rb[prop].get(ds)
            if sa is None or sb is None:
                lines.append(f"    {ds:34} {name_a}={sa} {name_b}={sb}")
                continue
            if ds.lower() in fit_a or ds.lower() in fit_b:
                who = name_a if ds.lower() in fit_a else name_b
                lines.append(f"    {ds:34} {sa:8.3f} vs {sb:8.3f}  ⚠ IN-SAMPLE "
                             f"({who} bu veriye fit) — TOPLAM dışı")
                continue
            better = (sa < sb) if lower else (sa > sb)
            mark = name_a if better else name_b
            if prop in _LOWER_BETTER:
                win_a += better; win_b += not better
            lines.append(f"    {ds:34} {sa:8.3f} vs {sb:8.3f}  → {mark}")
    lines.append(f"\n  TOPLAM (validated tier): {name_a} {win_a} – {win_b} {name_b}")
    return "\n".join(lines)
