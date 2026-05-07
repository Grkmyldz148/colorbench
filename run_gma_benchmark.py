#!/usr/bin/env python3
"""GMA (Gamut Mapping Algorithm) benchmark — Display-P3 → sRGB.

Compares 7 algorithms on 1000 P3-extreme test colors:
  · clip                — naive linear-RGB clamp (no perceptual awareness)
  · OKLab LCh           — CSS Color 4 reference algorithm
  · CIELab LCh          — legacy, 1976
  · Helmlab LCh         — production helmlab v0.11.9 GenSpace
  · IPT LCh             — Ebner & Fairchild (1998)
  · HelmCT-chroma LCh   — research champion (66W vs OKLab on ColorBench)
  · posthuewarp LCh     — research champion (68W vs OKLab on ColorBench, A=+0.03 φ=+90°)

Metrics use CAM16-UCS as the independent ground truth (any GMA space cannot win
its own metric — CIELab GMA scored 0° hue drift in CIELab tautologically).
CAM16-UCS is the CIE-recommended modern perceptual reference (CIE 224:2017).

  hue drift   (deg)  — |Δh| in CAM16-UCS, lower = better
  ΔE_CAM           — Euclidean distance in CAM16-UCS, lower = closer to source
  L* drift          — |ΔJ'| in CAM16-UCS lightness, lower = lightness preserved
  C* preserved (%)  — ratio of mapped chroma to original chroma in CAM16-UCS
  ΔE00 (CIELab)    — included for cross-reference

The point: when forced to drop chroma to enter sRGB, which space's LCh trajectory
preserves the most perceptual information?
"""

import os
import sys
import math
import json
import torch

torch.set_default_dtype(torch.float64)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_REPO, "research", "perflab", "mlp_helmct_chroma"))
sys.path.insert(0, os.path.join(_REPO, "research", "perflab", "mlp_helmct_chroma_posthuewarp"))

from core.spaces import OKLab, CIELab, HelmCT  # noqa: E402
from core.spaces_literature import IPT, CAM16UCS  # noqa: E402
from core.gamut_map import gamut_map, gamut_map_clip  # noqa: E402
from mlp_helmct_chroma_space import MLPHelmCTChroma  # noqa: E402
from mlp_helmct_chroma_posthuewarp_space import MLPHelmCTChromaPostHueWarp  # noqa: E402


# ─── Color matrices (D65) ──────────────────────────────────────────────────
M_SRGB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float64)
M_SRGB_INV = torch.linalg.inv(M_SRGB)

# Display-P3 (D65) — standard matrix from CSS Color 4 / Apple
M_P3 = torch.tensor([
    [0.4865709486482162, 0.26566769316909306, 0.1982172852343625],
    [0.2289745640697488, 0.6917385218365064,  0.079286914093745],
    [0.0,                0.04511338185890264, 1.0439443689009757],
], dtype=torch.float64)

D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)


# ─── CIE Lab ground-truth distance ─────────────────────────────────────────
def xyz_to_cielab(xyz):
    r = xyz / D65
    delta3 = (6.0 / 29.0) ** 3
    f = torch.where(r > delta3, r.clamp(min=1e-30).pow(1.0 / 3.0),
                    r / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def ciede2000(c1, c2):
    """ΔE00 (simplified form used elsewhere in colorbench)."""
    dL = c2[..., 0] - c1[..., 0]
    C1 = (c1[..., 1] ** 2 + c1[..., 2] ** 2).sqrt()
    C2 = (c2[..., 1] ** 2 + c2[..., 2] ** 2).sqrt()
    dC = C2 - C1
    dH = ((c2[..., 1] - c1[..., 1]) ** 2 + (c2[..., 2] - c1[..., 2]) ** 2 - dC ** 2).clamp(min=0).sqrt()
    SL = 1 + 0.015 * (c1[..., 0] - 50) ** 2 / (20 + (c1[..., 0] - 50) ** 2).sqrt()
    SC = 1 + 0.045 * C1
    SH = 1 + 0.015 * C1
    return ((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2).sqrt()


def hue_diff_deg(c1, c2):
    """Circular hue distance in CIE Lab, degrees."""
    h1 = torch.atan2(c1[..., 2], c1[..., 1]) * 180.0 / math.pi
    h2 = torch.atan2(c2[..., 2], c2[..., 1]) * 180.0 / math.pi
    d = (h2 - h1).abs()
    return torch.minimum(d, 360.0 - d)


# ─── Test set: P3-extreme colors that fall outside sRGB ────────────────────
def gen_test_colors(n=1000, seed=20260428):
    """Sample N P3-extreme colors that are OOG for sRGB.

    Strategy: random linear-P3 in [0,1]³, accept only those whose linear-sRGB
    representation has a component < -0.01 or > 1.01 (i.e. genuinely out-of-gamut,
    not just numerically near the boundary).
    """
    g = torch.Generator().manual_seed(seed)
    accepted = []
    target = n
    batch = 4000
    while sum(c.shape[0] for c in accepted) < target:
        rgb_p3 = torch.rand(batch, 3, generator=g, dtype=torch.float64)
        xyz = rgb_p3 @ M_P3.T
        rgb_srgb = xyz @ M_SRGB_INV.T
        oog = ((rgb_srgb < -0.01) | (rgb_srgb > 1.01)).any(dim=-1)
        accepted.append(xyz[oog])
    out = torch.cat(accepted)[:target]
    return out


# ─── Per-algorithm evaluation ──────────────────────────────────────────────
def evaluate_algorithm(name, mapper, xyz_src, ref_space):
    """`mapper` is a callable that takes xyz_src and returns mapped xyz.
    `ref_space` is the independent reference space (CAM16-UCS) used for judging."""
    xyz_dst = mapper(xyz_src)

    # Independent reference: CAM16-UCS
    ref_src = ref_space.forward(xyz_src)
    ref_dst = ref_space.forward(xyz_dst)
    hd = hue_diff_deg(ref_src, ref_dst)
    ld = (ref_src[:, 0] - ref_dst[:, 0]).abs()
    C_src = (ref_src[:, 1] ** 2 + ref_src[:, 2] ** 2).sqrt()
    C_dst = (ref_dst[:, 1] ** 2 + ref_dst[:, 2] ** 2).sqrt()
    de_cam = ((ref_src - ref_dst) ** 2).sum(dim=-1).sqrt()
    chroma_kept = (C_dst / C_src.clamp(min=1e-9) * 100.0).clamp(0, 200)

    # Cross-reference: CIELab ΔE00
    cl_src = xyz_to_cielab(xyz_src)
    cl_dst = xyz_to_cielab(xyz_dst)
    de00 = ciede2000(cl_src, cl_dst)

    rgb = xyz_dst @ M_SRGB_INV.T
    in_gamut = ((rgb >= -1e-3) & (rgb <= 1.0 + 1e-3)).all(dim=-1)

    return {
        "algorithm":          name,
        "hue_drift_mean":     hd.mean().item(),
        "hue_drift_p95":      hd.quantile(0.95).item(),
        "hue_drift_max":      hd.max().item(),
        "de_cam_mean":        de_cam.mean().item(),
        "de_cam_p95":         de_cam.quantile(0.95).item(),
        "de_cam_max":         de_cam.max().item(),
        "de00_mean":          de00.mean().item(),
        "de00_p95":           de00.quantile(0.95).item(),
        "L_drift_mean":       ld.mean().item(),
        "L_drift_max":        ld.max().item(),
        "chroma_kept_mean":   chroma_kept.mean().item(),
        "chroma_kept_median": chroma_kept.median().item(),
        "in_gamut_pct":       (in_gamut.float().mean().item()) * 100.0,
    }


def fmt_row(r):
    return (f"{r['algorithm']:14s}  "
            f"hue:μ={r['hue_drift_mean']:5.2f}° p95={r['hue_drift_p95']:5.2f}° max={r['hue_drift_max']:6.2f}°  "
            f"ΔE_CAM:μ={r['de_cam_mean']:5.2f} p95={r['de_cam_p95']:5.2f}  "
            f"ΔE00:μ={r['de00_mean']:5.2f} p95={r['de00_p95']:5.2f}  "
            f"L:μ={r['L_drift_mean']:4.2f}  "
            f"C*kept:μ={r['chroma_kept_mean']:5.1f}% med={r['chroma_kept_median']:5.1f}%  "
            f"in={r['in_gamut_pct']:5.1f}%")


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cpu")
    n = 1000
    print(f"Generating {n} P3-extreme test colors (OOG for sRGB)...")
    xyz_src = gen_test_colors(n=n)
    print(f"  → {xyz_src.shape[0]} colors loaded\n")

    helmct_json = os.path.join(_HERE, "..", "research", "checkpoints",
                               "genspace_v0.11.1.json")
    chroma_coef = os.path.join(_REPO, "research", "perflab", "mlp_helmct_chroma",
                               "helmct_chroma_v1.json")
    alpha0_rad = 2.0 * math.pi / 180.0
    phi_rad = 90.0 * math.pi / 180.0
    spaces = {
        "OKLab":          OKLab(device),
        "CIELab":         CIELab(device),
        "Helmlab":        HelmCT(helmct_json, device, label="Helmlab"),
        "IPT":            IPT(device),
        "HelmCT-chroma":  MLPHelmCTChroma(device, chroma_coef),
        "posthuewarp":    MLPHelmCTChromaPostHueWarp(
            device, chroma_coef, alpha0=alpha0_rad, A=0.030, phi=phi_rad,
        ),
    }
    ref_space = CAM16UCS(device)
    print(f"Reference (independent judge): {ref_space.name}\n")

    algorithms = []
    algorithms.append(("clip", lambda xyz: gamut_map_clip(xyz, M_SRGB_INV)))
    for sname, sp in spaces.items():
        algorithms.append((f"{sname} LCh",
                           lambda xyz, _sp=sp: gamut_map(xyz, _sp, M_SRGB_INV)))

    print("=" * 156)
    print(f"GMA benchmark — Display-P3 → sRGB, n={n} OOG colors  (lower=better for hue/ΔE00/L; higher=better for C*kept)")
    print("-" * 156)

    rows = []
    for name, mapper in algorithms:
        r = evaluate_algorithm(name, mapper, xyz_src, ref_space)
        rows.append(r)
        print(fmt_row(r))

    out_path = os.path.join(_HERE, "results", "gma_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n": n, "results": rows}, f, indent=2)
    print(f"\nsaved → {os.path.relpath(out_path, os.path.dirname(_HERE))}")

    # Champion summary
    print("\n=== Rankings ===")
    for key, label, lower_better in [
        ("hue_drift_mean", "hue drift μ (CAM16-UCS)", True),
        ("hue_drift_max",  "hue drift max (CAM16-UCS)", True),
        ("de_cam_mean",    "ΔE CAM16-UCS μ", True),
        ("de00_mean",      "ΔE00 (CIELab) μ", True),
        ("L_drift_mean",   "L drift μ (CAM16-UCS)", True),
        ("chroma_kept_mean", "chroma kept μ", False),
    ]:
        ranked = sorted(rows, key=lambda r: r[key], reverse=not lower_better)
        winner = ranked[0]
        runners = ", ".join(f"{r['algorithm']}={r[key]:.2f}" for r in ranked[1:3])
        print(f"  {label:18s} winner: {winner['algorithm']:14s} = {winner[key]:6.2f}    next: {runners}")


if __name__ == "__main__":
    main()
