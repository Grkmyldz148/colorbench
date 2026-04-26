#!/usr/bin/env python3
"""Near-monochrome palette classification for color spaces.

Answers: in space S's coordinates, how 1-dimensional does palette P look?

For each (space, palette) it reports:
  pc1_var_pct  — PCA variance explained by the dominant axis in space coords (Lab)
  cielab_hue_range_deg — ground-truth hue spread (CIE Lab) for context
  cielab_chroma_mean   — ground-truth mean chroma (CIE Lab) for context
  step_de2k_cv         — coefficient of variation of consecutive ciede2000 steps in space coords
  pc23_share_pct       — fraction of variance NOT on PC1 (hue/chroma residual the space sees)

Higher pc1_var_pct  => space "sees" the palette as more nearly 1D (near-monochrome).
Higher pc23_share   => space resolves chromatic spread between the swatches.

A native monochrome palette (pure grays) should give pc1 ~ 100% in any space; deviation
is numerical noise. Solarized's base sequence is the test case Dima asked about.
"""

import os
import sys
import math
import json
import torch

torch.set_default_dtype(torch.float64)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from core.spaces import OKLab, CIELab, HelmCT  # noqa: E402

# ---------------------------------------------------------------------------
# Palettes (hex). Source citations in inline comments.
# ---------------------------------------------------------------------------

PALETTES = {
    # Ethan Schoonover, https://ethanschoonover.com/solarized/ — 8 base tones,
    # designer-intended near-monochrome warm/cool ramp.
    "Solarized base (8)": [
        "#002b36", "#073642", "#586e75", "#657b83",
        "#839496", "#93a1a1", "#eee8d5", "#fdf6e3",
    ],
    # Solarized accents — 8 chromatic colors, control case (NOT monochrome).
    "Solarized accent (8)": [
        "#b58900", "#cb4b16", "#dc322f", "#d33682",
        "#6c71c4", "#268bd2", "#2aa198", "#859900",
    ],
    # Nord — Polar Night (4) + Snow Storm (3), https://www.nordtheme.com/
    "Nord polar+snow (7)": [
        "#2e3440", "#3b4252", "#434c5e", "#4c566a",
        "#d8dee9", "#e5e9f0", "#eceff4",
    ],
    # GitHub Primer neutrals — https://primer.style/foundations/color
    "GitHub neutrals (10)": [
        "#f6f8fa", "#eaeef2", "#d0d7de", "#afb8c1", "#8c959f",
        "#6e7781", "#57606a", "#424a53", "#32383f", "#24292f",
    ],
    # True grays — control, must give pc1 ~ 100% in any sane space.
    "Pure grays (9)": [
        "#101010", "#303030", "#505050", "#707070", "#909090",
        "#a8a8a8", "#c0c0c0", "#d8d8d8", "#f0f0f0",
    ],
}

# sRGB → XYZ (D65), Bradford-adapted matrix per IEC 61966-2-1 / Lindbloom.
_M_SRGB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float64)
_D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)


def _srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))


def _hex_to_xyz(hexstr, device):
    h = hexstr.lstrip("#")
    rgb = torch.tensor([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)],
                       dtype=torch.float64, device=device)
    return _srgb_to_linear(rgb) @ _M_SRGB.to(device).T


def _xyz_to_cielab(xyz):
    d65 = _D65.to(xyz.device)
    r = xyz / d65
    delta3 = (6.0 / 29.0) ** 3
    f = torch.where(r > delta3, r.pow(1.0 / 3.0),
                    r / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def _ciede2000(cl1, cl2):
    """Same simplified ciede2000 form used elsewhere in colorbench."""
    dL = cl2[..., 0] - cl1[..., 0]
    C1 = (cl1[..., 1] ** 2 + cl1[..., 2] ** 2).sqrt()
    C2 = (cl2[..., 1] ** 2 + cl2[..., 2] ** 2).sqrt()
    dC = C2 - C1
    dH = ((cl2[..., 1] - cl1[..., 1]) ** 2
          + (cl2[..., 2] - cl1[..., 2]) ** 2 - dC ** 2).clamp(min=0).sqrt()
    SL = 1 + 0.015 * (cl1[..., 0] - 50) ** 2 / (20 + (cl1[..., 0] - 50) ** 2).sqrt()
    SC = 1 + 0.045 * C1
    SH = 1 + 0.015 * C1
    return ((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2).sqrt()


def _hue_range_deg(cielab):
    """Span of hue across the palette, accounting for circular wrap.
    Returns the smallest arc covering all hues, in degrees [0,180]."""
    h = torch.atan2(cielab[:, 2], cielab[:, 1]) * 180.0 / math.pi  # [-180,180]
    h = (h % 360.0).sort().values
    gaps = torch.cat([h[1:] - h[:-1], (h[0] + 360.0 - h[-1]).unsqueeze(0)])
    largest_gap = gaps.max().item()
    return 360.0 - largest_gap


def _pca_variance_ratios(lab):
    """Variance ratios of PCA components in `lab` coords (centered, full Lab)."""
    x = lab - lab.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (x.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov)  # ascending
    eig = eig.flip(0).clamp(min=0)
    return (eig / eig.sum()).tolist()


def evaluate(space, hexes, device):
    xyz = torch.stack([_hex_to_xyz(h, device) for h in hexes])     # (N,3)
    lab = space.forward(xyz)                                        # (N,3) in space coords
    cielab = _xyz_to_cielab(xyz)                                    # (N,3) ground truth

    var_ratios = _pca_variance_ratios(lab)

    # consecutive ciede2000 steps (sort by space's L first so order is deterministic)
    order = lab[:, 0].argsort()
    cl_sorted = cielab[order]
    de = _ciede2000(cl_sorted[:-1], cl_sorted[1:])
    de_cv = (de.std() / de.mean()).item() if de.mean() > 1e-9 else float("nan")

    chroma_cielab = (cielab[:, 1] ** 2 + cielab[:, 2] ** 2).sqrt()

    return {
        "pc1_var_pct":          var_ratios[0] * 100.0,
        "pc23_share_pct":       (var_ratios[1] + var_ratios[2]) * 100.0,
        "cielab_hue_range_deg": _hue_range_deg(cielab),
        "cielab_chroma_mean":   chroma_cielab.mean().item(),
        "cielab_L_range":       (cielab[:, 0].max() - cielab[:, 0].min()).item(),
        "step_de2k_cv":         de_cv,
        "step_de2k_mean":       de.mean().item(),
    }


def fmt(r):
    return (f"PC1={r['pc1_var_pct']:6.2f}%  "
            f"PC2+3={r['pc23_share_pct']:5.2f}%  "
            f"hueRange={r['cielab_hue_range_deg']:5.1f}°  "
            f"meanC={r['cielab_chroma_mean']:5.1f}  "
            f"Lrange={r['cielab_L_range']:5.1f}  "
            f"stepΔE00 μ={r['step_de2k_mean']:5.2f} CV={r['step_de2k_cv']*100:5.1f}%")


def main():
    device = torch.device("cpu")
    spaces = [OKLab(device), CIELab(device)]

    helmct_json = os.path.join(_HERE, "..", "research", "checkpoints",
                               "genspace_v0.11.1.json")
    if os.path.exists(helmct_json):
        try:
            spaces.append(HelmCT(helmct_json, device, label="HelmCT(v0.11.1)"))
        except Exception as e:
            print(f"[skip HelmCT: {e}]")

    print("=" * 116)
    print(f"{'palette':24s} {'space':20s}  metrics")
    print("-" * 116)

    out = {}
    for pname, hexes in PALETTES.items():
        out[pname] = {}
        for sp in spaces:
            r = evaluate(sp, hexes, device)
            out[pname][sp.name] = r
            print(f"{pname:24s} {sp.name:20s}  {fmt(r)}")
        print()

    # JSON dump for reproducibility
    out_path = os.path.join(_HERE, "results", "near_monochrome.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved → {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
