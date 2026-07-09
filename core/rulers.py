"""Modular ruler layer — each metric measures with the ruler appropriate to WHAT
it measures. Rulers are FIXED instruments (fit to HUMAN data, not to ColorBench
metrics); the variable being tested is the candidate GENERATION space.

  - difference / discrimination ("how distinguishable / error magnitude")
        -> CIEDE2000 (chroma-compressed S_C=1+0.045C; correct for difference judgment)
  - spacing / uniformity ("are these steps equal in appearance"; generation regime)
        -> Perceptia-Spacing (our specialist uniform embedding, fit to Munsell
           human equal-step data, held-out). CIE76 kept as a no-compression baseline.

Why this layer exists: using a difference ruler (CIEDE2000) to score a SPACING
property biases the verdict — its chroma compression cancels a candidate space's
compression, systematically rewarding chroma-compressed generation spaces that look
visually non-uniform (proven: winner flips, GenSpace under-rated). See research
notebook 2026-05-30 ruler-mismatch audit + impact proof + perceptia_spacing_module.

Discipline: rulers come from HUMAN data (model-level non-circularity); when used to
test generation spaces, the ruler-fit / genspace-fit / ColorBench-test splits must be
disjoint (data-level non-circularity).
"""
import torch
from .gpu_de import ciede2000

# ── Perceptia-Spacing frozen coeffs (fit to Munsell held-out, ~9 params) ──────
_Q = 1.024          # lightness power
_P = 0.984          # chroma power
_HUE = (-0.004, 0.065, -0.222, 0.081, 0.043, 0.005, 0.026)  # w(h)=exp(harmonics)


def _spacing_coords(lab: torch.Tensor) -> torch.Tensor:
    L = lab[..., 0].clamp(min=1e-6)
    a = lab[..., 1]; b = lab[..., 2]
    C = torch.sqrt(a * a + b * b) + 1e-12
    h = torch.atan2(b, a)
    c = _HUE
    logw = (c[0] + c[1]*torch.cos(h) + c[2]*torch.sin(h)
            + c[3]*torch.cos(2*h) + c[4]*torch.sin(2*h)
            + c[5]*torch.cos(3*h) + c[6]*torch.sin(3*h))
    Cp = torch.exp(logw) * C ** _P
    Lp = L ** _Q
    return torch.stack([Lp, Cp * torch.cos(h), Cp * torch.sin(h)], dim=-1)


def perceptia_spacing(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """Spacing ruler: Euclidean distance in the Perceptia-Spacing uniform embedding."""
    d = _spacing_coords(lab1) - _spacing_coords(lab2)
    return torch.sqrt((d * d).sum(dim=-1).clamp(min=0))


def cie76(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """CIELAB ΔEab — Euclidean in CIE Lab (no-compression spacing baseline)."""
    d = lab1 - lab2
    return torch.sqrt((d * d).sum(dim=-1).clamp(min=0))


# ── Perceptia difference: CIEDE2000-form with free S-factor k's ──────────────
# Perceptia's PATCH (suprathreshold) member converged to k=(0.015,0.045,0.015) =
# canonical CIEDE2000 -> for suprathreshold difference, CIEDE2000 IS our best ruler.
# Perceptia's THRESHOLD (just-noticeable) member uses low k_C=0.0124 -> the better
# ruler for NEAR-THRESHOLD "is this step just visible/distinguishable" metrics.
def _de00_k(lab1, lab2, kL, kC, kH):
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = torch.hypot(a1, b1); C2 = torch.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - torch.sqrt(Cb**7 / (Cb**7 + 25.0**7 + 1e-30)))
    a1p = (1 + G) * a1; a2p = (1 + G) * a2
    C1p = torch.hypot(a1p, b1); C2p = torch.hypot(a2p, b2)
    h1p = torch.rad2deg(torch.atan2(b1, a1p)) % 360
    h2p = torch.rad2deg(torch.atan2(b2, a2p)) % 360
    dLp = L2 - L1; dCp = C2p - C1p
    diff = h2p - h1p
    diff = torch.where(diff.abs() <= 180, diff, torch.where(diff > 180, diff - 360, diff + 360))
    dhp = torch.where(C1p * C2p == 0, torch.zeros_like(diff), diff)
    dHp = 2 * torch.sqrt(C1p * C2p) * torch.sin(torch.deg2rad(dhp / 2))
    Lbp = (L1 + L2) / 2; Cbp = (C1p + C2p) / 2
    s = h1p + h2p
    hbp = torch.where(C1p * C2p == 0, s, torch.where((h1p - h2p).abs() <= 180, s/2,
                      torch.where(s < 360, (s+360)/2, (s-360)/2)))
    T = (1 - 0.17*torch.cos(torch.deg2rad(hbp-30)) + 0.24*torch.cos(torch.deg2rad(2*hbp))
         + 0.32*torch.cos(torch.deg2rad(3*hbp+6)) - 0.20*torch.cos(torch.deg2rad(4*hbp-63)))
    dTh = 30 * torch.exp(-((hbp-275)/25)**2)
    Rc = 2 * torch.sqrt(Cbp**7 / (Cbp**7 + 25.0**7 + 1e-30))
    Sl = 1 + (kL*(Lbp-50)**2) / torch.sqrt(20 + (Lbp-50)**2)
    Sc = 1 + kC*Cbp; Sh = 1 + kH*Cbp*T
    Rt = -torch.sin(torch.deg2rad(2*dTh)) * Rc
    return torch.sqrt((dLp/Sl)**2 + (dCp/Sc)**2 + (dHp/Sh)**2 + Rt*(dCp/Sc)*(dHp/Sh))


def perceptia_threshold(lab1, lab2):
    """Near-threshold difference ruler: Perceptia threshold member (low k_C=0.0124)."""
    return _de00_k(lab1, lab2, 0.0, 0.0124, 0.0016)


# ── Spacing CONSENSUS (fairness fix, 2026-06-08) ─────────────────────────────
# A single spacing ruler is a bias lever: Perceptia-Spacing (L^1.024/C^0.984) sits
# close to CIELAB and ranks gradients very differently from the third-party gold
# CAM16-UCS (proven: ranking flips by ruler). To remove the "whose ruler wins"
# lever, the spacing verdict is the CONSENSUS of three UNIFORM-appearance spaces —
# our Perceptia-Spacing, CAM16-UCS (Li 2017), and Jzazbz (Safdar 2017) — each
# normalized so a mid-gray ΔE*ab=1 step ≈ 1.0 (commensurate), then averaged.
# No single space's coordinate idiosyncrasy can swing the result.
def _cielab_to_xyz_d65(lab):
    import numpy as _np
    lab = _np.asarray(lab, dtype=_np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0; fx = a / 500.0 + fy; fz = fy - b / 200.0
    d = 6.0 / 29.0
    f = lambda t: _np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))
    w = _np.array([0.95047, 1.0, 1.08883])
    return _np.stack([f(fx), f(fy), f(fz)], axis=-1) * w


def _uniform_de(lab1, lab2, conv):
    """Euclidean ΔE in a colour-science uniform space (CAM16-UCS / Jzazbz)."""
    import numpy as _np
    x1 = _cielab_to_xyz_d65(_to_np(lab1)); x2 = _cielab_to_xyz_d65(_to_np(lab2))
    j1 = conv(_np.atleast_2d(x1)); j2 = conv(_np.atleast_2d(x2))
    return _np.sqrt(((j1 - j2) ** 2).sum(-1))


def _to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    import numpy as _np
    return _np.asarray(x, dtype=_np.float64)


# per-ruler scale = distance of a gray ΔE*ab=1 step (L 50→51), set lazily
_GRAY1 = ([50.0, 0.0, 0.0], [51.0, 0.0, 0.0])
_consensus_scales = {}


def _cam16ucs_de(lab1, lab2):
    import colour
    return _uniform_de(lab1, lab2, lambda xyz: colour.XYZ_to_CAM16UCS(xyz))


def _jzazbz_de(lab1, lab2):
    import colour
    return _uniform_de(lab1, lab2, lambda xyz: colour.XYZ_to_Jzazbz(xyz) * 100.0)


def _osa_ucs_de(lab1, lab2):
    # OSA-UCS (L,j,g) Euclidean — the OSA committee's visual equal-spacing
    # scale, fit to human uniform-spacing judgments and INDEPENDENT of Munsell.
    # Adding it dilutes Perceptia-Spacing's Munsell-fit weight (1/3 -> 1/4).
    import colour
    return _uniform_de(lab1, lab2, lambda xyz: colour.XYZ_to_OSA_UCS(xyz))


def spacing_consensus(lab1, lab2):
    """Normalized mean of {Perceptia-Spacing, CAM16-UCS, Jzazbz, OSA-UCS}
    spacing rulers. Returns a numpy array. Falls back to Perceptia-Spacing
    alone if colour-science is unavailable.

    OSA-UCS (2026-07) was added specifically to reduce the residual Munsell
    circularity: Perceptia-Spacing is Munsell-fit, so on Munsell-fit candidate
    generation spaces it can be in-sample. CAM16-UCS, Jzazbz, and OSA-UCS are
    all Munsell-INDEPENDENT, so 3/4 of the consensus is now non-Munsell and
    no single member's data can swing the spacing verdict."""
    import numpy as _np
    members = [("psp", perceptia_spacing)]
    try:
        import colour  # noqa: F401
        members += [("cam16", _cam16ucs_de), ("jzazbz", _jzazbz_de),
                    ("osa", _osa_ucs_de)]
    except Exception:
        pass
    vals = []
    for key, fn in members:
        if key not in _consensus_scales:
            import torch
            g1 = torch.tensor([_GRAY1[0]], dtype=torch.float64)
            g2 = torch.tensor([_GRAY1[1]], dtype=torch.float64)
            s = float(_to_np(fn(g1, g2)).ravel()[0])
            _consensus_scales[key] = s if s > 1e-9 else 1.0
        d = _to_np(fn(lab1, lab2)).ravel() / _consensus_scales[key]
        vals.append(d)
    out = _np.mean(_np.vstack(vals), axis=0)
    # drop-in: return torch tensor (on lab1's device) when inputs are torch
    try:
        import torch
        if isinstance(lab1, torch.Tensor):
            return torch.as_tensor(out, dtype=lab1.dtype, device=lab1.device)
    except Exception:
        pass
    return out


def _member_scale(key, fn):
    """Gray-step normalization scale for one consensus member (cached)."""
    if key not in _consensus_scales:
        import torch
        g1 = torch.tensor([_GRAY1[0]], dtype=torch.float64)
        g2 = torch.tensor([_GRAY1[1]], dtype=torch.float64)
        s = float(_to_np(fn(g1, g2)).ravel()[0])
        _consensus_scales[key] = s if s > 1e-9 else 1.0
    return _consensus_scales[key]


def spacing_members():
    """The individually-normalized members of the spacing consensus, for the
    RULER-SENSITIVITY check: a metric verdict is only trustworthy if it holds
    under EACH member ruler separately, not just under their average (a win
    that flips with the ruler is a property of the ruler, not the space).
    Returns {key: fn(lab1, lab2) -> numpy ΔE}, gray-step normalized."""
    members = {"psp": perceptia_spacing}
    try:
        import colour  # noqa: F401
        members["cam16"] = _cam16ucs_de
        members["jzazbz"] = _jzazbz_de
        members["osa"] = _osa_ucs_de
    except Exception:
        pass
    out = {}
    for key, fn in members.items():
        s = _member_scale(key, fn)
        out[key] = (lambda l1, l2, _fn=fn, _s=s: _to_np(_fn(l1, l2)).ravel() / _s)
    return out


RULERS = {
    "difference": ciede2000,            # suprathreshold = Perceptia patch (≈canonical) — best
    "threshold":  perceptia_threshold,  # near-threshold "just visible/distinguishable"
    "spacing":    spacing_consensus,    # FAIR: consensus of 3 uniform spaces (no single-ruler bias)
    "spacing_perceptia": perceptia_spacing,  # old single ruler (kept for reference/ablation)
    "spacing_baseline": cie76,          # no-compression baseline (CIELAB ΔEab)
}


def get_ruler(name: str):
    if name not in RULERS:
        raise KeyError(f"unknown ruler {name!r}; have {list(RULERS)}")
    return RULERS[name]
