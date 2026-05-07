"""Gamut geometry metrics — cusp scan + smoothness across sRGB/P3/Rec.2020.

For each gamut, sweep 360 hues × n_L × n_C in the test space's Lab grid,
inverse-map to gamut RGB, and check which (L, C) cells are inside the gamut
([0, 1]³ with small tolerance). Per-hue cusp = highest-chroma in-gamut cell.

Aggregate metrics (per gamut):
  valid_cusps          : hues with non-trivial cusps (0.05 < L < 0.99)
  monotonicity_violations : hues whose chroma profile dips post-cusp
  smoothness_max_jump  : largest cusp_L jump between adjacent hues
  cliff_max            : largest relative chroma drop 2 steps past cusp
  volume_fraction      : fraction of L×C grid in-gamut
  boundary_*           : non-monotonic drops on rising edge (gamut folds)
  anomalies            : per-hue cusp_L jumps > 0.2
  dead_zones           : consecutive hues with cusp_L < 0.05
  cusps                : full per-hue cusp record (360 entries)
"""
import math
import torch

from ._common import _M_SRGB_LIST, _M_P3_LIST, _M_REC2020_LIST, matrix

PI = math.pi


def _scan_gamut(space, gamut_matrix, n_hues, n_L, n_C):
    """Cusp scan for one gamut.

    Returns:
        cusp_L : (n_hues,)  L of highest-chroma in-gamut cell per hue
        cusp_C : (n_hues,)  matching chroma value
        mc_all : (n_hues, n_L)  per-(hue, L) maximum in-gamut chroma
        Ls     : (n_L,)  L-axis sample positions
    """
    dev, dt = space.device, space.dtype
    msi = torch.linalg.inv(gamut_matrix)
    Ls = torch.linspace(0.02, 0.998, n_L, device=dev, dtype=dt)
    Cs = torch.linspace(0.001, 0.5, n_C, device=dev, dtype=dt)

    cusp_L = torch.zeros(n_hues, device=dev, dtype=dt)
    cusp_C = torch.zeros(n_hues, device=dev, dtype=dt)
    mc_all = torch.zeros(n_hues, n_L, device=dev, dtype=dt)

    HB = 6  # hue batch size — limits VRAM at n_hues*n_L*n_C tensor build
    for hs in range(0, n_hues, HB):
        he = min(hs + HB, n_hues)
        nh = he - hs
        Le = Ls.view(1, n_L, 1).expand(nh, n_L, n_C)
        Ce = Cs.view(1, 1, n_C).expand(nh, n_L, n_C)
        angles = torch.arange(hs, he, device=dev, dtype=dt) * (2 * PI / n_hues)
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
        cusp_L[hs:he] = Ls[ci]
        cusp_C[hs:he] = mc[torch.arange(nh, device=dev), ci]

    return cusp_L, cusp_C, mc_all, Ls


def _per_gamut_metrics(cL, cC, mc_all, n_hues, n_L, dev):
    """Aggregate metrics for a single gamut from raw cusp scan output."""
    cL_np = cL.cpu()
    cC_np = cC.cpu()
    cL_list = cL_np.tolist()
    cC_list = cC_np.tolist()

    valid = ((cL > 0.05) & (cL < 0.99)).sum().item()

    # Post-cusp monotonicity (chroma should decrease past cusp)
    ci_all = mc_all.argmax(dim=1)
    mc_diff = mc_all[:, 1:] - mc_all[:, :-1]
    idx_range = torch.arange(n_L - 1, device=dev).unsqueeze(0)
    after_cusp = idx_range >= ci_all.unsqueeze(1)
    violations = ((mc_diff > 0.001) & after_cusp).any(dim=1).sum().item()

    # Cusp_L smoothness across hues (wrap-aware)
    jumps = (cL_np[1:] - cL_np[:-1]).abs()
    wrap_jump = abs(cL_np[-1] - cL_np[0])
    all_jumps = torch.cat([jumps, wrap_jump.unsqueeze(0)])

    # Cliff: relative chroma drop 2 steps past cusp, sampled every 5 hues
    sample_hues = torch.arange(0, n_hues, 5, device=dev)
    ci_s = ci_all[sample_hues]
    cc_s = mc_all[sample_hues, ci_s]
    post_idx = (ci_s + 2).clamp(max=n_L - 1)
    post_s = mc_all[sample_hues, post_idx]
    valid_mask = (ci_s < n_L - 2) & (cc_s > 0.01)
    cliffs = torch.where(
        valid_mask,
        (cc_s - post_s) / cc_s.clamp(min=1e-30),
        torch.zeros_like(cc_s),
    )
    max_cliff = max(cliffs.max().item() if cliffs.numel() > 0 else 0.0, 0.0)

    # In-gamut volume fraction
    volume_fraction = (mc_all > 0.001).float().mean().item()

    # Boundary continuity — drops on rising (pre-cusp) edge indicate gamut folds
    before_cusp = idx_range < ci_all.unsqueeze(1)
    rising_drops = (-mc_diff).clamp(min=0) * before_cusp.float()
    drops_per_hue = rising_drops.max(dim=1).values
    boundary_max_abs = drops_per_hue.max().item()
    boundary_mean_rel = drops_per_hue.mean().item()
    cusp_safe = cC.clamp(min=0.01)
    rel_per_hue = drops_per_hue / cusp_safe
    boundary_max_rel = rel_per_hue.max().item()
    boundary_bad_hues = (rel_per_hue > 0.05).sum().item()
    worst_hue_idx = rel_per_hue.argmax().item()
    worst_hue_jump = rel_per_hue[worst_hue_idx].item()

    # Cusp anomaly + dead zones (Python-level on pre-materialized lists)
    anomalies = []
    for i in range(n_hues):
        j = (i + 1) % n_hues
        jump = abs(cL_list[j] - cL_list[i])
        if jump > 0.2:
            anomalies.append({
                "hue_from": i, "hue_to": j,
                "L_from": cL_list[i], "L_to": cL_list[j],
                "jump": jump,
            })

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

    return {
        "valid_cusps": int(valid),
        "invalid_cusps": n_hues - int(valid),
        "monotonicity_violations": int(violations),
        "smoothness_max_jump": all_jumps.max().item(),
        "smoothness_mean_jump": all_jumps.mean().item(),
        "cliff_max": max_cliff,
        "volume_fraction": volume_fraction,
        "boundary_max_rel_jump": boundary_max_rel,
        "boundary_mean_rel_jump": boundary_mean_rel,
        "boundary_max_abs_jump": boundary_max_abs,
        "boundary_bad_hues": int(boundary_bad_hues),
        "boundary_worst_hue": int(worst_hue_idx),
        "boundary_worst_jump": worst_hue_jump,
        "anomalies": anomalies,
        "dead_zones": dead_zones,
        "cusps": [
            {"hue": i, "L": cL_list[i], "C": cC_list[i]}
            for i in range(n_hues)
        ],
    }


def measure_gamut(space, device=None, n_hues=360, n_L=300, n_C=200) -> dict:
    """Cusp scan + boundary geometry for sRGB / P3 / Rec.2020."""
    dev, dt = space.device, space.dtype
    matrices = {
        "sRGB":    matrix(_M_SRGB_LIST, dev, dt),
        "P3":      matrix(_M_P3_LIST, dev, dt),
        "Rec2020": matrix(_M_REC2020_LIST, dev, dt),
    }
    results = {}
    for name, mat in matrices.items():
        cL, cC, mc_all, _ = _scan_gamut(space, mat, n_hues, n_L, n_C)
        results[name] = _per_gamut_metrics(cL, cC, mc_all, n_hues, n_L, dev)
    return results
