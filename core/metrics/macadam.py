"""MacAdam ellipse isotropy — real 1942 JND ellipses.

Every point on a MacAdam ellipse is one just-noticeable difference from its
center, so a perceptually-uniform space must map each ellipse PERIMETER to
equal distance from its center: max/min ratio 1.0 is the perceptual target.

The pre-2026-07 version perturbed each center by a fixed CIRCLE in xy and
rewarded ratio→1 — i.e. it rewarded spaces for being isotropic in raw xy,
which means IGNORING MacAdam anisotropy. That was anti-perceptual (a correct
space scored ~the ellipse elongation, 2-3×). This version uses the full
a/b/theta ellipse geometry, so ratio→1 genuinely means "matches human
discrimination thresholds".

Ellipses were measured under illuminant C; points are Bradford-adapted to
D65 before entering the candidate space (all candidate spaces are D65-native).
"""
import math

import numpy as np
import torch

from ..constants import MACADAM_ELLIPSES

# Illuminant C white point (x=0.31006, y=0.31616) as XYZ at Y=1
_WHITE_C = np.array([0.31006 / 0.31616, 1.0,
                     (1.0 - 0.31006 - 0.31616) / 0.31616], dtype=np.float64)


def measure_macadam_isotropy(space, device=None) -> dict:
    """Anisotropy of the candidate space against the 25 real MacAdam ellipses.

    Per ellipse: sample the JND perimeter, measure distance from center in the
    candidate space, report max/min ratio (1.0 = perfect threshold match) and
    CV of the distances. Aggregates: mean/max ratio, mean CV.
    """
    from ..metric_eval import _cat_to_d65

    dev, dt = space.device, space.dtype
    n_dirs = 16
    Y_val = 0.5
    phis = np.linspace(0.0, 2.0 * math.pi, n_dirs, endpoint=False)

    ratios = []
    cvs = []
    for xc, yc, a, b, theta_deg in MACADAM_ELLIPSES:
        th = math.radians(theta_deg)
        ex = a * np.cos(phis)
        ey = b * np.sin(phis)
        px = xc + ex * math.cos(th) - ey * math.sin(th)
        py = yc + ex * math.sin(th) + ey * math.cos(th)

        xs = np.concatenate([[xc], px])
        ys = np.concatenate([[yc], py])
        if np.any(ys < 1e-10):
            continue
        X = xs / ys * Y_val
        Z = (1.0 - xs - ys) / ys * Y_val
        xyz = np.stack([X, np.full_like(X, Y_val), Z], axis=-1)
        xyz = _cat_to_d65(xyz, _WHITE_C)

        lab = space.forward(torch.tensor(xyz, device=dev, dtype=dt))
        d = (lab[1:] - lab[0]).pow(2).sum(-1).sqrt()
        dmin = float(d.min())
        dmax = float(d.max())
        dmean = float(d.mean())
        if dmin <= 0 or dmean <= 0:
            continue
        ratios.append(dmax / dmin)
        cvs.append(float(d.std(unbiased=False)) / dmean)

    return {
        "mean_ratio": sum(ratios) / len(ratios) if ratios else 0,
        "max_ratio": max(ratios) if ratios else 0,
        "mean_cv": sum(cvs) / len(cvs) if cvs else 0,
        "n_centers": len(ratios),
        # per-ellipse items for the paired-bootstrap tie decision
        "_bootstrap": {"mean_ratio": {"items": ratios, "stat": "mean"}},
    }
