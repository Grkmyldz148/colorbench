"""Validate ColorBench's CVD simulation against REAL CVD-observer data.

ColorBench's cvd metric simulates protan/deutan/tritan vision with the Brettel
1997 LMS matrices. That simulation has a CONFUSION AXIS per defect type — the
chromaticity direction a dichromat cannot distinguish (the simulation collapses
along it). Regan, Reffin & Mollon (1994) measured, on REAL protan/deutan/tritan
observers, discrimination ellipses whose LONG axis lies along exactly that
confusion axis. So the simulation's confusion-axis orientation should match the
real observers' ellipse orientations.

This is a "judge the tool" check — it validates ColorBench's own CVD simulation,
NOT a candidate colour space. It never enters the human panel or any verdict.
"""
import math

import numpy as np

from .metrics.cvd import _CVD_LIST, _M_HPE_LIST

_M_HPE = np.array(_M_HPE_LIST)                 # XYZ -> LMS
_M_HPE_INV = np.linalg.inv(_M_HPE)


def _xyz_to_uv(xyz):
    X, Y, Z = xyz
    d = X + 15 * Y + 3 * Z
    if d <= 0:
        return None
    return np.array([4 * X / d, 9 * Y / d])    # CIE 1976 u', v'


def _uv_to_xyz(u, v, Y=20.0):
    # invert u'v' + luminance Y to XYZ
    x = 9 * u / (6 * u - 16 * v + 12)
    y = 4 * v / (6 * u - 16 * v + 12)
    X = x / y * Y
    Z = (1 - x - y) / y * Y
    return np.array([X, Y, Z])


def _simulate(xyz, cvd_type):
    lms = _M_HPE @ xyz
    lms2 = np.array(_CVD_LIST[cvd_type]) @ lms
    return _M_HPE_INV @ lms2


def confusion_axis_deg(u, v, cvd_type, n_dir=180, Y=20.0):
    """Orientation (deg, 0-180 in u'v') of the simulation's confusion axis at
    (u,v): the perturbation direction that produces the SMALLEST change in the
    simulated colour — the dichromat's non-discriminable direction."""
    base = _uv_to_xyz(u, v, Y)
    eps = 0.01
    best_dir, best_change = None, None
    for k in range(n_dir):
        ang = math.pi * k / n_dir             # 0..pi (axis, not vector)
        du, dv = eps * math.cos(ang), eps * math.sin(ang)
        p1 = _simulate(_uv_to_xyz(u + du, v + dv, Y), cvd_type)
        p2 = _simulate(_uv_to_xyz(u - du, v - dv, Y), cvd_type)
        change = np.linalg.norm(p1 - p2)
        if best_change is None or change < best_change:
            best_change, best_dir = change, math.degrees(ang)
    return best_dir % 180.0


def _ang_diff(a, b):
    """Smallest difference between two axis orientations (mod 180)."""
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def validate_against_regan(pool_dir=None):
    """Compare the Brettel-simulation confusion axes to the real Regan-Reffin-
    Mollon (1994) ellipse orientations per CVD type. Returns a dict with, per
    type, the mean observed ellipse angle, the mean simulated confusion axis at
    the same centres, and their agreement (mean axis difference, degrees)."""
    import csv
    import os
    if pool_dir is None:
        from .data import pool_dir as _pd
        pool_dir = _pd(auto_fetch=False)
    path = os.path.join(pool_dir, "regan_1994_cvd_ellipses", "canonical.csv")
    if not os.path.exists(path):
        return {"skipped": "regan_1994_cvd_ellipses not in pool"}
    rows = list(csv.DictReader(open(path)))
    # map dichromat classes to a simulation type (only the pure dichromat +
    # extreme anomalous observers have a well-defined confusion axis)
    TYPE = {"P": "protan", "D": "deutan", "T": "tritan"}
    out = {}
    for cls, ctype in TYPE.items():
        ell = [r for r in rows if r["classification"] == cls]
        if not ell:
            continue
        obs_ax, sim_ax = [], []
        for r in ell:
            obs_ax.append(float(r["angle_deg"]) % 180.0)
            u, v = float(r["center_u"]), float(r["center_v"])
            sim_ax.append(confusion_axis_deg(u, v, ctype))
        # circular-axis means (mod 180 -> double angle)
        def axis_mean(a):
            r = np.deg2rad(np.array(a) * 2)
            m = math.atan2(np.sin(r).mean(), np.cos(r).mean())
            return (math.degrees(m) / 2) % 180.0
        om, sm = axis_mean(obs_ax), axis_mean(sim_ax)
        diffs = [_ang_diff(o, s) for o, s in zip(obs_ax, sim_ax)]
        out[ctype] = {
            "n_ellipses": len(ell),
            "observed_axis_deg": round(om, 1),
            "simulated_axis_deg": round(sm, 1),
            "mean_axis_diff_deg": round(float(np.mean(diffs)), 1),
        }
    return out
