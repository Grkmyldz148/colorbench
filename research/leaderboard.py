"""Build the ColorBench leaderboard — TWO separate boards, because a color
space that measures difference and a color space that generates color are
judged on different things (the project's core principle: ölçüm ≠ üretim).

  MEASUREMENT board  — "which model best predicts human color difference?"
      Every entrant scores a STRESS on the real pair datasets (COMBVD,
      MacAdam-1974) using ITS OWN ΔE. This is where helmlab's *metricspace*
      (a learned, non-Euclidean distance) competes against CIEDE2000,
      CAM16-UCS, CIECAM02-UCS, DIN99, CIE94, CIELAB, OKLab, Jzazbz.

  GENERATION board  — "which space is best to generate color in?"
      Invertible forward spaces scored on synthesis geometry: round-trip
      precision, gamut-mapping hue monotonicity, gamut ΔE smoothness, plus
      two human-data geometry checks (constant-hue straightness, Munsell
      spacing). This is where helmlab's *genspace* competes against OKLab,
      CIELAB, IPT, Jzazbz, ICtCp, CAM16-UCS, DIN99d.

Writes docs/leaderboard.json + docs/leaderboard-data.js for the site.
Run:  python3 research/leaderboard.py
"""
import contextlib
import io
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core import data  # noqa: E402
from core import human_pool as hp  # noqa: E402

HELMGEN = "/Volumes/harici_ssd/color-space/helmlab-main-repo/checkpoints/genspace_v0.11.1.json"
HELMMETRIC = "/Volumes/harici_ssd/color-space/helmlab-main-repo/checkpoints/metricspace_v21.json"
HELM_SRC = "/Volumes/harici_ssd/color-space/helmlab-main-repo/src"


# ───────────────────────── MEASUREMENT board ──────────────────────────────
# STRESS (lower = better) on real difference-scaling datasets, each model with
# its own ΔE. Straight from core.metric_eval — the same engine `run.py metric`
# uses. metricspace competes as a full learned distance, not Euclidean.
def measurement_board():
    if HELM_SRC not in sys.path:
        sys.path.insert(0, HELM_SRC)
    from core.metric_eval import run_metric_evaluation
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = run_metric_evaluation(HELMMETRIC, data.baseline_dir(),
                                  os.path.dirname(HELM_SRC))
    combvd, macadam = r["COMBVD"], r["MacAdam1974"]
    holdout = r.get("_holdout", {})
    # pretty labels + who is who
    LABEL = {"MetricSpace": "helmlab metricspace", "CIE Lab": "CIELAB"}
    models = sorted(set(combvd) | set(macadam))
    rows = {}
    for m in models:
        c, a = combvd.get(m), macadam.get(m)
        vals = [v for v in (c, a) if isinstance(v, (int, float))]
        rows[LABEL.get(m, m)] = {
            "combvd": c, "macadam": a,
            "mean": round(float(np.mean(vals)), 2) if vals else None,
            "is_helm": m == "MetricSpace",
        }
    order = sorted(rows, key=lambda n: rows[n]["mean"] if rows[n]["mean"] is not None else 99)
    cols = ["combvd", "macadam", "mean"]
    rank = {c: {} for c in cols}
    for c in cols:
        sc = sorted([(v[c], n) for n, v in rows.items() if isinstance(v[c], (int, float))])
        for i, (_, n) in enumerate(sc, 1):
            rank[c][n] = i
    return {
        "title": "Measurement — color-difference prediction",
        "subtitle": "STRESS on real difference datasets (each model's own ΔE). Lower = closer to human.",
        "metrics": [
            {"key": "combvd", "label": "COMBVD"},
            {"key": "macadam", "label": "MacAdam 1974"},
            {"key": "mean", "label": "Mean STRESS"},
        ],
        # metricspace declares no trained_on → its COMBVD may be in-sample; flag it.
        "holdout_note": ("helmlab metricspace declares no training-data manifest, so "
                         "its COMBVD score may be in-sample; MacAdam 1974 is a fair "
                         "held-out test where it still ranks near the top."),
        "holdout": holdout,
        "spaces": [
            {"name": n, "is_helm": rows[n]["is_helm"],
             "scores": {c: rows[n][c] for c in cols},
             "ranks": {c: rank[c].get(n) for c in cols},
             "overall_rank": rank["mean"].get(n)}
            for n in order
        ],
        "winner": order[0] if order else None,
    }


# ───────────────────────── GENERATION board ───────────────────────────────
# Invertible forward spaces scored on synthesis geometry. run_test-compatible
# built-ins + helmlab genspace. Lower = better on every column.
GEN_SPACES = ["oklab", "cielab", "ipt", "jzazbz", "ictcp", "cam16ucs", "din99d"]
GEN_PRETTY = {"oklab": "OKLab", "cielab": "CIELAB", "ipt": "IPT", "jzazbz": "Jzazbz",
              "ictcp": "ICtCp", "cam16ucs": "CAM16-UCS", "din99d": "DIN99d"}


def _gen_geometry(space, device):
    """round-trip + gamut-mapping quality for one forward space."""
    from core.metrics import measure_roundtrip, measure_gamut_mapping
    rt = measure_roundtrip(space, device)
    gm = measure_gamut_mapping(space, device)
    # aggregate the per-L gamut-mapping numbers
    mono = [v for k, v in _flat(gm) if k.endswith("non_monotonic_hues")]
    djump = [v for k, v in _flat(gm) if k.endswith("max_de_jump")]
    rt_err = next((v for k, v in _flat(rt) if "srgb_full" in k and k.endswith("max_error")), None)
    # Guard: a space whose gamut-mapping ΔE-jumps are ALL exactly zero didn't
    # earn a perfect smoothness score — the mapping degenerated (constant output
    # / scale mismatch, seen with the CAM16-UCS build). Mark it unmeasured (None)
    # so a broken measurement can't win the column.
    degenerate = bool(djump) and all(v == 0.0 for v in djump)
    return {
        "round_trip": float(rt_err) if rt_err is not None else None,
        "gamut_mono": None if degenerate else (float(np.mean(mono)) if mono else None),
        "gamut_smooth": None if degenerate else (float(np.mean(djump)) if djump else None),
    }


def _flat(d, pre=""):
    for k, v in (d.items() if isinstance(d, dict) else []):
        if isinstance(v, dict):
            yield from _flat(v, pre + k + ".")
        elif isinstance(v, (int, float)):
            yield pre + k, v


def generation_board():
    from run import build_space, get_device
    device, dtype, _ = get_device()
    jobs = [(n, None, GEN_PRETTY[n]) for n in GEN_SPACES]
    jobs.append(("genspace", HELMGEN, "helmlab genspace"))

    rows = {}
    for name, ck, pretty in jobs:
        try:
            sp = build_space(name, ck, device, dtype=dtype)
            sp.name = pretty
            geo = _gen_geometry(sp, device)
            panel = hp.evaluate_space_on_pool(sp, validated_only=True)["by_property"]
        except Exception as e:
            print(f"  gen skip {pretty}: {type(e).__name__}: {e}"); continue

        def _mean(prop):
            vals = [v for v in panel.get(prop, {}).values() if isinstance(v, (int, float))]
            return round(float(np.mean(vals)), 3) if vals else None

        rows[pretty] = {
            "round_trip": geo["round_trip"], "gamut_mono": geo["gamut_mono"],
            "gamut_smooth": geo["gamut_smooth"], "hue": _mean("hue"),
            "spacing": _mean("spacing"), "is_helm": ck is not None,
        }
        _f = lambda v, s: (s % v) if isinstance(v, (int, float)) else "  —"
        print(f"  {pretty:20} rt={_f(geo['round_trip'], '%.1e')} mono={_f(geo['gamut_mono'], '%.1f')} "
              f"smooth={_f(geo['gamut_smooth'], '%.2f')} hue={rows[pretty]['hue']} spacing={rows[pretty]['spacing']}")

    cols = ["round_trip", "gamut_mono", "gamut_smooth", "hue", "spacing"]
    rank = {c: {} for c in cols}
    for c in cols:
        sc = sorted([(v[c], n) for n, v in rows.items() if isinstance(v[c], (int, float))])
        for i, (_, n) in enumerate(sc, 1):
            rank[c][n] = i
    overall = {}
    for n in rows:
        rs = [rank[c][n] for c in cols if n in rank[c]]
        overall[n] = round(sum(rs) / len(rs), 2) if rs else None
    order = sorted(rows, key=lambda n: overall[n] if overall[n] is not None else 99)
    return {
        "title": "Generation — color-synthesis geometry",
        "subtitle": "Round-trip, gamut-mapping smoothness, and human-data hue/spacing. Lower = better.",
        "metrics": [
            {"key": "round_trip", "label": "Round-trip err"},
            {"key": "gamut_mono", "label": "Gamut hue-mono"},
            {"key": "gamut_smooth", "label": "Gamut ΔE-jump"},
            {"key": "hue", "label": "Hue (human)"},
            {"key": "spacing", "label": "Spacing (human)"},
        ],
        "spaces": [
            {"name": n, "is_helm": rows[n]["is_helm"],
             "scores": {c: rows[n][c] for c in cols},
             "ranks": {c: rank[c].get(n) for c in cols},
             "overall_rank": overall[n]}
            for n in order
        ],
        "winner": order[0] if order else None,
    }


def main():
    print("── Measurement board ─────────────────────────────")
    meas = measurement_board()
    print(f"  winner: {meas['winner']}")
    for s in meas["spaces"]:
        print(f"  {s['name']:22} combvd={s['scores']['combvd']}  macadam={s['scores']['macadam']}  mean={s['scores']['mean']}")

    print("\n── Generation board ──────────────────────────────")
    gen = generation_board()
    print(f"  winner: {gen['winner']}")

    out = {"generated": "2026-07-09", "boards": {"measurement": meas, "generation": gen}}
    dest = os.path.join(_ROOT, "docs", "leaderboard.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    with open(os.path.join(os.path.dirname(dest), "leaderboard-data.js"), "w") as f:
        f.write("window.LEADERBOARD = " + json.dumps(out) + ";\n")
    print(f"\n  wrote {dest} + leaderboard-data.js")


if __name__ == "__main__":
    main()
