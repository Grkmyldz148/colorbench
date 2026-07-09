"""Build the ColorBench leaderboard — TWO boards (ölçüm ≠ üretim), each a
DETAILED, grouped, spec-by-spec comparison (Epey/Akakçe style): per-gamut
breakdowns, per-dataset human scores, and a generalization/overfit column.

  MEASUREMENT board  — "which model best predicts human color difference?"
      STRESS on real pair datasets (COMBVD, MacAdam-1974), each model with its
      OWN ΔE. Overfit check: a fitted model that wins big on its likely training
      set (COMBVD) but drops on the held-out set (MacAdam) shows an in-sample
      gap. helmlab metricspace competes here.

  GENERATION board  — "which space is best to generate color in?"
      Invertible forward spaces, scored on synthesis geometry BROKEN DOWN BY
      GAMUT (sRGB / Display-P3 / Rec2020): round-trip precision, gamut-mapping
      hue monotonicity and ΔE smoothness; plus gradient evenness and per-dataset
      human hue / discrimination / spacing. Generalization = how much a space's
      rank swings across the independent human datasets (a narrow specialist
      that wins one and loses another is the empirical fingerprint of overfit).
      helmlab genspace competes here.

Writes docs/leaderboard.json + docs/leaderboard-data.js.
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


def _flat(d, pre=""):
    for k, v in (d.items() if isinstance(d, dict) else []):
        if isinstance(v, dict):
            yield from _flat(v, pre + k + ".")
        elif isinstance(v, (int, float)):
            yield pre + k, v


def _ranks(rows, keys):
    """rank (1=best/lowest) for each (space, key); None where a space lacks it."""
    rank = {k: {} for k in keys}
    for k in keys:
        sc = sorted([(v["scores"][k], n) for n, v in rows.items()
                     if isinstance(v["scores"].get(k), (int, float))])
        for i, (_, n) in enumerate(sc, 1):
            rank[k][n] = i
    return rank


def _generalization(rank, keys, name):
    """Rank swing across independent datasets: (worst - best) rank. Low = robust
    generalist; high = narrow specialist (the empirical fingerprint of overfit)."""
    rs = [rank[k][name] for k in keys if name in rank.get(k, {})]
    if len(rs) < 2:
        return None
    return max(rs) - min(rs)


# ───────────────────────── MEASUREMENT board ──────────────────────────────
# STRESS (+ bootstrap CI95) on 5 INDEPENDENT difference sources — COMBVD is
# unpacked into its real components (BFD-P, Leeds, Witt, RIT-DuPont) plus the
# held-out MacAdam 1974 — each model scoring with its OWN ΔE. This turns the old
# 2-dataset board into a real cross-source difference-model ranking with a
# generalization / overfit read.
MEAS_MODELS = ["helmlab metricspace", "CIEDE2000", "CIE94", "CIELAB", "DIN99",
               "OKLab", "CAM16-UCS", "CIECAM02-UCS", "Jzazbz"]


def _score_dataset(metric, x1, x2, wh, dv):
    from core.metric_eval import (_ciede2000, _cie94_de, _cielab_de, _din99_de,
                                  _oklab_de, _cam16_ucs_de, _ciecam02_ucs_de,
                                  _jzazbz_de, _cat_to_d65, stress)
    from core.bootstrap import stress_ci
    x1, x2, wh, dv = np.asarray(x1), np.asarray(x2), np.asarray(wh), np.asarray(dv)
    n = len(dv)
    x1d = np.array([_cat_to_d65(x1[i], wh[i]) for i in range(n)])
    x2d = np.array([_cat_to_d65(x2[i], wh[i]) for i in range(n)])
    pp = lambda f: np.array([f(x1[i:i+1], x2[i:i+1], wh[i]) for i in range(n)]).ravel()
    fns = {
        "helmlab metricspace": lambda: metric.distance(x1d, x2d),
        "CIEDE2000": lambda: pp(_ciede2000), "CIE94": lambda: pp(_cie94_de),
        "CIELAB": lambda: pp(_cielab_de), "DIN99": lambda: pp(_din99_de),
        "OKLab": lambda: _oklab_de(x1d, x2d), "CAM16-UCS": lambda: _cam16_ucs_de(x1d, x2d),
        "CIECAM02-UCS": lambda: _ciecam02_ucs_de(x1d, x2d), "Jzazbz": lambda: _jzazbz_de(x1d, x2d),
    }
    out = {}
    for name in MEAS_MODELS:
        de = np.asarray(fns[name]()).ravel()
        s = float(stress(de, dv))
        lo, hi = stress_ci(de, dv)
        out[name] = (round(s, 2), [round(float(lo), 1), round(float(hi), 1)])
    return out


def measurement_board():
    if HELM_SRC not in sys.path:
        sys.path.insert(0, HELM_SRC)
    from core.metric_eval import load_combvd_from_xlsx, load_macadam1974, _load_metric_space
    base = data.baseline_dir()
    metric = _load_metric_space(HELMMETRIC, os.path.dirname(HELM_SRC))

    recs = load_combvd_from_xlsx(base)
    def subset(pred):
        r = [x for x in recs if pred(x["dataset"])]
        return ([x["xyz1"] for x in r], [x["xyz2"] for x in r],
                [x["white"] for x in r], [x["dv"] for x in r])
    mx1, mx2, mw, mdv = load_macadam1974(base)
    mac = (list(mx1), list(mx2), [mw] * len(mdv), list(mdv))

    # 4 COMBVD components (possibly in-sample for the COMBVD-fit metricspace) +
    # MacAdam 1974 (held-out). BFD-P's three illuminant variants merge into one.
    DATASETS = [
        ("bfd", "BFD-P", "in", subset(lambda s: s.startswith("BFD-P"))),
        ("leeds", "Leeds", "in", subset(lambda s: s == "LEEDS")),
        ("witt", "Witt", "in", subset(lambda s: s == "WITT")),
        ("rit", "RIT-DuPont", "in", subset(lambda s: s == "RIT-DuPont")),
        ("macadam", "MacAdam 74", "held", mac),
    ]
    dkeys = [d[0] for d in DATASETS]
    in_keys = [d[0] for d in DATASETS if d[2] == "in"]

    rows = {m: {"is_helm": m == "helmlab metricspace", "scores": {}, "ci": {}}
            for m in MEAS_MODELS}
    n_by = {}
    for key, label, kind, (x1, x2, wh, dv) in DATASETS:
        n_by[key] = len(dv)
        print(f"  scoring {label} (n={len(dv)}) ...", flush=True)
        res = _score_dataset(metric, x1, x2, wh, dv)
        for m in MEAS_MODELS:
            rows[m]["scores"][key], rows[m]["ci"][key] = res[m]

    for m in MEAS_MODELS:
        vals = [rows[m]["scores"][k] for k in dkeys]
        rows[m]["scores"]["mean"] = round(float(np.mean(vals)), 2)

    rank = _ranks(rows, dkeys + ["mean"])
    for m, v in rows.items():
        # overfit Δrank: held-out MacAdam rank minus mean rank over the COMBVD
        # components (the metricspace's possible training pool). + = relatively
        # better on the possible-training data than on the held-out set.
        r_in = np.mean([rank[k][m] for k in in_keys if m in rank.get(k, {})])
        r_held = rank["macadam"].get(m)
        v["scores"]["overfit"] = int(round(r_held - r_in)) if r_held else None
        v["scores"]["gen_spread"] = _generalization(rank, dkeys, m)
        v["overall_rank"] = rank["mean"].get(m)
    order = sorted(rows, key=lambda m: rows[m]["overall_rank"] or 99)

    return {
        "title": "Measurement — color-difference prediction",
        "subtitle": ("STRESS (+ CI95 on hover) on 5 independent difference sources, each model "
                     "with its own ΔE. Lower = closer to human. Overall = mean STRESS rank."),
        "holdout_note": ("The 4 COMBVD components (BFD-P, Leeds, Witt, RIT-DuPont) may be in-sample "
                         "for the COMBVD-fit helmlab metricspace; MacAdam 1974 is the held-out check. "
                         "Overfit Δrank = held-out rank − mean in-sample rank (+ = relatively better "
                         "where it may have been fit). Rank swing = worst−best rank across all 5."),
        "groups": [
            {"label": "COMBVD components · STRESS (possible in-sample)", "metrics": [
                {"key": "bfd", "label": f"BFD-P"}, {"key": "leeds", "label": "Leeds"},
                {"key": "witt", "label": "Witt"}, {"key": "rit", "label": "RIT-DuPont"}]},
            {"label": "Held-out", "metrics": [{"key": "macadam", "label": "MacAdam 74"}]},
            {"label": "Composite", "metrics": [{"key": "mean", "label": "Mean STRESS"}]},
            {"label": "Generalization", "metrics": [
                {"key": "overfit", "label": "Overfit Δrank", "signed": True,
                 "hint": "held-out − in-sample rank"},
                {"key": "gen_spread", "label": "Rank swing", "hint": "worst−best rank over 5 sources"}]},
        ],
        "spaces": [{"name": m, "is_helm": rows[m]["is_helm"], "scores": rows[m]["scores"],
                    "ci": rows[m]["ci"], "overall_rank": rows[m]["overall_rank"]}
                   for m in order],
        "winner": order[0] if order else None,
    }


# ───────────────────────── GENERATION board ───────────────────────────────
GEN_SPACES = ["oklab", "cielab", "ipt", "jzazbz", "ictcp", "cam16ucs", "din99d"]
GEN_PRETTY = {"oklab": "OKLab", "cielab": "CIELAB", "ipt": "IPT", "jzazbz": "Jzazbz",
              "ictcp": "ICtCp", "cam16ucs": "CAM16-UCS", "din99d": "DIN99d"}
GAMUTS = [("sRGB", "sRGB"), ("P3", "P3"), ("Rec2020", "Rec2020")]


def _gen_metrics(space, device):
    from core.metrics import measure_roundtrip, measure_gamut_mapping, measure_gradients
    from core.pairs import generate_all_pairs
    rt = dict(_flat(measure_roundtrip(space, device)))
    gm = dict(_flat(measure_gamut_mapping(space, device)))
    pairs_xyz, labels = generate_all_pairs(device)
    gr = dict(_flat(measure_gradients(space, pairs_xyz, labels, device)))

    def rt_err(tag):
        for k, v in rt.items():
            if k.startswith(tag) and k.endswith("max_error"):
                return float(v)
        return None

    def gm_agg(gamut, suffix):
        vals = [v for k, v in gm.items() if k.startswith(gamut + "_L") and k.endswith(suffix)]
        # all-zero ΔE-jumps ⇒ the mapping degenerated (e.g. CAM16-UCS build): mark unmeasured
        if suffix == "max_de_jump" and vals and all(v == 0.0 for v in vals):
            return None
        return float(np.mean(vals)) if vals else None

    rts = [rt_err("srgb_full"), rt_err("p3_full"), rt_err("rec2020_2M")]
    out = {"grad_drift": gr.get("overall.drift_mean"),
           # invertibility: worst gamut (the 3 are near-tied ~1e-15 except the
           # two spaces that drop to ~1e-13 — one column carries that signal)
           "rt_worst": max([v for v in rts if v is not None], default=None)}
    monos, djs = [], []
    for g, _ in GAMUTS:
        smooth = gm_agg(g, "max_de_jump")
        djs.append(smooth)
        monos.append(None if smooth is None else gm_agg(g, "non_monotonic_hues"))
    good_dj = [v for v in djs if v is not None]
    good_mono = [v for v in monos if v is not None]
    out["gm_dj_worst"] = max(good_dj) if good_dj else None
    out["gm_mono_worst"] = max(good_mono) if good_mono else None
    return out


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
            m = _gen_metrics(sp, device)
            panel = hp.evaluate_space_on_pool(sp, validated_only=True)["by_property"]
        except Exception as e:
            print(f"  gen skip {pretty}: {type(e).__name__}: {e}"); continue
        h = panel.get("hue", {}); d = panel.get("discrimination", {}); s = panel.get("spacing", {})
        m.update({
            "hue_hb": h.get("hung_berns"), "hue_ef": h.get("ebner_fairchild"),
            "hue_mun": h.get("munsell"),
            "disc_mac": d.get("macadam1942"), "disc_lr": d.get("luo_rigg_ellipses"),
            "disc_regan": d.get("regan_1994_cvd_ellipses"),
            "sp_osa": s.get("osa_ucs_1974"),
        })
        rows[pretty] = {"is_helm": ck is not None, "scores": m}
        print(f"  {pretty:18} hue_hb={m['hue_hb']:.2f} regan={m['disc_regan']:.3f} "
              f"osa={m['sp_osa']:.3f} | rt_worst={m['rt_worst']:.0e} gm_dj={m['gm_dj_worst']}")

    # SCORED groups drive the overall rank — human-data only, because those are
    # the only columns measured with an unbiased ruler (real observers).
    scored_groups = [
        {"label": "Hue · human data", "scored": True, "metrics": [
            {"key": "hue_hb", "label": "Hung-Berns"}, {"key": "hue_ef", "label": "Ebner-F."},
            {"key": "hue_mun", "label": "Munsell"}]},
        {"label": "Discrimination · human", "scored": True, "metrics": [
            {"key": "disc_mac", "label": "MacAdam"}, {"key": "disc_lr", "label": "Luo-Rigg"},
            {"key": "disc_regan", "label": "Regan"}]},
        {"label": "Spacing · human", "scored": True, "metrics": [{"key": "sp_osa", "label": "OSA-UCS"}]},
    ]
    # DIAGNOSTIC groups are shown but NOT scored. Round-trip is a correctness gate
    # (all pass to ~1e-13). Gamut-mapping & gradient are measured in CIELab /
    # CIEDE2000 (see core/metrics/gamut_mapping.py) — a CIELab-referenced ruler
    # that structurally flatters CIELAB and its transforms, exactly the
    # measure-CIELab-with-CIELab circularity the project excludes from any
    # verdict. Collapsed to worst-gamut to keep the table desktop-width.
    diag_groups = [
        {"label": "Invertibility (gate)", "scored": False, "metrics": [
            {"key": "rt_worst", "label": "round-trip", "hint": "worst gamut; all pass"}]},
        {"label": "Gamut-map · CIELab-ref (diagnostic)", "scored": False, "metrics": [
            {"key": "gm_dj_worst", "label": "ΔE-jump", "hint": "worst gamut; CIELab-referenced"},
            {"key": "gm_mono_worst", "label": "hue-mono", "hint": "worst gamut; CIELab-referenced"}]},
        {"label": "Gradient · CIELab-ref", "scored": False, "metrics": [
            {"key": "grad_drift", "label": "hue-drift°", "hint": "CIELab-referenced"}]},
    ]
    human_keys = ["hue_hb", "hue_ef", "hue_mun", "disc_mac", "disc_lr", "disc_regan", "sp_osa"]
    all_keys = [m["key"] for g in scored_groups + diag_groups for m in g["metrics"]]
    rank = _ranks(rows, all_keys)
    # overall = mean of the SCORED (human) category ranks only — each of the 3
    # human categories weighs equally; the CIELab-referenced engineering columns
    # are diagnostics and do not move the ranking.
    for n, v in rows.items():
        grp = []
        for g in scored_groups:
            grs = [rank[m["key"]][n] for m in g["metrics"] if n in rank.get(m["key"], {})]
            if grs:
                grp.append(sum(grs) / len(grs))
        v["overall_rank"] = round(sum(grp) / len(grp), 2) if grp else None
        v["scores"]["gen_spread"] = _generalization(rank, human_keys, n)
    order = sorted(rows, key=lambda n: rows[n]["overall_rank"] or 99)
    groups = scored_groups + diag_groups + [
        {"label": "Generalization", "scored": True, "metrics": [
            {"key": "gen_spread", "label": "Rank swing", "hint": "worst−best rank over 7 human datasets"}]}]
    return {
        "title": "Generation — color-synthesis geometry",
        "subtitle": ("Ranked on HUMAN data only — hue (Hung-Berns/Ebner/Munsell), discrimination "
                     "(MacAdam/Luo-Rigg/Regan), spacing (OSA-UCS), each category weighing equally. "
                     "The engineering columns are diagnostics: round-trip is a pass/fail gate, and "
                     "gamut-mapping / gradient are measured in CIELab (a CIELab-referenced ruler that "
                     "would unfairly flatter CIELAB), so they don't drive the rank. Lower = better."),
        "groups": groups,
        "spaces": [{"name": n, "is_helm": rows[n]["is_helm"],
                    "scores": rows[n]["scores"], "overall_rank": rows[n]["overall_rank"]}
                   for n in order],
        "winner": order[0] if order else None,
    }


def main():
    print("── Measurement board ─────────────────────────────")
    meas = measurement_board()
    print(f"  winner: {meas['winner']}")
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
