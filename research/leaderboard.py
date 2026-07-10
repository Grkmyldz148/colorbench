"""Build the ColorBench leaderboard - comprehensive, permanent, cleanly split.

TWO boards, and each one carries ONLY the metrics that belong to its axis:

  MEASUREMENT - "how accurately does the space represent human color perception?"
      Everything about DIFFERENCE / discrimination:
        · difference prediction - STRESS (+ CI95) on 5 sources (COMBVD's BFD-P /
          Leeds / Witt / RIT-DuPont components + held-out MacAdam 1974), each
          model with its own ΔE (metricspace = learned distance; CIEDE2000 /
          CIE94 = formulas; every colour space = Euclidean ΔE)
        · discrimination (JND-ellipse roundness), 3-D discrimination, tolerance
        · appearance diagnostics: H-K brightness/lightness, chromatic
          adaptation, observer metamerism
      This is metricspace's home. A pure distance model (metricspace / CIEDE2000
      / CIE94) is scored on the discrimination / 3-D / tolerance ellipses with
      ITS OWN distance, so it competes on those columns too - not blank.

  GENERATION - "how well does the space GENERATE color (gradients, palettes)?"
      Only the properties that decide generation quality:
        · hue-constancy (Hung-Berns, Ebner-Fairchild, Munsell, Xiao) - gradients
          and shades must keep their hue
        · spacing (OSA-UCS) - even perceptual steps (no banding)
        · robustness gate (physics): round-trip invertibility + wide-gamut
          finiteness - can you generate valid color across gamuts
      This is genspace's home. Discrimination / tolerance are DIFFERENCE
      properties and live on the measurement board, not here.

Writes docs/leaderboard.json + docs/leaderboard-data.js.
Run:  python3 research/leaderboard.py
"""
import io
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import colour  # noqa: E402
from core import data  # noqa: E402
from core import human_pool as hp  # noqa: E402

HELMGEN = "/Volumes/harici_ssd/color-space/helmlab-main-repo/checkpoints/genspace_v0.11.1.json"
HELMMETRIC = "/Volumes/harici_ssd/color-space/helmlab-main-repo/checkpoints/metricspace_v21.json"
HELM_SRC = "/Volumes/harici_ssd/color-space/helmlab-main-repo/src"

COLOUR_SPACES = ["Lab", "Luv", "IPT", "IPT_Ragoo2021", "Jzazbz", "ICtCp",
                 "ICaCb", "IgPgTg", "Oklab", "DIN99", "ProLab", "Yrg",
                 "hdr_CIELab", "CAM02UCS", "CAM16UCS", "CAM02LCD", "CAM16LCD",
                 "CAM02SCD", "CAM16SCD", "sUCS"]
PRETTY = {"Lab": "CIELAB", "Luv": "CIELUV", "Oklab": "OKLab", "Jzazbz": "Jzazbz",
          "IPT_Ragoo2021": "IPT (Ragoo)", "hdr_CIELab": "hdr-CIELAB",
          "CAM02UCS": "CAM02-UCS", "CAM16UCS": "CAM16-UCS", "CAM02LCD": "CAM02-LCD",
          "CAM16LCD": "CAM16-LCD", "CAM02SCD": "CAM02-SCD", "CAM16SCD": "CAM16-SCD"}
_D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
_ILLUM_NEEDED = {"Lab", "Luv", "DIN99", "ProLab"}


class ColourWrapper:
    def __init__(self, name):
        self.name = PRETTY.get(name, name)
        self._f = getattr(colour, f"XYZ_to_{name}")
        self._i = getattr(colour, f"{name}_to_XYZ", None)
        self._illum = name in _ILLUM_NEEDED

    def forward(self, xyz):
        xyz = np.atleast_2d(np.asarray(xyz, float))
        out = self._f(xyz, _D65) if self._illum else self._f(xyz)
        return np.asarray(out, float)

    def inverse(self, coords):
        coords = np.atleast_2d(np.asarray(coords, float))
        out = self._i(coords, _D65) if self._illum else self._i(coords)
        return np.asarray(out, float)


def _ranks(rows, keys):
    rank = {k: {} for k in keys}
    for k in keys:
        sc = sorted([(v["scores"][k], n) for n, v in rows.items()
                     if isinstance(v["scores"].get(k), (int, float))])
        for i, (_, n) in enumerate(sc, 1):
            rank[k][n] = i
    return rank


def _swing(rank, keys, name):
    rs = [rank[k][name] for k in keys if name in rank.get(k, {})]
    return (max(rs) - min(rs)) if len(rs) >= 2 else None


def _overall(rank, keys, name):
    """DATASET-equal overall: every human dataset is one equal vote (so a
    property measured by 4 datasets carries 4× the evidence of a 1-dataset one,
    and no single-dataset category is over-weighted by being alone)."""
    r = [rank[k][name] for k in keys if name in rank.get(k, {})]
    return round(sum(r) / len(r), 2) if r else None


# ── physics robustness gate (generation) ───────────────────────────────────
def _sample_gamut_xyz(cs_name, n=20000, seed=0):
    cs = colour.RGB_COLOURSPACES[cs_name]
    rgb = np.random.RandomState(seed).rand(n, 3)
    return np.asarray(colour.RGB_to_XYZ(rgb, cs), float)


def _robustness(fwd, inv):
    out = {"rt_srgb": None, "rt_rec2020": None, "nan_rec2020": None}
    for key, csname in [("rt_srgb", "sRGB"), ("rt_rec2020", "ITU-R BT.2020")]:
        try:
            xyz = _sample_gamut_xyz(csname)
            c = np.asarray(fwd(xyz), float)
            back = np.asarray(inv(c), float)
            fin = np.isfinite(c).all(-1) & np.isfinite(back).all(-1)
            if key == "rt_rec2020":
                out["nan_rec2020"] = round(100.0 * float((~fin).mean()), 1)
            out[key] = float(np.abs(xyz[fin] - back[fin]).max()) if fin.any() else None
        except Exception:
            if key == "rt_rec2020":
                out["nan_rec2020"] = 100.0
    return out


# ── property catalogue (per board) ──────────────────────────────────────────
# generation = generation-relevant properties only
GEN_SCORED = [
    ("Hue-constancy · human", "hue", [
        ("hung_berns", "Hung-Berns"), ("ebner_fairchild", "Ebner-F."),
        ("munsell", "Munsell"), ("xiao_unique_hues", "Xiao")]),
    ("Even spacing · human", "spacing", [("osa_ucs_1974", "OSA-UCS")]),
]
# measurement = difference / discrimination properties (forward-geometry)
MEAS_GEOM = [
    ("Discrimination · THRESHOLD JND · human", "discrimination", [
        ("macadam1942", "MacAdam42"), ("luo_rigg_ellipses", "Luo-Rigg"),
        ("alder1982", "Alder"), ("regan_1994_cvd_ellipses", "Regan"),
        ("hong_2025_ellipsoids", "Hong")]),
    ("3-D discrimination · THRESHOLD JND · human", "3d_discrim", [
        ("koenderink_2026_3d_metric_field", "Koenderink"),
        ("brown_1957_12obs_ellipsoids", "Brown-57"),
        ("wyszecki_fielder_1971_ellipsoids", "Wyszecki-F"),
        ("brown_macadam_1949_ellipsoids", "Brown-MacAdam")]),
    ("Tolerance · near-threshold · human", "tolerance", [
        ("berns_1991_rit_dupont_tolerance_vectors", "RIT-DuPont"),
        ("huang_2012_cielab_ellipses", "Huang")]),
]
MEAS_DIAG = [
    ("H-K brightness · ρ ↑ better · diagnostic", "hk_mechanism", [
        ("wyszecki_1967_osa_tiles", "Wyszecki-67"),
        ("zhang_2023_laser_display_brightness", "Zhang-23"),
        ("sanders_wyszecki_1964_HK", "Sanders-64")]),
    ("H-K object lightness · diagnostic", "hk_object", [
        ("fairchild_pirrotta_1991", "Fairchild-P")]),
    ("Chromatic adaptation · ΔE · diagnostic", "adaptation", [
        ("corresponding_colours", "Corr-colours")]),
    ("Observer metamerism · diagnostic", "observer_variance", [
        ("asano_observers", "Asano")]),
]


def _build_genspace():
    from run import build_space, get_device
    device, dtype, _ = get_device()
    sp = build_space("genspace", HELMGEN, device, dtype=dtype)
    sp.name = "helmlab genspace"
    return sp


def compute_forward():
    """human_pool (all properties) + robustness for the 20 colour spaces +
    genspace. Returns {name: {"is_helm", "props": {prop:{ds:val}}, "rob": {...}}}."""
    import torch
    spaces = [ColourWrapper(n) for n in COLOUR_SPACES]
    try:
        spaces.append(_build_genspace())
    except Exception as e:
        print(f"  genspace skipped: {e}")
    out = {}
    for sp in spaces:
        try:
            try:
                panel = hp.evaluate_space_on_pool(sp, validated_only=False)["by_property"]
            except Exception:
                panel = hp.evaluate_space_on_pool(sp, validated_only=True)["by_property"]
        except Exception as e:
            print(f"  skip {sp.name}: {type(e).__name__}: {e}"); continue
        if isinstance(sp, ColourWrapper):
            rob = _robustness(sp.forward, sp.inverse)
        else:
            fwd = lambda x: sp.forward(torch.as_tensor(x, dtype=sp.dtype, device=sp.device)).detach().cpu().numpy()
            inv = lambda c: sp.inverse(torch.as_tensor(c, dtype=sp.dtype, device=sp.device)).detach().cpu().numpy()
            rob = _robustness(fwd, inv)
        out[sp.name] = {"is_helm": sp.name.startswith("helmlab"), "props": panel, "rob": rob}
        print(f"  {sp.name:16} hue_hb={panel.get('hue',{}).get('hung_berns')} "
              f"disc={panel.get('discrimination',{}).get('macadam1942')} rt={rob['rt_srgb']}", flush=True)
    return out


def _val(props, prop, ds):
    v = props.get(prop, {}).get(ds)
    return float(v) if isinstance(v, (int, float)) else None


# ═══════════════════════════ MEASUREMENT board ═════════════════════════════
def measurement_board(fwd):
    from core.metric_eval import (load_combvd_from_xlsx, load_macadam1974,
                                  _ciede2000, _cie94_de, _cat_to_d65, stress,
                                  _load_metric_space)
    from core.bootstrap import stress_ci
    base = data.baseline_dir()
    if HELM_SRC not in sys.path:
        sys.path.insert(0, HELM_SRC)
    metric = _load_metric_space(HELMMETRIC, os.path.dirname(HELM_SRC))
    wrappers = [ColourWrapper(n) for n in COLOUR_SPACES]

    recs = load_combvd_from_xlsx(base)
    def subset(pred):
        r = [x for x in recs if pred(x["dataset"])]
        return ([x["xyz1"] for x in r], [x["xyz2"] for x in r],
                [x["white"] for x in r], [x["dv"] for x in r])
    mx1, mx2, mw, mdv = load_macadam1974(base)
    mac = (list(mx1), list(mx2), [mw] * len(mdv), list(mdv))
    DATASETS = [("bfd", "BFD-P", "in", subset(lambda s: s.startswith("BFD-P"))),
                ("leeds", "Leeds", "in", subset(lambda s: s == "LEEDS")),
                ("witt", "Witt", "in", subset(lambda s: s == "WITT")),
                ("rit", "RIT-DuPont", "in", subset(lambda s: s == "RIT-DuPont")),
                ("macadam", "MacAdam 74", "held", mac)]
    diff_keys = [d[0] for d in DATASETS]
    in_keys = [d[0] for d in DATASETS if d[2] == "in"]

    # rows: difference-only models (metricspace + formulas) + full forward spaces
    rows = {}
    rows["helmlab metricspace"] = {"is_helm": True, "scores": {}, "ci": {}, "diff_only": True}
    rows["CIEDE2000"] = {"is_helm": False, "scores": {}, "ci": {}, "diff_only": True}
    rows["CIE94"] = {"is_helm": False, "scores": {}, "ci": {}, "diff_only": True}
    for w in wrappers:
        rows[w.name] = {"is_helm": False, "scores": {}, "ci": {}, "diff_only": False}
    # genspace shown too (a forward space, off its home turf)
    if "helmlab genspace" in fwd:
        rows["helmlab genspace"] = {"is_helm": True, "scores": {}, "ci": {}, "diff_only": False}

    # ── difference STRESS ──
    for key, label, kind, (x1r, x2r, wh, dv) in DATASETS:
        x1r = np.asarray(x1r, float); x2r = np.asarray(x2r, float)
        wh = np.asarray(wh, float); dv = np.asarray(dv, float); n = len(dv)
        x1d = np.array([_cat_to_d65(x1r[i], wh[i]) for i in range(n)])
        x2d = np.array([_cat_to_d65(x2r[i], wh[i]) for i in range(n)])
        print(f"  difference {label} (n={n}) ...", flush=True)
        des = {"helmlab metricspace": np.asarray(metric.distance(x1d, x2d)).ravel(),
               "CIEDE2000": np.array([_ciede2000(x1r[i:i+1], x2r[i:i+1], wh[i]) for i in range(n)]).ravel(),
               "CIE94": np.array([_cie94_de(x1r[i:i+1], x2r[i:i+1], wh[i]) for i in range(n)]).ravel()}
        for w in wrappers:
            des[w.name] = np.sqrt(((w.forward(x1d) - w.forward(x2d)) ** 2).sum(-1))
        for name, de in des.items():
            de = np.asarray(de, float).ravel(); ok = np.isfinite(de)
            if ok.sum() < 3:
                continue
            rows[name]["scores"][key] = round(float(stress(de[ok], dv[ok])), 2)
            lo, hi = stress_ci(de[ok], dv[ok])
            rows[name]["ci"][key] = [round(float(lo), 1), round(float(hi), 1)]

    # genspace difference via its own forward (torch) - compute Euclidean ΔE
    if "helmlab genspace" in rows:
        import torch
        gsp = _build_genspace()
        for key, label, kind, (x1r, x2r, wh, dv) in DATASETS:
            x1r = np.asarray(x1r, float); x2r = np.asarray(x2r, float); wh = np.asarray(wh, float)
            dv = np.asarray(dv, float); n = len(dv)
            x1d = np.array([_cat_to_d65(x1r[i], wh[i]) for i in range(n)])
            x2d = np.array([_cat_to_d65(x2r[i], wh[i]) for i in range(n)])
            c1 = gsp.forward(torch.as_tensor(x1d, dtype=gsp.dtype, device=gsp.device)).detach().cpu().numpy()
            c2 = gsp.forward(torch.as_tensor(x2d, dtype=gsp.dtype, device=gsp.device)).detach().cpu().numpy()
            de = np.sqrt(((c1 - c2) ** 2).sum(-1)); ok = np.isfinite(de)
            if ok.sum() >= 3:
                rows["helmlab genspace"]["scores"][key] = round(float(stress(de[ok], dv[ok])), 2)
                lo, hi = stress_ci(de[ok], dv[ok])
                rows["helmlab genspace"]["ci"][key] = [round(float(lo), 1), round(float(hi), 1)]

    # ── forward-geometry columns: discrimination / 3d / tolerance (all entrants)
    # + appearance diagnostics (forward spaces only) ──
    # forward spaces read from the human_pool; the distance-only models
    # (metricspace / CIEDE2000 / CIE94) are scored on the SAME JND ellipses with
    # THEIR OWN distance (via the judges' `metric=` hook), so they're no longer
    # blank there - every entrant competes on discrimination with its own ΔE.
    from core.human_pool import (judge_jnd_ellipse, judge_regan_normal_ellipse, judge_hong_2dw,
                                 judge_jnd_ellipsoid_koenderink, judge_brown_1957,
                                 judge_jnd_ellipsoid_g, judge_lab_ellipsoid, judge_tolerance_vectors)
    from core.metric_eval import _ciede2000, _cie94_de
    _D65X = np.array([0.95047, 1.0, 1.08883])

    def _metric_geom(mfn):
        g = lambda fn, *a: (fn(None, *a, metric=mfn) or {}).get("mean_cv")
        return {
            "macadam1942": g(judge_jnd_ellipse, "macadam1942"),
            "luo_rigg_ellipses": g(judge_jnd_ellipse, "luo_rigg_ellipses"),
            "alder1982": g(judge_jnd_ellipse, "alder1982"),
            "regan_1994_cvd_ellipses": g(judge_regan_normal_ellipse),
            "hong_2025_ellipsoids": g(judge_hong_2dw),
            "koenderink_2026_3d_metric_field": g(judge_jnd_ellipsoid_koenderink),
            "brown_1957_12obs_ellipsoids": g(judge_brown_1957),
            "wyszecki_fielder_1971_ellipsoids": g(judge_jnd_ellipsoid_g, "wyszecki_fielder_1971_ellipsoids"),
            "brown_macadam_1949_ellipsoids": g(judge_jnd_ellipsoid_g, "brown_macadam_1949_ellipsoids"),
            "berns_1991_rit_dupont_tolerance_vectors": g(judge_tolerance_vectors),
            "huang_2012_cielab_ellipses": g(judge_lab_ellipsoid),
        }
    def _loop_de(fn):
        return lambda a, b: np.array([fn(np.asarray(a, float)[i:i+1], np.asarray(b, float)[i:i+1], _D65X)
                                      for i in range(len(a))]).ravel()
    DIST_METRICS = {
        "helmlab metricspace": lambda a, b: np.asarray(metric.distance(np.asarray(a, float), np.asarray(b, float))).ravel(),
        "CIEDE2000": _loop_de(_ciede2000), "CIE94": _loop_de(_cie94_de),
    }
    geom_keys = []
    diag_keys = []
    for _, prop, metrics in MEAS_GEOM:
        for k, _ in metrics:
            geom_keys.append(k)
            for name, v in rows.items():
                if not v["diff_only"]:
                    rows[name]["scores"][k] = _val(fwd.get(name, {}).get("props", {}), prop, k)
    for _, prop, metrics in MEAS_DIAG:
        for k, _ in metrics:
            diag_keys.append(k)
            for name, v in rows.items():
                if not v["diff_only"]:
                    rows[name]["scores"][k] = _val(fwd.get(name, {}).get("props", {}), prop, k)
    for mname, mfn in DIST_METRICS.items():
        if mname not in rows:
            continue
        print(f"  {mname} discrimination via own distance ...", flush=True)
        for k, cv in _metric_geom(mfn).items():
            rows[mname]["scores"][k] = float(cv) if isinstance(cv, (int, float)) else None

    # ── ranks + overall ──
    # Now every entrant (spaces AND distance models) has difference + discrimination
    # + 3-D + tolerance, so rank on all four (dataset-equal). Appearance diagnostics
    # stay out of the overall (distance models can't produce them).
    meas_scored_keys = diff_keys + geom_keys
    all_score_keys = meas_scored_keys + diag_keys
    rank = _ranks(rows, all_score_keys)
    for name, v in rows.items():
        v["overall_rank"] = _overall(rank, meas_scored_keys, name)
        if v["is_helm"] and name == "helmlab metricspace":
            r_in = [rank[k][name] for k in in_keys if name in rank.get(k, {})]
            r_h = rank["macadam"].get(name)
            v["scores"]["overfit"] = int(round(r_h - np.mean(r_in))) if (r_h and r_in) else None
        else:
            v["scores"]["overfit"] = None
        v["scores"]["gen_spread"] = _swing(rank, all_score_keys, name)
    order = sorted(rows, key=lambda n: rows[n]["overall_rank"] or 999)

    groups = [
        {"label": "Difference · SUPRATHRESHOLD STRESS (COMBVD; possible in-sample)", "metrics": [
            {"key": "bfd", "label": "BFD-P"}, {"key": "leeds", "label": "Leeds"},
            {"key": "witt", "label": "Witt"}, {"key": "rit", "label": "RIT-DuPont"}]},
        {"label": "Difference · suprathreshold, held-out", "metrics": [{"key": "macadam", "label": "MacAdam 74"}]},
    ]
    for gl, prop, metrics in MEAS_GEOM:
        groups.append({"label": gl, "metrics": [{"key": k, "label": lb} for k, lb in metrics]})
    for gl, prop, metrics in MEAS_DIAG:
        groups.append({"label": gl, "scored": False,
                       "metrics": [{"key": k, "label": lb} for k, lb in metrics]})
    groups.append({"label": "Generalization", "metrics": [
        {"key": "overfit", "label": "Overfit Δrank", "signed": True, "hint": "helmlab only"},
        {"key": "gen_spread", "label": "Rank swing", "hint": "worst-best rank"}]})
    return {
        "title": "Measurement - perceptual accuracy (color difference)",
        "subtitle": ("How accurately a model represents human color DIFFERENCE, each entrant with its "
                     "OWN ΔE. TWO REGIMES: the Difference columns are SUPRATHRESHOLD - the magnitude "
                     "of clearly-perceptible differences (ΔE ~1-10, STRESS on COMBVD; +CI95 on hover); "
                     "discrimination / 3-D / tolerance are THRESHOLD - just-noticeable JND-ellipse "
                     "roundness. A model can lead one regime and not the other: metricspace (fit to "
                     "suprathreshold COMBVD) tops Difference, while the CAM 'uniform color spaces' are "
                     "built for threshold isotropy and lead discrimination. A pure distance model is "
                     "scored on the ellipses with its own distance (not blank). Overall = one equal "
                     "vote per dataset; grey appearance diagnostics are shown, not scored."),
        "holdout_note": ("metricspace is fit to COMBVD, so its BFD-P/Leeds/Witt/RIT scores may be "
                         "in-sample; on the held-out MacAdam 1974 it does NOT win (CAM16-UCS does). "
                         "Overfit Δrank (metricspace only) = held-out rank - mean in-sample rank."),
        "groups": groups,
        "spaces": [{"name": n, "is_helm": rows[n]["is_helm"], "scores": rows[n]["scores"],
                    "ci": rows[n]["ci"], "overall_rank": rows[n]["overall_rank"]} for n in order],
        "winner": order[0] if order else None,
    }


# ═══════════════════════════ GENERATION board ═════════════════════════════
def generation_board(fwd):
    rows = {}
    for name, rec in fwd.items():
        sc = {}
        for _, prop, metrics in GEN_SCORED:
            for k, _ in metrics:
                sc[k] = _val(rec["props"], prop, k)
        sc.update(rec["rob"])
        rows[name] = {"is_helm": rec["is_helm"], "scores": sc}

    scored_keys = [k for _, _, ms in GEN_SCORED for k, _ in ms]
    rank = _ranks(rows, scored_keys)
    for name, v in rows.items():
        v["overall_rank"] = _overall(rank, scored_keys, name)
        v["scores"]["gen_spread"] = _swing(rank, scored_keys, name)
    order = sorted(rows, key=lambda n: rows[n]["overall_rank"] or 999)

    groups = [{"label": gl, "metrics": [{"key": k, "label": lb} for k, lb in ms]}
              for gl, _, ms in GEN_SCORED]
    groups.append({"label": "Generalization", "metrics": [
        {"key": "gen_spread", "label": "Rank swing", "hint": "worst-best over hue+spacing"}]})
    groups.append({"label": "Robustness · physics (gate, not scored)", "scored": False, "metrics": [
        {"key": "rt_srgb", "label": "RT sRGB", "hint": "round-trip max error"},
        {"key": "rt_rec2020", "label": "RT Rec2020", "hint": "round-trip at wide gamut"},
        {"key": "nan_rec2020", "label": "Rec2020 NaN%", "hint": "% non-finite at wide gamut"}]})
    return {
        "title": "Generation - color-synthesis quality",
        "subtitle": ("Only generation-relevant properties: hue-constancy (gradients & shades keep "
                     "their hue) and even spacing (no banding), plus a physics robustness gate. "
                     "Difference / discrimination metrics live on the Measurement board - they judge "
                     "color-matching, not generation. Lower = better."),
        "groups": groups,
        "spaces": [{"name": n, "is_helm": rows[n]["is_helm"], "scores": rows[n]["scores"],
                    "overall_rank": rows[n]["overall_rank"]} for n in order],
        "winner": order[0] if order else None,
    }


def main():
    print("── forward pool (human_pool + robustness) ────────")
    fwd = compute_forward()
    print("\n── Measurement board ─────────────────────────────")
    meas = measurement_board(fwd)
    print(f"  winner: {meas['winner']} | {len(meas['spaces'])} entrants")
    print("\n── Generation board ──────────────────────────────")
    gen = generation_board(fwd)
    print(f"  winner: {gen['winner']} | {len(gen['spaces'])} spaces")

    out = {"generated": "2026-07-10", "boards": {"measurement": meas, "generation": gen}}
    dest = os.path.join(_ROOT, "docs", "leaderboard.json")
    json.dump(out, open(dest, "w"), indent=2)
    with open(os.path.join(os.path.dirname(dest), "leaderboard-data.js"), "w") as f:
        f.write("window.LEADERBOARD = " + json.dumps(out) + ";\n")
    print(f"\n  wrote {dest} + leaderboard-data.js")


if __name__ == "__main__":
    main()
