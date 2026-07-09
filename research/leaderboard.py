"""Build the ColorBench leaderboard — the comprehensive, permanent version.

TWO boards (ölçüm ≠ üretim), every invertible colour-science space + helmlab,
scored on the FULL validated human-data pool.

  MEASUREMENT — "which model best predicts human color difference?"
      STRESS (+ bootstrap CI95) on 5 independent difference sources (COMBVD
      unpacked into BFD-P / Leeds / Witt / RIT-DuPont, plus held-out MacAdam
      1974). Entrants: helmlab metricspace (learned distance), the CIEDE2000 /
      CIE94 formulas, and every colour-science space via its Euclidean ΔE.
      Overfit Δrank (helmlab only) + Rank swing.

  GENERATION — "which space best matches human vision to generate color in?"
      Ranked on the FULL validated human pool — 16 datasets across 5 categories:
      hue (4), discrimination (5), 3-D discrimination (4), tolerance (2),
      spacing (1) — each category weighing equally. Entrants: every invertible
      colour-science space + helmlab genspace. Rank swing over all 16 datasets.
      (No CIELab-referenced engineering metric is scored — that ruler flatters
      CIELAB; it is left off entirely.)

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

import colour  # noqa: E402
from core import data  # noqa: E402
from core import human_pool as hp  # noqa: E402

HELMGEN = "/Volumes/harici_ssd/color-space/helmlab-main-repo/checkpoints/genspace_v0.11.1.json"
HELMMETRIC = "/Volumes/harici_ssd/color-space/helmlab-main-repo/checkpoints/metricspace_v21.json"
HELM_SRC = "/Volumes/harici_ssd/color-space/helmlab-main-repo/src"

# ── colour-science invertible perceptual/uniform spaces ─────────────────────
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
    """A colour-science model as a ColorBench forward-space (numpy forward+inverse)."""
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


# ── Robustness gate: pure physics (round-trip + wide-gamut finiteness), no
# rival-space ruler. Measures whether a space produces valid, invertible colors
# across the sRGB / Rec2020 gamuts — it can't flatter any family because it
# compares only to the identity and to "is this a finite number". Not scored.
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


# ═══════════════════════════ MEASUREMENT board ═════════════════════════════
def _build_metricspace():
    if HELM_SRC not in sys.path:
        sys.path.insert(0, HELM_SRC)
    from core.metric_eval import _load_metric_space
    return _load_metric_space(HELMMETRIC, os.path.dirname(HELM_SRC))


def measurement_board():
    from core.metric_eval import (load_combvd_from_xlsx, load_macadam1974,
                                  _ciede2000, _cie94_de, _cat_to_d65, stress)
    from core.bootstrap import stress_ci
    base = data.baseline_dir()
    metric = _build_metricspace()
    wrappers = [ColourWrapper(n) for n in COLOUR_SPACES]

    recs = load_combvd_from_xlsx(base)
    def subset(pred):
        r = [x for x in recs if pred(x["dataset"])]
        return ([x["xyz1"] for x in r], [x["xyz2"] for x in r],
                [x["white"] for x in r], [x["dv"] for x in r])
    mx1, mx2, mw, mdv = load_macadam1974(base)
    mac = (list(mx1), list(mx2), [mw] * len(mdv), list(mdv))
    DATASETS = [
        ("bfd", "BFD-P", "in", subset(lambda s: s.startswith("BFD-P"))),
        ("leeds", "Leeds", "in", subset(lambda s: s == "LEEDS")),
        ("witt", "Witt", "in", subset(lambda s: s == "WITT")),
        ("rit", "RIT-DuPont", "in", subset(lambda s: s == "RIT-DuPont")),
        ("macadam", "MacAdam 74", "held", mac),
    ]
    dkeys = [d[0] for d in DATASETS]
    in_keys = [d[0] for d in DATASETS if d[2] == "in"]

    # every entrant is (name, is_helm, de-function taking D65-adapted or raw pairs)
    rows = {}
    def add(name, is_helm):
        rows[name] = {"is_helm": is_helm, "scores": {}, "ci": {}}
    add("helmlab metricspace", True)
    add("CIEDE2000", False)
    add("CIE94", False)
    for w in wrappers:
        add(w.name, False)

    for key, label, kind, (x1r, x2r, wh, dv) in DATASETS:
        x1r = np.asarray(x1r, float); x2r = np.asarray(x2r, float)
        wh = np.asarray(wh, float); dv = np.asarray(dv, float)
        n = len(dv)
        x1d = np.array([_cat_to_d65(x1r[i], wh[i]) for i in range(n)])
        x2d = np.array([_cat_to_d65(x2r[i], wh[i]) for i in range(n)])
        print(f"  scoring {label} (n={n}) ...", flush=True)

        des = {"helmlab metricspace": np.asarray(metric.distance(x1d, x2d)).ravel(),
               "CIEDE2000": np.array([_ciede2000(x1r[i:i+1], x2r[i:i+1], wh[i]) for i in range(n)]).ravel(),
               "CIE94": np.array([_cie94_de(x1r[i:i+1], x2r[i:i+1], wh[i]) for i in range(n)]).ravel()}
        for w in wrappers:
            c1 = w.forward(x1d); c2 = w.forward(x2d)
            des[w.name] = np.sqrt(((c1 - c2) ** 2).sum(-1))

        for name, de in des.items():
            de = np.asarray(de, float).ravel()
            ok = np.isfinite(de)
            if ok.sum() < 3:
                rows[name]["scores"][key] = None; rows[name]["ci"][key] = None; continue
            s = float(stress(de[ok], dv[ok]))
            lo, hi = stress_ci(de[ok], dv[ok])
            rows[name]["scores"][key] = round(s, 2)
            rows[name]["ci"][key] = [round(float(lo), 1), round(float(hi), 1)]

    for m in rows:
        vals = [rows[m]["scores"][k] for k in dkeys if isinstance(rows[m]["scores"].get(k), (int, float))]
        rows[m]["scores"]["mean"] = round(float(np.mean(vals)), 2) if vals else None

    rank = _ranks(rows, dkeys + ["mean"])
    for m, v in rows.items():
        r_in = [rank[k][m] for k in in_keys if m in rank.get(k, {})]
        r_held = rank["macadam"].get(m)
        v["scores"]["overfit"] = (int(round(r_held - np.mean(r_in)))
                                  if (r_held and r_in and v["is_helm"]) else None)
        v["scores"]["gen_spread"] = _swing(rank, dkeys, m)
        v["overall_rank"] = rank["mean"].get(m)
    order = sorted(rows, key=lambda m: rows[m]["overall_rank"] or 999)

    return {
        "title": "Measurement — color-difference prediction",
        "subtitle": ("STRESS (+ CI95 on hover) on 5 independent difference sources, each model with "
                     "its own ΔE (metricspace = learned distance; CIEDE2000 / CIE94 = formulas; every "
                     "colour-science space = Euclidean ΔE). Lower = closer to human."),
        "holdout_note": ("The 4 COMBVD components (BFD-P, Leeds, Witt, RIT-DuPont) may be in-sample for "
                         "the COMBVD-fit helmlab metricspace; MacAdam 1974 is the held-out check — where "
                         "it does NOT win (CAM16-UCS does). Overfit Δrank (helmlab only) = held-out rank "
                         "− mean in-sample rank. Rank swing = worst−best rank across all 5 sources."),
        "groups": [
            {"label": "COMBVD components · STRESS (possible in-sample)", "metrics": [
                {"key": "bfd", "label": "BFD-P"}, {"key": "leeds", "label": "Leeds"},
                {"key": "witt", "label": "Witt"}, {"key": "rit", "label": "RIT-DuPont"}]},
            {"label": "Held-out", "metrics": [{"key": "macadam", "label": "MacAdam 74"}]},
            {"label": "Composite", "metrics": [{"key": "mean", "label": "Mean STRESS"}]},
            {"label": "Generalization", "metrics": [
                {"key": "overfit", "label": "Overfit Δrank", "signed": True, "hint": "helmlab only"},
                {"key": "gen_spread", "label": "Rank swing", "hint": "worst−best over 5 sources"}]},
        ],
        "spaces": [{"name": m, "is_helm": rows[m]["is_helm"], "scores": rows[m]["scores"],
                    "ci": rows[m]["ci"], "overall_rank": rows[m]["overall_rank"]}
                   for m in order],
        "winner": order[0] if order else None,
    }


# ═══════════════════════════ GENERATION board ═════════════════════════════
# full validated human pool: 16 datasets across 5 categories (all forward-only)
HUMAN = [
    ("Hue · human data", [
        ("hung_berns", "Hung-Berns"), ("ebner_fairchild", "Ebner-F."),
        ("munsell", "Munsell"), ("xiao_unique_hues", "Xiao")]),
    ("Discrimination · human", [
        ("macadam1942", "MacAdam42"), ("luo_rigg_ellipses", "Luo-Rigg"),
        ("alder1982", "Alder"), ("regan_1994_cvd_ellipses", "Regan"),
        ("hong_2025_ellipsoids", "Hong")]),
    ("3-D discrimination · human", [
        ("koenderink_2026_3d_metric_field", "Koenderink"),
        ("brown_1957_12obs_ellipsoids", "Brown-57"),
        ("wyszecki_fielder_1971_ellipsoids", "Wyszecki-F"),
        ("brown_macadam_1949_ellipsoids", "Brown-MacAdam")]),
    ("Tolerance · human", [
        ("berns_1991_rit_dupont_tolerance_vectors", "RIT-DuPont"),
        ("huang_2012_cielab_ellipses", "Huang")]),
    ("Spacing · human", [("osa_ucs_1974", "OSA-UCS")]),
]
_PROP_OF = {"Hue · human data": "hue", "Discrimination · human": "discrimination",
            "3-D discrimination · human": "3d_discrim", "Tolerance · human": "tolerance",
            "Spacing · human": "spacing"}

# diagnostic human judges — real human data but lightly validated, so SHOWN (with
# honest labels) but NOT scored. (naming/WCS is dropped: it's space-insensitive,
# ~0.61 for every space, so it carries no signal.) Each group notes its ruler +
# direction. lower=better unless the label says ↑.
DIAG_HUMAN = [
    ("H-K brightness · Spearman ρ, ↑ better · diagnostic", "hk_mechanism", [
        ("wyszecki_1967_osa_tiles", "Wyszecki-67"),
        ("zhang_2023_laser_display_brightness", "Zhang-23"),
        ("sanders_wyszecki_1964_HK", "Sanders-64")]),
    ("H-K object lightness · STRESS · diagnostic", "hk_object", [
        ("fairchild_pirrotta_1991", "Fairchild-P")]),
    ("Chromatic adaptation · ΔE · diagnostic", "adaptation", [
        ("corresponding_colours", "Corr-colours")]),
    ("Observer metamerism · spread · diagnostic", "observer_variance", [
        ("asano_observers", "Asano")]),
]


def _build_genspace():
    from run import build_space, get_device
    device, dtype, _ = get_device()
    sp = build_space("genspace", HELMGEN, device, dtype=dtype)
    sp.name = "helmlab genspace"
    return sp


def generation_board():
    spaces = [ColourWrapper(n) for n in COLOUR_SPACES]
    try:
        spaces.append(_build_genspace())
    except Exception as e:
        print(f"  genspace skipped: {e}")

    import torch
    rows = {}
    for sp in spaces:
        try:
            # validated_only=False also runs the diagnostic judges (H-K,
            # adaptation, observer). Fall back to validated-only if a diagnostic
            # judge crashes on some space, so the space still appears.
            try:
                panel = hp.evaluate_space_on_pool(sp, validated_only=False)["by_property"]
            except Exception:
                panel = hp.evaluate_space_on_pool(sp, validated_only=True)["by_property"]
        except Exception as e:
            print(f"  gen skip {sp.name}: {type(e).__name__}: {e}"); continue
        sc = {}
        for gl, metrics in HUMAN:
            prop = panel.get(_PROP_OF[gl], {})
            for key, _ in metrics:
                v = prop.get(key)
                sc[key] = float(v) if isinstance(v, (int, float)) else None
        for _, propname, metrics in DIAG_HUMAN:
            prop = panel.get(propname, {})
            for key, _ in metrics:
                v = prop.get(key)
                sc[key] = float(v) if isinstance(v, (int, float)) else None
        # robustness gate (physics) — numpy for colour wrappers, torch adapters
        # for genspace
        if isinstance(sp, ColourWrapper):
            rob = _robustness(sp.forward, sp.inverse)
        else:
            fwd = lambda x: sp.forward(torch.as_tensor(x, dtype=sp.dtype, device=sp.device)).detach().cpu().numpy()
            inv = lambda c: sp.inverse(torch.as_tensor(c, dtype=sp.dtype, device=sp.device)).detach().cpu().numpy()
            rob = _robustness(fwd, inv)
        sc.update(rob)
        rows[sp.name] = {"is_helm": sp.name.startswith("helmlab"), "scores": sc}
        print(f"  {sp.name:16} hue_hb={sc.get('hung_berns')} koen={sc.get('koenderink_2026_3d_metric_field')} "
              f"rit={sc.get('berns_1991_rit_dupont_tolerance_vectors')}", flush=True)

    all_keys = [k for _, ms in HUMAN for k, _ in ms]
    rank = _ranks(rows, all_keys)
    for n, v in rows.items():
        cats = []
        for gl, metrics in HUMAN:
            crs = [rank[k][n] for k, _ in metrics if n in rank.get(k, {})]
            if crs:
                cats.append(sum(crs) / len(crs))
        v["overall_rank"] = round(sum(cats) / len(cats), 2) if cats else None
        v["scores"]["gen_spread"] = _swing(rank, all_keys, n)
    order = sorted(rows, key=lambda n: rows[n]["overall_rank"] or 999)

    groups = [{"label": gl, "scored": True,
               "metrics": [{"key": k, "label": lb} for k, lb in ms]} for gl, ms in HUMAN]
    groups.append({"label": "Generalization", "scored": True, "metrics": [
        {"key": "gen_spread", "label": "Rank swing", "hint": "worst−best rank over 16 datasets"}]})
    # diagnostic human judges (real data, lightly validated) — shown, not scored
    for gl, propname, metrics in DIAG_HUMAN:
        groups.append({"label": gl, "scored": False,
                       "metrics": [{"key": k, "label": lb} for k, lb in metrics]})
    # physics-only robustness gate — NOT scored (no ruler, so no family bias);
    # flags spaces that lose invertibility or go non-finite at wide gamut
    groups.append({"label": "Robustness · physics (gate, not scored)", "scored": False, "metrics": [
        {"key": "rt_srgb", "label": "RT sRGB", "hint": "round-trip max error (invertibility)"},
        {"key": "rt_rec2020", "label": "RT Rec2020", "hint": "round-trip max error at wide gamut"},
        {"key": "nan_rec2020", "label": "Rec2020 NaN%", "hint": "% of wide-gamut points that go non-finite"}]})
    return {
        "title": "Generation — match to human vision",
        "subtitle": ("RANKED on the full validated human pool — 16 datasets in 5 categories (hue, "
                     "discrimination, 3-D discrimination, tolerance, spacing), each category equal "
                     "weight. Everything else is SHOWN but not scored, honestly labeled: diagnostic "
                     "human judges (H-K brightness/lightness, chromatic adaptation, observer "
                     "metamerism) and a physics robustness gate. Lower = better except columns marked "
                     "↑. No CIELab-referenced engineering metric is scored (that ruler flatters CIELAB)."),
        "groups": groups,
        "spaces": [{"name": n, "is_helm": rows[n]["is_helm"], "scores": rows[n]["scores"],
                    "overall_rank": rows[n]["overall_rank"]} for n in order],
        "winner": order[0] if order else None,
    }


def main():
    print("── Measurement board ─────────────────────────────")
    meas = measurement_board()
    print(f"  winner: {meas['winner']} | {len(meas['spaces'])} models")
    print("\n── Generation board ──────────────────────────────")
    gen = generation_board()
    print(f"  winner: {gen['winner']} | {len(gen['spaces'])} spaces")

    out = {"generated": "2026-07-09", "boards": {"measurement": meas, "generation": gen}}
    dest = os.path.join(_ROOT, "docs", "leaderboard.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    with open(os.path.join(os.path.dirname(dest), "leaderboard-data.js"), "w") as f:
        f.write("window.LEADERBOARD = " + json.dumps(out) + ";\n")
    print(f"\n  wrote {dest} + leaderboard-data.js")


if __name__ == "__main__":
    main()
