"""Human-perception validation of the ENGINEERING metrics — without new observers.

ColorBench's compare-mode metrics (gradient CV, banding, hue geometry, ...) are
algorithmic. The honest question Grok raised: do they PREDICT human perception,
or are they just numbers? We can't run new observers, but we have the human_pool
(46 real datasets). So we ask the answerable version:

    across many colour spaces, does each engineering metric TRACK the
    human-grounded property it claims to measure?

Method: build N built-in spaces; for each, collect (a) its engineering metric
scores and (b) its human_pool per-property scores; then Spearman-correlate each
engineering metric against each human property ACROSS spaces. A high |rho| means
the engineering metric is human-predictive; near-zero means it measures
something the human data doesn't care about.

This is a meta-analysis of the metrics themselves, not a candidate ranking.
Run:  python3 research/metric_predictivity.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# literature spaces buildable with no --json checkpoint
SPACES = ["oklab", "cielab", "ipt", "jzazbz", "ictcp", "cam16ucs",
          "din99d", "perceptia", "engineered"]

# engineering metric (result_key, dot-path, lower_is_better) -> the human
# property it is SUPPOSED to track. We validate each pairing.
CLAIMS = [
    # engineering spacing metrics vs human uniform spacing / discrimination
    ("gradients", "overall.cv_mean", True, "spacing"),
    ("gradients", "overall.cv_high_chroma", True, "discrimination"),
    ("banding", "total_duplicate_pct", True, "spacing"),
    # engineering hue geometry vs human constant-hue
    ("hue_leaf", "mean_max_drift_deg", True, "hue"),
    ("tint_shade_hue", "mean_max_drift_deg", True, "hue"),
    # engineering palette uniformity vs human spacing
    ("palette_uniformity", "mean_cv", True, "spacing"),
]


def _extract(results, key, path):
    from core.comparison import _extract_score
    return _extract_score(results, key, path)


def main():
    from run import build_space, run_test, get_device
    from core import human_pool as hp
    device, dtype, dname = get_device()
    if not hp.pool_available():
        print("human pool unavailable — cannot validate."); return

    eng = {}     # space -> {(key,path): value}
    human = {}   # space -> {property: value}
    for name in SPACES:
        try:
            sp = build_space(name, None, device, dtype=dtype)
            print(f"  evaluating {sp.name} ...", flush=True)
            rep = run_test(sp, device, dname)
            panel = hp.evaluate_space_on_pool(sp, validated_only=True)["by_property"]
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {e}"); continue
        eng[sp.name] = {(k, p): _extract(rep, k, p) for k, p, _, _ in CLAIMS}
        # per property, use the mean over that property's datasets (lower=better
        # for all validated properties used here)
        human[sp.name] = {
            prop: float(np.mean([v for v in d.values() if isinstance(v, (int, float))]))
            for prop, d in panel.items()
            if any(isinstance(v, (int, float)) for v in d.values())
        }

    names = list(eng)
    if len(names) < 4:
        print("too few spaces for a correlation."); return

    def spearman(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        ra = ra - ra.mean(); rb = rb - rb.mean()
        den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        return float((ra * rb).sum() / den) if den else 0.0

    print(f"\n  Human-predictivity of engineering metrics (N={len(names)} spaces)")
    print(f"  {'engineering metric':32} {'human property':14} {'rho':>6}  verdict")
    print("  " + "-" * 68)
    for k, p, lower, prop in CLAIMS:
        ev = [eng[n].get((k, p)) for n in names]
        hv = [human[n].get(prop) for n in names]
        pair = [(e, h) for e, h in zip(ev, hv)
                if isinstance(e, (int, float)) and isinstance(h, (int, float))]
        if len(pair) < 4:
            print(f"  {k+'.'+p:32.32} {prop:14} {'  n/a':>6}  (insufficient)")
            continue
        e_arr, h_arr = zip(*pair)
        rho = spearman(e_arr, h_arr)   # both lower=better -> positive rho = agrees
        verdict = ("STRONG human-predictive" if rho >= 0.6 else
                   "weak" if rho >= 0.3 else
                   "NOT predictive" if abs(rho) < 0.3 else
                   "ANTI-correlated (suspect)")
        print(f"  {k+'.'+p:32.32} {prop:14} {rho:>6.2f}  {verdict}")
    print("\n  rho>0 = engineering metric ranks spaces the same way the human "
          "data does (both lower=better).\n  Metrics near 0 measure something "
          "the human panel doesn't reward — read them structurally, not as "
          "perceptual quality.")


if __name__ == "__main__":
    main()
