"""Regression tests for the 2026-07-08 fairness/correctness fixes.

Each test pins one fix from the full-repo audit:
  1. NaN/inf scores can never win/tie and the verdict is CLI-order-independent
  2. All three verdict paths (solo, tiered, fair) share ONE tie tolerance
  3. HumanFB is rank-only: Spearman ρ correct, excluded from pooled STRESS
  4. MacAdam isotropy uses real ellipse geometry (OKLab beats CIELab — the
     pre-fix metric rewarded the OPPOSITE direction)
  5. Gradient subset bucketing is space-independent (fixed CIELab frame)
  6. heuristic_proxy tier exists and is excluded from headline + fair weight
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from colorbench.core.comparison import compare_spaces, TIE_TOLERANCE


def _rt(err):
    return {"roundtrip": {"srgb_full_16M": {"max_error": err},
                          "p3_full_16M": {"max_error": err},
                          "rec2020_2M_uniform": {"max_error": err}}}


def test_nan_inf_never_wins_and_order_independent():
    good = _rt(1e-8)
    bad = {"roundtrip": {"srgb_full_16M": {"max_error": float("nan")},
                         "p3_full_16M": {"max_error": float("inf")},
                         "rec2020_2M_uniform": {"max_error": 1e-8}}}
    for order in [("GOOD", "BAD"), ("BAD", "GOOD")]:
        rbs = {n: (good if n == "GOOD" else bad) for n in order}
        comp = compare_spaces(rbs)
        assert comp.solo_wins["GOOD"] == 2, comp.solo_wins
        assert comp.solo_wins["BAD"] == 0, comp.solo_wins
        key = next(iter(comp.head_to_head))
        h = comp.head_to_head[key]
        wins_good = h["w1"] if key[0] == "GOOD" else h["w2"]
        assert wins_good == 2 and h["tie"] == 1, (order, h)


def test_tie_tolerance_agrees_across_verdict_paths():
    from colorbench.core.judge_provenance import tiered_winhist
    from colorbench.core.fair_verdict import fair_winhist

    just_inside = 1.0 + TIE_TOLERANCE * 0.7   # tie in every path
    just_outside = 1.0 + TIE_TOLERANCE * 2.0  # win in every path

    for other, expect_tie in [(just_inside, True), (just_outside, False)]:
        comp = compare_spaces({
            "A": {"roundtrip": {"srgb_full_16M": {"max_error": 1.0}}},
            "B": {"roundtrip": {"srgb_full_16M": {"max_error": other}}},
        })
        th = tiered_winhist(comp.tests, "A", "B")["_headline"]
        fh = fair_winhist(comp, "A", "B")
        solo_tied = comp.solo_wins == {"A": 0, "B": 0}
        assert solo_tied == expect_tie, (other, comp.solo_wins)
        assert (th["tie"] == 1) == expect_tie, (other, th)
        assert (fh["raw_tie"] == 1) == expect_tie, (other, fh)


def test_spearman_rho_ties_and_order_independence():
    from colorbench.core.metric_eval import spearman_rho

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(spearman_rho(a, a) - 1.0) < 1e-12
    assert abs(spearman_rho(a, -a) + 1.0) < 1e-12

    rng = np.random.default_rng(0)
    de = rng.random(1000)
    dv = np.digitize(de + rng.normal(0, 0.2, 1000), [0.2, 0.4, 0.6, 0.8]).astype(float)
    p = rng.permutation(1000)
    assert abs(spearman_rho(de, dv) - spearman_rho(de[p], dv[p])) < 1e-12


def test_humanfb_rank_only_in_summary():
    """_print_summary must keep *_rank_only datasets out of the STRESS table
    and the avg ranking (they'd otherwise pollute the pooled verdict)."""
    import io
    from contextlib import redirect_stdout
    from colorbench.core.metric_eval import _print_summary

    results = {
        "COMBVD": {"MetricSpace": 30.0, "OKLab": 35.0},
        "HumanFeedback_rank_only": {"MetricSpace": 0.7, "OKLab": 0.65},
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_summary(results)
    out = buf.getvalue()
    # avg ranking must be COMBVD-only: avg(MetricSpace)=30.00, not (30+0.7)/2
    assert "avg=30.00" in out, out
    assert "RANK-ONLY" in out and "ρ=0.7000" in out, out


def test_humanfb_registry_is_rank_only():
    from colorbench.core import human_pool as hp
    entries = [e for e in hp.REGISTRY if e[1] == "helmlabfb"]
    assert len(entries) == 1
    prop, _, _, key, _ = entries[0]
    assert key == "spearman_rho", "helmlabfb must be scored rank-only"
    assert prop not in hp._LOWER_BETTER, \
        "helmlabfb's property must stay out of the headline win count"


def test_macadam_uses_real_ellipses():
    """With real JND ellipse geometry, OKLab must beat CIELab (lower ratio).
    The pre-fix circle metric rewarded the opposite direction."""
    import torch  # noqa: F401  (skip cleanly if torch missing)
    from colorbench.run import build_space
    from colorbench.core.metrics.macadam import measure_macadam_isotropy

    r = {}
    for name in ("oklab", "cielab"):
        sp = build_space(name, None, "cpu")
        r[name] = measure_macadam_isotropy(sp)
        assert r[name]["n_centers"] == 25, r[name]
        assert r[name]["mean_ratio"] > 1.5, \
            "real anisotropy should be visible (not the old circle artifact)"
    assert r["oklab"]["mean_ratio"] < r["cielab"]["mean_ratio"], r


def test_gradient_buckets_space_independent():
    """Subset masks depend only on the sRGB pair endpoints (fixed CIELab
    frame), never on the candidate space's own coordinates."""
    import torch
    from colorbench.core.metrics.gradients import _subset_cvs, _subset_masks
    from colorbench.core.metrics._common import xyz_to_cielab, vec, _D65_LIST

    torch.manual_seed(0)
    xyz = torch.rand(200, 2, 3, dtype=torch.float64) * 0.8 + 0.05
    d65 = vec(_D65_LIST, "cpu", torch.float64)
    ref1 = xyz_to_cielab(xyz[:, 0], d65)
    ref2 = xyz_to_cielab(xyz[:, 1], d65)
    cvs_a = torch.rand(200, dtype=torch.float64)
    masks = _subset_masks(ref1, ref2)
    out1 = _subset_cvs(cvs_a, masks)
    out2 = _subset_cvs(cvs_a, _subset_masks(ref1, ref2))
    assert out1 == out2
    assert set(out1) == {"cv_bright", "cv_dark", "cv_high_chroma",
                         "cv_cross_lightness", "cv_near_achromatic"}


def test_proxy_tier_excluded_everywhere():
    from colorbench.core.judge_provenance import (
        provenance_of, tiered_winhist, TIER_PROXY, TRUSTWORTHY_TIERS)
    from colorbench.core.fair_verdict import metric_weight

    assert provenance_of("hue.rms") == TIER_PROXY
    assert provenance_of("specials.yellow_chroma") == TIER_PROXY
    assert TIER_PROXY not in TRUSTWORTHY_TIERS
    assert metric_weight("hue") == 0.0
    assert metric_weight("specials") == 0.0

    comp = compare_spaces({
        "A": {"specials": {"yellow_chroma": 0.9},
              "roundtrip": {"srgb_full_16M": {"max_error": 1e-9}}},
        "B": {"specials": {"yellow_chroma": 0.5},
              "roundtrip": {"srgb_full_16M": {"max_error": 5e-8}}},
    })
    th = tiered_winhist(comp.tests, "A", "B")
    assert th["_headline"]["n"] == 1          # only roundtrip
    assert th[TIER_PROXY]["n"] == 1           # specials visible, separate


def test_bootstrap_ci_overrides_threshold_both_ways():
    """decide_outcome: paired-bootstrap CI decides when per-item data exists.
    Case A: 0.5% score gap (threshold would say TIE) but every item is
    consistently lower → statistically a WIN.
    Case B: 5% score gap (threshold would say WIN) but item-level noise
    swamps it → statistically a TIE."""
    rng = np.random.default_rng(7)
    base = rng.random(400) + 1.0

    # Case A: consistent tiny improvement
    items_a = (base * 0.995).tolist()
    res_a = {"roundtrip": {"srgb_full_16M": {"max_error": float(np.mean(items_a))},
                           "_bootstrap": {"srgb_full_16M.max_error":
                                          {"items": items_a, "stat": "mean"}}}}
    res_b = {"roundtrip": {"srgb_full_16M": {"max_error": float(base.mean())},
                           "_bootstrap": {"srgb_full_16M.max_error":
                                          {"items": base.tolist(), "stat": "mean"}}}}
    comp = compare_spaces({"A": res_a, "B": res_b})
    assert comp.tests[0].ci_based
    assert comp.solo_wins["A"] == 1, comp.solo_wins  # threshold alone would tie

    # Case B: 5% gap, huge unpaired noise
    noisy_a = (base + rng.normal(0, 1.5, 400) + 0.05).tolist()
    noisy_b = (base + rng.normal(0, 1.5, 400)).tolist()
    res_a2 = {"roundtrip": {"srgb_full_16M": {"max_error": float(np.mean(noisy_a))},
                            "_bootstrap": {"srgb_full_16M.max_error":
                                           {"items": noisy_a, "stat": "mean"}}}}
    res_b2 = {"roundtrip": {"srgb_full_16M": {"max_error": float(np.mean(noisy_b))},
                            "_bootstrap": {"srgb_full_16M.max_error":
                                           {"items": noisy_b, "stat": "mean"}}}}
    comp2 = compare_spaces({"A": res_a2, "B": res_b2})
    assert comp2.solo_wins == {"A": 0, "B": 0}, comp2.solo_wins


def test_bootstrap_metrics_expose_items():
    """The bootstrap-enabled metric groups must ship per-item payloads."""
    import torch  # noqa: F401
    from colorbench.run import build_space
    from colorbench.core.metrics.macadam import measure_macadam_isotropy
    from colorbench.core.metrics.independent import measure_hung_berns

    sp = build_space("oklab", None, "cpu")
    mac = measure_macadam_isotropy(sp)
    assert len(mac["_bootstrap"]["mean_ratio"]["items"]) == 25
    hb = measure_hung_berns(sp)
    items = hb["_bootstrap"]["mean_mad_deg"]["items"]
    assert len(items) == hb["n_hues"] and len(items) >= 4


def test_contamination_excludes_in_sample_wins():
    """A space declaring trained_on=[munsell] cannot win munsell_value, and
    the pair is skipped (not counted) in the tiered head-to-head."""
    from colorbench.core.judge_provenance import tiered_winhist

    fit = {"trained_on": ["munsell"],
           "munsell_value": {"dL_cv": 0.5},
           "roundtrip": {"srgb_full_16M": {"max_error": 1e-8}}}
    clean = {"munsell_value": {"dL_cv": 4.0},
             "roundtrip": {"srgb_full_16M": {"max_error": 1e-8}}}
    comp = compare_spaces({"FitSpace": fit, "Clean": clean})
    mv = [tr for tr in comp.tests if tr.metric.result_key == "munsell_value"][0]
    assert mv.contaminated == {"FitSpace": "full"}
    assert mv.winner == "Clean"
    th = tiered_winhist(comp.tests, "FitSpace", "Clean")["_headline"]
    assert th["n"] == 1 and th["tie"] == 1  # munsell pair skipped entirely


def test_new_pool_judges_sane():
    """2026-07 judge expansion: new judges return sane values and score a
    degenerate space (raw XYZ) WORSE than OKLab — the direction check that
    gates validated=True."""
    import torch  # noqa: F401
    from colorbench.run import build_space
    from colorbench.core import human_pool as hp

    if not hp.pool_available():
        return  # pool not on this machine

    class RawXYZ:
        name = "rawxyz"
        def forward(self, x):
            return x

    ok = build_space("oklab", None, "cpu")
    raw = RawXYZ()
    for fn in (hp.judge_brown_1957, hp.judge_lab_ellipsoid,
               hp.judge_tolerance_vectors):
        r_ok, r_raw = fn(ok), fn(raw)
        assert r_ok.get("mean_cv") is not None, r_ok
        assert 0 < r_ok["mean_cv"] < 2.0, r_ok
        assert r_raw["mean_cv"] > r_ok["mean_cv"], (fn.__name__, r_raw, r_ok)

    # registry: the expansion datasets are wired
    wired = {e[1] for e in hp.REGISTRY}
    for ds in ("brown_1957_12obs_ellipsoids", "wyszecki_fielder_1971_ellipsoids",
               "huang_2012_cielab_ellipses",
               "berns_1991_rit_dupont_tolerance_vectors", "hong_2025_ellipsoids",
               "sanders_wyszecki_1964_HK"):
        assert ds in wired, ds
    # hong promoted to validated (2026-07-08: OSF measured-primaries
    # calibration replaced the sRGB approximation)
    hong = [e for e in hp.REGISTRY if e[1] == "hong_2025_ellipsoids"][0]
    assert hong[4] is True

    # xiao unique-hue judge: constant-hue property on unique-hue loci —
    # degenerate raw-XYZ must be worse than a real perceptual space
    x_ok = hp.judge_unique_hues(ok)
    x_raw = hp.judge_unique_hues(raw)
    if "skipped" not in x_ok:
        assert 0 < x_ok["mean_mad_deg"] < 15
        assert x_raw["mean_mad_deg"] > x_ok["mean_mad_deg"]

    # naming + observer_variance stay diagnostic (no canonical direction /
    # known space-insensitive)
    for ds in ("wcs", "asano_observers"):
        entry = [e for e in hp.REGISTRY if e[1] == ds][0]
        assert entry[4] is False
        assert entry[0] not in hp._LOWER_BETTER

    # OSA-UCS spacing judge (validated): the independent, non-Munsell spacing
    # anchor — degenerate raw-XYZ must be worse than a real uniform space
    osa_ok = hp.judge_osa_spacing(ok)
    osa_raw = hp.judge_osa_spacing(raw)
    if "skipped" not in osa_ok:
        assert 0 < osa_ok["mean_cv"] < 1
        assert osa_raw["mean_cv"] > osa_ok["mean_cv"]
        entry = [e for e in hp.REGISTRY if e[1] == "osa_ucs_1974"][0]
        assert entry[4] is True and entry[0] == "spacing"


def test_ruler_sensitivity_flag():
    """Gradient metrics ship per-ruler aggregates; compare_spaces flags each
    verdict as robust (same winner under every consensus member) or sensitive."""
    import torch
    from colorbench.run import build_space
    from colorbench.core.pairs import generate_all_pairs
    from colorbench.core.metrics.gradients import measure_gradients

    pairs, labels = generate_all_pairs("cpu")
    pairs, labels = pairs[:120], labels[:120]
    res = {}
    for name in ("oklab", "cielab"):
        sp = build_space(name, None, "cpu")
        res[sp.name] = {"gradients": measure_gradients(sp, pairs, labels)}

    rs = res["OKLab"]["gradients"]["_ruler_sensitivity"]
    assert "overall.cv_mean" in rs and len(rs["overall.cv_mean"]) >= 2

    comp = compare_spaces(res)
    flagged = [tr for tr in comp.tests if tr.ruler_flag in ("robust", "sensitive")]
    assert flagged, "no gradient metric received a ruler flag"


def test_api_wrap_and_scorecard():
    """cb.wrap: any space enters via two callables; scorecard renders the
    property × space karne with per-property winners."""
    import torch
    import colorbench as cb

    fwd = lambda xyz: xyz.clamp(min=0).pow(1.0 / 3.0)
    inv = lambda lab: lab.pow(3)
    sp = cb.wrap(fwd, inv, name="CubeRootXYZ", trained_on=["munsell"])
    x = torch.rand(50, 3, dtype=torch.float64)
    assert (sp.inverse(sp.forward(x)) - x).abs().max().item() < 1e-12
    assert sp.trained_on == ["munsell"]

    from colorbench.core import human_pool as hp
    if not hp.pool_available():
        return
    from colorbench.run import build_space
    from colorbench.core.scorecard import scorecard
    txt = scorecard({"OKLab": build_space("oklab", None, "cpu"),
                     "CubeRootXYZ": sp})
    assert "★" in txt and "ÖZELLİK KAZANANLARI" in txt
    # contaminated munsell cell must be marked in-sample, not starred
    munsell_line = [l for l in txt.splitlines() if l.strip().startswith("munsell ")]
    assert munsell_line and "⚠" in munsell_line[0]


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        try:
            fn()
            print(f"  ✓ PASS  {name}")
        except AssertionError as e:
            print(f"  ✗ FAIL  {name}: {e}")
