"""N-space comparison engine with winner logic and head-to-head matrix.

Every comparison metric has an explicit path from raw results to scalar score.
No implicit scoring. Fully auditable.
"""

from dataclasses import dataclass, field
import math


@dataclass
class MetricDef:
    """Definition of one comparison metric."""
    result_key: str       # Key in results dict (e.g. "roundtrip")
    score_path: str       # Dot-separated path to scalar (e.g. "srgb_full_16M.max_error")
    name: str             # Human-readable name
    category: str         # Category for grouping
    unit: str             # Display unit (%, deg, ratio, dE, etc.)
    lower_is_better: bool # Direction
    format_str: str = ""  # Optional format string (e.g. ".2e", ".1f")
    abs_tie: float = 0.0  # Absolute TIE threshold (0 = use relative only)


@dataclass
class TestResult:
    """Result of one metric across all spaces."""
    metric: MetricDef
    scores: dict          # {space_name: float}
    winner: str | None    # space name or None (tie)
    is_tie: bool
    ref_spaces: list      # spaces marked as self-referential for this metric
    items: dict = field(default_factory=dict)   # {space: {"items": [...], "stat": str}}
    winners: list = None  # all spaces statistically tied with the best
    ci_based: bool = False  # True = tie decided by paired-bootstrap CI
    ruler_flag: str = None  # "robust" | "sensitive" | None (no per-ruler data)
    contaminated: dict = field(default_factory=dict)  # {space: "full"|"partial"}


@dataclass
class Comparison:
    """Complete comparison result."""
    tests: list           # list[TestResult]
    space_names: list     # list[str]
    solo_wins: dict       # {space: int}
    shared_wins: dict     # {space: int}
    head_to_head: dict    # {(s1, s2): {"w1": int, "w2": int, "tie": int}}


# ═══════════════════════════════════════════════════════════════
#  METRIC DEFINITIONS — THE SINGLE SOURCE OF TRUTH
# ═══════════════════════════════════════════════════════════════
#
# Each entry maps a raw result path to a named comparison metric.
# This list completely defines what gets compared.

METRIC_DEFS = [
    # ── Numerical Stability ──
    MetricDef("roundtrip", "srgb_full_16M.max_error",
              "Round-trip sRGB 16.7M", "Numerical", "", True, ".2e"),
    MetricDef("roundtrip", "p3_full_16M.max_error",
              "Round-trip P3 16.7M", "Numerical", "", True, ".2e", 1e-13),
    MetricDef("roundtrip", "rec2020_2M_uniform.max_error",
              "Round-trip Rec2020 2.1M", "Numerical", "", True, ".2e", 1e-13),
    # Note: condition numbers come from jacobian, not stability
    # Stability has perturbation + near_black/near_white

    # ── Achromatic ──
    MetricDef("achromatic", "gray_ramp_srgb.max_chroma",
              "Gray ramp sRGB C*", "Achromatic", "C*", True, ".2e"),
    MetricDef("achromatic", "gray_ramp_pure.max_chroma",
              "Gray ramp pure C*", "Achromatic", "C*", True, ".2e"),

    # ── Gradient Quality ──
    MetricDef("gradients", "overall.cv_mean",
              "Gradient CV (mean)", "Gradient", "%", True, ".2f"),
    MetricDef("gradients", "overall.cv_p95",
              "Gradient CV (p95)", "Gradient", "%", True, ".2f"),
    MetricDef("gradients", "overall.drift_max_noncrossing",
              "Max hue drift (non-crossing)", "Gradient", "deg", True, ".1f"),
    MetricDef("gradients", "overall.banding_mean",
              "Banding mean", "Gradient", "", True, ".1f"),
    MetricDef("gradients", "overall.cv_max",
              "Worst-case gradient CV", "Gradient", "%", True, ".1f"),

    # ── Gradient Subsets ──
    MetricDef("gradients", "overall.cv_bright",
              "Bright gradient CV (L>0.6)", "Gradient", "%", True, ".2f"),
    MetricDef("gradients", "overall.cv_dark",
              "Dark gradient CV (L<0.4)", "Gradient", "%", True, ".2f"),
    MetricDef("gradients", "overall.cv_high_chroma",
              "High-chroma gradient CV", "Gradient", "%", True, ".2f"),
    MetricDef("gradients", "overall.cv_cross_lightness",
              "Cross-lightness gradient CV", "Gradient", "%", True, ".2f"),
    MetricDef("gradients", "overall.cv_near_achromatic",
              "Near-achromatic gradient CV", "Gradient", "%", True, ".2f"),

    # ── Hue ──
    MetricDef("hue", "hue_rms",
              "Hue RMS", "Hue", "deg", True, ".1f"),

    # ── Gamut Geometry ──
    MetricDef("gamut", "sRGB.valid_cusps",
              "sRGB valid cusps", "Gamut", "/360", False, "d"),
    MetricDef("gamut", "sRGB.monotonicity_violations",
              "sRGB mono violations", "Gamut", "", True, "d"),
    MetricDef("gamut", "sRGB.cliff_max",
              "sRGB cliff max", "Gamut", "%", True, ".1f"),
    MetricDef("gamut", "sRGB.volume_fraction",
              "Gamut volume fill", "Gamut", "%", False, ".1f"),
    MetricDef("gamut", "P3.valid_cusps",
              "P3 valid cusps", "Gamut", "/360", False, "d"),
    MetricDef("gamut", "P3.monotonicity_violations",
              "P3 mono violations", "Gamut", "", True, "d"),
    MetricDef("gamut", "P3.cliff_max",
              "P3 cliff max", "Gamut", "%", True, ".1f"),
    MetricDef("gamut", "Rec2020.monotonicity_violations",
              "Rec2020 mono violations", "Gamut", "", True, "d"),
    MetricDef("gamut", "Rec2020.cliff_max",
              "Rec2020 cliff max", "Gamut", "%", True, ".1f"),
    MetricDef("gamut", "P3.smoothness_max_jump",
              "P3 cusp smoothness", "Gamut", "", True, ".3f"),
    MetricDef("gamut", "Rec2020.smoothness_max_jump",
              "Rec2020 cusp smoothness", "Gamut", "", True, ".3f"),
    MetricDef("gamut", "P3.anomalies",
              "P3 gamut anomalies", "Gamut", "", True, "d"),
    MetricDef("gamut", "Rec2020.anomalies",
              "Rec2020 gamut anomalies", "Gamut", "", True, "d"),
    MetricDef("gamut", "P3.dead_zones",
              "P3 dead zones", "Gamut", "", True, "d"),
    MetricDef("gamut", "Rec2020.dead_zones",
              "Rec2020 dead zones", "Gamut", "", True, "d"),
    MetricDef("gamut", "sRGB.invalid_cusps",
              "sRGB invalid cusps", "Gamut", "", True, "d"),
    MetricDef("gamut", "P3.invalid_cusps",
              "P3 invalid cusps", "Gamut", "", True, "d"),
    MetricDef("gamut", "sRGB.smoothness_mean_jump",
              "sRGB cusp mean smoothness", "Gamut", "", True, ".4f"),
    MetricDef("gamut", "sRGB.boundary_bad_hues",
              "sRGB boundary bad hues", "Gamut", "", True, "d"),
    MetricDef("gamut", "P3.boundary_bad_hues",
              "P3 boundary bad hues", "Gamut", "", True, "d"),
    MetricDef("gamut", "Rec2020.boundary_bad_hues",
              "Rec2020 boundary bad hues", "Gamut", "", True, "d"),
    MetricDef("gamut", "P3.smoothness_mean_jump",
              "P3 cusp mean smoothness", "Gamut", "", True, ".4f"),
    MetricDef("gamut", "Rec2020.smoothness_mean_jump",
              "Rec2020 cusp mean smoothness", "Gamut", "", True, ".4f"),
    MetricDef("gamut", "sRGB.boundary_mean_rel_jump",
              "sRGB boundary mean jump", "Gamut", "", True, ".4f"),
    MetricDef("gamut", "P3.boundary_mean_rel_jump",
              "P3 boundary mean jump", "Gamut", "", True, ".4f"),
    MetricDef("gamut", "Rec2020.boundary_mean_rel_jump",
              "Rec2020 boundary mean jump", "Gamut", "", True, ".4f"),

    # ── Special Gradients ──
    MetricDef("specials", "yellow_chroma",
              "Yellow chroma", "Special", "", False, ".4f"),
    MetricDef("specials", "blue_white_midpoint.G_over_R",
              "Blue-White midpoint G/R", "Special", "", False, ".3f"),

    # ── Banding ──
    MetricDef("banding", "total_invisible_pct",
              "Invisible gradient steps", "Banding", "%", False, ".1f"),
    MetricDef("banding", "total_duplicate_pct",
              "Duplicate 8-bit steps", "Banding", "%", True, ".1f"),

    # ── CVD Accessibility ──
    MetricDef("cvd", "protan.worst_min_de",
              "CVD protan min step dE", "Accessibility", "dE", False, ".2f"),
    MetricDef("cvd", "deutan.worst_min_de",
              "CVD deutan min step dE", "Accessibility", "dE", False, ".2f"),

    # ── Hue Leaf ──
    MetricDef("hue_leaf", "max_deviation",
              "Hue leaf constancy", "Perceptual", "deg", True, ".1f"),

    # ── Animation ──
    MetricDef("animation", "_mean_cv",
              "Animation frame-to-frame CV", "Advanced", "%", True, ".1f"),

    # ── Advanced ──
    MetricDef("jacobian", "mean",
              "Jacobian condition", "Advanced", "", True, ".2f"),
    MetricDef("double_rt", "trips_1000.max_error",
              "1000-trip RT", "Advanced", "", True, ".2e"),
    MetricDef("quantization", "random_10k_exact_count",
              "8-bit exact/10K", "Advanced", "", False, "d"),
    MetricDef("channel_mono", "_total_violations",
              "Channel mono violations", "Advanced", "", True, "d"),

    # ── Perceptual Uniformity (NEW — from gpu_metrics_perceptual.py) ──
    MetricDef("munsell_value", "dL_cv",
              "Munsell Value uniformity", "Perceptual", "%", True, ".2f"),
    MetricDef("munsell_hue", "spacing_cv",
              "Munsell Hue spacing", "Perceptual", "%", True, ".1f"),
    MetricDef("macadam_isotropy", "mean_ratio",
              "MacAdam isotropy", "Perceptual", "ratio", True, ".2f"),
    MetricDef("hue_agreement", "mae_deg",
              "Hue agreement with CIE Lab", "Perceptual", "deg", True, ".1f"),

    # ── Application (NEW) ──
    MetricDef("palette_uniformity", "mean_cv",
              "Palette L* spacing", "Application", "%", True, ".1f"),
    MetricDef("tint_shade_hue", "mean_max_drift_deg",
              "Tint/shade hue preservation", "Application", "deg", True, ".1f"),
    MetricDef("dataviz_distinguish", "mean_min_de",
              "Data viz min pairwise dE", "Application", "dE", False, ".2f"),
    MetricDef("multistop_gradient", "mean_cv",
              "Multi-stop gradient CV", "Application", "%", True, ".1f"),
    MetricDef("wcag_midpoint", "mean_min_contrast",
              "WCAG midpoint contrast", "Application", ":1", False, ".2f"),
    MetricDef("harmony_accuracy", "mean_error_deg",
              "Palette harmony accuracy", "Application", "deg", True, ".1f"),
    MetricDef("photo_gamut_map", "mean_hue_shift_deg",
              "Photo gamut map fidelity", "Application", "deg", True, ".2f"),
    MetricDef("eased_animation", "mean_cv",
              "Eased animation CV", "Application", "%", True, ".1f"),
    MetricDef("shade_hue_consistency", "overall_mean_max_drift_deg",
              "Shade palette hue drift", "Application", "deg", True, ".1f"),
    MetricDef("shade_hue_consistency", "overall_max_drift_deg",
              "Shade palette worst hue drift", "Application", "deg", True, ".1f"),
    MetricDef("chroma_preservation", "mean_preservation",
              "Chroma preservation (no mud)", "Application", "", False, ".3f"),
    MetricDef("chroma_preservation", "n_muddy",
              "Muddy gradients (C drop >50%)", "Application", "", True, "d"),

    # ── Computed but previously missing from comparison ──
    MetricDef("hue", "primary_L_range",
              "Primary L range", "Hue", "", False, ".3f"),
    MetricDef("specials", "red_white_midpoint.G_minus_B",
              "Red-White midpoint G-B", "Special", "", True, ".3f"),
    MetricDef("gamut", "sRGB.smoothness_max_jump",
              "Cusp smoothness (max jump)", "Gamut", "", True, ".3f"),
    MetricDef("gamut", "Rec2020.valid_cusps",
              "Rec2020 valid cusps", "Gamut", "/360", False, "d"),
    MetricDef("gamut", "sRGB.boundary_max_rel_jump",
              "sRGB boundary continuity", "Gamut", "", True, ".3f"),
    MetricDef("gamut", "P3.boundary_max_rel_jump",
              "P3 boundary continuity", "Gamut", "", True, ".3f"),
    MetricDef("gamut", "Rec2020.boundary_max_rel_jump",
              "Rec2020 boundary continuity", "Gamut", "", True, ".3f"),
    MetricDef("cross_gamut", "amplification_mean",
              "Cross-gamut amplification", "Advanced", "x", True, ".1f"),
    MetricDef("3color", "_mean_cv",
              "3-color gradient CV", "Gradient", "%", True, ".2f"),

    # ── Structural (NEW — from gpu_metrics_advanced.py) ──
    MetricDef("oog_excursion", "excursion_pct",
              "OOG excursion pairs", "Structural", "%", True, ".1f"),
    MetricDef("oog_excursion", "max_oog_dist",
              "OOG max distance", "Structural", "", True, ".4f"),
    MetricDef("hue_reversal", "hues_with_reversals",
              "Hue reversals (count)", "Structural", "", True, "d"),
    MetricDef("hue_reversal", "max_reversal_angle",
              "Hue reversal max angle", "Structural", "deg", True, ".1f"),
    MetricDef("primary_hue_disc", "srgb_max_jump",
              "Primary hue disc (sRGB)", "Structural", "deg", True, ".2f"),
    MetricDef("primary_hue_disc", "p3_max_jump",
              "Primary hue disc (P3)", "Structural", "deg", True, ".2f"),
    MetricDef("negative_lms", "pct_negative",
              "Negative LMS colors", "Structural", "%", True, ".2f"),
    MetricDef("extreme_chroma_stab", "max_amplification",
              "Extreme chroma amplification", "Structural", "x", True, ".2f"),

    # ── Independent Third-Party Benchmarks ──
    MetricDef("hung_berns", "mean_mad_deg",
              "Hung-Berns hue linearity (mean)", "Independent", "deg", True, ".2f"),
    MetricDef("hung_berns", "max_deviation_deg",
              "Hung-Berns hue linearity (max)", "Independent", "deg", True, ".1f"),
    MetricDef("ebner_fairchild", "mean_mad_deg",
              "Ebner-Fairchild hue surfaces (mean)", "Independent", "deg", True, ".2f"),
    MetricDef("ebner_fairchild", "max_deviation_deg",
              "Ebner-Fairchild hue surfaces (max)", "Independent", "deg", True, ".1f"),
    MetricDef("pointer_gamut", "chroma_cv",
              "Pointer gamut chroma isotropy", "Independent", "", True, ".3f"),
    MetricDef("pointer_gamut", "boundary_smoothness",
              "Pointer gamut boundary smoothness", "Independent", "", True, ".3f"),
    MetricDef("pointer_gamut", "hue_uniformity_cv",
              "Pointer gamut hue uniformity", "Independent", "", True, ".3f"),
]


def _extract_score(results: dict, result_key: str, score_path: str) -> float | None:
    """Extract scalar score from nested results dict.

    result_key: top-level key (e.g. "roundtrip")
    score_path: dot-separated path (e.g. "srgb_full_16M.max_error")

    Special handling:
    - "_total_violations" for channel_mono: sum all sub-dicts' total_violations
    """
    if result_key not in results:
        return None

    obj = results[result_key]

    # Special: channel_mono total
    if score_path == "_total_violations":
        try:
            return sum(d.get("total_violations", 0)
                       for d in obj.values() if isinstance(d, dict))
        except (TypeError, AttributeError):
            return None

    # Special: animation mean CV across all transitions
    if result_key == "animation" and score_path == "_mean_cv":
        try:
            cvs = [d.get("cv", 0) for d in obj.values()
                   if isinstance(d, dict) and "cv" in d]
            return sum(cvs) / len(cvs) * 100 if cvs else None  # Convert to %
        except (TypeError, AttributeError):
            return None

    # Special: 3-color gradient mean CV
    if result_key == "3color" and score_path == "_mean_cv":
        try:
            cvs = [d.get("cv", 0) for d in obj.values()
                   if isinstance(d, dict) and "cv" in d]
            return sum(cvs) / len(cvs) * 100 if cvs else None
        except (TypeError, AttributeError):
            return None

    # General dot-path traversal
    parts = score_path.split(".")
    for part in parts:
        if isinstance(obj, dict):
            if part not in obj:
                return None
            obj = obj[part]
        else:
            return None

    try:
        return float(obj)
    except (TypeError, ValueError):
        return None


def _is_self_referential(space_name: str, metric_name: str, score: float,
                         all_scores: dict) -> bool:
    """Detect self-referential scores.

    CIE Lab gets 0 on any test that measures deviation from CIE Lab
    (hue agreement, tint/shade hue, harmony accuracy) because CIE Lab
    IS the reference frame. These wins are meaningless.

    Detection: score essentially zero (< 1e-6) while all others are > 1.0.
    This catches both explicit (hue agreement) and implicit (tint/shade,
    harmony) self-referential cases.
    """
    if score is None:
        return False
    other_scores = [v for k, v in all_scores.items()
                    if k != space_name and v is not None]
    if not other_scores:
        return False
    min_other = min(other_scores)

    # Primary: score essentially zero while others are meaningfully higher
    # BUT only for metrics where zero is "suspicious" (hue agreement, harmony)
    # NOT for metrics where zero is a genuine achievement (violations, errors)
    is_violation_metric = any(kw in metric_name.lower() for kw in
                              ["violation", "nan", "inf", "negative", "error", "excursion",
                               "invalid", "cliff", "holes", "anomal", "dead", "bad hues"])
    if not is_violation_metric:
        if abs(score) < 1e-6 and min_other > 0.1:
            return True
        if abs(score) < 0.01 and min_other > 0.1:
            return True

    # Gamut tests: CIE Lab L range [0,100] vs scanner [0,1] → 0 cusps always.
    # CIE Lab's gamut scores are structurally incomparable.
    if "CIE" in space_name and ("cusps" in metric_name.lower() or
                                 "cliff" in metric_name.lower() or
                                 "smoothness" in metric_name.lower() or
                                 "mono violation" in metric_name.lower()):
        return True

    return False


# Canonical tie tolerance for EVERY verdict path (solo winners, head-to-head,
# tiered winhist, fair winhist). Convention: relative difference measured
# against the LOSER's magnitude. Do not hardcode another value elsewhere —
# a mismatched tolerance makes the headline and fair verdicts disagree on
# identical scores.
TIE_TOLERANCE = 0.01  # 1% relative tolerance for ties


def _extract_items(results: dict, result_key: str, score_path: str):
    """Per-item bootstrap payload for one metric, if the metric exposed it
    (results[group]["_bootstrap"][score_path] = {"items": [...], "stat": str})."""
    group = results.get(result_key)
    if not isinstance(group, dict):
        return None
    boot = group.get("_bootstrap")
    if not isinstance(boot, dict):
        return None
    payload = boot.get(score_path)
    if isinstance(payload, dict) and payload.get("items"):
        return payload
    return None


def decide_outcome(mdef, sc_a: float, sc_b: float,
                   items_a=None, items_b=None) -> str:
    """Pairwise outcome ('a' | 'b' | 'tie') — THE shared decision for every
    verdict path (solo winners, h2h, tiered winhist, fair winhist).

    Tie rule precedence:
      1. abs_tie / exact equality
      2. paired-bootstrap CI when both sides expose per-item data
         (statistical tie: 95% CI of the aggregate difference contains 0)
      3. TIE_TOLERANCE relative threshold (loser-denominator) fallback

    Callers handle None/NaN before calling (non-finite = loss, see h2h).
    """
    abs_tie = getattr(mdef, "abs_tie", 0) or 0
    diff = abs(sc_a - sc_b)
    if abs_tie > 0 and diff <= abs_tie:
        return "tie"
    if sc_a == sc_b:
        return "tie"
    lower = getattr(mdef, "lower_is_better", True)
    if (isinstance(items_a, dict) and isinstance(items_b, dict)
            and items_a.get("stat") == items_b.get("stat")):
        from .bootstrap import paired_decision
        r = paired_decision(items_a["items"], items_b["items"],
                            items_a["stat"], lower_is_better=lower)
        if r is not None:
            return r["outcome"]
    win_a = (sc_a < sc_b) if lower else (sc_a > sc_b)
    loser = sc_b if win_a else sc_a
    rel = diff / (abs(loser) + 1e-30) if loser != 0 else 1.0
    return "tie" if rel <= TIE_TOLERANCE else ("a" if win_a else "b")


def compare_spaces(results_by_space: dict) -> Comparison:
    """Compare N spaces across all METRIC_DEFS.

    Args:
        results_by_space: {space_name: full_results_dict}

    Returns:
        Comparison dataclass with wins, ties, head-to-head matrix.
    """
    from .contamination import trained_on_of, contamination_of

    space_names = list(results_by_space.keys())
    declared = {s: trained_on_of(r) for s, r in results_by_space.items()}
    test_results = []

    for mdef in METRIC_DEFS:
        scores = {}
        items = {}
        for sname, results in results_by_space.items():
            scores[sname] = _extract_score(results, mdef.result_key, mdef.score_path)
            it = _extract_items(results, mdef.result_key, mdef.score_path)
            if it:
                items[sname] = it

        # Skip metric if no space has a score.
        # None = not applicable (skip); NaN/inf = computed but failed —
        # a non-finite score can never win or tie, so it is excluded here
        # (h2h below counts it as a loss against any finite opponent).
        valid_scores = {k: v for k, v in scores.items()
                        if v is not None and math.isfinite(v)}
        if not valid_scores:
            continue

        # Detect self-referential
        ref_spaces = []
        for sname, score in valid_scores.items():
            if _is_self_referential(sname, mdef.name, score, valid_scores):
                ref_spaces.append(sname)

        # Fit-data contamination (three-way holdout enforced): a space fit on
        # the judge's own dataset scores in-sample there — "full" level is
        # excluded from winning like a reference space; "partial" is reported.
        contaminated = {}
        for sname in valid_scores:
            level = contamination_of(mdef.result_key, declared.get(sname) or set())
            if level:
                contaminated[sname] = level

        # Find winner among non-ref, non-contaminated spaces
        fair_scores = {k: v for k, v in valid_scores.items()
                       if k not in ref_spaces and contaminated.get(k) != "full"}
        if not fair_scores:
            test_results.append(TestResult(mdef, scores, None, True, ref_spaces,
                                           items, contaminated=contaminated))
            continue

        ci_based = len(items) >= 2
        if mdef.lower_is_better:
            best_space = min(fair_scores, key=fair_scores.get)
        else:
            best_space = max(fair_scores, key=fair_scores.get)
        best_val = fair_scores[best_space]

        # Winners = best space + every space whose outcome vs best is a tie
        # (decide_outcome: abs_tie → bootstrap CI → relative threshold)
        winners = [best_space]
        for sname, score in fair_scores.items():
            if sname == best_space:
                continue
            outcome = decide_outcome(mdef, score, best_val,
                                     items.get(sname), items.get(best_space))
            if outcome == "tie":
                winners.append(sname)

        is_tie = len(winners) > 1
        winner = winners[0] if len(winners) == 1 else None

        # Ruler-sensitivity: when the metric ships per-ruler values, check
        # whether the SAME space wins under every consensus-member ruler.
        # A win that flips with the ruler is a property of the ruler, not
        # the space — flagged, and worth less trust.
        ruler_flag = None
        per_ruler = {}
        for sname in fair_scores:
            rs = results_by_space[sname].get(mdef.result_key, {})
            payload = rs.get("_ruler_sensitivity", {}) if isinstance(rs, dict) else {}
            entry = payload.get(mdef.score_path)
            if isinstance(entry, dict) and len(entry) >= 2:
                per_ruler[sname] = entry
        if len(per_ruler) == len(fair_scores) and len(fair_scores) >= 2:
            rkeys = set.intersection(*(set(v) for v in per_ruler.values()))
            if len(rkeys) >= 2:
                pick = min if mdef.lower_is_better else max
                ruler_winners = {
                    pick(fair_scores, key=lambda s: per_ruler[s][rk])
                    for rk in rkeys}
                ruler_flag = "robust" if len(ruler_winners) == 1 else "sensitive"

        test_results.append(TestResult(mdef, scores, winner, is_tie, ref_spaces,
                                       items, winners, ci_based, ruler_flag,
                                       contaminated))

    # Count wins
    solo_wins = {s: 0 for s in space_names}
    shared_wins = {s: 0 for s in space_names}
    for tr in test_results:
        if tr.winner:
            solo_wins[tr.winner] += 1
        elif tr.is_tie and tr.winners:
            for sname in tr.winners:
                shared_wins[sname] += 1

    # Head-to-head matrix
    h2h = {}
    for i, s1 in enumerate(space_names):
        for j, s2 in enumerate(space_names):
            if i >= j:
                continue
            w1 = w2 = tie = 0
            for tr in test_results:
                sc1 = tr.scores.get(s1)
                sc2 = tr.scores.get(s2)
                if sc1 is None or sc2 is None:
                    continue
                # Skip if either is self-referential or in-sample (fit on the
                # judge's dataset) — the pair is uninformative
                if s1 in tr.ref_spaces or s2 in tr.ref_spaces:
                    continue
                if tr.contaminated.get(s1) == "full" or tr.contaminated.get(s2) == "full":
                    continue

                # NaN/inf = the space failed to compute this metric.
                # A failure loses to any finite opponent; two failures carry
                # no information. Without this, min()/comparisons involving
                # NaN made the verdict depend on CLI argument order.
                fin1 = math.isfinite(sc1)
                fin2 = math.isfinite(sc2)
                if not fin1 and not fin2:
                    continue
                if not fin1:
                    w2 += 1
                    continue
                if not fin2:
                    w1 += 1
                    continue

                outcome = decide_outcome(tr.metric, sc1, sc2,
                                         tr.items.get(s1), tr.items.get(s2))
                if outcome == "a":
                    w1 += 1
                elif outcome == "b":
                    w2 += 1
                else:
                    tie += 1

            h2h[(s1, s2)] = {"w1": w1, "w2": w2, "tie": tie}

    return Comparison(
        tests=test_results,
        space_names=space_names,
        solo_wins=solo_wins,
        shared_wins=shared_wins,
        head_to_head=h2h,
    )


def print_summary(comp: Comparison):
    """Print terminal summary of comparison results."""
    n_ci = sum(1 for tr in comp.tests if tr.ci_based)
    sensitive = [tr.metric.name for tr in comp.tests if tr.ruler_flag == "sensitive"]
    n_robust = sum(1 for tr in comp.tests if tr.ruler_flag == "robust")
    print(f"\n{'='*60}")
    print(f"  COMPARISON RESULTS ({len(comp.tests)} metrics)")
    print(f"  tie kararı: {n_ci} metrik paired-bootstrap CI (%95), "
          f"{len(comp.tests) - n_ci} metrik %{TIE_TOLERANCE*100:.0f} eşik")
    if n_robust or sensitive:
        print(f"  cetvel kontrolü: {n_robust} metrik RULER-ROBUST "
              f"(3 cetvelde de aynı kazanan), {len(sensitive)} metrik SENSITIVE")
        for name in sensitive:
            print(f"    ⚠ cetvele göre dönüyor: {name}")
    print(f"{'='*60}")

    print(f"\n  {'Space':20s} {'Solo':>6s} {'Shared':>8s}")
    print(f"  {'-'*36}")
    for s in comp.space_names:
        print(f"  {s:20s} {comp.solo_wins[s]:>6d} {comp.shared_wins[s]:>8d}")

    print(f"\n  Head-to-Head:")
    for (s1, s2), h in comp.head_to_head.items():
        print(f"  {s1:15s} vs {s2:15s}: {h['w1']}-{h['w2']} (tie {h['tie']})")

    # Per-category breakdown
    categories = []
    for tr in comp.tests:
        if tr.metric.category not in categories:
            categories.append(tr.metric.category)

    for cat in categories:
        cat_tests = [tr for tr in comp.tests if tr.metric.category == cat]
        print(f"\n  {cat}:")
        for tr in cat_tests:
            scores_str = "  ".join(
                f"{s[:8]:>8s}={_fmt(tr.scores.get(s), tr.metric)}"
                for s in comp.space_names
            )
            w = tr.winner or ("TIE" if tr.is_tie else "?")
            ref = " (ref:" + ",".join(tr.ref_spaces) + ")" if tr.ref_spaces else ""
            print(f"    {tr.metric.name:35s} {scores_str}  W={w}{ref}")


def _fmt(val, mdef: MetricDef) -> str:
    """Format a score value."""
    if val is None:
        return "    N/A"
    fmt = mdef.format_str or ".4f"
    # Only multiply by 100 for base-metric CVs stored as 0.xx (gradient overall)
    # NOT for perceptual metrics that already return percentages
    if mdef.unit == "%" and mdef.result_key == "gradients":
        val = val * 100
    # Integer format for float values
    if fmt == "d":
        return f"{int(val)}"
    return f"{val:{fmt}}"
