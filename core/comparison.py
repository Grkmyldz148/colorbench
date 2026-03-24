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


@dataclass
class TestResult:
    """Result of one metric across all spaces."""
    metric: MetricDef
    scores: dict          # {space_name: float}
    winner: str | None    # space name or None (tie)
    is_tie: bool
    ref_spaces: list      # spaces marked as self-referential for this metric


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
              "Round-trip P3 16.7M", "Numerical", "", True, ".2e"),
    MetricDef("roundtrip", "rec2020_2M_uniform.max_error",
              "Round-trip Rec2020 2.1M", "Numerical", "", True, ".2e"),
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
    MetricDef("cross_gamut", "amplification_mean",
              "Cross-gamut amplification", "Advanced", "x", True, ".1f"),
    MetricDef("3color", "_mean_cv",
              "3-color gradient CV", "Gradient", "%", True, ".2f"),
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
    if abs(score) < 1e-6 and min_other > 0.1:
        return True
    # Secondary: score very small (< 0.01) and ratio to closest competitor > 10x
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


TIE_TOLERANCE = 0.01  # 1% relative tolerance for ties


def compare_spaces(results_by_space: dict) -> Comparison:
    """Compare N spaces across all METRIC_DEFS.

    Args:
        results_by_space: {space_name: full_results_dict}

    Returns:
        Comparison dataclass with wins, ties, head-to-head matrix.
    """
    space_names = list(results_by_space.keys())
    test_results = []

    for mdef in METRIC_DEFS:
        scores = {}
        for sname, results in results_by_space.items():
            scores[sname] = _extract_score(results, mdef.result_key, mdef.score_path)

        # Skip metric if no space has a score
        valid_scores = {k: v for k, v in scores.items() if v is not None}
        if not valid_scores:
            continue

        # Detect self-referential
        ref_spaces = []
        for sname, score in valid_scores.items():
            if _is_self_referential(sname, mdef.name, score, valid_scores):
                ref_spaces.append(sname)

        # Find winner among non-ref spaces
        fair_scores = {k: v for k, v in valid_scores.items() if k not in ref_spaces}
        if not fair_scores:
            test_results.append(TestResult(mdef, scores, None, True, ref_spaces))
            continue

        if mdef.lower_is_better:
            best_val = min(fair_scores.values())
        else:
            best_val = max(fair_scores.values())

        # Check for ties (within tolerance)
        winners = []
        for sname, score in fair_scores.items():
            if best_val == 0:
                if score == 0:
                    winners.append(sname)
            else:
                rel_diff = abs(score - best_val) / (abs(best_val) + 1e-30)
                if rel_diff <= TIE_TOLERANCE:
                    winners.append(sname)

        is_tie = len(winners) > 1
        winner = winners[0] if len(winners) == 1 else None

        test_results.append(TestResult(mdef, scores, winner, is_tie, ref_spaces))

    # Count wins
    solo_wins = {s: 0 for s in space_names}
    shared_wins = {s: 0 for s in space_names}
    for tr in test_results:
        if tr.winner:
            solo_wins[tr.winner] += 1
        elif tr.is_tie:
            # Count which spaces are in the tie
            fair = {k: v for k, v in tr.scores.items()
                    if v is not None and k not in tr.ref_spaces}
            if fair:
                best = min(fair.values()) if tr.metric.lower_is_better else max(fair.values())
                for sname, score in fair.items():
                    if score is not None:
                        rel_diff = abs(score - best) / (abs(best) + 1e-30) if best != 0 else (0 if score == 0 else 1)
                        if rel_diff <= TIE_TOLERANCE:
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
                # Skip if either is self-referential
                if s1 in tr.ref_spaces or s2 in tr.ref_spaces:
                    continue

                if tr.metric.lower_is_better:
                    if sc1 < sc2:
                        rel = abs(sc2 - sc1) / (abs(sc2) + 1e-30) if sc2 != 0 else (0 if sc1 == 0 else 1)
                        if rel > TIE_TOLERANCE:
                            w1 += 1
                        else:
                            tie += 1
                    elif sc2 < sc1:
                        rel = abs(sc1 - sc2) / (abs(sc1) + 1e-30) if sc1 != 0 else (0 if sc2 == 0 else 1)
                        if rel > TIE_TOLERANCE:
                            w2 += 1
                        else:
                            tie += 1
                    else:
                        tie += 1
                else:
                    if sc1 > sc2:
                        rel = abs(sc1 - sc2) / (abs(sc1) + 1e-30) if sc1 != 0 else (0 if sc2 == 0 else 1)
                        if rel > TIE_TOLERANCE:
                            w1 += 1
                        else:
                            tie += 1
                    elif sc2 > sc1:
                        rel = abs(sc2 - sc1) / (abs(sc2) + 1e-30) if sc2 != 0 else (0 if sc1 == 0 else 1)
                        if rel > TIE_TOLERANCE:
                            w2 += 1
                        else:
                            tie += 1
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
    print(f"\n{'='*60}")
    print(f"  COMPARISON RESULTS ({len(comp.tests)} metrics)")
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
