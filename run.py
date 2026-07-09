#!/usr/bin/env python3
"""Color space test runner.

Usage:
  python run.py oklab                           # test OKLab
  python run.py cielab                          # test CIE Lab
  python run.py genspace path/to/params.json    # test GenSpace from JSON
  python run.py compare oklab genspace p.json   # compare spaces side by side

Output: terminal summary + JSON report in results/
"""

import json
import sys
import os
import time

# Force UTF-8 stdout on Windows (cp1254 can't handle unicode)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import argparse

# ── MetricSpace fast path (no torch needed) ────────────────────────────────
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1].lower() == "metric":
    _colorbench_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _colorbench_dir)
    from core.metric_eval import run_metric_evaluation
    _p = argparse.ArgumentParser()
    _p.add_argument("space")
    _p.add_argument("--json", default=None)
    _a = _p.parse_args()
    _repo_root = os.path.dirname(_colorbench_dir)
    _default = os.path.join(_repo_root, "research", "checkpoints", "metricspace_v21.json")
    run_metric_evaluation(_a.json or _default, os.path.join(_repo_root, "datasets"), _repo_root)
    sys.exit(0)

import torch

torch.set_default_dtype(torch.float64)

# ── Color spaces ─────────────────────────────────────────────────────────────
# Refactored modular spaces (Phase 1-6):
from core.cs import (
    OKLab, OKLab32, CIELab, Perceptia, Engineered, HelmCT, HelmCT_PathBA, HelmCT_minimal, HelmCT_M1M2,
    IPT, JzAzBz, ICtCp, CAM16UCS, DIN99d,
    IPTCanonical, JzAzBzCanonical, CAM16UCSCanonical, DIN99dCanonical,
)
# GenSpace adapter family (refactored Phase 7):
from core.cs import (
    GenSpaceAdapter, GenSpaceEnriched, NakaRushtonEnriched,
    GenSpaceBlueFix, NonlinearM1,
)
# clean-helmlab-mistral family (Mistral Vibe):
from core.cs import CleanMetricMistral, TriOppMistral

# ── Pairs + metrics ──────────────────────────────────────────────────────────
from core.pairs import generate_all_pairs
# All 81 metrics now live under core.metrics (Phase 5):
from core.metrics import (
    measure_roundtrip, measure_achromatic, measure_gradients,
    measure_gamut, measure_gamut_mapping, measure_hue, measure_special_gradients,
    measure_stability,
    # advanced
    measure_cvd, measure_animation, measure_extremes, measure_jacobian,
    measure_contrast, measure_hue_leaf, measure_3color_gradients,
    measure_double_roundtrip, measure_cross_gamut_consistency,
    measure_quantization_symmetry, measure_channel_monotonicity,
    measure_perceptual_banding, measure_oog_excursion, measure_hue_reversal,
    measure_primary_hue_discontinuity, measure_negative_lms,
    measure_extreme_chroma_stability,
    # perceptual
    measure_munsell_value, measure_munsell_hue, measure_macadam_isotropy,
    measure_palette_uniformity, measure_tint_shade_hue,
    measure_dataviz_distinguishability, measure_multistop_gradient,
    measure_wcag_midpoint_contrast, measure_harmony_accuracy,
    measure_photo_gamut_map, measure_eased_animation, measure_hue_agreement,
    measure_shade_hue_consistency, measure_chroma_preservation,
    # independent
    measure_hung_berns, measure_ebner_fairchild, measure_pointer_gamut,
    # user full (39)
    measure_user_image_synthetic_gradient, measure_user_color_grading_lut,
    measure_user_white_balance, measure_user_natural_scene_palette,
    measure_user_tailwind_palette, measure_user_material_palette,
    measure_user_diverging_colormap, measure_user_sequential_colormap,
    measure_user_categorical_palette, measure_user_theme_dark_mode,
    measure_user_skin_tone_fitzpatrick, measure_user_natural_colors,
    measure_user_brand_colors, measure_user_logo_color_preservation,
    measure_user_cinematic_lut, measure_user_picker_hue_continuity,
    measure_user_picker_chroma_envelope, measure_user_achromatic_visual,
    measure_user_hue_wheel_uniformity, measure_user_cvd_palette_spacing,
    measure_user_low_vision_contrast, measure_user_color_blind_safe_palettes,
    measure_user_p3_wide_gamut, measure_user_rec2020_hdr_gamut,
    measure_user_display_calibration_drift, measure_user_8bit_quantization,
    measure_user_hover_state_transition, measure_user_focus_ring_quality,
    measure_user_dark_mode_flip,
    measure_user_print_cmyk_fidelity, measure_user_pantone_spot,
    measure_user_hdr_tone_mapping, measure_user_cvd_tritanomaly,
    measure_user_newsprint_simulation, measure_user_cross_cultural_skin,
    measure_user_glassmorphism, measure_user_status_indicator_distinct,
    measure_user_real_photo_macbeth, measure_user_jnd_aware_summary,
)
from core.precision import select as _select_precision
from core.report import compile_report, save_json, print_summary


def get_device(precision_args=None):
    """Resolve (device, dtype, label).

    precision_args: namespace from argparse with .fast, .device, .precision
    Default: CPU float64 (bit-exact baseline). --fast picks best GPU + float32.
    Explicit --device/--precision combinations override.
    """
    if precision_args is None:
        return _select_precision("default")[:2] + ("CPU float64",)

    if precision_args.fast:
        return _select_precision("fast")
    if precision_args.device or precision_args.precision:
        return _select_precision(
            "explicit",
            device=precision_args.device or "cpu",
            precision=precision_args.precision or 64,
        )
    return _select_precision("default")


def build_space(space_arg, json_path, device, canonical=False, dtype=torch.float64):
    """Create a ColorSpace from CLI arguments.

    canonical=True: literature spaces use colour-science wrapper (bit-identical reference).
    canonical=False: literature spaces use ColorBench-tuned implementations.
    """
    s = space_arg.lower()
    # New refactored spaces accept (device=, dtype=) keywords:
    if s == "oklab":
        return OKLab(device=device, dtype=dtype)
    elif s == "oklab32":
        return OKLab32(device=device, dtype=dtype)
    elif s == "cielab":
        return CIELab(device=device, dtype=dtype)
    elif s == "perceptia":
        return Perceptia(device=device, dtype=dtype)
    elif s == "engineered" or s == "substrate":
        return Engineered(device=device, dtype=dtype)
    elif s == "helmct" or s == "ct" or s == "genspace":
        if not json_path:
            print(f"Error: {s} requires --json path", file=sys.stderr)
            sys.exit(1)
        return HelmCT(json_path, device=device, dtype=dtype)
    elif s == "helmctpathba" or s == "pathba":
        if not json_path:
            print(f"Error: {s} requires --json path (HelmCT v0.11.1 base params)", file=sys.stderr)
            sys.exit(1)
        # Cycle 25-28 default: α=-3°, σ=10° uniform across 6 primaries
        return HelmCT_PathBA(json_path, device=device, dtype=dtype)
    elif s == "helmctminimal" or s == "minimal":
        if not json_path:
            print(f"Error: {s} requires --json path (HelmCT v0.11.1 base params)", file=sys.stderr)
            sys.exit(1)
        # Cycle 40-41: 19 params (M1+M2+depcubic only). Hedef 3 candidate.
        return HelmCT_minimal(json_path, device=device, dtype=dtype)
    elif s == "helmctm1m2" or s == "m1m2":
        if not json_path:
            print(f"Error: {s} requires --json path (HelmCT v0.11.1 base params)", file=sys.stderr)
            sys.exit(1)
        # Cycle 43: 18 params (M1+M2+plain_cbrt). STRICT Hedef 3 candidate (= OKLab budget).
        return HelmCT_M1M2(json_path, device=device, dtype=dtype)
    # GenSpace family (refactored Phase 7, dtype/device-aware):
    elif s == "genenriched":
        if not json_path:
            print("Error: genenriched requires --json path", file=sys.stderr); sys.exit(1)
        return GenSpaceEnriched(json_path, device, dtype=dtype)
    elif s == "nonlinearm1" or s == "nlm1":
        if not json_path:
            print("Error: nonlinearm1 requires --json path", file=sys.stderr); sys.exit(1)
        return NonlinearM1(json_path, device, dtype=dtype)
    elif s == "bluefix":
        if not json_path:
            print("Error: bluefix requires --json path", file=sys.stderr); sys.exit(1)
        return GenSpaceBlueFix(json_path, device, dtype=dtype)
    elif s == "nr" or s == "nakarushton":
        if not json_path:
            print("Error: nr requires --json path", file=sys.stderr); sys.exit(1)
        return NakaRushtonEnriched(json_path, device, dtype=dtype)
    # Literature spaces (refactored, dtype/device-aware):
    elif s == "ipt":
        return IPTCanonical(device) if canonical else IPT(device)
    elif s == "jzazbz":
        return JzAzBzCanonical(device) if canonical else JzAzBz(device)
    elif s == "ictcp":
        return ICtCp(device)
    elif s == "cam16ucs" or s == "cam16-ucs":
        return CAM16UCSCanonical(device) if canonical else CAM16UCS(device)
    elif s == "din99d":
        return DIN99dCanonical(device) if canonical else DIN99d(device)
    # clean-helmlab-mistral spaces
    elif s == "cleanmetric" or s == "clean_metric" or s == "cleanmetricmistral":
        return CleanMetricMistral(device=device, dtype=dtype)
    elif s == "triopp" or s == "triopp_mistral" or s == "trioppmistral":
        return TriOppMistral(device=device, dtype=dtype)
    else:
        print(f"Unknown space: {space_arg}", file=sys.stderr)
        sys.exit(1)


def run_test(space, device, device_name):
    """Run full test suite on a single space."""
    print(f"\n{'=' * 60}")
    print(f"  Testing: {space.name}")
    print(f"  Device:  {device_name}")
    print(f"{'=' * 60}\n")

    results = {}

    t0 = time.time()
    print("  [1/42] Round-trip...", flush=True)
    results["roundtrip"] = measure_roundtrip(space, device)
    print(f"         {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [2/42] Achromatic...", flush=True)
    results["achromatic"] = measure_achromatic(space, device)
    print(f"         {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [3/42] Gradient pairs...", flush=True)
    pairs_xyz, pair_labels = generate_all_pairs(device)
    results["gradients"] = measure_gradients(space, pairs_xyz, pair_labels, device)
    print(f"         {time.time()-t0:.1f}s ({len(pair_labels)} pairs)")

    t0 = time.time()
    print("  [4/42] Gamut geometry (360°)...", flush=True)
    results["gamut"] = measure_gamut(space, device)
    print(f"         {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [5/42] Gamut mapping...", flush=True)
    results["gamut_mapping"] = measure_gamut_mapping(space, device)
    print(f"         {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [6/42] Hue properties...", flush=True)
    results["hue"] = measure_hue(space, device)
    print(f"         {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [7/42] Special gradients...", flush=True)
    results["specials"] = measure_special_gradients(space, device)
    print(f"         {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [8/42] Stability...", flush=True)
    results["stability"] = measure_stability(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [9/42] CVD simulation...", flush=True)
    results["cvd"] = measure_cvd(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [10/42] Animation smoothness...", flush=True)
    results["animation"] = measure_animation(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [11/42] Dark/light extremes...", flush=True)
    results["extremes"] = measure_extremes(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [12/42] Jacobian condition...", flush=True)
    results["jacobian"] = measure_jacobian(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [13/42] WCAG contrast...", flush=True)
    results["contrast"] = measure_contrast(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [14/42] Hue leaf constancy...", flush=True)
    results["hue_leaf"] = measure_hue_leaf(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [15/42] 3-color gradients...", flush=True)
    results["3color"] = measure_3color_gradients(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [16/42] Perceptual banding...", flush=True)
    results["banding"] = measure_perceptual_banding(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [17/42] Double round-trip...", flush=True)
    results["double_rt"] = measure_double_roundtrip(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [18/42] Cross-gamut consistency...", flush=True)
    results["cross_gamut"] = measure_cross_gamut_consistency(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [19/42] 8-bit quantization symmetry...", flush=True)
    results["quantization"] = measure_quantization_symmetry(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [20/42] Channel monotonicity...", flush=True)
    results["channel_mono"] = measure_channel_monotonicity(space, device)
    print(f"          {time.time()-t0:.1f}s")

    # ── Perceptual & Application metrics ──
    t0 = time.time()
    print("  [21/42] Munsell Value uniformity...", flush=True)
    results["munsell_value"] = measure_munsell_value(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [22/42] Munsell Hue spacing...", flush=True)
    results["munsell_hue"] = measure_munsell_hue(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [23/42] MacAdam ellipse isotropy...", flush=True)
    results["macadam_isotropy"] = measure_macadam_isotropy(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [24/42] Palette L* spacing...", flush=True)
    results["palette_uniformity"] = measure_palette_uniformity(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [25/42] Tint/shade hue preservation...", flush=True)
    results["tint_shade_hue"] = measure_tint_shade_hue(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [26/42] Data viz distinguishability...", flush=True)
    results["dataviz_distinguish"] = measure_dataviz_distinguishability(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [27/42] Multi-stop gradient CV...", flush=True)
    results["multistop_gradient"] = measure_multistop_gradient(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [28/42] WCAG midpoint contrast...", flush=True)
    results["wcag_midpoint"] = measure_wcag_midpoint_contrast(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [29/42] Palette harmony accuracy...", flush=True)
    results["harmony_accuracy"] = measure_harmony_accuracy(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [30/42] Photo gamut map fidelity...", flush=True)
    results["photo_gamut_map"] = measure_photo_gamut_map(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [31/42] Eased animation CV...", flush=True)
    results["eased_animation"] = measure_eased_animation(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [32/42] Hue agreement with CIE Lab...", flush=True)
    results["hue_agreement"] = measure_hue_agreement(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [33/42] Shade palette hue consistency...", flush=True)
    results["shade_hue_consistency"] = measure_shade_hue_consistency(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [34/42] Chroma preservation (muddy midpoints)...", flush=True)
    results["chroma_preservation"] = measure_chroma_preservation(space, device)
    print(f"          {time.time()-t0:.1f}s")

    # ── New Structural metrics ──
    t0 = time.time()
    print("  [35/42] Out-of-gamut excursion...", flush=True)
    results["oog_excursion"] = measure_oog_excursion(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [36/42] Hue reversal detection...", flush=True)
    results["hue_reversal"] = measure_hue_reversal(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [37/42] Near-primary hue discontinuity...", flush=True)
    results["primary_hue_disc"] = measure_primary_hue_discontinuity(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [38/42] Negative LMS detection...", flush=True)
    results["negative_lms"] = measure_negative_lms(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [39/42] Extreme chroma stability...", flush=True)
    results["extreme_chroma_stab"] = measure_extreme_chroma_stability(space, device)
    print(f"          {time.time()-t0:.1f}s")

    # ── Independent third-party benchmarks ──
    t0 = time.time()
    print("  [40/42] Hung & Berns (1995) hue linearity...", flush=True)
    results["hung_berns"] = measure_hung_berns(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [41/42] Ebner & Fairchild (1998) hue surfaces...", flush=True)
    results["ebner_fairchild"] = measure_ebner_fairchild(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [42/42] Pointer's Gamut (1980) distortion...", flush=True)
    results["pointer_gamut"] = measure_pointer_gamut(space, device)
    print(f"          {time.time()-t0:.1f}s")

    # ── 29 end-user perceptual tests (Phase 9) ────────────────────────────
    user_tests = [
        ("user_image_synthetic_gradient",   measure_user_image_synthetic_gradient),
        ("user_color_grading_lut",          measure_user_color_grading_lut),
        ("user_white_balance",              measure_user_white_balance),
        ("user_natural_scene_palette",      measure_user_natural_scene_palette),
        ("user_tailwind_palette",           measure_user_tailwind_palette),
        ("user_material_palette",           measure_user_material_palette),
        ("user_diverging_colormap",         measure_user_diverging_colormap),
        ("user_sequential_colormap",        measure_user_sequential_colormap),
        ("user_categorical_palette",        measure_user_categorical_palette),
        ("user_theme_dark_mode",            measure_user_theme_dark_mode),
        ("user_skin_tone_fitzpatrick",      measure_user_skin_tone_fitzpatrick),
        ("user_natural_colors",             measure_user_natural_colors),
        ("user_brand_colors",               measure_user_brand_colors),
        ("user_logo_color_preservation",    measure_user_logo_color_preservation),
        ("user_cinematic_lut",              measure_user_cinematic_lut),
        ("user_picker_hue_continuity",      measure_user_picker_hue_continuity),
        ("user_picker_chroma_envelope",     measure_user_picker_chroma_envelope),
        ("user_achromatic_visual",          measure_user_achromatic_visual),
        ("user_hue_wheel_uniformity",       measure_user_hue_wheel_uniformity),
        ("user_cvd_palette_spacing",        measure_user_cvd_palette_spacing),
        ("user_low_vision_contrast",        measure_user_low_vision_contrast),
        ("user_color_blind_safe_palettes",  measure_user_color_blind_safe_palettes),
        ("user_p3_wide_gamut",              measure_user_p3_wide_gamut),
        ("user_rec2020_hdr_gamut",          measure_user_rec2020_hdr_gamut),
        ("user_display_calibration_drift",  measure_user_display_calibration_drift),
        ("user_8bit_quantization",          measure_user_8bit_quantization),
        ("user_hover_state_transition",     measure_user_hover_state_transition),
        ("user_focus_ring_quality",         measure_user_focus_ring_quality),
        ("user_dark_mode_flip",             measure_user_dark_mode_flip),
        # Phase 10
        ("user_print_cmyk_fidelity",        measure_user_print_cmyk_fidelity),
        ("user_pantone_spot",               measure_user_pantone_spot),
        ("user_hdr_tone_mapping",           measure_user_hdr_tone_mapping),
        ("user_cvd_tritanomaly",            measure_user_cvd_tritanomaly),
        ("user_newsprint_simulation",       measure_user_newsprint_simulation),
        ("user_cross_cultural_skin",        measure_user_cross_cultural_skin),
        ("user_glassmorphism",              measure_user_glassmorphism),
        ("user_status_indicator_distinct",  measure_user_status_indicator_distinct),
        # Phase 11 — real-data + JND-aware
        ("user_real_photo_macbeth",         measure_user_real_photo_macbeth),
        ("user_jnd_aware_summary",          measure_user_jnd_aware_summary),
    ]
    for i, (key, fn) in enumerate(user_tests, 1):
        t0 = time.time()
        print(f"  [E{i:02d}/{len(user_tests)}] {key}...", flush=True)
        results[key] = fn(space, device)
        print(f"          {time.time()-t0:.1f}s")

    report = compile_report(space.name, device_name, results)
    return report


def main():
    parser = argparse.ArgumentParser(description="Color Space Test Suite")
    parser.add_argument("space", nargs="+",
                        help="Space(s) to test: oklab, cielab, genspace, metric")
    parser.add_argument("--json", help="JSON params file (for genspace / metricspace)")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--canonical", action="store_true",
                        help="Use colour-science wrapper for IPT/JzAzBz/CAM16-UCS/DIN99d (bit-identical reference). Default: ColorBench-tuned implementations.")
    parser.add_argument("--category", choices=["all", "mathematical", "structural", "perceptual_internal", "perceptual_visible"],
                        default="all",
                        help="Filter comparison summary to a single metric category. "
                             "Default 'all' shows all 4 categories. 'perceptual_visible' shows only end-user perceptible metrics.")
    # Precision / device routing (Phase 8 refactor):
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: GPU + float32 (~1e-6 numeric drift, sub-JND). "
                             "Auto-selects MPS (Apple) or CUDA (NVIDIA). USE FOR ITERATION ONLY — "
                             "snapshot regression and publication require default CPU float64.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None,
                        help="Explicit device override. Default: cpu.")
    parser.add_argument("--precision", type=int, choices=[32, 64], default=None,
                        help="Explicit precision. Default: 64 (bit-exact). MPS supports only 32.")
    args = parser.parse_args()

    # ── MetricSpace evaluation (completely separate path) ──────────────────
    if args.space[0].lower() == "metric":
        from core.metric_eval import run_metric_evaluation
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_json = os.path.join(repo_root, "research", "checkpoints", "metricspace_v21.json")
        metric_json = args.json or default_json
        from core.data import baseline_dir
        datasets_dir = baseline_dir()   # env / dev-layout / cache (auto-fetch)
        run_metric_evaluation(metric_json, datasets_dir, repo_root)
        return

    device, dtype, device_name = get_device(args)
    if args.fast:
        print(f"⚡ FAST MODE — {device_name}, ~1e-6 drift. Not for snapshot regression.\n")
    else:
        print(f"Device: {device_name}")

    os.makedirs(args.out, exist_ok=True)

    reports = []
    spaces_by_name = {}
    for space_name in args.space:
        space = build_space(space_name, args.json, device, canonical=args.canonical, dtype=dtype)
        spaces_by_name[space.name] = space

        # Fit-data declaration (three-way holdout): --json spaces may declare
        # "trained_on": [dataset, ...] in their params file; judges built on
        # those datasets are then flagged in-sample for this space.
        if args.json and not hasattr(space, "trained_on"):
            try:
                with open(args.json) as f:
                    decl = json.load(f).get("trained_on")
                if decl:
                    space.trained_on = decl
            except Exception:
                pass
        if args.json and not getattr(space, "trained_on", None):
            print(f"  NOT: '{space.name}' fit-verisi beyanı yok (params JSON'a "
                  f"\"trained_on\": [...] ekleyin) — kontaminasyon kontrolü yapılamıyor.")

        # Delete old JSON before test
        safe_name = space.name.replace("/", "_").replace(" ", "_")
        json_path = os.path.join(args.out, f"{safe_name}.json")
        if os.path.exists(json_path):
            os.remove(json_path)

        t_start = time.time()
        report = run_test(space, device, device_name)
        report["total_time"] = time.time() - t_start
        report["trained_on"] = list(getattr(space, "trained_on", []) or [])
        save_json(report, json_path)
        print(f"\n  JSON saved: {json_path}")

        # Print summary
        print_summary(report)
        reports.append(report)

    # If multiple spaces, run comparison + HTML report
    if len(reports) > 1:
        from core.comparison import compare_spaces, print_summary as print_comp_summary
        from core.html_report import generate as generate_html

        results_by_space = {r["space"]: r for r in reports}
        comp = compare_spaces(results_by_space)
        print_comp_summary(comp)

        from core.contamination import summarize as contamination_summary
        cs = contamination_summary(comp)
        if cs:
            print("\n" + cs)

        # ── Fairness-corrected verdict — THE HEADLINE ────────────────────
        # The raw W-L-T above is fully auditable but over-weights gamut
        # (31 metrics = ~10 tests × 3 gamuts) and includes CIELab-ceiling
        # metrics that penalize hue-correcting spaces. Never quote the raw
        # count as the verdict; quote this block.
        from core.judge_provenance import tiered_winhist, format_tiered_verdict
        from core.fair_verdict import fair_winhist

        names = [r["space"] for r in reports]
        print("\n" + "=" * 60)
        print("  ADİL VERDİCT (headline) — ham W-L-T yukarıda, referans için")
        print("=" * 60)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                print()
                print(format_tiered_verdict(tiered_winhist(comp.tests, a, b), a, b))
                fw = fair_winhist(comp, a, b)
                print(f"  AĞIRLIKLI (gamut×1/3, CIELab-ref×0): "
                      f"{a} {fw['a']} – {fw['b']} {b}  (tie {fw['tie']})")

        # Human-data panel (best-of-breed pool) for pairwise runs; skipped
        # gracefully when the color-perception-datasets pool isn't available.
        if len(reports) == 2:
            try:
                from core import human_pool as hp
                print()
                print(hp.compare_on_pool(spaces_by_name[names[0]], spaces_by_name[names[1]],
                                         names[0], names[1], validated_only=True))
            except Exception as e:
                print(f"\n  (insan-verisi paneli atlandı: {repr(e)[:100]})")

        # Karne: property × space matrix — the benchmark's primary output
        # (no single overall score, by design: no space is best at everything)
        try:
            from core.scorecard import scorecard as make_scorecard
            print()
            print(make_scorecard({n: spaces_by_name[n] for n in names}))
        except Exception as e:
            print(f"\n  (karne atlandı: {repr(e)[:100]})")

        html_path = os.path.join(args.out, "comparison.html")
        generate_html(comp, html_path)
        print(f"\n  HTML report: {html_path}")


if __name__ == "__main__":
    main()
