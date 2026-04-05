#!/usr/bin/env python3
"""Color space test runner.

Usage:
  python run.py oklab                           # test OKLab
  python run.py cielab                          # test CIE Lab
  python run.py genspace path/to/params.json    # test GenSpace from JSON
  python run.py compare oklab genspace p.json   # compare spaces side by side

Output: terminal summary + JSON report in results/
"""

import sys
import os
import time

# Force UTF-8 stdout on Windows (cp1254 can't handle unicode)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import argparse

import torch

torch.set_default_dtype(torch.float64)

from core.spaces import OKLab, OKLab32, CIELab, GenSpaceAdapter, GenSpaceEnriched, NakaRushtonEnriched, GenSpaceBlueFix, NonlinearM1, HelmCT
from core.pairs import generate_all_pairs
from core.gpu_metrics import (
    measure_roundtrip,
    measure_achromatic,
    measure_gradients,
    measure_gamut,
    measure_gamut_mapping,
    measure_hue,
    measure_special_gradients,
    measure_stability,
)
from core.gpu_metrics_advanced import (
    measure_cvd,
    measure_animation,
    measure_extremes,
    measure_jacobian,
    measure_contrast,
    measure_hue_leaf,
    measure_3color_gradients,
    measure_double_roundtrip,
    measure_cross_gamut_consistency,
    measure_quantization_symmetry,
    measure_channel_monotonicity,
    measure_perceptual_banding,
    measure_oog_excursion,
    measure_hue_reversal,
    measure_primary_hue_discontinuity,
    measure_negative_lms,
    measure_extreme_chroma_stability,
)
from core.gpu_metrics_perceptual import (
    measure_munsell_value,
    measure_munsell_hue,
    measure_macadam_isotropy,
    measure_palette_uniformity,
    measure_tint_shade_hue,
    measure_dataviz_distinguishability,
    measure_multistop_gradient,
    measure_wcag_midpoint_contrast,
    measure_harmony_accuracy,
    measure_photo_gamut_map,
    measure_eased_animation,
    measure_hue_agreement,
    measure_shade_hue_consistency,
    measure_chroma_preservation,
)
from core.report import compile_report, save_json, print_summary


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"CUDA ({name})"
    # MPS doesn't support float64 — use CPU for correctness
    return torch.device("cpu"), "CPU"


def build_space(space_arg, json_path, device):
    """Create a ColorSpace from CLI arguments."""
    s = space_arg.lower()
    if s == "oklab":
        return OKLab(device)
    elif s == "oklab32":
        return OKLab32(device)
    elif s == "cielab":
        return CIELab(device)
    elif s == "genspace":
        if not json_path:
            print("Error: genspace requires --json path", file=sys.stderr)
            sys.exit(1)
        return HelmCT(json_path, device)
    elif s == "genenriched":
        if not json_path:
            print("Error: genenriched requires --json path", file=sys.stderr)
            sys.exit(1)
        return GenSpaceEnriched(json_path, device)
    elif s == "nonlinearm1" or s == "nlm1":
        if not json_path:
            print("Error: nonlinearm1 requires --json path", file=sys.stderr)
            sys.exit(1)
        return NonlinearM1(json_path, device)
    elif s == "bluefix":
        if not json_path:
            print("Error: bluefix requires --json path", file=sys.stderr)
            sys.exit(1)
        return GenSpaceBlueFix(json_path, device)
    elif s == "nr" or s == "nakarushton":
        if not json_path:
            print("Error: nr requires --json path", file=sys.stderr)
            sys.exit(1)
        return NakaRushtonEnriched(json_path, device)
    elif s == "helmct" or s == "ct":
        if not json_path:
            print("Error: helmct requires --json path", file=sys.stderr)
            sys.exit(1)
        return HelmCT(json_path, device)
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
    print("  [1/39] Round-trip...", flush=True)
    results["roundtrip"] = measure_roundtrip(space, device)
    print(f"         {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [2/39] Achromatic...", flush=True)
    results["achromatic"] = measure_achromatic(space, device)
    print(f"         {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [3/39] Gradient pairs...", flush=True)
    pairs_xyz, pair_labels = generate_all_pairs(device)
    results["gradients"] = measure_gradients(space, pairs_xyz, pair_labels, device)
    print(f"         {time.time()-t0:.1f}s ({len(pair_labels)} pairs)")

    t0 = time.time()
    print("  [4/39] Gamut geometry (360°)...", flush=True)
    results["gamut"] = measure_gamut(space, device)
    print(f"         {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [5/39] Gamut mapping...", flush=True)
    results["gamut_mapping"] = measure_gamut_mapping(space, device)
    print(f"         {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [6/39] Hue properties...", flush=True)
    results["hue"] = measure_hue(space, device)
    print(f"         {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [7/39] Special gradients...", flush=True)
    results["specials"] = measure_special_gradients(space, device)
    print(f"         {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [8/39] Stability...", flush=True)
    results["stability"] = measure_stability(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [9/39] CVD simulation...", flush=True)
    results["cvd"] = measure_cvd(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [10/39] Animation smoothness...", flush=True)
    results["animation"] = measure_animation(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [11/39] Dark/light extremes...", flush=True)
    results["extremes"] = measure_extremes(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [12/39] Jacobian condition...", flush=True)
    results["jacobian"] = measure_jacobian(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [13/39] WCAG contrast...", flush=True)
    results["contrast"] = measure_contrast(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [14/39] Hue leaf constancy...", flush=True)
    results["hue_leaf"] = measure_hue_leaf(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [15/39] 3-color gradients...", flush=True)
    results["3color"] = measure_3color_gradients(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [16/39] Perceptual banding...", flush=True)
    results["banding"] = measure_perceptual_banding(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [17/39] Double round-trip...", flush=True)
    results["double_rt"] = measure_double_roundtrip(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [18/39] Cross-gamut consistency...", flush=True)
    results["cross_gamut"] = measure_cross_gamut_consistency(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [19/39] 8-bit quantization symmetry...", flush=True)
    results["quantization"] = measure_quantization_symmetry(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [20/39] Channel monotonicity...", flush=True)
    results["channel_mono"] = measure_channel_monotonicity(space, device)
    print(f"          {time.time()-t0:.1f}s")

    # ── Perceptual & Application metrics ──
    t0 = time.time()
    print("  [21/39] Munsell Value uniformity...", flush=True)
    results["munsell_value"] = measure_munsell_value(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [22/39] Munsell Hue spacing...", flush=True)
    results["munsell_hue"] = measure_munsell_hue(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [23/39] MacAdam ellipse isotropy...", flush=True)
    results["macadam_isotropy"] = measure_macadam_isotropy(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [24/39] Palette L* spacing...", flush=True)
    results["palette_uniformity"] = measure_palette_uniformity(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [25/39] Tint/shade hue preservation...", flush=True)
    results["tint_shade_hue"] = measure_tint_shade_hue(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [26/39] Data viz distinguishability...", flush=True)
    results["dataviz_distinguish"] = measure_dataviz_distinguishability(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [27/39] Multi-stop gradient CV...", flush=True)
    results["multistop_gradient"] = measure_multistop_gradient(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [28/39] WCAG midpoint contrast...", flush=True)
    results["wcag_midpoint"] = measure_wcag_midpoint_contrast(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [29/39] Palette harmony accuracy...", flush=True)
    results["harmony_accuracy"] = measure_harmony_accuracy(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [30/39] Photo gamut map fidelity...", flush=True)
    results["photo_gamut_map"] = measure_photo_gamut_map(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [31/39] Eased animation CV...", flush=True)
    results["eased_animation"] = measure_eased_animation(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [32/39] Hue agreement with CIE Lab...", flush=True)
    results["hue_agreement"] = measure_hue_agreement(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [33/39] Shade palette hue consistency...", flush=True)
    results["shade_hue_consistency"] = measure_shade_hue_consistency(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [34/39] Chroma preservation (muddy midpoints)...", flush=True)
    results["chroma_preservation"] = measure_chroma_preservation(space, device)
    print(f"          {time.time()-t0:.1f}s")

    # ── New Structural metrics ──
    t0 = time.time()
    print("  [35/39] Out-of-gamut excursion...", flush=True)
    results["oog_excursion"] = measure_oog_excursion(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [36/39] Hue reversal detection...", flush=True)
    results["hue_reversal"] = measure_hue_reversal(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [37/39] Near-primary hue discontinuity...", flush=True)
    results["primary_hue_disc"] = measure_primary_hue_discontinuity(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [38/39] Negative LMS detection...", flush=True)
    results["negative_lms"] = measure_negative_lms(space, device)
    print(f"          {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  [39/39] Extreme chroma stability...", flush=True)
    results["extreme_chroma_stab"] = measure_extreme_chroma_stability(space, device)
    print(f"          {time.time()-t0:.1f}s")

    report = compile_report(space.name, device_name, results)
    return report


def main():
    parser = argparse.ArgumentParser(description="Color Space Test Suite")
    parser.add_argument("space", nargs="+",
                        help="Space(s) to test: oklab, cielab, genspace")
    parser.add_argument("--json", help="JSON params file (for genspace)")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    device, device_name = get_device()
    print(f"Device: {device_name}")

    os.makedirs(args.out, exist_ok=True)

    reports = []
    for space_name in args.space:
        space = build_space(space_name, args.json, device)

        # Delete old JSON before test
        safe_name = space.name.replace("/", "_").replace(" ", "_")
        json_path = os.path.join(args.out, f"{safe_name}.json")
        if os.path.exists(json_path):
            os.remove(json_path)

        t_start = time.time()
        report = run_test(space, device, device_name)
        report["total_time"] = time.time() - t_start
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

        html_path = os.path.join(args.out, "comparison.html")
        generate_html(comp, html_path)
        print(f"\n  HTML report: {html_path}")


if __name__ == "__main__":
    main()
