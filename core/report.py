"""Report generation — JSON output + terminal summary.

NO pass/fail verdicts. Only raw data + comparative display.
Judgment is left to the analyst (human or AI).
"""

import json
from datetime import datetime


def compile_report(space_name, device_name, results):
    """Compile all metric results into a structured report. No verdicts."""
    report = {
        "space": space_name,
        "device": device_name,
        "timestamp": datetime.now().isoformat(),
    }

    # Copy all results as-is
    for key in ["roundtrip", "achromatic", "gamut_mapping",
                "hue", "specials", "stability",
                "cvd", "animation", "extremes", "jacobian",
                "contrast", "hue_leaf", "3color", "banding",
                "double_rt", "cross_gamut", "quantization", "channel_mono",
                # Perceptual & Application (Faz 2)
                "munsell_value", "munsell_hue", "macadam_isotropy",
                "palette_uniformity", "tint_shade_hue", "dataviz_distinguish",
                "multistop_gradient", "wcag_midpoint", "harmony_accuracy",
                "photo_gamut_map", "eased_animation", "hue_agreement",
                "chroma_preservation"]:
        if key in results:
            report[key] = results[key]

    # Gradients — separate raw pairs for AI analysis
    if "gradients" in results:
        report["gradients"] = {
            "overall": results["gradients"]["overall"],
            "by_category": results["gradients"]["by_category"],
        }
        report["_pairs_detail"] = results["gradients"]["pairs"]

    # Gamut — cusps go to detail section
    if "gamut" in results:
        report["gamut"] = {}
        for gname, gdata in results["gamut"].items():
            report["gamut"][gname] = {k: v for k, v in gdata.items() if k != "cusps"}
            report[f"_cusps_{gname}"] = gdata.get("cusps", [])

    # Methodology notes — fairness caveats for anyone reading the JSON
    report["_methodology"] = {
        "version": "colorbench v1.0",
        "total_metrics": 46,
        "total_gradient_pairs": 3038,
        "gamuts_tested": ["sRGB", "Display P3", "Rec.2020"],
        "perceptual_metric": "CIEDE2000 (simplified, no RT rotation term)",
        "fairness_notes": [
            {
                "severity": "medium",
                "issue": "CIEDE2000 structural bias",
                "detail": "Gradient CV, multi-stop CV, eased animation CV, data viz dE, "
                          "and banding tests all use CIEDE2000 as the perceptual distance metric. "
                          "CIEDE2000 is built on CIE Lab coordinates, which gives CIE Lab and "
                          "CIE Lab-adjacent spaces (like OKLab) a structural advantage on these tests. "
                          "No independent perceptual ground truth exists as an alternative.",
            },
            {
                "severity": "medium",
                "issue": "Munsell data favors CIE Lab",
                "detail": "Munsell Value scale uniformity test uses Y values from ASTM D1535. "
                          "CIE Lab was specifically designed to linearize Munsell Value, so it "
                          "will always score well on this test. A high score here means agreement "
                          "with CIE Lab's lightness model, not necessarily perceptual accuracy.",
            },
            {
                "severity": "medium",
                "issue": "MacAdam ellipse data is CIE Lab-era",
                "detail": "MacAdam 1942 ellipse centers are defined in CIE xy chromaticity. "
                          "The isotropy test measures local uniformity at these specific points. "
                          "A space optimized for different chromaticity regions may score poorly "
                          "here despite being perceptually superior in practice.",
            },
            {
                "severity": "low",
                "issue": "Hue agreement with CIE Lab is tautological for CIE Lab",
                "detail": "This test measures angular difference from CIE Lab hue angles. "
                          "CIE Lab trivially scores 0. Other spaces are penalized for disagreeing "
                          "with CIE Lab, even if their hue ordering is perceptually more accurate. "
                          "CIE Lab is marked as (ref) and excluded from win counting for this test.",
            },
            {
                "severity": "low",
                "issue": "No human judgment data",
                "detail": "All metrics are computed algorithmically. No test measures whether "
                          "gradients 'look good' to human observers. A space could score well on "
                          "all metrics but produce visually unappealing results, or vice versa.",
            },
        ],
        "self_referential_handling": "Scores that are structurally zero for a space "
                                     "(e.g., CIE Lab hue agreement = 0, CIE Lab gamut cusps = 0 "
                                     "due to L scale mismatch) are marked as (ref) and excluded "
                                     "from win counting. The best non-ref space wins instead.",
    }

    return report


def save_json(report, path):
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def print_summary(report):
    """Terminal summary — numbers only, no judgments."""
    name = report["space"]

    print(f"\n{'=' * 64}")
    print(f"  {name}")
    print(f"{'=' * 64}")

    # 1. Round-trip
    rt = report.get("roundtrip", {})
    print(f"\n  1. Round-Trip")
    if "srgb_full_16M" in rt:
        r = rt["srgb_full_16M"]
        print(f"     sRGB 16.7M:       max={r['max_error']:.2e}  "
              f"NaN={r['nan_count']}  Inf={r['inf_count']}")
    if "p3_full_16M" in rt:
        r = rt["p3_full_16M"]
        print(f"     P3 full 16.7M:    max={r['max_error']:.2e}  "
              f"NaN/Inf={r['nan_inf_count']}")
    if "rec2020_2M_uniform" in rt:
        print(f"     Rec2020 2.1M:     max={rt['rec2020_2M_uniform']['max_error']:.2e}")
    if "rec2020_50K_boundary" in rt:
        print(f"     Rec2020 boundary: max={rt['rec2020_50K_boundary']['max_error']:.2e}")
    if "srgb_boundary_360" in rt:
        print(f"     sRGB boundary:    max={rt['srgb_boundary_360']['max_error']:.2e}  "
              f"({rt['srgb_boundary_360']['n_colors']} colors)")

    # 2. Achromatic
    ach = report.get("achromatic", {})
    print(f"\n  2. Achromatic")
    if "gray_ramp_srgb" in ach:
        print(f"     sRGB 257-step:    C*={ach['gray_ramp_srgb']['max_chroma']:.2e}  "
              f"(includes sRGB matrix rounding)")
    if "gray_ramp_pure" in ach:
        print(f"     D65-pure 500:     C*={ach['gray_ramp_pure']['max_chroma']:.2e}  "
              f"(true achromatic fidelity)")
    if "white" in ach:
        w = ach["white"]
        print(f"     White:            L={w['L']:.8f}  a={w['a']:.2e}  b={w['b']:.2e}")
    if "black" in ach:
        b = ach["black"]
        print(f"     Black:            L={b['L']:.8f}  a={b['a']:.2e}  b={b['b']:.2e}")

    # 3. Gradients
    if "gradients" in report:
        gr = report["gradients"]["overall"]
        print(f"\n  3. Gradients ({gr['n_total']} pairs, {gr['n_crossing']} crossing)")
        print(f"     CV:  mean={gr['cv_mean']*100:.1f}%  "
              f"p50={gr['cv_p50']*100:.1f}%  p95={gr['cv_p95']*100:.1f}%  "
              f"p99={gr['cv_p99']*100:.1f}%  max={gr['cv_max']*100:.1f}%")
        print(f"     Drift: mean={gr['drift_mean']:.1f}deg  "
              f"p95={gr['drift_p95']:.1f}deg  "
              f"max(nc)={gr['drift_max_noncrossing']:.1f}deg  "
              f"max(cross)={gr['drift_max_crossing']:.1f}deg")
        print(f"     Banding: mean={gr['banding_mean']:.1f} dup  "
              f"max={gr['banding_max']} dup")
        print(f"\n     By category:")
        for cat, s in sorted(report["gradients"]["by_category"].items()):
            print(f"       {cat:15s}: CV={s['cv_mean']*100:5.1f}%  "
                  f"p95={s['cv_p95']*100:5.1f}%  "
                  f"drift={s['drift_max']:5.1f}deg  [{s['count']}]")

        # Worst pairs by CV (exclude near_achromatic s=0.05 — always high CV due to quantization)
        if "_pairs_detail" in report:
            real_pairs = [p for p in report["_pairs_detail"]
                          if "s0.05" not in p.get("description", "")]
            worst_cv = sorted(real_pairs, key=lambda p: -p["cv"])[:5]
            print(f"\n     Worst CV pairs (excl. near-ach s=0.05):")
            for p in worst_cv:
                print(f"       CV={p['cv']*100:6.1f}%  drift={p['drift_max']:5.1f}deg  "
                      f"{p['category']:15s}  {p['description']}")

            # Worst drift (non-crossing)
            nc = [p for p in real_pairs if not p.get("is_crossing", False)]
            worst_drift = sorted(nc, key=lambda p: -p["drift_max"])[:5]
            print(f"\n     Worst drift (non-crossing):")
            for p in worst_drift:
                print(f"       drift={p['drift_max']:5.1f}deg  CV={p['cv']*100:5.1f}%  "
                      f"{p['category']:15s}  {p['description']}")

    # 4. Gamut
    if "gamut" in report:
        print(f"\n  4. Gamut Geometry")
        for gname in ["sRGB", "P3", "Rec2020"]:
            gm = report["gamut"].get(gname, {})
            if not gm:
                continue
            print(f"     {gname:8s}: cusps={gm.get('valid_cusps',0)}/360  "
                  f"mono_viol={gm.get('monotonicity_violations',0)}  "
                  f"cliff={gm.get('cliff_max',0)*100:.0f}%  "
                  f"smooth={gm.get('smoothness_max_jump',0):.4f}  "
                  f"vol={gm.get('volume_fraction',0)*100:.0f}%")
            if gm.get("anomalies"):
                for a in gm["anomalies"]:
                    print(f"       ! h={a['hue_from']}->{a['hue_to']}: "
                          f"L={a['L_from']:.3f}->{a['L_to']:.3f} "
                          f"(jump={a['jump']:.3f})")
            if gm.get("dead_zones"):
                for dz in gm["dead_zones"]:
                    print(f"       X DEAD ZONE h={dz['start']}-{dz['end']}deg "
                          f"({dz['span']}deg span, cusp_L<0.05)")

    # 5. Gamut mapping
    if "gamut_mapping" in report:
        print(f"\n  5. Gamut Mapping")
        for k, gm in report["gamut_mapping"].items():
            flag = "  !" if gm['non_monotonic_hues'] > 0 or gm.get('max_hue_jump', 0) > 10 else ""
            print(f"     {k}: non_mono={gm['non_monotonic_hues']}  "
                  f"max_dE={gm['max_de_jump']:.2f}  "
                  f"hue_jump={gm.get('max_hue_jump', 0):.1f}deg{flag}")

    # 6. Hue
    if "hue" in report:
        hu = report["hue"]
        print(f"\n  6. Hue")
        print(f"     RMS={hu['hue_rms']:.1f}deg  "
              f"ordered={'yes' if hu['hue_ordered'] else 'NO'}  "
              f"L_range={hu['primary_L_range']:.3f}")
        for name, d in hu["per_primary"].items():
            print(f"       {name:8s}: h={d['hue']:6.1f}deg "
                  f"(exp {d['expected']:.0f}deg, err={d['error']:+.1f}deg) "
                  f"L={d['L']:.3f} C={d['C']:.3f}")

    # 7. Specials
    if "specials" in report:
        sp = report["specials"]
        print(f"\n  7. Special Gradients")
        print(f"     Blue->White G/R: {sp['blue_white_midpoint']['G_over_R']:.3f}  "
              f"sRGB=({', '.join(f'{x:.3f}' for x in sp['blue_white_midpoint']['srgb'])})")
        print(f"     Red->White G-B:  {sp['red_white_midpoint']['G_minus_B']:+.4f}  "
              f"sRGB=({', '.join(f'{x:.3f}' for x in sp['red_white_midpoint']['srgb'])})")
        print(f"     Yellow chroma:  {sp['yellow_chroma']:.4f}")

    # 8. Stability
    if "stability" in report:
        st = report["stability"]
        print(f"\n  8. Stability")
        print(f"     Perturbation 1e-8: max dLab={st['perturbation_1e8']['max_lab_change']:.2e}  "
              f"mean={st['perturbation_1e8']['mean_lab_change']:.2e}")
        print(f"     Near-black: NaN={st['near_black']['nan']} Inf={st['near_black']['inf']}")
        print(f"     Near-white: NaN={st['near_white']['nan']} Inf={st['near_white']['inf']}")

    # 9. CVD
    if "cvd" in report:
        print(f"\n  9. CVD")
        for ctype in ["protan", "deutan", "tritan"]:
            if ctype in report["cvd"]:
                d = report["cvd"][ctype]
                print(f"     {ctype:7s}: worst_min_dE={d['worst_min_de']:.2f}  "
                      f"mean_dE={d['mean_de']:.1f}")
                # Show worst 3 pairs per type
                worst = sorted(d["pairs"], key=lambda p: p["min_de"])[:3]
                for p in worst:
                    print(f"       ! {p['pair']:15s} min_dE={p['min_de']:.2f}")

    # 10. Animation
    if "animation" in report:
        print(f"\n  10. Animation (60fps, 120 frames)")
        for name, d in report["animation"].items():
            print(f"       {name:5s}: CV={d['cv']:.2f}  "
                  f"ratio={d['step_ratio']:.1f}  "
                  f"dE=[{d['de_min']:.2f}-{d['de_max']:.2f}]")

    # 11. Extremes
    if "extremes" in report:
        ex = report["extremes"]
        print(f"\n  11. Extremes")
        print(f"     Dark hue max var: {ex['near_black_max_variance']:.4f}")
        print(f"     Near-white L rev: {ex['near_white_L_reversals']}")
        print(f"     Full L reversals: {ex['full_L_reversals']}")
        print(f"     L range:          [{ex['L_range'][0]:.4f}, {ex['L_range'][1]:.4f}]")

    # 12. Jacobian
    if "jacobian" in report:
        j = report["jacobian"]
        print(f"\n  12. Jacobian Condition")
        print(f"     mean={j['mean']:.1f}  p95={j['p95']:.1f}  max={j['max']:.1f}")
        print(f"     dark={j['by_region']['dark']:.1f}  "
              f"mid={j['by_region']['mid']:.1f}  "
              f"bright={j['by_region']['bright']:.1f}")

    # 13. Contrast
    if "contrast" in report:
        cr = report["contrast"]
        print(f"\n  13. WCAG Contrast (L=0.3 vs L=0.7)")
        print(f"     CR: mean={cr['cr_mean']:.2f}  "
              f"min={cr['cr_min']:.2f}  max={cr['cr_max']:.2f}  "
              f"CV={cr['cr_cv']:.3f}")

    # 14. Hue leaf
    if "hue_leaf" in report:
        hl = report["hue_leaf"]
        print(f"\n  14. Hue Leaf Constancy")
        print(f"     Max CIELab hue dev: {hl['max_deviation']:.1f}deg  "
              f"mean_std={hl['mean_std']:.1f}deg")
        if hl.get("per_hue"):
            worst_hl = sorted(hl["per_hue"].items(), key=lambda x: -x[1]["max_deviation"])[:5]
            for h, d in worst_hl:
                print(f"       ! h={h:>3s}deg: max_dev={d['max_deviation']:.1f}deg  "
                      f"std={d['std_deviation']:.1f}deg  ({d['n_points']} pts)")

    # 15. 3-color
    if "3color" in report:
        print(f"\n  15. 3-Color Gradients")
        for name, d in report["3color"].items():
            print(f"       {name:8s}: CV={d['cv']:.2f}  "
                  f"dE={d['de_mean']:.1f} mean  {d['de_max']:.1f} max")

    # 16. Banding
    if "banding" in report:
        bd = report["banding"]
        print(f"\n  16. Perceptual Banding (256-step)")
        print(f"     Total invisible: {bd['total_invisible_pct']:.1f}%  "
              f"duplicate: {bd['total_duplicate_pct']:.1f}%")
        for name, d in bd["per_gradient"].items():
            print(f"       {name:5s}: {d['invisible_pct']:4.0f}% invis  "
                  f"{d['duplicate_rgb']} dup  "
                  f"dE=[{d['de_min']:.2f}-{d['de_max']:.2f}]")

    # Double round-trip
    if "double_rt" in report:
        dr = report["double_rt"]
        print(f"\n  17. Double Round-Trip (error accumulation)")
        for k in sorted(dr.keys()):
            d = dr[k]
            print(f"     {k}: max={d['max_error']:.2e}  mean={d['mean_error']:.2e}")

    # Cross-gamut
    if "cross_gamut" in report:
        cg = report["cross_gamut"]
        print(f"\n  18. Cross-Gamut Consistency (sRGBvsP3)")
        print(f"     Max Lab diff:     {cg['max_lab_diff']:.2e}")
        print(f"     Amplification:    mean={cg['amplification_mean']:.1f}x  "
              f"max={cg['amplification_max']:.1f}x")

    # Quantization
    if "quantization" in report:
        q = report["quantization"]
        print(f"\n  19. 8-Bit Quantization Symmetry")
        print(f"     Grays exact:      {q['grays_exact']}")
        print(f"     Web-safe exact:   {q['websafe_exact']}")
        print(f"     Random 10K exact: {q['random_10k_exact']}")
        print(f"     Max channel err:  {q['max_channel_error']}")

    # Channel monotonicity
    if "channel_mono" in report:
        cm = report["channel_mono"]
        print(f"\n  20. Channel Monotonicity")
        for name, d in cm.items():
            v = d["violations"]
            total = d["total_violations"]
            print(f"       {name:5s}: {total} violations  "
                  f"(R={v['R']} G={v['G']} B={v['B']})")

    print(f"\n{'=' * 64}")
