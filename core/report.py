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
                "shade_hue_consistency", "chroma_preservation",
                # Structural (Faz 3)
                "oog_excursion", "hue_reversal", "primary_hue_disc",
                "negative_lms", "extreme_chroma_stab",
                # Independent third-party (Faz 4)
                "hung_berns", "ebner_fairchild", "pointer_gamut",
                # End-user perceptual (Phase 9 — 29 tests)
                "user_image_synthetic_gradient", "user_color_grading_lut",
                "user_white_balance", "user_natural_scene_palette",
                "user_tailwind_palette", "user_material_palette",
                "user_diverging_colormap", "user_sequential_colormap",
                "user_categorical_palette", "user_theme_dark_mode",
                "user_skin_tone_fitzpatrick", "user_natural_colors",
                "user_brand_colors", "user_logo_color_preservation",
                "user_cinematic_lut", "user_picker_hue_continuity",
                "user_picker_chroma_envelope", "user_achromatic_visual",
                "user_hue_wheel_uniformity", "user_cvd_palette_spacing",
                "user_low_vision_contrast", "user_color_blind_safe_palettes",
                "user_p3_wide_gamut", "user_rec2020_hdr_gamut",
                "user_display_calibration_drift", "user_8bit_quantization",
                "user_hover_state_transition", "user_focus_ring_quality",
                "user_dark_mode_flip",
                # Phase 10
                "user_print_cmyk_fidelity", "user_pantone_spot",
                "user_hdr_tone_mapping", "user_cvd_tritanomaly",
                "user_newsprint_simulation", "user_cross_cultural_skin",
                "user_glassmorphism", "user_status_indicator_distinct",
                # Phase 11
                "user_real_photo_macbeth", "user_jnd_aware_summary"]:
        if key in results:
            report[key] = results[key]

    # Gradients — separate raw pairs for AI analysis
    if "gradients" in results:
        report["gradients"] = {
            "overall": results["gradients"]["overall"],
            "by_category": results["gradients"]["by_category"],
            # per-item payload for the paired-bootstrap tie decision —
            # must survive into the saved report so comparisons from JSON
            # decide ties the same way as live runs
            "_bootstrap": results["gradients"].get("_bootstrap", {}),
            # per-ruler aggregates for the ruler-sensitivity check
            "_ruler_sensitivity": results["gradients"].get("_ruler_sensitivity", {}),
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
        "version": "colorbench v1.1",
        "total_metrics": 94,
        "total_gradient_pairs": 3038,
        "gamuts_tested": ["sRGB", "Display P3", "Rec.2020"],
        "perceptual_metric": "spacing metrics: 3-uniform-space consensus ruler "
                             "(Perceptia-Spacing / CAM16-UCS / Jzazbz, core/rulers.py); "
                             "difference metrics: CIEDE2000 (full Sharma 2005, incl. RT term)",
        "verdict_layers": "raw 94-metric W-L-T is auditable but NOT the verdict; "
                          "the headline is the tiered/fair verdict "
                          "(core/judge_provenance.py + core/fair_verdict.py: gamut 1/3 weight, "
                          "CIELab-reference and heuristic-proxy metrics excluded)",
        "fairness_notes": [
            {
                "severity": "medium",
                "issue": "Difference-ruler (CIEDE2000) structural bias",
                "detail": "Difference/distinguishability tests (3-color, data viz dE, CVD, "
                          "photo gamut map) use CIEDE2000, which is built on CIE Lab coordinates "
                          "and gives Lab-adjacent spaces a structural advantage there. Spacing "
                          "tests (gradient/banding/animation CV) were moved OFF CIEDE2000 to the "
                          "3-uniform-space consensus ruler for exactly this reason.",
            },
            {
                "severity": "medium",
                "issue": "Munsell data favors CIE Lab",
                "detail": "Munsell Value scale uniformity test uses Y values from ASTM D1535. "
                          "CIE Lab was specifically designed to linearize Munsell Value, so it "
                          "will always score well on this test. A high score here means agreement "
                          "with CIE Lab's lightness model, not necessarily perceptual accuracy. "
                          "The judge is also a CV in the candidate's own coordinates (inherent "
                          "to spacing tests) — human-data input, own-coordinate judge.",
            },
            {
                "severity": "low",
                "issue": "MacAdam isotropy fixed 2026-07 to real ellipse geometry",
                "detail": "The test now samples each real 1942 JND ellipse perimeter (a/b/theta, "
                          "Bradford C→D65) so ratio 1.0 genuinely means matching human "
                          "discrimination thresholds. The previous fixed-xy-circle version "
                          "rewarded IGNORING MacAdam anisotropy and its scores are not "
                          "comparable with current ones.",
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
            brj = gm.get('boundary_max_rel_jump', 0)
            bah = gm.get('boundary_bad_hues', 0)
            bwh = gm.get('boundary_worst_hue', 0)
            if brj > 0.05 or bah > 0:
                print(f"              boundary: max_rel_jump={brj:.3f}  "
                      f"bad_hues={bah}/360  worst_hue={bwh}deg")
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

    # 21. OOG excursion
    if "oog_excursion" in report:
        oog = report["oog_excursion"]
        print(f"\n  21. Out-of-Gamut Excursion")
        print(f"     Pairs with excursion: {oog['excursion_pairs']}/{oog['total_pairs']} "
              f"({oog['excursion_pct']:.1f}%)")
        print(f"     Max OOG distance:     {oog['max_oog_dist']:.4f}")
        if oog.get("worst_pairs"):
            for p in oog["worst_pairs"][:5]:
                print(f"       ! {p['pair']:15s} OOG steps={p['oog_steps']}  "
                      f"max_dist={p['max_oog_dist']:.4f}")

    # 22. Hue reversal
    if "hue_reversal" in report:
        hr = report["hue_reversal"]
        print(f"\n  22. Hue Reversal Detection")
        print(f"     Hues with reversals:  {hr['hues_with_reversals']}/{hr['total_hues_tested']}")
        print(f"     Max reversal angle:   {hr['max_reversal_angle']:.1f}deg")
        if hr.get("worst_hues"):
            for h in hr["worst_hues"][:5]:
                print(f"       ! h={h['hue']:>3d}deg: {h['n_reversals']} reversals  "
                      f"max={h['max_angle']:.1f}deg")

    # 23. Primary hue discontinuity
    if "primary_hue_disc" in report:
        phd = report["primary_hue_disc"]
        print(f"\n  23. Near-Primary Hue Discontinuity")
        print(f"     sRGB max jump:  {phd['srgb_max_jump']:.2f}deg  "
              f"mean={phd['srgb_mean_jump']:.2f}deg")
        print(f"     P3 max jump:    {phd['p3_max_jump']:.2f}deg  "
              f"mean={phd['p3_mean_jump']:.2f}deg")
        if phd.get("per_primary"):
            for name, d in phd["per_primary"].items():
                if d["max_hue_jump_deg"] > 5:
                    print(f"       ! {name:8s}: max_jump={d['max_hue_jump_deg']:.2f}deg")

    # 24. Negative LMS
    if "negative_lms" in report:
        nl = report["negative_lms"]
        print(f"\n  24. Negative LMS Detection")
        print(f"     Colors with neg LMS:  {nl['n_negative']}/10000 "
              f"({nl['pct_negative']:.2f}%)")
        print(f"     Max negative value:   {nl['max_negative']:.4f}")
        if nl.get("per_channel"):
            for ch_name, ch_data in nl["per_channel"].items():
                if ch_data["n_negative"] > 0:
                    print(f"       {ch_name}: {ch_data['n_negative']} neg  "
                          f"min={ch_data['min_value']:.6f}")

    # 25. Extreme chroma stability
    if "extreme_chroma_stab" in report:
        ecs = report["extreme_chroma_stab"]
        print(f"\n  25. Extreme Chroma Stability")
        print(f"     Max amplification:    {ecs['max_amplification']:.2f}x")
        print(f"     NaN: {ecs['nan_count']}  Inf: {ecs['inf_count']}")
        if ecs.get("per_color"):
            worst = sorted(ecs["per_color"].items(),
                           key=lambda x: -x[1].get("amplification", 0))[:5]
            for name, d in worst:
                amp = d.get("amplification", 0)
                if amp > 1.0:
                    print(f"       {name:15s}: {amp:.2f}x")

    print(f"\n{'=' * 64}")
