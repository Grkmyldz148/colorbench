"""Phase 5i: 39 user_full perceptual metrics — drop-in port verification."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import torch
torch.set_default_dtype(torch.float64)

from core.cs import OKLab
from core import metrics


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, path))
        elif isinstance(v, (int, float, bool)):
            out[path] = v
    return out


USER_TESTS = [
    "measure_user_image_synthetic_gradient",
    "measure_user_color_grading_lut",
    "measure_user_white_balance",
    "measure_user_natural_scene_palette",
    "measure_user_tailwind_palette",
    "measure_user_material_palette",
    "measure_user_diverging_colormap",
    "measure_user_sequential_colormap",
    "measure_user_categorical_palette",
    "measure_user_theme_dark_mode",
    "measure_user_skin_tone_fitzpatrick",
    "measure_user_natural_colors",
    "measure_user_brand_colors",
    "measure_user_logo_color_preservation",
    "measure_user_cinematic_lut",
    "measure_user_picker_hue_continuity",
    "measure_user_picker_chroma_envelope",
    "measure_user_achromatic_visual",
    "measure_user_hue_wheel_uniformity",
    "measure_user_cvd_palette_spacing",
    "measure_user_low_vision_contrast",
    "measure_user_color_blind_safe_palettes",
    "measure_user_p3_wide_gamut",
    "measure_user_rec2020_hdr_gamut",
    "measure_user_display_calibration_drift",
    "measure_user_8bit_quantization",
    "measure_user_hover_state_transition",
    "measure_user_focus_ring_quality",
    "measure_user_dark_mode_flip",
    "measure_user_print_cmyk_fidelity",
    "measure_user_pantone_spot",
    "measure_user_hdr_tone_mapping",
    "measure_user_cvd_tritanomaly",
    "measure_user_newsprint_simulation",
    "measure_user_cross_cultural_skin",
    "measure_user_glassmorphism",
    "measure_user_status_indicator_distinct",
    "measure_user_real_photo_macbeth",
    "measure_user_jnd_aware_summary",
]


def main():
    sp = OKLab(device=torch.device("cpu"), dtype=torch.float64)
    snap = json.load(open(os.path.join(COLORBENCH, "tests", "snapshots",
                                        "OKLab_2026-05-07_v6.json")))

    pass_count = 0
    fail_list = []
    for fn_name in USER_TESTS:
        fn = getattr(metrics, fn_name)
        snap_key = fn_name.replace("measure_", "")
        if snap_key not in snap:
            fail_list.append(f"  ⚠ {snap_key}: snap key missing")
            continue
        try:
            new = fn(sp, torch.device("cpu"))  # legacy device arg accepted
        except Exception as e:
            fail_list.append(f"  ✗ {snap_key}: ERROR {type(e).__name__}: {e}")
            continue
        f_new = _flat(new)
        f_snap = _flat(snap[snap_key])
        diffs = []
        for k in sorted(set(f_new.keys()) | set(f_snap.keys())):
            if k not in f_new or k not in f_snap:
                diffs.append(k); continue
            a, b = f_new[k], f_snap[k]
            if a == b: continue
            if isinstance(a, (bool,)) or isinstance(b, (bool,)):
                if a != b: diffs.append(k)
                continue
            if abs(a - b) < 1e-13: continue
            if (a == 0) and (b == 0): continue
            diffs.append(k)
        if not diffs:
            pass_count += 1
            print(f"  ✓ {snap_key}: BIT-EXACT ({len(f_new)} keys)")
        else:
            fail_list.append(f"  ✗ {snap_key}: {len(diffs)} diffs (first: {diffs[0]})")

    print(f"\n{pass_count}/{len(USER_TESTS)} PASS")
    if fail_list:
        for line in fail_list[:15]:
            print(line)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
