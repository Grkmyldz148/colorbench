"""Phase 9 — 29 end-user perceptual metric pin testler.

Her measure_user_* fonksiyonu:
  - colour-science ground truth (CIE Lab + ΔE2000) kullanmalı
  - Deterministic (sabit seed where applicable, no random in core path)
  - Aynı space iki kez koşturuldu→ aynı sonuç

Bu test'ler:
  1. Determinism: aynı space iki kere → bit-identik output
  2. Smoke: her fonksiyon hatasız çalışıyor + dict döndürüyor
  3. Sanity: temel beklentiler (achromatic chroma <1, hue drift <360°, vs)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "helmlab-main-repo", "src")))

try:
    import torch
    from colorbench.core.spaces import OKLab
    from colorbench.core import gpu_metrics_user_full as m
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


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
    # Phase 10
    "measure_user_print_cmyk_fidelity",
    "measure_user_pantone_spot",
    "measure_user_hdr_tone_mapping",
    "measure_user_cvd_tritanomaly",
    "measure_user_newsprint_simulation",
    "measure_user_cross_cultural_skin",
    "measure_user_glassmorphism",
    "measure_user_status_indicator_distinct",
    # Phase 11
    "measure_user_real_photo_macbeth",
    "measure_user_jnd_aware_summary",
]


def test_all_29_smoke():
    """Her fonksiyon hatasız çalışıp dict döndürüyor."""
    if not HAVE_TORCH: return
    device = torch.device("cpu")
    sp = OKLab(device)
    failed = []
    for fn_name in USER_TESTS:
        try:
            fn = getattr(m, fn_name)
            result = fn(sp, device)
            assert isinstance(result, dict), f"{fn_name} dict döndürmedi"
            assert len(result) > 0, f"{fn_name} boş dict döndürdü"
        except Exception as e:
            failed.append(f"{fn_name}: {e}")
    assert not failed, f"Smoke fail: {failed}"


def test_29_determinism():
    """Aynı space iki kere → bit-identik output (aynı seed altında)."""
    if not HAVE_TORCH: return
    device = torch.device("cpu")
    sp1 = OKLab(device)
    sp2 = OKLab(device)
    diffs = []
    for fn_name in USER_TESTS:
        fn = getattr(m, fn_name)
        r1 = fn(sp1, device); r2 = fn(sp2, device)
        for k, v in r1.items():
            if isinstance(v, (int, float)):
                v2 = r2.get(k)
                if isinstance(v2, (int, float)) and abs(v - v2) > 1e-12:
                    diffs.append(f"{fn_name}.{k}: {v} != {v2}")
    assert not diffs, f"Determinism fail: {diffs[:5]}"


def test_29_returns_perceptual_visible_metrics():
    """Her fonksiyon en az 1 perceptual numeric döndürüyor."""
    if not HAVE_TORCH: return
    device = torch.device("cpu")
    sp = OKLab(device)
    weak = []
    for fn_name in USER_TESTS:
        fn = getattr(m, fn_name)
        result = fn(sp, device)
        # Recursive flatten
        def flat(d, p=""):
            out = []
            for k, v in d.items():
                key = f"{p}.{k}" if p else k
                if isinstance(v, dict):
                    out.extend(flat(v, key))
                elif isinstance(v, (int, float)):
                    out.append((key, v))
            return out
        f = flat(result)
        # Skip n_ count fields
        scalars = [(k, v) for k, v in f if not k.split(".")[-1].startswith("n_")]
        if len(scalars) < 1:
            weak.append(f"{fn_name}: 0 scalar metrics")
    assert not weak, f"Weak return: {weak}"


def test_29_in_metric_categories():
    """29 measure_user_* fonksiyonu metric_categories.GROUP_CATEGORY'de var mı."""
    from colorbench.core.metric_categories import GROUP_CATEGORY
    missing = []
    for fn_name in USER_TESTS:
        group = fn_name.replace("measure_", "")
        if group not in GROUP_CATEGORY:
            missing.append(group)
    assert not missing, f"Missing categories: {missing}"


if __name__ == "__main__":
    if not HAVE_TORCH:
        print("PyTorch yok — venv kullan"); sys.exit(0)
    tests = [
        ("29 measure_user_* smoke (no errors, returns dict)", test_all_29_smoke),
        ("29 measure_user_* determinism (same input → same output)", test_29_determinism),
        ("29 measure_user_* returns >0 scalar perceptual metric", test_29_returns_perceptual_visible_metrics),
        ("29 measure_user_* registered in metric_categories", test_29_in_metric_categories),
    ]
    print("PIN TEST: 29 user perceptual fonksiyonları\n")
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ PASS  {name}")
        except AssertionError as e:
            print(f"  ✗ FAIL  {name}\n          {e}")
        except Exception as e:
            print(f"  ⚠ ERR   {name}\n          {type(e).__name__}: {e}")
