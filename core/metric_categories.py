"""ColorBench metric category taxonomy.

Her ColorBench metric grubu 4 kategoriye ayrılır:

1. **mathematical** — Matematiksel correctness/sağlık. Kullanıcı görmez.
   (round-trip error, NaN/Inf count, channel monotonicity, double round-trip,
    cross-gamut consistency, negative LMS detection, quantization symmetry)

2. **structural** — Uzayın iç yapısal düzgünlüğü. Sub-JND seviyede.
   Görsel kazanca **dolaylı** çevrilir (ancak uzayın native API'si kullanılırsa).
   (achromatic axis, cusp count, primary discontinuity, hue reversals,
    OOG excursion, extreme chroma stability)

3. **perceptual_internal** — Uzay-içi perceptual uniformity. Akademik
   referans dataset'lerine bağlı; "uzay perceptually uniform mı" ölçer.
   Görsel kazanca **kısmen** çevrilir.
   (Munsell V uniformity, Munsell hue spacing, MacAdam isotropy, Hung-Berns,
    Ebner-Fairchild, Pointer gamut, hue agreement, hue leaf, jacobian)

4. **perceptual_visible** — End-user'ın görme şansı en yüksek.
   Görev-spesifik gradient/palette/animation/gamut metric'leri.
   "Tasarımcı bu uzayı kullanırsa gözle iyi sonuç alır mı" sorusunu cevaplıyor.
   (gradients CV, multistop gradient, banding, photo gamut map, palette
    uniformity, tint/shade hue drift, shade palette, eased animation,
    chroma preservation, harmony accuracy, dataviz distinguishability,
    contrast, WCAG midpoint, CVD)

Kullanım:
    from colorbench.core.metric_categories import CATEGORIES, category_of

    cat = category_of("munsell_value.dL_cv")  # → "perceptual_internal"
    cat = category_of("gradients.overall_cv_mean")  # → "perceptual_visible"
"""

# Top-level group → category mapping
GROUP_CATEGORY = {
    # ─── MATHEMATICAL ─────────────────────────────────────────────
    "roundtrip":            "mathematical",
    "double_rt":            "mathematical",
    "cross_gamut":          "mathematical",
    "channel_mono":         "mathematical",
    "negative_lms":         "mathematical",
    "quantization":         "mathematical",

    # ─── STRUCTURAL (sub-JND, gamut topology) ─────────────────────
    "achromatic":           "structural",
    "extremes":             "structural",
    "primary_hue_disc":     "structural",
    "hue_reversal":         "structural",
    "oog_excursion":        "structural",
    "extreme_chroma_stab":  "structural",
    "stability":            "structural",
    "gamut":                "structural",  # mostly cusp count/smoothness — affects topology

    # ─── PERCEPTUAL_INTERNAL (uniformity benchmarks) ──────────────
    "munsell_value":        "perceptual_internal",
    "munsell_hue":          "perceptual_internal",
    "macadam_isotropy":     "perceptual_internal",
    "hung_berns":           "perceptual_internal",
    "ebner_fairchild":      "perceptual_internal",
    "pointer_gamut":        "perceptual_internal",
    "hue_agreement":        "perceptual_internal",
    "hue_leaf":             "perceptual_internal",
    "jacobian":             "perceptual_internal",

    # ─── PERCEPTUAL_VISIBLE (end-user görsel) ────────────────────
    "gradients":            "perceptual_visible",
    "multistop_gradient":   "perceptual_visible",
    "banding":              "perceptual_visible",
    "3color":               "perceptual_visible",
    "photo_gamut_map":      "perceptual_visible",
    "gamut_mapping":        "perceptual_visible",
    "palette_uniformity":   "perceptual_visible",
    "tint_shade_hue":       "perceptual_visible",
    "shade_hue_consistency":"perceptual_visible",
    "chroma_preservation":  "perceptual_visible",
    "harmony_accuracy":     "perceptual_visible",
    "dataviz_distinguish":  "perceptual_visible",
    "eased_animation":      "perceptual_visible",
    "animation":            "perceptual_visible",
    "wcag_midpoint":        "perceptual_visible",
    "contrast":             "perceptual_visible",
    "cvd":                  "perceptual_visible",
    "specials":             "perceptual_visible",  # blue-white midpoint, yellow chroma, etc.
    "hue":                  "perceptual_visible",  # primary hue ordering visible in palette work

    # ─── Phase 9: 29 end-user perceptual tests (all visible) ──────
    "user_image_synthetic_gradient":  "perceptual_visible",
    "user_color_grading_lut":         "perceptual_visible",
    "user_white_balance":             "perceptual_visible",
    "user_natural_scene_palette":     "perceptual_visible",
    "user_tailwind_palette":          "perceptual_visible",
    "user_material_palette":          "perceptual_visible",
    "user_diverging_colormap":        "perceptual_visible",
    "user_sequential_colormap":       "perceptual_visible",
    "user_categorical_palette":       "perceptual_visible",
    "user_theme_dark_mode":           "perceptual_visible",
    "user_skin_tone_fitzpatrick":     "perceptual_visible",
    "user_natural_colors":            "perceptual_visible",
    "user_brand_colors":              "perceptual_visible",
    "user_logo_color_preservation":   "perceptual_visible",
    "user_cinematic_lut":             "perceptual_visible",
    "user_picker_hue_continuity":     "perceptual_visible",
    "user_picker_chroma_envelope":    "perceptual_visible",
    "user_achromatic_visual":         "perceptual_visible",
    "user_hue_wheel_uniformity":      "perceptual_visible",
    "user_cvd_palette_spacing":       "perceptual_visible",
    "user_low_vision_contrast":       "perceptual_visible",
    "user_color_blind_safe_palettes": "perceptual_visible",
    "user_p3_wide_gamut":             "perceptual_visible",
    "user_rec2020_hdr_gamut":         "perceptual_visible",
    "user_display_calibration_drift": "perceptual_visible",
    "user_8bit_quantization":         "perceptual_visible",
    "user_hover_state_transition":    "perceptual_visible",
    "user_focus_ring_quality":        "perceptual_visible",
    "user_dark_mode_flip":            "perceptual_visible",

    # ─── Phase 10: 8 ek end-user perceptual test ──────────────────
    "user_print_cmyk_fidelity":       "perceptual_visible",
    "user_pantone_spot":              "perceptual_visible",
    "user_hdr_tone_mapping":          "perceptual_visible",
    "user_cvd_tritanomaly":           "perceptual_visible",
    "user_newsprint_simulation":      "perceptual_visible",
    "user_cross_cultural_skin":       "perceptual_visible",
    "user_glassmorphism":             "perceptual_visible",
    "user_status_indicator_distinct": "perceptual_visible",

    # ─── Phase 11: 2 yeni real-data test ──────────────────────────
    "user_real_photo_macbeth":        "perceptual_visible",
    "user_jnd_aware_summary":         "perceptual_visible",
}


CATEGORIES = ["mathematical", "structural", "perceptual_internal", "perceptual_visible"]


# ColorBench's existing 12 fine-grained categories (in MetricDef.category)
# → mapped to our 4 meta-categories.
COLORBENCH_TO_META = {
    "Numerical":     "mathematical",
    "Achromatic":    "structural",
    "Cusp":          "structural",
    "Structural":    "structural",
    "Stability":     "structural",
    "Perceptual":    "perceptual_internal",
    "Independent":   "perceptual_internal",
    "Gradient":      "perceptual_visible",
    "Banding":       "perceptual_visible",
    "Application":   "perceptual_visible",
    "Accessibility": "perceptual_visible",
    "Advanced":      "perceptual_visible",
    "Special":       "perceptual_visible",
}


def meta_category_of_colorbench(cb_category: str) -> str:
    return COLORBENCH_TO_META.get(cb_category, "unknown")

CATEGORY_LABEL = {
    "mathematical":         "Matematiksel sağlık (kullanıcı görmez)",
    "structural":           "Yapısal düzgünlük (sub-JND, dolaylı görünür)",
    "perceptual_internal":  "İç perceptual uniformity (akademik dataset)",
    "perceptual_visible":   "End-user görsel (tasarımcı görür)",
}


def category_of(metric_id):
    """Flat metric ID → category. ID format: 'group.metric' or 'group'.

    Returns 'unknown' if group not in GROUP_CATEGORY map.
    """
    group = metric_id.split(".")[0] if "." in metric_id else metric_id
    return GROUP_CATEGORY.get(group, "unknown")


def categorize_results(results_dict):
    """Take a flat dict of metric_id → value, return dict of category → list of (id, value)."""
    by_cat = {c: [] for c in CATEGORIES}
    by_cat["unknown"] = []
    for mid, val in results_dict.items():
        cat = category_of(mid)
        by_cat[cat].append((mid, val))
    return by_cat


def category_summary(by_cat):
    """Pretty-print category counts."""
    lines = []
    for cat in CATEGORIES + ["unknown"]:
        items = by_cat.get(cat, [])
        if items:
            lines.append(f"  {cat:<22} {len(items)} metric — {CATEGORY_LABEL.get(cat, cat)}")
    return "\n".join(lines)
