"""ColorBench metric modules — public API.

Each metric is a focused module exporting one measure_* function.
"""
from .roundtrip import measure_roundtrip
from .achromatic import measure_achromatic
from .gradients import measure_gradients
from .gamut import measure_gamut
from .gamut_mapping import measure_gamut_mapping
from .hue import measure_hue, measure_special_gradients
from .stability import measure_stability
from .munsell import measure_munsell_value, measure_munsell_hue
from .macadam import measure_macadam_isotropy
from .palette import measure_palette_uniformity, measure_tint_shade_hue
from .dataviz import measure_dataviz_distinguishability, measure_multistop_gradient
from .wcag import measure_wcag_midpoint_contrast
from .harmony import measure_harmony_accuracy, measure_hue_agreement
from .photo import measure_photo_gamut_map
from .shade import measure_shade_hue_consistency, measure_chroma_preservation
from .animation import measure_eased_animation, measure_animation
from .cvd import measure_cvd
from .extremes import measure_extremes, measure_jacobian
from .contrast import measure_contrast
from .hue_leaf import measure_hue_leaf
from .multi_gradient import measure_3color_gradients
from .double_rt import measure_double_roundtrip
from .cross_gamut import measure_cross_gamut_consistency
from .quantization import measure_quantization_symmetry
from .channel_mono import measure_channel_monotonicity
from .banding import measure_perceptual_banding
from .oog import measure_oog_excursion
from .hue_reversal import measure_hue_reversal
from .primary_disc import measure_primary_hue_discontinuity
from .negative_lms import measure_negative_lms
from .extreme_chroma import measure_extreme_chroma_stability
from .independent import (
    measure_hung_berns, measure_ebner_fairchild, measure_pointer_gamut,
)
from .user_full import (
    measure_user_image_synthetic_gradient,
    measure_user_color_grading_lut,
    measure_user_white_balance,
    measure_user_natural_scene_palette,
    measure_user_tailwind_palette,
    measure_user_material_palette,
    measure_user_diverging_colormap,
    measure_user_sequential_colormap,
    measure_user_categorical_palette,
    measure_user_theme_dark_mode,
    measure_user_skin_tone_fitzpatrick,
    measure_user_natural_colors,
    measure_user_brand_colors,
    measure_user_logo_color_preservation,
    measure_user_cinematic_lut,
    measure_user_picker_hue_continuity,
    measure_user_picker_chroma_envelope,
    measure_user_achromatic_visual,
    measure_user_hue_wheel_uniformity,
    measure_user_cvd_palette_spacing,
    measure_user_low_vision_contrast,
    measure_user_color_blind_safe_palettes,
    measure_user_p3_wide_gamut,
    measure_user_rec2020_hdr_gamut,
    measure_user_display_calibration_drift,
    measure_user_8bit_quantization,
    measure_user_hover_state_transition,
    measure_user_focus_ring_quality,
    measure_user_dark_mode_flip,
    measure_user_print_cmyk_fidelity,
    measure_user_pantone_spot,
    measure_user_hdr_tone_mapping,
    measure_user_cvd_tritanomaly,
    measure_user_newsprint_simulation,
    measure_user_cross_cultural_skin,
    measure_user_glassmorphism,
    measure_user_status_indicator_distinct,
    measure_user_real_photo_macbeth,
    measure_user_jnd_aware_summary,
)

__all__ = [
    "measure_roundtrip",
    "measure_achromatic",
    "measure_gradients",
    "measure_gamut",
    "measure_gamut_mapping",
    "measure_hue",
    "measure_special_gradients",
    "measure_stability",
    "measure_munsell_value",
    "measure_munsell_hue",
    "measure_macadam_isotropy",
    "measure_palette_uniformity",
    "measure_tint_shade_hue",
    "measure_dataviz_distinguishability",
    "measure_multistop_gradient",
    "measure_wcag_midpoint_contrast",
    "measure_harmony_accuracy",
    "measure_hue_agreement",
    "measure_photo_gamut_map",
    "measure_eased_animation",
    "measure_shade_hue_consistency",
    "measure_chroma_preservation",
    "measure_animation",
    "measure_cvd",
    "measure_extremes",
    "measure_jacobian",
    "measure_contrast",
    "measure_hue_leaf",
    "measure_3color_gradients",
    "measure_double_roundtrip",
    "measure_cross_gamut_consistency",
    "measure_quantization_symmetry",
    "measure_channel_monotonicity",
    "measure_perceptual_banding",
    "measure_oog_excursion",
    "measure_hue_reversal",
    "measure_primary_hue_discontinuity",
    "measure_negative_lms",
    "measure_extreme_chroma_stability",
    "measure_hung_berns",
    "measure_ebner_fairchild",
    "measure_pointer_gamut",
]
