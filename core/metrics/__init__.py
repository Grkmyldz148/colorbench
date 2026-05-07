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
]
