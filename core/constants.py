"""Hardcoded perceptual data constants — no external file dependencies.

Sources:
  - Munsell Value→Y: ASTM D1535 (Munsell renotation)
  - Munsell Hue Chips: Approximate sRGB at V=5, C≈6
  - MacAdam Ellipse Centers: MacAdam 1942, CIE xy chromaticity (illum C)
  - Multi-stop gradients: Common CSS gradient patterns
  - WCAG contrast pairs: Representative accessibility test cases
"""

# ── Munsell Value → Y (ASTM D1535) ──────────────────────────
# Y values for neutral grays at Munsell Values 1-9
# Used to test L-channel uniformity: ideal space maps equal
# Munsell value steps to equal L* steps.
MUNSELL_VALUE_Y = {
    1: 0.01221,
    2: 0.03126,
    3: 0.06552,
    4: 0.12000,
    5: 0.19770,
    6: 0.30049,
    7: 0.43060,
    8: 0.59100,
    9: 0.78660,
}

# ── Munsell 10 Principal Hue Chips (sRGB 8-bit) ─────────────
# At V=5, C≈6. 10 hues equally spaced around the Munsell circle.
# Ideal space should place these at equal angular intervals (~36°).
MUNSELL_HUE_CHIPS_RGB = {
    '5R':  (176, 103, 101),
    '5YR': (169, 117,  82),
    '5Y':  (155, 135,  80),
    '5GY': (115, 143,  87),
    '5G':  ( 75, 148, 115),
    '5BG': ( 58, 146, 140),
    '5B':  ( 69, 138, 159),
    '5PB': (101, 118, 162),
    '5P':  (132, 106, 149),
    '5RP': (159,  99, 126),
}

# ── MacAdam 1942 Ellipse Centers ─────────────────────────────
# 25 chromaticity coordinates (CIE xy) from MacAdam's 1942 paper.
# Used to measure local isotropy: ideal space has equal distance
# in all directions from each center (ratio = 1.0).
MACADAM_CENTERS = [
    (0.160, 0.057), (0.187, 0.118), (0.253, 0.125), (0.150, 0.680),
    (0.131, 0.521), (0.212, 0.550), (0.258, 0.450), (0.152, 0.365),
    (0.280, 0.385), (0.380, 0.498), (0.160, 0.200), (0.228, 0.250),
    (0.305, 0.323), (0.385, 0.393), (0.472, 0.399), (0.527, 0.350),
    (0.475, 0.300), (0.510, 0.236), (0.596, 0.283), (0.344, 0.284),
    (0.390, 0.237), (0.441, 0.198), (0.278, 0.223), (0.300, 0.163),
    (0.365, 0.153),
]

# ── CSS Multi-Stop Gradient Patterns ─────────────────────────
# Common gradient patterns for testing multi-point interpolation quality.
MULTI_STOP_GRADIENTS = {
    "Rainbow 5": ['#ff0000', '#ffff00', '#00ff00', '#0000ff', '#ff00ff'],
    "Warm 4":    ['#ff0000', '#ff8800', '#ffff00', '#ffffff'],
    "Cool 4":    ['#0000ff', '#00aaff', '#00ffff', '#ffffff'],
    "Brand 3":   ['#6366f1', '#ec4899', '#f59e0b'],
}

# ── WCAG Contrast Test Pairs ─────────────────────────────────
# Representative pairs for testing contrast ratio preservation at midpoint.
WCAG_CONTRAST_PAIRS = [
    ('#000000', '#ffffff'),   # Black-White (21:1)
    ('#1a1a2e', '#e0e0e0'),   # Dark-Light (~12:1)
    ('#0000ff', '#ffff00'),   # Blue-Yellow
    ('#ff0000', '#ffffff'),   # Red-White
    ('#006600', '#ffffff'),   # Green-White
]
