"""Hardcoded perceptual data constants — no external file dependencies.

Sources:
  - Munsell Value→Y: ASTM D1535 (Munsell renotation)
  - Munsell Hue Chips: Approximate sRGB at V=5, C≈6
  - MacAdam Ellipse Centers: MacAdam 1942, CIE xy chromaticity (illum C)
  - Multi-stop gradients: Common CSS gradient patterns
  - WCAG contrast pairs: Representative accessibility test cases
"""

# Bump whenever a metric's behavior changes — cached baseline reports are
# keyed on this, so a stale cache can never be compared against new code.
METHODOLOGY_VERSION = "2026-07-08"

# ── Munsell Value → Y (ASTM D1535) ──────────────────────────
# Y values for neutral grays at Munsell Values 1-9
# Used to test L-channel uniformity: ideal space maps equal
# Munsell value steps to equal L* steps.
# Values match the Munsell renotation (NNJ 1943 quintic) and the pool's
# munsell/canonical.csv exactly. 2026-07-08 dataset audit fixed V=1: the old
# 0.01221 was the quintic's linear coefficient (1.2219) slipped in — correct
# Y(V=1) is 1.210% (a ~0.9% error that biased the munsell_value metric).
MUNSELL_VALUE_Y = {
    1: 0.01210,
    2: 0.03126,
    3: 0.06555,
    4: 0.12000,
    5: 0.19770,
    6: 0.30050,
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

# ── MacAdam 1942 Ellipses ────────────────────────────────────
# 25 JND ellipses from MacAdam's 1942 paper (illuminant C):
# (x_c, y_c, semi_major_a, semi_minor_b, theta_deg).
# Every point on an ellipse is ONE just-noticeable difference from its
# center, so a perceptually-uniform space maps the PERIMETER to equal
# distance from the center — max/min ratio 1.0 is the perceptual target.
# (Perturbing by a fixed xy CIRCLE instead — as the pre-2026-07 metric
# did — rewards spaces for being isotropic in raw xy, i.e. for IGNORING
# MacAdam anisotropy. Anti-perceptual; never do that.)
MACADAM_ELLIPSES = [
    (0.160, 0.057, 0.00085, 0.00035,  62.5),
    (0.187, 0.118, 0.00220, 0.00055,  77.0),
    (0.253, 0.125, 0.00250, 0.00050,  55.5),
    (0.150, 0.680, 0.00960, 0.00230, 105.0),
    (0.131, 0.521, 0.00470, 0.00200, 112.5),
    (0.212, 0.550, 0.00580, 0.00230, 100.0),
    (0.258, 0.450, 0.00500, 0.00200,  92.0),
    (0.152, 0.365, 0.00380, 0.00190, 110.0),
    (0.280, 0.385, 0.00400, 0.00150,  75.5),
    (0.380, 0.498, 0.00440, 0.00120,  70.0),
    (0.160, 0.200, 0.00210, 0.00095, 104.0),
    (0.228, 0.250, 0.00310, 0.00090,  72.0),
    (0.305, 0.323, 0.00230, 0.00090,  58.0),
    (0.385, 0.393, 0.00380, 0.00160,  65.5),
    (0.472, 0.399, 0.00320, 0.00140,  51.0),
    (0.527, 0.350, 0.00260, 0.00130,  20.0),
    (0.475, 0.300, 0.00290, 0.00110,  28.5),
    (0.510, 0.236, 0.00240, 0.00120,  29.5),
    (0.596, 0.283, 0.00260, 0.00130,  13.0),
    (0.344, 0.284, 0.00230, 0.00090,  60.0),
    (0.390, 0.237, 0.00250, 0.00100,  47.0),
    (0.441, 0.198, 0.00280, 0.00095,  34.5),
    (0.278, 0.223, 0.00240, 0.00055,  57.5),
    (0.300, 0.163, 0.00290, 0.00060,  54.0),
    (0.365, 0.153, 0.00360, 0.00095,  40.0),
]

# Centers only (legacy consumers)
MACADAM_CENTERS = [(e[0], e[1]) for e in MACADAM_ELLIPSES]

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
