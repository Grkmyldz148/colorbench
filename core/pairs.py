"""Gradient pair generation — comprehensive sRGB gamut coverage.

Generates ~700 pairs in categories:
  primary     (15) — all 6-choose-2 primary/secondary combos
  to_white    (6)  — each primary → white
  to_black    (6)  — each primary → black
  hue_sweep   (72) — adjacent hues every 5° at full saturation
  saturation  (36) — same hue, varying saturation → white
  lightness   (42) — same hue, dark → light
  gray        (10) — achromatic pairs at different levels
  complementary (6) — opposite hues (R-C, G-M, B-Y)
  random     (500) — uniform random sRGB
"""

import torch
import math


def srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))


def generate_all_pairs(device):
    """Returns (pairs_xyz, pair_labels) where pairs_xyz is (N, 2, 3) XYZ tensor
    and pair_labels is list of (category, description) tuples."""

    M_SRGB = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], device=device, dtype=torch.float64)

    pairs = []
    labels = []

    def add(cat, desc, rgb1, rgb2):
        r1 = torch.tensor(rgb1, device=device, dtype=torch.float64)
        r2 = torch.tensor(rgb2, device=device, dtype=torch.float64)
        x1 = M_SRGB @ srgb_to_linear(r1)
        x2 = M_SRGB @ srgb_to_linear(r2)
        pairs.append(torch.stack([x1, x2]))
        labels.append((cat, desc))

    # Primary/secondary combos
    primaries = {"R": [1,0,0], "G": [0,1,0], "B": [0,0,1],
                 "Y": [1,1,0], "C": [0,1,1], "M": [1,0,1]}
    names = list(primaries.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            add("primary", f"{names[i]}-{names[j]}",
                primaries[names[i]], primaries[names[j]])

    # To white / black
    for n, rgb in primaries.items():
        add("to_white", f"{n}->W", rgb, [1, 1, 1])
        add("to_black", f"{n}->K", rgb, [0, 0, 0])

    # Hue sweep — every 5° at full saturation
    for h_start in range(0, 360, 5):
        h_end = (h_start + 30) % 360
        r1, g1, b1 = _hsv_to_rgb(h_start / 360, 1.0, 1.0)
        r2, g2, b2 = _hsv_to_rgb(h_end / 360, 1.0, 1.0)
        add("hue_sweep", f"h{h_start}-h{h_end}", [r1, g1, b1], [r2, g2, b2])

    # Saturation sweep — desaturate to white
    for h in [0, 60, 120, 180, 240, 300]:
        for s in [1.0, 0.7, 0.4]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, s, 1.0)
            add("saturation", f"h{h}s{s:.1f}->W", [r1, g1, b1], [1, 1, 1])

    # Lightness sweep — same hue, dark to light
    for h in [0, 60, 120, 180, 240, 300]:
        for v_lo, v_hi in [(0.2, 0.8), (0.1, 0.5), (0.5, 1.0)]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, 0.8, v_lo)
            r2, g2, b2 = _hsv_to_rgb(h / 360, 0.8, v_hi)
            add("lightness", f"h{h}v{v_lo}-v{v_hi}", [r1, g1, b1], [r2, g2, b2])
        # Full dark to light
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.8, 0.15)
        r2, g2, b2 = _hsv_to_rgb(h / 360, 0.8, 0.95)
        add("lightness", f"h{h}dark-light", [r1, g1, b1], [r2, g2, b2])

    # Gray pairs
    grays = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(grays)):
        for j in range(i + 2, len(grays)):
            add("gray", f"g{grays[i]:.1f}-g{grays[j]:.1f}",
                [grays[i]] * 3, [grays[j]] * 3)

    # Complementary (exact opposites, both directions)
    for n1, n2 in [("R", "C"), ("C", "R"), ("G", "M"), ("M", "G"), ("B", "Y"), ("Y", "B")]:
        add("complementary", f"{n1}-{n2}",
            primaries[n1], primaries[n2])

    # Near-achromatic pairs (low chroma, where spaces struggle most)
    for h in range(0, 360, 15):
        for s in [0.05, 0.10, 0.15]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, s, 0.5)
            r2, g2, b2 = _hsv_to_rgb(((h + 30) % 360) / 360, s, 0.5)
            add("near_achromatic", f"h{h}s{s:.2f}", [r1, g1, b1], [r2, g2, b2])

    # Dark-to-dark pairs
    for h in range(0, 360, 30):
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.6, 0.15)
        r2, g2, b2 = _hsv_to_rgb(((h + 60) % 360) / 360, 0.6, 0.15)
        add("dark_dark", f"dark_h{h}", [r1, g1, b1], [r2, g2, b2])

    # Pastel pairs
    for h in range(0, 360, 20):
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.25, 0.95)
        r2, g2, b2 = _hsv_to_rgb(((h + 40) % 360) / 360, 0.25, 0.95)
        add("pastel", f"pastel_h{h}", [r1, g1, b1], [r2, g2, b2])

    # Hue wrap-around (near red, h≈0°/360°)
    for h_start in [350, 355, 358]:
        for h_end in [2, 5, 10]:
            r1, g1, b1 = _hsv_to_rgb(h_start / 360, 1.0, 1.0)
            r2, g2, b2 = _hsv_to_rgb(h_end / 360, 1.0, 1.0)
            add("hue_wrap", f"h{h_start}-h{h_end}", [r1, g1, b1], [r2, g2, b2])

    # L extremes — very dark and very bright gradients (12 hues, not 4)
    for h in range(0, 360, 30):
        # Dark: V≈0.02 to V≈0.08
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.5, 0.02)
        r2, g2, b2 = _hsv_to_rgb(h / 360, 0.5, 0.08)
        add("L_extreme_dark", f"vdark_h{h}", [r1, g1, b1], [r2, g2, b2])
        # Very dark: V≈0.01 to V≈0.04
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.7, 0.01)
        r2, g2, b2 = _hsv_to_rgb(h / 360, 0.7, 0.04)
        add("L_extreme_dark", f"vvdark_h{h}", [r1, g1, b1], [r2, g2, b2])
        # Bright: V≈0.93 to V≈0.99
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.3, 0.93)
        r2, g2, b2 = _hsv_to_rgb(h / 360, 0.3, 0.99)
        add("L_extreme_bright", f"vbright_h{h}", [r1, g1, b1], [r2, g2, b2])
        # Very bright: V≈0.97 to V≈1.0
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.15, 0.97)
        r2, g2, b2 = _hsv_to_rgb(h / 360, 0.15, 1.0)
        add("L_extreme_bright", f"vvbright_h{h}", [r1, g1, b1], [r2, g2, b2])

    # Gamut boundary stress — sRGB colors near the edge (high saturation, various L)
    for h in range(0, 360, 15):
        for v in [0.3, 0.5, 0.7, 0.9]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, 0.95, v)
            r2, g2, b2 = _hsv_to_rgb(h / 360, 0.95, min(v + 0.2, 1.0))
            add("boundary_srgb", f"bnd_h{h}_v{v}", [r1, g1, b1], [r2, g2, b2])

    # ── Skin tone gradients ──────────────────────────────────────
    # Real-world skin tones across Fitzpatrick scale (I-VI)
    # Most common gradient use case: portrait editing, cosmetics, avatars
    skin_tones = [
        ([0.976, 0.906, 0.839], [0.941, 0.800, 0.682]),  # I-II (very light)
        ([0.941, 0.800, 0.682], [0.871, 0.643, 0.486]),  # II-III
        ([0.871, 0.643, 0.486], [0.737, 0.494, 0.345]),  # III-IV
        ([0.737, 0.494, 0.345], [0.545, 0.329, 0.208]),  # IV-V
        ([0.545, 0.329, 0.208], [0.361, 0.208, 0.129]),  # V-VI (very dark)
        ([0.976, 0.906, 0.839], [0.361, 0.208, 0.129]),  # I-VI full range
        ([0.871, 0.643, 0.486], [1.0, 1.0, 1.0]),         # mid skin → white (highlight)
        ([0.545, 0.329, 0.208], [0.0, 0.0, 0.0]),         # dark skin → black (shadow)
        ([0.941, 0.800, 0.682], [0.871, 0.580, 0.451]),   # blush/warmth shift
        ([0.737, 0.494, 0.345], [0.659, 0.510, 0.392]),   # olive undertone shift
    ]
    for i, (rgb1, rgb2) in enumerate(skin_tones):
        add("skin_tone", f"skin_{i}", rgb1, rgb2)

    # ── Earth tone gradients ─────────────────────────────────────
    # Landscape, architecture, nature photography
    earth_tones = [
        ([0.545, 0.353, 0.169], [0.282, 0.471, 0.208]),  # brown → forest green
        ([0.824, 0.706, 0.549], [0.545, 0.353, 0.169]),   # sand → brown
        ([0.282, 0.471, 0.208], [0.133, 0.267, 0.133]),   # forest → dark green
        ([0.824, 0.706, 0.549], [0.529, 0.596, 0.627]),   # sand → stone gray
        ([0.545, 0.353, 0.169], [0.824, 0.412, 0.118]),   # brown → terracotta
        ([0.412, 0.325, 0.220], [0.608, 0.537, 0.396]),   # dark earth → light earth
        ([0.133, 0.267, 0.133], [0.824, 0.706, 0.549]),   # forest → sand (full range)
        ([0.490, 0.490, 0.412], [0.282, 0.224, 0.141]),   # olive → dark umber
    ]
    for i, (rgb1, rgb2) in enumerate(earth_tones):
        add("earth_tone", f"earth_{i}", rgb1, rgb2)

    # ── Warm↔Cool transitions ────────────────────────────────────
    # Photography white balance, cinematic color grading
    warm_cool = [
        ([1.0, 0.600, 0.200], [0.200, 0.400, 1.0]),       # warm orange → cool blue
        ([0.957, 0.263, 0.212], [0.129, 0.588, 0.953]),   # red → blue (classic)
        ([1.0, 0.757, 0.027], [0.247, 0.318, 0.710]),     # amber → indigo
        ([0.984, 0.549, 0.235], [0.475, 0.643, 0.776]),   # peach → steel blue
        ([0.804, 0.361, 0.361], [0.361, 0.612, 0.804]),   # warm pink → cool blue
        ([1.0, 0.843, 0.600], [0.600, 0.800, 1.0]),       # warm white → cool white
        ([0.933, 0.510, 0.165], [0.165, 0.651, 0.745]),   # tangerine → teal
        ([0.698, 0.133, 0.133], [0.098, 0.098, 0.698]),   # dark red → dark blue
        ([1.0, 0.922, 0.804], [0.804, 0.878, 1.0]),       # warm highlight → cool highlight
        ([0.600, 0.200, 0.000], [0.000, 0.200, 0.600]),   # dark warm → dark cool
    ]
    for i, (rgb1, rgb2) in enumerate(warm_cool):
        add("warm_cool", f"wc_{i}", rgb1, rgb2)

    # ── Very similar colors (dE < 5 stress) ──────────────────────
    # Banding and quantization stress — where spaces struggle most
    similar_pairs = [
        ([0.500, 0.500, 0.500], [0.510, 0.500, 0.500]),   # near-gray, tiny R shift
        ([0.500, 0.500, 0.500], [0.500, 0.510, 0.510]),   # near-gray, tiny GB shift
        ([0.200, 0.400, 0.700], [0.210, 0.405, 0.695]),   # blue, tiny perturbation
        ([0.800, 0.300, 0.300], [0.790, 0.310, 0.305]),   # red, tiny perturbation
        ([0.100, 0.600, 0.100], [0.105, 0.595, 0.110]),   # green, tiny perturbation
        ([0.900, 0.900, 0.100], [0.895, 0.895, 0.110]),   # yellow, tiny perturbation
        ([0.300, 0.300, 0.300], [0.305, 0.305, 0.305]),   # dark gray, minimal step
        ([0.700, 0.700, 0.700], [0.705, 0.705, 0.705]),   # light gray, minimal step
        ([0.950, 0.800, 0.650], [0.945, 0.805, 0.655]),   # skin-like, minimal step
        ([0.400, 0.200, 0.100], [0.405, 0.205, 0.105]),   # brown, minimal step
        ([0.100, 0.100, 0.400], [0.105, 0.100, 0.395]),   # dark blue, minimal step
        ([0.600, 0.300, 0.600], [0.595, 0.305, 0.605]),   # purple, minimal step
    ]
    for i, (rgb1, rgb2) in enumerate(similar_pairs):
        add("very_similar", f"sim_{i}", rgb1, rgb2)

    # ── Neon/fluorescent colors ──────────────────────────────────
    # UI design, gaming, high-saturation displays
    neon_pairs = [
        ([0.0, 1.0, 0.0], [1.0, 1.0, 0.0]),               # neon green → neon yellow
        ([1.0, 0.0, 1.0], [0.0, 1.0, 1.0]),               # magenta → cyan
        ([1.0, 0.0, 0.5], [0.5, 0.0, 1.0]),               # hot pink → electric purple
        ([0.0, 1.0, 0.5], [0.0, 0.5, 1.0]),               # spring green → ocean blue
        ([1.0, 0.4, 0.0], [1.0, 0.0, 0.4]),               # electric orange → neon red
        ([0.5, 1.0, 0.0], [0.0, 1.0, 0.5]),               # chartreuse → spring green
        ([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]),               # pure red → pure yellow
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),               # pure blue → pure green
    ]
    for i, (rgb1, rgb2) in enumerate(neon_pairs):
        add("neon", f"neon_{i}", rgb1, rgb2)

    # ── Cross-gamut gradients (P3 → sRGB) ────────────────────────
    # Real scenario: photo editing, wide-gamut display to web export
    M_P3_early = torch.tensor([
        [0.4865709486482162, 0.26566769316909306, 0.1982172852343625],
        [0.2289745640697488, 0.6917385218365064, 0.079286914093745],
        [0.0, 0.04511338185890264, 1.0439443689009757],
    ], device=device, dtype=torch.float64)
    cross_gamut_specific = [
        ([1.0, 0.0, 0.0], [1.0, 0.5, 0.0], True, False),   # P3 red → sRGB orange
        ([0.0, 1.0, 0.0], [0.0, 0.8, 0.2], True, False),   # P3 green → sRGB green
        ([0.0, 0.8, 0.8], [0.0, 0.7, 0.7], True, False),   # P3 teal → sRGB teal
        ([0.6, 0.0, 1.0], [0.5, 0.0, 0.8], True, False),   # P3 purple → sRGB purple
    ]
    for i, (rgb1, rgb2, is_p3_1, is_p3_2) in enumerate(cross_gamut_specific):
        r1 = torch.tensor(rgb1, device=device, dtype=torch.float64)
        r2 = torch.tensor(rgb2, device=device, dtype=torch.float64)
        m1 = M_P3_early if is_p3_1 else M_SRGB
        m2 = M_P3_early if is_p3_2 else M_SRGB
        x1 = m1 @ srgb_to_linear(r1)
        x2 = m2 @ srgb_to_linear(r2)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("cross_gamut_specific", f"xgamut_{i}"))

    # ── High-luminance transitions ───────────────────────────────
    # Near-white chromatic gradients (HDR-like, Y > 0.8)
    high_lum = [
        ([1.0, 0.95, 0.85], [0.85, 0.95, 1.0]),           # warm white → cool white
        ([1.0, 0.90, 0.90], [0.90, 1.0, 0.90]),           # pink white → mint white
        ([1.0, 1.0, 0.85], [0.85, 0.85, 1.0]),             # cream → lavender
        ([0.95, 0.85, 0.75], [0.75, 0.85, 0.95]),         # warm light → cool light
        ([1.0, 0.92, 0.80], [0.92, 0.96, 1.0]),           # sunset glow → sky glow
        ([0.98, 0.90, 0.85], [0.90, 0.92, 0.98]),         # warm peach → cool periwinkle
    ]
    for i, (rgb1, rgb2) in enumerate(high_lum):
        add("high_luminance", f"hilum_{i}", rgb1, rgb2)

    # ── UI shade palette pairs ───────────────────────────────────
    # Same hue, 10 lightness levels — Material Design / Tailwind pattern
    for h_deg in [0, 30, 60, 120, 200, 270, 330]:
        shades = []
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            r, g, b = _hsv_to_rgb(h_deg / 360, 0.7, v)
            shades.append([r, g, b])
        # Adjacent shade pairs
        for j in range(len(shades) - 1):
            add("ui_shade", f"shade_h{h_deg}_v{j}", shades[j], shades[j + 1])
        # Wide jumps (50→900, 100→800)
        add("ui_shade", f"shade_h{h_deg}_wide1", shades[1], shades[8])
        add("ui_shade", f"shade_h{h_deg}_wide2", shades[0], shades[9])

    # Random sRGB (1000)
    gen = torch.Generator(device=device).manual_seed(42)
    for k in range(1000):
        rgb1 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        rgb2 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        x1 = M_SRGB @ srgb_to_linear(rgb1)
        x2 = M_SRGB @ srgb_to_linear(rgb2)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("random_srgb", f"rnd_s{k}"))

    # ── P3 gamut pairs ──
    M_P3 = torch.tensor([
        [0.4865709486482162, 0.26566769316909306, 0.1982172852343625],
        [0.2289745640697488, 0.6917385218365064, 0.079286914093745],
        [0.0, 0.04511338185890264, 1.0439443689009757],
    ], device=device, dtype=torch.float64)

    # P3 primaries
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rgb1_t = torch.tensor(primaries[names[i]], device=device, dtype=torch.float64)
            rgb2_t = torch.tensor(primaries[names[j]], device=device, dtype=torch.float64)
            x1 = M_P3 @ srgb_to_linear(rgb1_t)  # P3 linear primaries
            x2 = M_P3 @ srgb_to_linear(rgb2_t)
            pairs.append(torch.stack([x1, x2]))
            labels.append(("p3_primary", f"P3_{names[i]}-{names[j]}"))

    # P3 to sRGB cross-gamut: P3 primary → sRGB white
    for n, rgb in primaries.items():
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        x1 = M_P3 @ srgb_to_linear(rgb_t)
        x2 = M_SRGB @ srgb_to_linear(torch.tensor([1., 1., 1.], device=device))
        pairs.append(torch.stack([x1, x2]))
        labels.append(("p3_to_srgb", f"P3_{n}->sRGB_W"))

    # P3 to_white / to_black
    for n, rgb in primaries.items():
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        x1 = M_P3 @ srgb_to_linear(rgb_t)
        x_w = M_P3 @ srgb_to_linear(torch.ones(3, device=device, dtype=torch.float64))
        x_k = M_P3 @ srgb_to_linear(torch.zeros(3, device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x_w]))
        labels.append(("p3_to_white", f"P3_{n}->W"))
        pairs.append(torch.stack([x1, x_k]))
        labels.append(("p3_to_black", f"P3_{n}->K"))

    # P3 hue sweep
    for h_start in range(0, 360, 15):
        h_end = (h_start + 30) % 360
        r1, g1, b1 = _hsv_to_rgb(h_start / 360, 1.0, 1.0)
        r2, g2, b2 = _hsv_to_rgb(h_end / 360, 1.0, 1.0)
        x1 = M_P3 @ srgb_to_linear(torch.tensor([r1, g1, b1], device=device, dtype=torch.float64))
        x2 = M_P3 @ srgb_to_linear(torch.tensor([r2, g2, b2], device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x2]))
        labels.append(("p3_hue_sweep", f"P3_h{h_start}-h{h_end}"))

    # P3 near-achromatic (same density as sRGB: every 15°, 3 saturation levels)
    for h in range(0, 360, 15):
        for s in [0.05, 0.10, 0.15]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, s, 0.5)
            r2, g2, b2 = _hsv_to_rgb(((h + 30) % 360) / 360, s, 0.5)
            x1 = M_P3 @ srgb_to_linear(torch.tensor([r1, g1, b1], device=device, dtype=torch.float64))
            x2 = M_P3 @ srgb_to_linear(torch.tensor([r2, g2, b2], device=device, dtype=torch.float64))
            pairs.append(torch.stack([x1, x2]))
            labels.append(("p3_near_achromatic", f"P3_na_h{h}_s{s:.2f}"))

    # P3 boundary stress
    for h in range(0, 360, 15):
        for v in [0.3, 0.5, 0.7, 0.9]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, 0.95, v)
            r2, g2, b2 = _hsv_to_rgb(h / 360, 0.95, min(v + 0.2, 1.0))
            x1 = M_P3 @ srgb_to_linear(torch.tensor([r1, g1, b1], device=device, dtype=torch.float64))
            x2 = M_P3 @ srgb_to_linear(torch.tensor([r2, g2, b2], device=device, dtype=torch.float64))
            pairs.append(torch.stack([x1, x2]))
            labels.append(("p3_boundary", f"P3_bnd_h{h}_v{v}"))

    # P3 lightness sweep
    for h in [0, 60, 120, 180, 240, 300]:
        for v_lo, v_hi in [(0.1, 0.5), (0.5, 1.0), (0.2, 0.8)]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, 0.8, v_lo)
            r2, g2, b2 = _hsv_to_rgb(h / 360, 0.8, v_hi)
            x1 = M_P3 @ srgb_to_linear(torch.tensor([r1, g1, b1], device=device, dtype=torch.float64))
            x2 = M_P3 @ srgb_to_linear(torch.tensor([r2, g2, b2], device=device, dtype=torch.float64))
            pairs.append(torch.stack([x1, x2]))
            labels.append(("p3_lightness", f"P3_L_h{h}_v{v_lo}-{v_hi}"))

    # Random P3 (500)
    for k in range(500):
        rgb1 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        rgb2 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        x1 = M_P3 @ srgb_to_linear(rgb1)
        x2 = M_P3 @ srgb_to_linear(rgb2)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("random_p3", f"rnd_p3_{k}"))

    # ── Rec.2020 pairs ──
    M_R2020 = torch.tensor([
        [0.6369580483012914, 0.14461690358620832, 0.1688809751641721],
        [0.2627002120112671, 0.6779980715188708, 0.05930171646986196],
        [0.0, 0.028072693049087428, 1.0609850577107909],
    ], device=device, dtype=torch.float64)

    # Rec.2020→P3 cross-gamut
    for n, rgb in primaries.items():
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        x1 = M_R2020 @ srgb_to_linear(rgb_t)
        x2 = M_P3 @ srgb_to_linear(rgb_t)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("rec2020_to_p3", f"R2020_{n}->P3_{n}"))

    # Rec.2020 primaries
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rgb1_t = torch.tensor(primaries[names[i]], device=device, dtype=torch.float64)
            rgb2_t = torch.tensor(primaries[names[j]], device=device, dtype=torch.float64)
            x1 = M_R2020 @ srgb_to_linear(rgb1_t)
            x2 = M_R2020 @ srgb_to_linear(rgb2_t)
            pairs.append(torch.stack([x1, x2]))
            labels.append(("rec2020_primary", f"R2020_{names[i]}-{names[j]}"))

    # Rec.2020 to_white / to_black
    for n, rgb in primaries.items():
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        x1 = M_R2020 @ srgb_to_linear(rgb_t)
        x_w = M_R2020 @ srgb_to_linear(torch.ones(3, device=device, dtype=torch.float64))
        x_k = M_R2020 @ srgb_to_linear(torch.zeros(3, device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x_w]))
        labels.append(("rec2020_to_white", f"R2020_{n}->W"))
        pairs.append(torch.stack([x1, x_k]))
        labels.append(("rec2020_to_black", f"R2020_{n}->K"))

    # Rec.2020 hue sweep
    for h_start in range(0, 360, 15):
        h_end = (h_start + 30) % 360
        r1, g1, b1 = _hsv_to_rgb(h_start / 360, 1.0, 1.0)
        r2, g2, b2 = _hsv_to_rgb(h_end / 360, 1.0, 1.0)
        x1 = M_R2020 @ srgb_to_linear(torch.tensor([r1, g1, b1], device=device, dtype=torch.float64))
        x2 = M_R2020 @ srgb_to_linear(torch.tensor([r2, g2, b2], device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x2]))
        labels.append(("rec2020_hue_sweep", f"R2020_h{h_start}-h{h_end}"))

    # Rec.2020 near-achromatic
    for h in range(0, 360, 15):
        for s in [0.05, 0.10, 0.15]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, s, 0.5)
            r2, g2, b2 = _hsv_to_rgb(((h + 30) % 360) / 360, s, 0.5)
            x1 = M_R2020 @ srgb_to_linear(torch.tensor([r1, g1, b1], device=device, dtype=torch.float64))
            x2 = M_R2020 @ srgb_to_linear(torch.tensor([r2, g2, b2], device=device, dtype=torch.float64))
            pairs.append(torch.stack([x1, x2]))
            labels.append(("rec2020_near_achromatic", f"R2020_na_h{h}_s{s:.2f}"))

    # Rec.2020 boundary stress
    for h in range(0, 360, 15):
        for v in [0.3, 0.5, 0.7, 0.9]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, 0.95, v)
            r2, g2, b2 = _hsv_to_rgb(h / 360, 0.95, min(v + 0.2, 1.0))
            x1 = M_R2020 @ srgb_to_linear(torch.tensor([r1, g1, b1], device=device, dtype=torch.float64))
            x2 = M_R2020 @ srgb_to_linear(torch.tensor([r2, g2, b2], device=device, dtype=torch.float64))
            pairs.append(torch.stack([x1, x2]))
            labels.append(("rec2020_boundary", f"R2020_bnd_h{h}_v{v}"))

    # Rec.2020 lightness sweep
    for h in [0, 60, 120, 180, 240, 300]:
        for v_lo, v_hi in [(0.1, 0.5), (0.5, 1.0), (0.2, 0.8)]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, 0.8, v_lo)
            r2, g2, b2 = _hsv_to_rgb(h / 360, 0.8, v_hi)
            x1 = M_R2020 @ srgb_to_linear(torch.tensor([r1, g1, b1], device=device, dtype=torch.float64))
            x2 = M_R2020 @ srgb_to_linear(torch.tensor([r2, g2, b2], device=device, dtype=torch.float64))
            pairs.append(torch.stack([x1, x2]))
            labels.append(("rec2020_lightness", f"R2020_L_h{h}_v{v_lo}-{v_hi}"))

    # Random Rec.2020 (500)
    for k in range(500):
        rgb1 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        rgb2 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        x1 = M_R2020 @ srgb_to_linear(rgb1)
        x2 = M_R2020 @ srgb_to_linear(rgb2)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("random_rec2020", f"rnd_r2020_{k}"))

    return torch.stack(pairs), labels  # (N, 2, 3), list[N]


def _hsv_to_rgb(h, s, v):
    """HSV [0-1] → RGB [0-1]."""
    if s == 0:
        return v, v, v
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]
