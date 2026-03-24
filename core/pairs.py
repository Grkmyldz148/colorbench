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
