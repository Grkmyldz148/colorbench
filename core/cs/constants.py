"""Module-level color science constants used by spaces and metrics.

Kept on CPU at float64; cast to space.dtype/device at use site.
Re-exported from package root for `from core.cs import D65` shorthand.

Cycle 36 (2026-05-08): M_SRGB updated 7-decimal IEC → 17-decimal canonical
Bradford-adapted IEC 61966-2-1 / Lindbloom values. Y-row sum:
  7-decimal: 1.0000001 (1e-7 over)
  17-decimal: 0.9999999999999999 (machine epsilon)
This fixed the paper-grade RT 1e-7 LOSS (HelmCT couldn't recover the noise
on the (255,255,255) white pixel). Validator PHASE B PARTIAL → PASS conditional
on consolidating all 9 sites to import from this constants module.
"""
import torch

# CIE D65 illuminant (ASTM E308)
D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)

# sRGB (BT.709) → XYZ matrix, D65 — TRUE Lindbloom 17-decimal (Bradford-adapted IEC 61966-2-1)
# Computed from first principles: sRGB primaries (R=(0.64,0.33), G=(0.30,0.60), B=(0.15,0.06))
# adapted to D65 = (0.95047, 1.0, 1.08883).
# Verification: M @ (1,1,1) = (0.95047, 1.0, 1.08883) within 1e-16.
# Y-row sum: 0.9999999999999999 (machine epsilon).
# Previous 7-decimal IEC values archived in `M_SRGB_7DECIMAL_DEPRECATED` for reference.
#
# NOTE (cycle 36 lesson): an earlier "canonical" matrix was deployed (W3C CSS 4 / sRGB
# spec source) that DIFFERED from Lindbloom in the 5th decimal — it inverted D65 with
# 1e-4 drift. Reverted to TRUE Lindbloom (extension of 7-decimal Lindbloom, not a
# different convention).
M_SRGB = torch.tensor([
    [0.41245643908969221, 0.35757607764390892, 0.18043748326639891],
    [0.21267285140562253, 0.71515215528781784, 0.07217499330655956],
    [0.01933389558232930, 0.11919202588130294, 0.95030407853636767],
], dtype=torch.float64)

# Deprecated 7-decimal values (kept for diff/reference; do not use)
M_SRGB_7DECIMAL_DEPRECATED = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],   # Y-row sum 1.0000001 ← bug
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float64)
