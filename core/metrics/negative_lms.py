"""Negative LMS detection — cone-fundamental sanity check.

For 10000 random sRGB colors, compute LMS (post-M1, pre-transfer) using the
space's exposed M1/M1_mod matrix. Count colors with any negative LMS channel.
A perfectly-conditioned space has zero negative LMS.

Spaces without an M1 matrix (e.g. CIELab) report 0 with a "not applicable" note.
"""
import torch

from ._common import _M_SRGB_LIST, matrix, srgb_to_linear


def measure_negative_lms(space, device=None) -> dict:
    """Count of 10k random sRGB colors with any LMS channel < 0."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)

    gen = torch.Generator(device=dev).manual_seed(42)
    srgb = torch.rand(10000, 3, generator=gen, device=dev, dtype=dt)
    xyz = srgb_to_linear(srgb) @ ms.T

    # Locate space's M1 by attribute name (HelmCT uses _M1_mod, OKLab uses M1)
    M1 = None
    for attr in ("_M1_mod", "_M1", "M1"):
        if hasattr(space, attr):
            M1 = getattr(space, attr)
            break
    if M1 is None:
        return {
            "n_negative": 0,
            "max_negative": 0.0,
            "pct_negative": 0.0,
            "note": "Space has no M1 matrix (LMS check not applicable)",
        }

    lms = xyz @ M1.T
    neg_mask = lms < 0
    n_with_neg = int(neg_mask.any(dim=1).sum().item())
    max_neg = (-lms[neg_mask]).max().item() if neg_mask.any() else 0.0

    per_channel = {}
    for ch, ch_name in enumerate(["L", "M", "S"]):
        per_channel[ch_name] = {
            "n_negative": int((lms[:, ch] < 0).sum().item()),
            "min_value": lms[:, ch].min().item(),
        }

    return {
        "n_negative": n_with_neg,
        "max_negative": max_neg,
        "pct_negative": n_with_neg / 10000 * 100,
        "per_channel": per_channel,
    }
