"""Hue properties + special gradient endpoints.

measure_hue
-----------
Maps the 6 sRGB primaries (R, Y, G, C, B, M) into the test space's Lab.
Reports per-primary (L, C, hue, expected, error), RMS error, lightness range,
and ordering check (hues should monotonically increase mod 360).

measure_special_gradients
-------------------------
Three classic visual fidelity probes:
  - Blue→White midpoint G/R (high = blue stays blue, low = lavender)
  - Red→White midpoint G−B (deviation from neutral red ramp)
  - Yellow chroma at full saturation
"""
import math
import torch

from ._common import _M_SRGB_LIST, matrix, srgb_to_linear, linear_to_srgb

PI = math.pi


_PRIMARY_NAMES = ["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"]
_PRIMARY_RGB = [
    [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 1, 1], [0, 0, 1], [1, 0, 1],
]
_PRIMARY_HUE = [0, 60, 120, 180, 240, 300]


def measure_hue(space, device=None) -> dict:
    """Primary hue ordering, RMS deviation, lightness spread."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    primaries = torch.tensor(_PRIMARY_RGB, device=dev, dtype=dt)
    expected = torch.tensor(_PRIMARY_HUE, device=dev, dtype=dt)

    xyz = srgb_to_linear(primaries) @ ms.T
    lab = space.forward(xyz)

    h = torch.atan2(lab[:, 2], lab[:, 1]) * (180.0 / PI) % 360
    dh = h - expected
    dh = torch.where(dh > 180, dh - 360, dh)
    dh = torch.where(dh < -180, dh + 360, dh)
    rms = (dh ** 2).mean().sqrt().item()

    per_primary = {}
    for i, name in enumerate(_PRIMARY_NAMES):
        per_primary[name] = {
            "hue": h[i].item(),
            "expected": expected[i].item(),
            "error": dh[i].item(),
            "L": lab[i, 0].item(),
            "C": (lab[i, 1] ** 2 + lab[i, 2] ** 2).sqrt().item(),
        }

    # Hue ordering R→Y→G→C→B→M with proper 360° wrap
    h_list = h.tolist()
    hue_ordered = True
    for i in range(5):
        diff = h_list[i + 1] - h_list[i]
        if diff < -180: diff += 360
        if diff > 180:  diff -= 360
        if diff <= 0:
            hue_ordered = False
            break

    L_vals = lab[:, 0]
    return {
        "hue_rms": rms,
        "hue_ordered": hue_ordered,
        "primary_L_range": (L_vals.max() - L_vals.min()).item(),
        "per_primary": per_primary,
    }


def _srgb_to_xyz_single(rgb_list, dev, dt, ms):
    """Single sRGB triplet → XYZ (3,)."""
    t = torch.tensor(rgb_list, device=dev, dtype=dt)
    return (srgb_to_linear(t) @ ms.T).unsqueeze(0)


def measure_special_gradients(space, device=None) -> dict:
    """Blue→White, Red→White midpoints + yellow chroma."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)

    bl = space.forward(_srgb_to_xyz_single([0, 0, 1], dev, dt, ms))[0]
    wl = space.forward(_srgb_to_xyz_single([1, 1, 1], dev, dt, ms))[0]
    rl = space.forward(_srgb_to_xyz_single([1, 0, 0], dev, dt, ms))[0]
    yl = space.forward(_srgb_to_xyz_single([1, 1, 0], dev, dt, ms))[0]

    mid_bw = space.inverse(((bl + wl) / 2).unsqueeze(0))[0]
    mid_bw_srgb = linear_to_srgb((mid_bw @ msi.T).clamp(0, 1))

    mid_rw = space.inverse(((rl + wl) / 2).unsqueeze(0))[0]
    mid_rw_srgb = linear_to_srgb((mid_rw @ msi.T).clamp(0, 1))

    return {
        "blue_white_midpoint": {
            "G_over_R": mid_bw_srgb[1].item() / max(mid_bw_srgb[0].item(), 1e-10),
            "srgb": [mid_bw_srgb[i].item() for i in range(3)],
        },
        "red_white_midpoint": {
            "G_minus_B": mid_rw_srgb[1].item() - mid_rw_srgb[2].item(),
            "srgb": [mid_rw_srgb[i].item() for i in range(3)],
        },
        "yellow_chroma": (yl[1] ** 2 + yl[2] ** 2).sqrt().item(),
    }
