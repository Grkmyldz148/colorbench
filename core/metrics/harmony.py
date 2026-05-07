"""Color harmony + hue agreement metrics.

measure_harmony_accuracy
------------------------
For 12 base hues, perform 3 rotations (180° complementary, 120° triadic, 30°
analogous) in the test space. Compare actual CIE Lab rotation to intended.

measure_hue_agreement
---------------------
Mean absolute hue difference between test space and CIE Lab over 36 hues
sampled from HSV (s=0.8, v=0.8). CIE Lab self-comparison = 0.
"""
import math
import torch

from ._common import (
    _M_SRGB_LIST, _D65_LIST,
    matrix, vec, srgb_to_linear, xyz_to_cielab,
)

PI = math.pi


def _hsv_to_rgb_scalar(h, s, v):
    if s == 0:
        return v, v, v
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]


def measure_harmony_accuracy(space, device=None) -> dict:
    """Hue rotation accuracy: 12 hues × 3 rotations vs CIE Lab ground truth."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    d65 = vec(_D65_LIST, dev, dt)

    errors = []
    for base_h_deg in range(0, 360, 30):
        r, g, b = _hsv_to_rgb_scalar(base_h_deg / 360, 0.8, 0.8)
        rgb = torch.tensor([r, g, b], device=dev, dtype=dt)
        xyz_base = ms @ srgb_to_linear(rgb)
        lab_base = space.forward(xyz_base.unsqueeze(0))[0]
        C_base = (lab_base[1] ** 2 + lab_base[2] ** 2).sqrt()
        h_base = torch.atan2(lab_base[2], lab_base[1])

        cielab_base = xyz_to_cielab(xyz_base.unsqueeze(0), d65)[0]
        h_cielab_base = torch.atan2(cielab_base[2], cielab_base[1])

        for rot_deg in [180, 120, 30]:
            rot_rad = rot_deg * PI / 180
            h_new = h_base + rot_rad
            lab_new = torch.tensor(
                [lab_base[0].item(),
                 (C_base * torch.cos(h_new)).item(),
                 (C_base * torch.sin(h_new)).item()],
                device=dev, dtype=dt,
            )
            xyz_new = space.inverse(lab_new.unsqueeze(0))[0]
            cielab_new = xyz_to_cielab(xyz_new.unsqueeze(0), d65)[0]
            h_cielab_new = torch.atan2(cielab_new[2], cielab_new[1])

            actual_rot = torch.atan2(
                torch.sin(h_cielab_new - h_cielab_base),
                torch.cos(h_cielab_new - h_cielab_base),
            )
            actual_deg = actual_rot.item() * 180 / PI
            error = abs(actual_deg - rot_deg)
            if error > 180:
                error = 360 - error
            errors.append(error)

    return {
        "mean_error_deg": sum(errors) / len(errors) if errors else 0,
        "max_error_deg": max(errors) if errors else 0,
    }


def measure_hue_agreement(space, device=None) -> dict:
    """Mean absolute hue difference vs CIE Lab over 36 hues."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    d65 = vec(_D65_LIST, dev, dt)

    xyzs = []
    for h_deg in range(0, 360, 10):
        r, g, b = _hsv_to_rgb_scalar(h_deg / 360, 0.8, 0.8)
        rgb = torch.tensor([r, g, b], device=dev, dtype=dt)
        xyzs.append(ms @ srgb_to_linear(rgb))
    xyzs = torch.stack(xyzs)

    lab = space.forward(xyzs)
    h_space = torch.atan2(lab[:, 2], lab[:, 1])

    cielab = xyz_to_cielab(xyzs, d65)
    h_ref = torch.atan2(cielab[:, 2], cielab[:, 1])

    dh = torch.atan2(torch.sin(h_space - h_ref), torch.cos(h_space - h_ref))
    return {
        "mae_deg": (dh.abs() * 180 / PI).mean().item(),
        "max_diff_deg": (dh.abs() * 180 / PI).max().item(),
    }
