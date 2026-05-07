"""Photo gamut mapping fidelity (P3 → sRGB).

Generate 500 random P3 colors. For those out of sRGB gamut, perform 50-step
binary-search chroma reduction in the test space. Measure CIE Lab hue shift
between the original and the mapped color.
"""
import math
import torch

from ._common import (
    _M_SRGB_LIST, _M_P3_LIST, _D65_LIST,
    matrix, vec, srgb_to_linear, xyz_to_cielab,
)

PI = math.pi


def measure_photo_gamut_map(space, device=None) -> dict:
    """Hue shift during chroma-reduction P3 → sRGB."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    mp3 = matrix(_M_P3_LIST, dev, dt)
    d65 = vec(_D65_LIST, dev, dt)

    gen = torch.Generator(device=dev).manual_seed(99)
    p3_rgb = torch.rand(500, 3, generator=gen, device=dev, dtype=dt)
    p3_xyz = srgb_to_linear(p3_rgb) @ mp3.T

    srgb_lin = p3_xyz @ msi.T
    out_of_gamut = (srgb_lin < -0.001).any(dim=1) | (srgb_lin > 1.001).any(dim=1)
    if out_of_gamut.sum() == 0:
        return {"mean_hue_shift_deg": 0.0, "n_mapped": 0}

    xyz_oog = p3_xyz[out_of_gamut]
    cielab_orig = xyz_to_cielab(xyz_oog, d65)
    h_orig = torch.atan2(cielab_orig[:, 2], cielab_orig[:, 1])

    # Bisect chroma in test space's Lab
    lab_oog = space.forward(xyz_oog)
    L_oog = lab_oog[:, 0:1]
    a_oog = lab_oog[:, 1:2]
    b_oog = lab_oog[:, 2:3]
    C_oog = (a_oog ** 2 + b_oog ** 2 + 1e-30).sqrt()
    h_space = torch.atan2(b_oog, a_oog)

    lo = torch.zeros_like(C_oog)
    hi = torch.ones_like(C_oog)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        C_test = C_oog * mid
        lab_test = torch.cat(
            [L_oog, C_test * torch.cos(h_space), C_test * torch.sin(h_space)], dim=1
        )
        xyz_test = space.inverse(lab_test)
        linear_rgb = xyz_test @ msi.T
        in_gamut = (
            (linear_rgb >= -0.001).all(dim=1, keepdim=True)
            & (linear_rgb <= 1.001).all(dim=1, keepdim=True)
        )
        lo = torch.where(in_gamut, mid, lo)
        hi = torch.where(in_gamut, hi, mid)

    C_mapped = C_oog * lo
    lab_mapped = torch.cat(
        [L_oog, C_mapped * torch.cos(h_space), C_mapped * torch.sin(h_space)], dim=1
    )
    xyz_mapped = space.inverse(lab_mapped)
    cielab_mapped = xyz_to_cielab(xyz_mapped, d65)
    h_mapped = torch.atan2(cielab_mapped[:, 2], cielab_mapped[:, 1])

    dh = torch.atan2(torch.sin(h_mapped - h_orig), torch.cos(h_mapped - h_orig))
    hue_shifts = dh.abs() * 180 / PI

    return {
        "mean_hue_shift_deg": hue_shifts.mean().item(),
        "max_hue_shift_deg": hue_shifts.max().item(),
        "n_mapped": int(out_of_gamut.sum().item()),
    }
