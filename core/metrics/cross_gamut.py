"""sRGB ⊂ P3 cross-gamut consistency.

For 1000 random sRGB colors, compute Lab via two paths:
  1. XYZ → Lab directly
  2. XYZ → P3 linear → P3 sRGB encode → P3 linear back → XYZ → Lab

Measure whether the space amplifies P3 round-trip quantization noise.
"""
import torch

from ._common import _M_SRGB_LIST, _M_P3_LIST, matrix, srgb_to_linear, linear_to_srgb


def measure_cross_gamut_consistency(space, device=None) -> dict:
    """Lab agreement when Same XYZ takes sRGB-direct vs P3-roundabout path."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    mp3 = matrix(_M_P3_LIST, dev, dt)
    mp3i = torch.linalg.inv(mp3)

    gen = torch.Generator(device="cpu").manual_seed(66)
    srgb = torch.rand(1000, 3, generator=gen, dtype=torch.float64).to(device=dev, dtype=dt)
    xyz = srgb_to_linear(srgb) @ ms.T

    lab1 = space.forward(xyz)

    p3_lin = xyz @ mp3i.T
    p3_srgb = linear_to_srgb(p3_lin.clamp(0, 1))
    p3_lin_back = srgb_to_linear(p3_srgb)
    xyz_via_p3 = p3_lin_back @ mp3.T
    lab2 = space.forward(xyz_via_p3)

    xyz_diff = (xyz - xyz_via_p3).abs()
    lab_diff = (lab1 - lab2).abs()

    xyz_norm = xyz_diff.norm(dim=1).clamp(min=1e-15)
    lab_norm = lab_diff.norm(dim=1).clamp(min=1e-15)
    ratio = lab_norm / xyz_norm

    return {
        "max_lab_diff": lab_diff.max().item(),
        "mean_lab_diff": lab_diff.mean().item(),
        "amplification_mean": ratio.mean().item(),
        "amplification_max": ratio.max().item(),
    }
