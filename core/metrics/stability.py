"""Numerical stability — perturbation sensitivity + near-boundary behavior.

Three probes:
  - perturbation_1e8 : forward(x + ε) vs forward(x), max/mean Lab change
                       at ε = 1e-8 XYZ noise. 5000 random sRGB samples.
  - near_black       : NaN/Inf in forward(sRGB ∈ [0, 0.01]³)
  - near_white       : NaN/Inf in forward(sRGB ∈ [0.99, 1]³)
"""
import torch

from ._common import _M_SRGB_LIST, matrix, srgb_to_linear


def measure_stability(space, device=None) -> dict:
    """Perturbation sensitivity + NaN/Inf at gamut boundaries."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)

    gen = torch.Generator(device=dev).manual_seed(99)

    # Perturbation: 1e-8 XYZ noise on 5k random sRGB samples.
    # Use generator-based randn (legacy used randn_like which is global RNG and
    # thus non-deterministic — this caused max_lab_change to drift across runs).
    srgb = torch.rand(5000, 3, generator=gen, device=dev, dtype=dt)
    xyz = srgb_to_linear(srgb) @ ms.T
    lab = space.forward(xyz)
    perturb = torch.randn(xyz.shape, generator=gen, device=dev, dtype=dt) * 1e-8
    lab2 = space.forward(xyz + perturb)
    diff = (lab - lab2).abs()

    # Near-black (sRGB < 0.01)
    dark = torch.rand(1000, 3, generator=gen, device=dev, dtype=dt) * 0.01
    dark_lab = space.forward(srgb_to_linear(dark) @ ms.T)
    dark_nan = dark_lab.isnan().sum().item()
    dark_inf = dark_lab.isinf().sum().item()

    # Near-white (sRGB > 0.99)
    bright = 0.99 + torch.rand(1000, 3, generator=gen, device=dev, dtype=dt) * 0.01
    bright_lab = space.forward(srgb_to_linear(bright) @ ms.T)
    bright_nan = bright_lab.isnan().sum().item()
    bright_inf = bright_lab.isinf().sum().item()

    return {
        "perturbation_1e8": {
            "max_lab_change": diff.max().item(),
            "mean_lab_change": diff.mean().item(),
        },
        "near_black": {"nan": int(dark_nan), "inf": int(dark_inf)},
        "near_white": {"nan": int(bright_nan), "inf": int(bright_inf)},
    }
