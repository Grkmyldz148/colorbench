"""Double round-trip — error accumulation under repeated forward∘inverse.

10000 random sRGB samples. Run forward∘inverse N ∈ {1, 10, 100, 1000} times.
Reports max + mean accumulated error vs original XYZ. A bit-exact bijective
space holds error at machine epsilon even at 1000 trips.
"""
import torch

from ._common import _M_SRGB_LIST, matrix, srgb_to_linear


def measure_double_roundtrip(space, device=None) -> dict:
    """Forward∘inverse repeated N times; error accumulation report."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)

    gen = torch.Generator(device=dev).manual_seed(33)
    srgb = torch.rand(10000, 3, generator=gen, device=dev, dtype=dt)
    xyz_orig = srgb_to_linear(srgb) @ ms.T

    results = {}
    for n_trips in [1, 10, 100, 1000]:
        xyz_test = xyz_orig.clone()
        for _ in range(n_trips):
            xyz_test = space.inverse(space.forward(xyz_test))
        err = (xyz_orig - xyz_test).abs()
        results[f"trips_{n_trips}"] = {
            "max_error": err.max().item(),
            "mean_error": err.mean().item(),
        }
    return results
