"""Achromatic axis fidelity metric.

Tests whether grays remain at (a, b) = (0, 0) under forward mapping.

Two probe families:
  1. sRGB 8-bit gray ramp (257 levels) — includes sRGB matrix rounding noise
  2. D65-proportional pure grays (350 samples, log-spaced) — exact achromatic input

Plus the white and black endpoints individually.
"""
import torch

from ._common import _M_SRGB_LIST, _D65_LIST, matrix, vec, srgb_to_linear


def measure_achromatic(space, device=None) -> dict:
    """Achromatic preservation across gray ramps + white/black endpoints.

    Args:
        space: ColorSpace instance
        device: legacy compatibility (ignored)

    Returns: dict matching legacy schema with gray_ramp_srgb, gray_ramp_pure,
             white, black sub-dicts.
    """
    dev, dt = space.device, space.dtype
    d65 = vec(_D65_LIST, dev, dt)
    ms = matrix(_M_SRGB_LIST, dev, dt)

    # 257-level sRGB gray ramp (matrix rounding noise included)
    g = torch.linspace(0.0, 1.0, 257, device=dev, dtype=dt)
    gray_srgb = g.unsqueeze(1).expand(257, 3)
    gray_xyz_srgb = srgb_to_linear(gray_srgb) @ ms.T
    lab_srgb = space.forward(gray_xyz_srgb)
    chroma_srgb = (lab_srgb[:, 1] ** 2 + lab_srgb[:, 2] ** 2).sqrt()

    # 350 D65-proportional grays — log-distributed, captures HDR range too
    Y_pure = torch.cat([
        torch.linspace(0.0001, 0.01, 50, device=dev, dtype=dt),
        torch.linspace(0.01, 0.1, 50, device=dev, dtype=dt),
        torch.linspace(0.1, 1.0, 200, device=dev, dtype=dt),
        torch.linspace(1.0, 2.0, 50, device=dev, dtype=dt),
    ])
    pure_xyz = Y_pure.unsqueeze(1) * d65.unsqueeze(0)
    lab_pure = space.forward(pure_xyz)
    chroma_pure = (lab_pure[:, 1] ** 2 + lab_pure[:, 2] ** 2).sqrt()

    # White + Black endpoints (single sample each)
    white_lab = space.forward(d65.unsqueeze(0))[0]
    black_lab = space.forward(torch.zeros(1, 3, device=dev, dtype=dt))[0]

    return {
        "gray_ramp_srgb": {
            "max_chroma": chroma_srgb.max().item(),
            "mean_chroma": chroma_srgb.mean().item(),
            "max_a": lab_srgb[:, 1].abs().max().item(),
            "max_b": lab_srgb[:, 2].abs().max().item(),
            "note": "includes sRGB matrix rounding (~1e-7 XYZ offset)",
        },
        "gray_ramp_pure": {
            "max_chroma": chroma_pure.max().item(),
            "mean_chroma": chroma_pure.mean().item(),
            "max_a": lab_pure[:, 1].abs().max().item(),
            "max_b": lab_pure[:, 2].abs().max().item(),
            "note": "D65-proportional grays, no matrix rounding",
            "n_samples": int(Y_pure.shape[0]),
        },
        "white": {
            "L": white_lab[0].item(),
            "a": white_lab[1].item(),
            "b": white_lab[2].item(),
            "L_error": abs(white_lab[0].item() - 1.0),
        },
        "black": {
            "L": black_lab[0].item(),
            "a": black_lab[1].item(),
            "b": black_lab[2].item(),
        },
    }
