"""Space-agnostic gamut mapping via LCh chroma reduction.

Given a `ColorSpace` (any class with `.forward(xyz) -> Lab` and `.inverse(lab) -> xyz`),
reduce chroma in that space's polar (LCh) form until the color fits inside a target
RGB gamut. The target gamut is parameterized by its 3×3 XYZ→linear-RGB matrix.

This is the same shape as the CSS Color 4 reference algorithm (binary search in OKLCh),
just generalized so we can compare which perceptual space gives the best chroma-reduction
trajectory — i.e. which space hides hue drift, lightness drift, and gradient kinks best.
"""

import torch


def _in_gamut_mask(xyz, ms_inv, eps=1e-6):
    """True where XYZ projects to linear RGB inside [0,1] under the target gamut."""
    rgb = xyz @ ms_inv.T
    return ((rgb >= -eps) & (rgb <= 1.0 + eps)).all(dim=-1)


def gamut_map(xyz_src, space, ms_inv, n_iter=20, eps=1e-6):
    """Binary chroma reduction in `space`'s LCh.

    Parameters
    ----------
    xyz_src : (N,3) source XYZ (D65)
    space   : object exposing .forward(xyz) -> Lab and .inverse(lab) -> xyz, both (N,3)
    ms_inv  : (3,3) XYZ → linear-RGB matrix for the TARGET gamut (e.g. sRGB)
    n_iter  : binary-search iterations (20 ≈ 1e-6 chroma resolution)

    Returns
    -------
    xyz_out : (N,3) gamut-mapped XYZ (in target gamut up to numerical residual)
    """
    in_gamut = _in_gamut_mask(xyz_src, ms_inv, eps=eps)
    lab = space.forward(xyz_src)
    L = lab[:, 0]
    a, b = lab[:, 1], lab[:, 2]
    C = (a * a + b * b).sqrt().clamp(min=1e-12)
    cos_h = a / C
    sin_h = b / C

    c_lo = torch.zeros_like(C)  # known in-gamut
    c_hi = C.clone()             # current upper bound (start at original chroma)

    for _ in range(n_iter):
        c_mid = (c_lo + c_hi) * 0.5
        lab_mid = torch.stack([L, c_mid * cos_h, c_mid * sin_h], dim=-1)
        xyz_mid = space.inverse(lab_mid)
        mid_in = _in_gamut_mask(xyz_mid, ms_inv, eps=eps)
        c_lo = torch.where(mid_in, c_mid, c_lo)
        c_hi = torch.where(mid_in, c_hi, c_mid)

    # Reconstruct at the largest known-in-gamut chroma
    lab_out = torch.stack([L, c_lo * cos_h, c_lo * sin_h], dim=-1)
    xyz_out = space.inverse(lab_out)

    # Numerical residual: clamp linear RGB to [0,1] and project back.
    ms = torch.linalg.inv(ms_inv)
    rgb_out = (xyz_out @ ms_inv.T).clamp(0.0, 1.0)
    xyz_out = rgb_out @ ms.T

    # If source was already in gamut, leave it untouched.
    return torch.where(in_gamut.unsqueeze(-1), xyz_src, xyz_out)


def gamut_map_clip(xyz_src, ms_inv):
    """Naive baseline: clamp linear-target RGB to [0,1]. No perceptual awareness."""
    ms = torch.linalg.inv(ms_inv)
    rgb = (xyz_src @ ms_inv.T).clamp(0.0, 1.0)
    return rgb @ ms.T
