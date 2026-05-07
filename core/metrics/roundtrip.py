"""Round-trip integrity metric.

Tests forward∘inverse identity across:
  - all 16,777,216 sRGB 8-bit colors
  - all 16,777,216 P3 8-bit colors
  - 2.1M Rec.2020 uniform grid (128³)
  - 50K Rec.2020 boundary stress samples
  - 360 sRGB gamut-edge samples (high saturation × 5 lightness levels)

Performance design
------------------
1. Sync deferral: per-chunk results accumulated as device-resident tensor
   scalars; single .item() at end of each result family. Eliminates the
   ~85 mid-loop CPU↔GPU stalls in the legacy implementation.
2. Chunk size 1M (~25 MB at float64) — minimizes Python loop overhead while
   keeping memory bounded. Bit-exact regardless of chunk size.
3. Boundary loop kept sequential (50k iter) to preserve random-draw order
   from existing snapshots — vectorizing would change RNG sequence.
"""
import torch

from ._common import (
    _M_SRGB_LIST, _M_P3_LIST, _M_REC2020_LIST,
    matrix, srgb_to_linear,
)

CHUNK = 1_000_000


def _build_8bit_xyz(start: int, end: int, mat, dtype, device):
    """Generate XYZ for 8-bit RGB values [start, end), via mat @ linearized RGB.T."""
    n = end - start
    idx = torch.arange(start, end, device=device, dtype=torch.int64)
    r = (idx // 65536).to(dtype) / 255.0
    g = ((idx % 65536) // 256).to(dtype) / 255.0
    b = (idx % 256).to(dtype) / 255.0
    rgb = torch.stack([r, g, b], dim=1)
    return srgb_to_linear(rgb) @ mat.T


def _grid_test(space, mat) -> dict:
    """Per-chunk forward+inverse over full 8-bit cube. Sync deferred to end."""
    chunk_max_errs, chunk_nans, chunk_infs = [], [], []
    total = 256 ** 3
    for start in range(0, total, CHUNK):
        end = min(start + CHUNK, total)
        xyz = _build_8bit_xyz(start, end, mat, space.dtype, space.device)
        lab = space.forward(xyz)
        xyz_rt = space.inverse(lab)
        chunk_nans.append(lab.isnan().sum() + xyz_rt.isnan().sum())
        chunk_infs.append(lab.isinf().sum() + xyz_rt.isinf().sum())
        chunk_max_errs.append((xyz - xyz_rt).abs().max())
    return {
        "max_error": torch.stack(chunk_max_errs).max().item(),
        "nan_count": int(torch.stack(chunk_nans).sum().item()),
        "inf_count": int(torch.stack(chunk_infs).sum().item()),
    }


def _rec2020_uniform(space, mat_r2020) -> dict:
    """128³ Rec.2020 uniform grid round-trip (fits in single batch ~50MB)."""
    steps = 128
    vals = torch.linspace(0, 1, steps, device=space.device, dtype=space.dtype)
    rr = vals.view(steps, 1, 1).expand(steps, steps, steps).reshape(-1)
    gg = vals.view(1, steps, 1).expand(steps, steps, steps).reshape(-1)
    bb = vals.view(1, 1, steps).expand(steps, steps, steps).reshape(-1)
    rgb = torch.stack([rr, gg, bb], dim=1)
    xyz = rgb.pow(2.4) @ mat_r2020.T  # simplified Rec.2020 transfer
    rt = space.inverse(space.forward(xyz))
    return {"max_error": (xyz - rt).abs().max().item()}


def _rec2020_boundary(space, mat_r2020) -> dict:
    """50k near-primary samples. Loop kept sequential for snapshot stability:
    vectorized random-draw order differs from original, breaking bit-exactness.
    """
    gen = torch.Generator(device=space.device).manual_seed(42)
    rgb = torch.zeros(50_000, 3, device=space.device, dtype=space.dtype)
    for i in range(50_000):
        ch = i % 3
        rgb[i, ch] = 0.8 + torch.rand(1, generator=gen, device=space.device,
                                       dtype=space.dtype).item() * 0.2
        for j in range(3):
            if j != ch:
                rgb[i, j] = torch.rand(1, generator=gen, device=space.device,
                                        dtype=space.dtype).item() * 0.3
    xyz = rgb.pow(2.4) @ mat_r2020.T
    rt = space.inverse(space.forward(xyz))
    return {"max_error": (xyz - rt).abs().max().item()}


def _srgb_boundary_360(space, mat_srgb) -> dict:
    """360 high-saturation sRGB colors at 5 lightness levels. Vectorized HSV→RGB.

    ⚠ Algorithm preserved for snapshot stability: every 5° hue × {0.2..1.0} V
    at s=0.98, sequential builder. Small (360 iter); kept Python for clarity.
    """
    rows = []
    for h_i in range(72):
        h = h_i / 72
        for v in [0.2, 0.4, 0.6, 0.8, 1.0]:
            i = int(h * 6.0) % 6
            f = h * 6.0 - int(h * 6.0)
            p_ = v * 0.02
            q_ = v * (1.0 - 0.98 * f)
            t_ = v * (1.0 - 0.98 * (1.0 - f))
            rgb = [(v, t_, p_), (q_, v, p_), (p_, v, t_),
                   (p_, q_, v), (t_, p_, v), (v, p_, q_)][i]
            rows.append(rgb)
    bnd = torch.tensor(rows, device=space.device, dtype=space.dtype)
    xyz = srgb_to_linear(bnd) @ mat_srgb.T
    rt = space.inverse(space.forward(xyz))
    err = (xyz - rt).abs()
    return {
        "max_error": err.max().item(),
        "mean_error": err.mean().item(),
        "n_colors": 360,
    }


def measure_roundtrip(space, device=None) -> dict:
    """Full round-trip suite. Returns dict matching legacy schema.

    Args:
        space: ColorSpace instance
        device: legacy compatibility (ignored, space.device is authoritative)

    Returns:
        dict with keys: srgb_full_16M, p3_full_16M, rec2020_2M_uniform,
                       rec2020_50K_boundary, srgb_boundary_360
    """
    ms = matrix(_M_SRGB_LIST, space.device, space.dtype)
    mp3 = matrix(_M_P3_LIST, space.device, space.dtype)
    mr2020 = matrix(_M_REC2020_LIST, space.device, space.dtype)

    out = {}
    out["srgb_full_16M"] = _grid_test(space, ms)
    p3 = _grid_test(space, mp3)
    # Legacy schema combines NaN+Inf for P3
    out["p3_full_16M"] = {
        "max_error": p3["max_error"],
        "nan_inf_count": p3["nan_count"] + p3["inf_count"],
    }
    out["rec2020_2M_uniform"] = _rec2020_uniform(space, mr2020)
    out["rec2020_50K_boundary"] = _rec2020_boundary(space, mr2020)
    out["srgb_boundary_360"] = _srgb_boundary_360(space, ms)
    return out
