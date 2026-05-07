"""8-bit quantization symmetry — sRGB→Lab→sRGB exactness.

Three probes:
  - 256 grays: how many round-trip exactly to the same 8-bit value?
  - 216 web-safe colors: same check
  - 10000 random 8-bit triplets: max channel error + exact-match count
"""
import torch

from ._common import _M_SRGB_LIST, matrix, srgb_to_linear, linear_to_srgb


def measure_quantization_symmetry(space, device=None) -> dict:
    """sRGB 8-bit round-trip exactness across grays, web-safe, random."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)

    # 256 grays
    g = torch.arange(256, device=dev, dtype=dt) / 255.0
    gray_srgb = g.unsqueeze(1).expand(256, 3)
    gray_xyz = srgb_to_linear(gray_srgb) @ ms.T
    gray_xyz_rt = space.inverse(space.forward(gray_xyz))
    gray_8bit_rt = (linear_to_srgb((gray_xyz_rt @ msi.T).clamp(0, 1)) * 255).round()
    gray_8bit_orig = (gray_srgb * 255).round()
    gray_exact = (gray_8bit_rt == gray_8bit_orig).all(dim=1).sum().item()

    # 216 web-safe (6³)
    vals_ws = torch.tensor([0, 51, 102, 153, 204, 255], device=dev, dtype=dt) / 255.0
    ws = torch.stack(torch.meshgrid(vals_ws, vals_ws, vals_ws, indexing='ij'),
                     dim=-1).reshape(-1, 3)
    ws_xyz_rt = space.inverse(space.forward(srgb_to_linear(ws) @ ms.T))
    ws_8bit_rt = (linear_to_srgb((ws_xyz_rt @ msi.T).clamp(0, 1)) * 255).round()
    ws_exact = (ws_8bit_rt == (ws * 255).round()).all(dim=1).sum().item()

    # 10000 random 8-bit triplets
    gen = torch.Generator(device=dev).manual_seed(55)
    rnd_8bit = torch.randint(0, 256, (10000, 3), generator=gen, device=dev).to(dt)
    rnd_xyz = srgb_to_linear(rnd_8bit / 255.0) @ ms.T
    rnd_xyz_rt = space.inverse(space.forward(rnd_xyz))
    rnd_8bit_rt = (linear_to_srgb((rnd_xyz_rt @ msi.T).clamp(0, 1)) * 255).round()
    channel_err = (rnd_8bit - rnd_8bit_rt).abs()
    exact_10k = (channel_err == 0).all(dim=1).sum().item()

    return {
        "grays_exact": f"{gray_exact}/256",
        "grays_exact_count": gray_exact,
        "websafe_exact": f"{ws_exact}/216",
        "websafe_exact_count": ws_exact,
        "random_10k_exact": f"{exact_10k}/10000",
        "random_10k_exact_count": exact_10k,
        "max_channel_error": int(channel_err.max().item()),
        "mean_channel_error": channel_err.float().mean().item(),
    }
