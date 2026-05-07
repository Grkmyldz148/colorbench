"""Near-primary hue discontinuity — singularity check at gamut corners.

For each sRGB and P3 primary/secondary, perturb ±0.01 in each RGB channel
and measure the hue jump in test space. Large jumps indicate hue-angle
singularities (typical at fully saturated primaries).
"""
import math
import torch

from ._common import _M_SRGB_LIST, _M_P3_LIST, matrix, srgb_to_linear

PI = math.pi


_SRGB_PRIMARIES = {
    "R": [1, 0, 0], "G": [0, 1, 0], "B": [0, 0, 1],
    "C": [0, 1, 1], "M": [1, 0, 1], "Y": [1, 1, 0],
}
_P3_PRIMARIES = {
    "P3_R": [1, 0, 0], "P3_G": [0, 1, 0], "P3_B": [0, 0, 1],
    "P3_C": [0, 1, 1], "P3_M": [1, 0, 1], "P3_Y": [1, 1, 0],
}


def measure_primary_hue_discontinuity(space, device=None) -> dict:
    """Max hue jump on ±0.01 perturbation around each primary."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    mp3 = matrix(_M_P3_LIST, dev, dt)
    delta = 0.01

    results = {}
    for gamut_name, primaries, gamut_mat in [
        ("sRGB", _SRGB_PRIMARIES, ms),
        ("P3", _P3_PRIMARIES, mp3),
    ]:
        for name, rgb in primaries.items():
            rgb_t = torch.tensor(rgb, device=dev, dtype=dt)
            xyz_center = (srgb_to_linear(rgb_t) @ gamut_mat.T).unsqueeze(0)
            lab_center = space.forward(xyz_center)[0]
            h_center = math.atan2(lab_center[2].item(), lab_center[1].item())

            max_jump = 0.0
            for ch in range(3):
                for sign in [-1, 1]:
                    perturbed = rgb_t.clone()
                    perturbed[ch] = (perturbed[ch] + sign * delta).clamp(0.0, 1.0)
                    if (perturbed == rgb_t).all():
                        continue
                    xyz_p = (srgb_to_linear(perturbed) @ gamut_mat.T).unsqueeze(0)
                    lab_p = space.forward(xyz_p)[0]
                    C_p = (lab_p[1] ** 2 + lab_p[2] ** 2).sqrt().item()
                    C_c = (lab_center[1] ** 2 + lab_center[2] ** 2).sqrt().item()
                    if C_p > 0.01 and C_c > 0.01:
                        h_p = math.atan2(lab_p[2].item(), lab_p[1].item())
                        dh = abs(h_p - h_center)
                        if dh > PI:
                            dh = 2 * PI - dh
                        max_jump = max(max_jump, dh * (180 / PI))

            results[name] = {
                "max_hue_jump_deg": max_jump,
                "lab": [lab_center[0].item(), lab_center[1].item(), lab_center[2].item()],
            }

    srgb_jumps = [v["max_hue_jump_deg"] for k, v in results.items() if not k.startswith("P3_")]
    p3_jumps = [v["max_hue_jump_deg"] for k, v in results.items() if k.startswith("P3_")]
    return {
        "per_primary": results,
        "srgb_max_jump": max(srgb_jumps) if srgb_jumps else 0.0,
        "srgb_mean_jump": sum(srgb_jumps) / len(srgb_jumps) if srgb_jumps else 0.0,
        "p3_max_jump": max(p3_jumps) if p3_jumps else 0.0,
        "p3_mean_jump": sum(p3_jumps) / len(p3_jumps) if p3_jumps else 0.0,
    }
