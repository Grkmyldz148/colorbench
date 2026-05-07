"""Channel monotonicity — RGB channel direction along Lab-interpolated gradients.

For each gradient (e.g. R→W), interpolate 256 steps in test space, decode to
sRGB and check whether each channel moves in the expected direction. A
violation = the channel decreased where it should have increased (or vice
versa) by more than 0.005.

12 gradients across sRGB and P3 gamuts.
"""
import torch

from ._common import (
    _M_SRGB_LIST, _M_P3_LIST,
    matrix, srgb_to_linear, linear_to_srgb,
)


_GRADIENTS = [
    ("R→W", [1, 0, 0], [1, 1, 1], [False, True, True], "srgb"),
    ("G→W", [0, 1, 0], [1, 1, 1], [True, False, True], "srgb"),
    ("B→W", [0, 0, 1], [1, 1, 1], [True, True, False], "srgb"),
    ("R→K", [1, 0, 0], [0, 0, 0], [False, False, False], "srgb"),
    ("K→W", [0, 0, 0], [1, 1, 1], [True, True, True], "srgb"),
    ("R→C", [1, 0, 0], [0, 1, 1], [False, True, True], "srgb"),
    ("Y→B", [1, 1, 0], [0, 0, 1], [False, False, True], "srgb"),
    ("P3_R→W", [1, 0, 0], [1, 1, 1], [False, True, True], "p3"),
    ("P3_G→W", [0, 1, 0], [1, 1, 1], [True, False, True], "p3"),
    ("P3_B→W", [0, 0, 1], [1, 1, 1], [True, True, False], "p3"),
    ("P3_K→W", [0, 0, 0], [1, 1, 1], [True, True, True], "p3"),
    ("P3_R→C", [1, 0, 0], [0, 1, 1], [False, True, True], "p3"),
]


def measure_channel_monotonicity(space, device=None) -> dict:
    """For 12 gradients: count of channel direction violations (>0.005 drift)."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    mp3 = matrix(_M_P3_LIST, dev, dt)
    gamut_mats = {
        "srgb": (ms, torch.linalg.inv(ms)),
        "p3": (mp3, torch.linalg.inv(mp3)),
    }

    results = {}
    for name, rgb1, rgb2, expected_increase, gamut_key in _GRADIENTS:
        gmat, ginv = gamut_mats[gamut_key]
        xyz1 = srgb_to_linear(torch.tensor(rgb1, device=dev, dtype=dt)) @ gmat.T
        xyz2 = srgb_to_linear(torch.tensor(rgb2, device=dev, dtype=dt)) @ gmat.T
        lab1 = space.forward(xyz1.unsqueeze(0))[0]
        lab2 = space.forward(xyz2.unsqueeze(0))[0]

        t = torch.linspace(0, 1, 256, device=dev, dtype=dt)
        labs = lab1.unsqueeze(0) + t.unsqueeze(1) * (lab2 - lab1).unsqueeze(0)
        xyz_path = space.inverse(labs)
        srgb_path = linear_to_srgb((xyz_path @ ginv.T).clamp(0, 1))

        violations = {}
        for ch, ch_name in enumerate(["R", "G", "B"]):
            diffs = srgb_path[1:, ch] - srgb_path[:-1, ch]
            if expected_increase[ch]:
                n_viol = (diffs < -0.005).sum().item()
            else:
                n_viol = (diffs > 0.005).sum().item()
            violations[ch_name] = int(n_viol)

        results[name] = {
            "violations": violations,
            "total_violations": sum(violations.values()),
        }
    return results
