"""3-color (and 4-color) multi-stop gradient quality.

15 path patterns spanning RGB primaries, monochrome, pastel, dark variants.
For each path, interpolate 13 steps per segment in test space, quantize,
re-encode CIE Lab, measure CV of consecutive ΔE2000.
"""
import torch

from ._common import (
    _M_SRGB_LIST, _D65_LIST,
    matrix, vec, srgb_to_linear, linear_to_srgb, xyz_to_cielab,
)
from ..rulers import get_ruler as _get_ruler

_step_ruler = _get_ruler("spacing")  # uniformity -> spacing ruler (Perceptia-Spacing); notebook 2026-05-30


_PATHS = [
    ("R→G→B", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ("C→M→Y", [[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
    ("R→W→B", [[1, 0, 0], [1, 1, 1], [0, 0, 1]]),
    ("K→R→W", [[0, 0, 0], [1, 0, 0], [1, 1, 1]]),
    ("B→G→Y", [[0, 0, 1], [0, 1, 0], [1, 1, 0]]),
    ("K→G→W", [[0, 0, 0], [0, 1, 0], [1, 1, 1]]),
    ("K→B→W", [[0, 0, 0], [0, 0, 1], [1, 1, 1]]),
    ("R→Y→G", [[1, 0, 0], [1, 1, 0], [0, 1, 0]]),
    ("G→C→B", [[0, 1, 0], [0, 1, 1], [0, 0, 1]]),
    ("B→M→R", [[0, 0, 1], [1, 0, 1], [1, 0, 0]]),
    ("R→Y→G→B", [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]]),
    ("R→G→B→W", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]),
    ("pR→pG→pB", [[1, 0.7, 0.7], [0.7, 1, 0.7], [0.7, 0.7, 1]]),
    ("dR→dG→dB", [[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]]),
    ("K→g50→W", [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]]),
]


def measure_3color_gradients(space, device=None) -> dict:
    """CV per multi-stop path; 13 steps per segment, junction-deduplicated."""
    dev, dt = space.device, space.dtype
    ms = matrix(_M_SRGB_LIST, dev, dt)
    msi = torch.linalg.inv(ms)
    d65 = vec(_D65_LIST, dev, dt)

    results = {}
    for name, stops in _PATHS:
        stop_xyz = []
        for rgb in stops:
            t = torch.tensor(rgb, device=dev, dtype=dt)
            stop_xyz.append(srgb_to_linear(t) @ ms.T)
        stop_labs = [space.forward(x.unsqueeze(0))[0] for x in stop_xyz]

        all_labs = []
        for seg in range(len(stops) - 1):
            t = torch.linspace(0, 1, 13, device=dev, dtype=dt)
            seg_labs = stop_labs[seg].unsqueeze(0) + t.unsqueeze(1) * (
                stop_labs[seg + 1] - stop_labs[seg]).unsqueeze(0)
            all_labs.append(seg_labs if seg == 0 else seg_labs[1:])

        labs = torch.cat(all_labs, dim=0)
        xyz_path = space.inverse(labs)
        lin = (xyz_path @ msi.T).clamp(0, 1)
        s8 = (linear_to_srgb(lin) * 255).round() / 255.0
        xyz_q = srgb_to_linear(s8) @ ms.T
        cielab = xyz_to_cielab(xyz_q.clamp(min=1e-10), d65)

        de = _step_ruler(cielab[:-1], cielab[1:])
        md = de.mean()
        cv = (de.std() / md).item() if md > 0.001 else 0
        results[name] = {
            "cv": cv,
            "de_mean": md.item(),
            "de_max": de.max().item(),
            "n_steps": labs.shape[0],
        }
    return results
