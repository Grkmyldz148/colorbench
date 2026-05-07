"""Extreme chroma stability — perturbation sensitivity at gamut corners.

For P3 and Rec.2020 primaries+secondaries, compute Lab, perturb ±0.001 in each
of L/a/b independently (6 directions), inverse, measure XYZ amplification
factor. NaN/Inf indicate numerical breakdown at extreme chroma; high
amplification indicates the space is ill-conditioned in that region.
"""
import torch

from ._common import _M_P3_LIST, _M_REC2020_LIST, matrix


def measure_extreme_chroma_stability(space, device=None) -> dict:
    """Inverse stability + amplification at P3/Rec.2020 primary corners."""
    dev, dt = space.device, space.dtype
    mp3 = matrix(_M_P3_LIST, dev, dt)
    mr2020 = matrix(_M_REC2020_LIST, dev, dt)

    test_colors = {}
    for gname, gmat in [("P3", mp3), ("Rec2020", mr2020)]:
        primaries = torch.eye(3, device=dev, dtype=dt)
        for i, pname in enumerate(["R", "G", "B"]):
            test_colors[f"{gname}_{pname}"] = (primaries[i] @ gmat.T).unsqueeze(0)
        secondaries = torch.tensor([
            [1, 1, 0], [0, 1, 1], [1, 0, 1],
        ], device=dev, dtype=dt)
        for i, sname in enumerate(["Y", "C", "M"]):
            test_colors[f"{gname}_{sname}"] = (secondaries[i] @ gmat.T).unsqueeze(0)

    eps = 0.001
    max_amplification = 0.0
    nan_count = 0
    inf_count = 0
    per_color = {}

    for name, xyz_orig in test_colors.items():
        lab = space.forward(xyz_orig)
        lab_val = lab[0]

        perturbations = torch.zeros(6, 3, device=dev, dtype=dt)
        perturbations[0, 0] = eps
        perturbations[1, 0] = -eps
        perturbations[2, 1] = eps
        perturbations[3, 1] = -eps
        perturbations[4, 2] = eps
        perturbations[5, 2] = -eps

        lab_perturbed = lab_val.unsqueeze(0) + perturbations
        xyz_perturbed = space.inverse(lab_perturbed)

        n_nan = int(xyz_perturbed.isnan().sum().item())
        n_inf = int(xyz_perturbed.isinf().sum().item())
        nan_count += n_nan
        inf_count += n_inf

        if n_nan > 0 or n_inf > 0:
            per_color[name] = {
                "amplification": float("inf"),
                "nan": n_nan,
                "inf": n_inf,
            }
            continue

        xyz_diff = (xyz_perturbed - xyz_orig).norm(dim=1)
        lab_diff = perturbations.norm(dim=1)
        amp = xyz_diff / lab_diff.clamp(min=1e-15)
        max_amp = amp.max().item()
        max_amplification = max(max_amplification, max_amp)
        per_color[name] = {
            "amplification": max_amp,
            "max_xyz_diff": xyz_diff.max().item(),
        }

    return {
        "max_amplification": max_amplification,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "per_color": per_color,
    }
