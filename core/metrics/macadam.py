"""MacAdam ellipse isotropy.

For 25 MacAdam color centers, perturb in 8 directions in CIE xy and measure
the ratio of max/min Lab distance. Ideal = 1.0 (isotropic). Reports the mean
and max ratio across all valid centers.
"""
import math
import torch

from ..constants import MACADAM_CENTERS

PI = math.pi


def measure_macadam_isotropy(space, device=None) -> dict:
    """Mean anisotropy ratio at 25 MacAdam centers."""
    dev, dt = space.device, space.dtype
    eps = 0.002
    n_dirs = 8
    Y_val = 0.5

    ratios = []
    for cx, cy in MACADAM_CENTERS:
        if cy < 1e-10:
            continue
        X = cx / cy * Y_val
        Z = (1 - cx - cy) / cy * Y_val
        xyz_c = torch.tensor([X, Y_val, Z], device=dev, dtype=dt)
        lab_c = space.forward(xyz_c.unsqueeze(0))[0]

        dists = []
        for k in range(n_dirs):
            angle = k * 2 * PI / n_dirs
            dx = eps * math.cos(angle)
            dy = eps * math.sin(angle)
            nx, ny = cx + dx, cy + dy
            if ny < 1e-10:
                continue
            nX = nx / ny * Y_val
            nZ = (1 - nx - ny) / ny * Y_val
            xyz_p = torch.tensor([nX, Y_val, nZ], device=dev, dtype=dt)
            lab_p = space.forward(xyz_p.unsqueeze(0))[0]
            dists.append((lab_p - lab_c).pow(2).sum().sqrt().item())

        if len(dists) >= 4:
            ratios.append(max(dists) / (min(dists) + 1e-30))

    return {
        "mean_ratio": sum(ratios) / len(ratios) if ratios else 0,
        "max_ratio": max(ratios) if ratios else 0,
        "n_centers": len(ratios),
    }
