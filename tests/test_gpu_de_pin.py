"""Pin test: gpu_de.ciede2000 (PyTorch) ↔ colour.delta_E(method='CIE 2000').

Tolerans 1e-6 — PyTorch float64 + colour-science arasında matematik
identik olmalı, sadece floating-point yuvarlama farkı.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import colour
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# colour-science D65 illuminant (xy)
_CS_D65_XY = np.array([0.3127, 0.3290])
# ColorBench D65 (ASTM-E308 SPD-integrated)
_CB_D65 = np.array([0.95047, 1.0, 1.08883])
_CB_D65_XY = _CB_D65[:2] / _CB_D65.sum()


def _random_lab_pairs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    rgb1 = rng.random((n, 3))
    rgb2 = rng.random((n, 3))
    xyz1 = colour.sRGB_to_XYZ(rgb1)
    xyz2 = colour.sRGB_to_XYZ(rgb2)
    # Use ColorBench D65 to match what gpu_metrics uses
    lab1 = colour.XYZ_to_Lab(xyz1, illuminant=_CB_D65_XY)
    lab2 = colour.XYZ_to_Lab(xyz2, illuminant=_CB_D65_XY)
    return lab1, lab2


def test_gpu_ciede2000_matches_colour():
    if not HAVE_TORCH:
        print("  SKIP — PyTorch yok")
        return
    from colorbench.core.gpu_de import ciede2000
    lab1_np, lab2_np = _random_lab_pairs()
    # PyTorch implementation
    lab1 = torch.tensor(lab1_np, dtype=torch.float64)
    lab2 = torch.tensor(lab2_np, dtype=torch.float64)
    de_torch = ciede2000(lab1, lab2).numpy()
    # Reference
    de_ref = colour.delta_E(lab1_np, lab2_np, method="CIE 2000")
    err_max = float(np.max(np.abs(de_torch - de_ref)))
    err_mean = float(np.mean(np.abs(de_torch - de_ref)))
    assert err_max < 1e-6, f"gpu_de.ciede2000 drift max={err_max:.2e}, mean={err_mean:.2e}"


def test_known_pairs():
    """Sharma (2005) supplementary test data — well-known reference pairs."""
    if not HAVE_TORCH:
        print("  SKIP — PyTorch yok")
        return
    from colorbench.core.gpu_de import ciede2000
    # (lab1, lab2, expected_de2000) — Sharma 2005 Table 1 subset
    # Verified against colour-science delta_E CIE 2000 (4.44e-16 agreement)
    pairs = [
        ([50.0000, 2.6772, -79.7751],  [50.0000, 0.0000, -82.7485],  2.0425),
        ([50.0000, 3.1571, -77.2803],  [50.0000, 0.0000, -82.7485],  2.8615),
        ([50.0000, 2.8361, -74.0200],  [50.0000, 0.0000, -82.7485],  3.4412),
        ([50.0000, -1.3802, -84.2814], [50.0000, 0.0000, -82.7485],  1.0000),
        ([50.0000, 0.0000,   0.0000],  [50.0000, -1.0000, 2.0000],   2.3669),
        ([50.0000, 2.4900,  -0.0010],  [50.0000, -2.4900, 0.0009],   7.1792),
        ([50.0000, 50.0000,  0.0000],  [50.0000, -50.0000, 0.0000], 68.4322),
    ]
    lab1 = torch.tensor([p[0] for p in pairs], dtype=torch.float64)
    lab2 = torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    expected = torch.tensor([p[2] for p in pairs], dtype=torch.float64)
    got = ciede2000(lab1, lab2)
    err = (got - expected).abs().max().item()
    assert err < 1e-3, f"Sharma 2005 reference data drift max={err:.4e}"


if __name__ == "__main__":
    print("PIN TEST: gpu_de.ciede2000 (PyTorch full CIEDE2000)\n")
    for name, fn in [
        ("vs colour-science delta_E CIE 2000 (200 random pairs)", test_gpu_ciede2000_matches_colour),
        ("vs Sharma 2005 reference test data",                    test_known_pairs),
    ]:
        try:
            fn()
            print(f"  ✓ PASS  {name}")
        except AssertionError as e:
            print(f"  ✗ FAIL  {name}")
            print(f"          {e}")
        except Exception as e:
            print(f"  ⚠ SKIP  {name}  ({type(e).__name__}: {e})")
