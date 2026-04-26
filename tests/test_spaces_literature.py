"""Tests for literature color spaces (IPT, JzAzBz, ICtCp, CAM16-UCS, DIN99d).

Each space must satisfy:
 1. Roundtrip:  max |sRGB→XYZ→space→XYZ→sRGB - sRGB| < 1e-8
                (CAM16-UCS loosened to 1e-6 since inverse uses branch logic.)
 2. Finite:      no NaN/Inf for random inputs
 3. Achromatic:  |a|, |b| < 1e-6 for pure gray inputs (D65 * s)

Run:
    python3 -m colorbench.tests.test_spaces_literature
"""

import math
import os
import sys

import torch

# Make colorbench package importable when run as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float64)

from colorbench.core.spaces import M_SRGB, D65  # noqa: E402
from colorbench.core.spaces_literature import (  # noqa: E402
    IPT,
    JzAzBz,
    ICtCp,
    CAM16UCS,
    DIN99d,
)


def srgb_to_xyz(srgb_lin: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Linear sRGB (N,3) in [0,1] → XYZ (N,3) with D65 Y=1."""
    return srgb_lin @ M_SRGB.to(device).T


def xyz_to_srgb(xyz: torch.Tensor, device: torch.device) -> torch.Tensor:
    M = M_SRGB.to(device)
    M_inv = torch.linalg.inv(M)
    return xyz @ M_inv.T


def gen_random_srgb(n: int, device: torch.device, seed: int = 42) -> torch.Tensor:
    """Random sRGB linear values in [0.01, 0.99]^3."""
    g = torch.Generator(device=device).manual_seed(seed)
    return 0.01 + 0.98 * torch.rand((n, 3), generator=g, device=device, dtype=torch.float64)


def gray_xyz(n: int, device: torch.device, seed: int = 7) -> torch.Tensor:
    """Pure gray inputs: XYZ = D65 * s for varying s in [0.02, 1.0]."""
    g = torch.Generator(device=device).manual_seed(seed)
    s = 0.02 + 0.98 * torch.rand((n, 1), generator=g, device=device, dtype=torch.float64)
    return s * D65.to(device).unsqueeze(0)


def run_roundtrip(space, device, tol, n=1000):
    srgb_lin = gen_random_srgb(n, device)
    xyz = srgb_to_xyz(srgb_lin, device)
    lab = space.forward(xyz)
    xyz2 = space.inverse(lab)
    err = (xyz - xyz2).abs().max().item()
    return err, torch.isfinite(lab).all().item() and torch.isfinite(xyz2).all().item()


def run_finite(space, device, n=1000):
    # Wider range (includes out-of-gamut) to stress nonlinearity
    g = torch.Generator(device=device).manual_seed(123)
    srgb = -0.1 + 1.2 * torch.rand((n, 3), generator=g, device=device, dtype=torch.float64)
    xyz = srgb_to_xyz(srgb, device)
    lab = space.forward(xyz)
    ok_fwd = torch.isfinite(lab).all().item()
    xyz2 = space.inverse(lab)
    ok_inv = torch.isfinite(xyz2).all().item()
    return ok_fwd, ok_inv


def run_achromatic(space, device, n=50):
    xyz = gray_xyz(n, device)
    lab = space.forward(xyz)
    # Channel 0 is lightness-like; channels 1 & 2 should be ~0
    max_a = lab[:, 1].abs().max().item()
    max_b = lab[:, 2].abs().max().item()
    return max_a, max_b


def test_space(name, space_cls, device, rt_tol=1e-8, ach_tol=1e-6):
    space = space_cls(device)
    rt_err, rt_finite = run_roundtrip(space, device, rt_tol)
    ok_fwd, ok_inv = run_finite(space, device)
    ach_a, ach_b = run_achromatic(space, device)

    rt_pass = rt_err < rt_tol
    fin_pass = ok_fwd and ok_inv
    ach_pass = max(ach_a, ach_b) < ach_tol

    status = "PASS" if (rt_pass and fin_pass and ach_pass) else "FAIL"
    print(f"[{status}] {name:12s}  rt={rt_err:.3e}  "
          f"finite_fwd={ok_fwd}  finite_inv={ok_inv}  "
          f"ach_a={ach_a:.3e}  ach_b={ach_b:.3e}")
    return {
        "name": name,
        "pass": rt_pass and fin_pass and ach_pass,
        "rt_err": rt_err,
        "rt_tol": rt_tol,
        "rt_pass": rt_pass,
        "fin_pass": fin_pass,
        "ach_pass": ach_pass,
        "ach_a": ach_a,
        "ach_b": ach_b,
    }


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")
    print("=" * 72)

    results = []
    results.append(test_space("IPT",        IPT,       device))
    results.append(test_space("JzAzBz",     JzAzBz,    device))
    results.append(test_space("ICtCp",      ICtCp,     device))
    results.append(test_space("CAM16-UCS",  CAM16UCS,  device, rt_tol=1e-6))
    results.append(test_space("DIN99d",     DIN99d,    device))

    print("=" * 72)
    n_pass = sum(1 for r in results if r["pass"])
    print(f"Summary: {n_pass}/{len(results)} PASS")
    if n_pass != len(results):
        print("\nFailed details:")
        for r in results:
            if not r["pass"]:
                print(f"  {r['name']}: rt_pass={r['rt_pass']} "
                      f"fin_pass={r['fin_pass']} ach_pass={r['ach_pass']} "
                      f"rt_err={r['rt_err']:.3e}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
