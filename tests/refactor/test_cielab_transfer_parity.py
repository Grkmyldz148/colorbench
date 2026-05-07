"""Phase 2 parity tests:
  - CIELab new vs legacy: bit-exact forward + inverse
  - All transfer functions: round-trip + match against legacy HelmCT branch
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from core.cs import CIELab as CIELab_NEW
from core.cs.transfer import (
    CbrtTransfer, DepCubicTransfer, NakaRushtonTransfer,
    SoftcbrtTransfer, CielabDeltaTransfer, PowerTransfer, RationalTransfer,
)
from core.spaces import CIELab as CIELab_OLD  # legacy monolithic


def test_cielab_parity():
    """CIELab new vs old: bit-exact forward + inverse over 1M random XYZ."""
    np.random.seed(0)
    xyz_np = np.random.uniform(0, 1.5, (1_000_000, 3)).astype(np.float64)
    xyz = torch.from_numpy(xyz_np)

    sp_new = CIELab_NEW(device=torch.device("cpu"), dtype=torch.float64)
    sp_old = CIELab_OLD(torch.device("cpu"))

    lab_new = sp_new.forward(xyz)
    lab_old = sp_old.forward(xyz)
    fwd_diff = (lab_new - lab_old).abs().max().item()

    rt_new = sp_new.inverse(lab_new)
    rt_old = sp_old.inverse(lab_old)
    inv_diff = (rt_new - rt_old).abs().max().item()

    rt_err_new = (rt_new - xyz).abs().max().item()
    rt_err_old = (rt_old - xyz).abs().max().item()

    print(f"  CIELab forward new vs old max diff: {fwd_diff:.2e}")
    print(f"  CIELab inverse new vs old max diff: {inv_diff:.2e}")
    print(f"  CIELab round-trip new: {rt_err_new:.2e}, old: {rt_err_old:.2e}")
    assert fwd_diff < 1e-13, f"CIELab forward mismatch: {fwd_diff}"
    assert inv_diff < 1e-13, f"CIELab inverse mismatch: {inv_diff}"


def test_transfer_roundtrips():
    """Each transfer: forward∘inverse should be identity to float64 epsilon."""
    np.random.seed(0)
    xs = torch.from_numpy(np.random.uniform(-1, 1, 100_000).astype(np.float64))

    transfers = [
        ("CbrtTransfer", CbrtTransfer()),
        ("DepCubicTransfer(α=0.021)", DepCubicTransfer(0.021)),
        ("NakaRushtonTransfer", NakaRushtonTransfer()),
        ("SoftcbrtTransfer(ε=0.001)", SoftcbrtTransfer(0.001)),
        ("CielabDeltaTransfer", CielabDeltaTransfer()),
        ("PowerTransfer(γ=0.43)", PowerTransfer(0.43)),
        ("RationalTransfer", RationalTransfer()),
    ]
    for name, t in transfers:
        # Naka-Rushton has bounded input range (|x| < s); clip
        if isinstance(t, NakaRushtonTransfer):
            xs_t = xs * 0.3  # ensure |y| < s = 0.71
        else:
            xs_t = xs
        y = t.forward(xs_t)
        x_rt = t.inverse(y)
        err = (x_rt - xs_t).abs().max().item()
        # Tolerances: most are bit-exact; CielabDelta has piecewise discontinuity
        tol = 1e-13 if name not in ("CielabDeltaTransfer",) else 1e-12
        status = "✓" if err < tol else "✗"
        print(f"  {status} {name:<32} round-trip err: {err:.2e}")
        assert err < tol, f"{name} round-trip too lossy: {err}"


def test_depcubic_vs_legacy():
    """DepCubicTransfer matches legacy HelmCT depcubic branch (forward output)."""
    alpha = 0.021
    np.random.seed(0)
    xs = torch.from_numpy(np.random.uniform(-1, 1, 1_000_000).astype(np.float64))

    # New
    t_new = DepCubicTransfer(alpha)
    y_new = t_new.forward(xs)

    # Legacy (matches colorbench/core/spaces.py:914-924 with 1 Halley iter)
    s = (alpha / 3) ** 0.5
    t = xs / (2 * s ** 3)
    y = 2 * s * torch.sinh(torch.arcsinh(t) / 3)
    f = y ** 3 + alpha * y - xs
    fp = 3 * y ** 2 + alpha
    fpp = 6 * y
    denom = 2 * fp * fp - f * fpp
    safe = denom.abs() > 1e-30
    y_old = torch.where(safe, y - 2 * f * fp / torch.where(safe, denom, torch.ones_like(denom)), y)

    diff = (y_new - y_old).abs().max().item()
    print(f"  DepCubic new vs legacy forward diff: {diff:.2e}")
    assert diff < 1e-13, f"DepCubic mismatch: {diff}"


if __name__ == "__main__":
    print("Test 1: CIELab parity (1M XYZ)")
    test_cielab_parity()

    print("\nTest 2: Transfer round-trips")
    test_transfer_roundtrips()

    print("\nTest 3: DepCubic vs legacy")
    test_depcubic_vs_legacy()

    print("\n✓ All Phase 2 parity tests PASS")
