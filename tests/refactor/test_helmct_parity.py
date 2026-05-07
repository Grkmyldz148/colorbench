"""Phase 4: HelmCT new vs legacy bit-exact parity.

Loads same JSON checkpoint with both implementations, compares forward and
inverse output across 1M random XYZ samples.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import time
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from core.cs import HelmCT as HelmCT_NEW
from core.spaces import HelmCT as HelmCT_OLD


CKPT = "/Volumes/harici_ssd/color-space/helmlab-experimental/checkpoints/genspace_v0.11.1_api_legacy.json"


def test_forward_parity():
    np.random.seed(0)
    xyz_np = np.random.uniform(0, 1.5, (1_000_000, 3)).astype(np.float64)
    xyz = torch.from_numpy(xyz_np)

    sp_new = HelmCT_NEW(CKPT, device=torch.device("cpu"), dtype=torch.float64)
    sp_old = HelmCT_OLD(CKPT, torch.device("cpu"))

    t0 = time.time()
    lab_new = sp_new.forward(xyz)
    t_new_fwd = time.time() - t0

    t0 = time.time()
    lab_old = sp_old.forward(xyz)
    t_old_fwd = time.time() - t0

    diff_max = (lab_new - lab_old).abs().max().item()
    diff_mean = (lab_new - lab_old).abs().mean().item()
    print(f"  Forward new: {t_new_fwd*1000:.0f}ms  old: {t_old_fwd*1000:.0f}ms")
    print(f"  Forward max diff: {diff_max:.2e}, mean: {diff_mean:.2e}")
    assert diff_max < 1e-13, f"HelmCT forward mismatch: {diff_max}"


def test_inverse_parity():
    np.random.seed(1)
    # Use realistic Lab values: forward of random XYZ to ensure they're in valid range
    xyz_np = np.random.uniform(0, 1.5, (1_000_000, 3)).astype(np.float64)
    xyz = torch.from_numpy(xyz_np)
    sp_new = HelmCT_NEW(CKPT, device=torch.device("cpu"), dtype=torch.float64)
    sp_old = HelmCT_OLD(CKPT, torch.device("cpu"))

    lab_new = sp_new.forward(xyz)
    lab_old = sp_old.forward(xyz)

    t0 = time.time()
    rt_new = sp_new.inverse(lab_new)
    t_new = time.time() - t0
    t0 = time.time()
    rt_old = sp_old.inverse(lab_old)
    t_old = time.time() - t0

    diff_inv = (rt_new - rt_old).abs().max().item()
    rt_err_new = (rt_new - xyz).abs().max().item()
    rt_err_old = (rt_old - xyz).abs().max().item()
    print(f"  Inverse new: {t_new*1000:.0f}ms  old: {t_old*1000:.0f}ms")
    print(f"  Inverse new vs old max diff: {diff_inv:.2e}")
    print(f"  Round-trip new: {rt_err_new:.2e}, old: {rt_err_old:.2e}")
    assert diff_inv < 1e-13, f"HelmCT inverse mismatch: {diff_inv}"


if __name__ == "__main__":
    print(f"Checkpoint: {os.path.basename(CKPT)}\n")
    print("Test 1: Forward parity")
    test_forward_parity()

    print("\nTest 2: Inverse parity")
    test_inverse_parity()

    print("\n✓ All Phase 4 HelmCT parity tests PASS")
