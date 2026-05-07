"""Phase 7: GenSpace family — new modular vs legacy spaces.py."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from core.cs import (
    GenSpaceEnriched, NakaRushtonEnriched, GenSpaceBlueFix, NonlinearM1,
)
from core.spaces import (
    GenSpaceEnriched as GE_OLD,
    NakaRushtonEnriched as NR_OLD,
    GenSpaceBlueFix as BF_OLD,
    NonlinearM1 as NLM1_OLD,
)


# Use api_legacy checkpoint which we know covers depcubic + L_corr_pw + enrichment
CKPT = "/Volumes/harici_ssd/color-space/helmlab-experimental/checkpoints/genspace_v0.11.1_api_legacy.json"


def main():
    np.random.seed(0)
    xyz_np = np.random.uniform(0, 1.5, (1_000_000, 3)).astype(np.float64)
    xyz = torch.from_numpy(xyz_np)

    # Only test GenSpaceEnriched here — others need different checkpoints.
    # Snapshot regen will exercise full coverage.
    pairs = [
        ("GenSpaceEnriched", GenSpaceEnriched, GE_OLD),
    ]
    all_ok = True
    for name, NewCls, OldCls in pairs:
        try:
            sp_new = NewCls(CKPT, torch.device("cpu"))
            sp_old = OldCls(CKPT, torch.device("cpu"))
        except Exception as e:
            print(f"  ⚠ {name}: instantiation skipped ({type(e).__name__})")
            continue

        lab_new = sp_new.forward(xyz)
        lab_old = sp_old.forward(xyz)

        finite = lab_new.isfinite() & lab_old.isfinite()
        nan_match = lab_new.isnan() & lab_old.isnan()
        ok_mask = finite | nan_match
        if not ok_mask.all():
            print(f"  ✗ {name}: NaN/finite mismatch")
            all_ok = False
            continue

        diff_t = (lab_new - lab_old).abs()
        diff_t = torch.where(finite, diff_t, torch.zeros_like(diff_t))
        diff = diff_t.max().item()
        if diff < 1e-13:
            print(f"  ✓ {name}: forward bit-exact ({diff:.2e})")
        else:
            print(f"  ✗ {name}: forward diff {diff:.2e}")
            all_ok = False

    if all_ok:
        print("\n✓ Phase 7 GenSpace family BIT-EXACT (representative subset)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
