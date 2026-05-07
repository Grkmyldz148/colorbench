"""Phase 6: 5 literature spaces + 4 canonical wrappers vs legacy."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from core.cs import (
    IPT, JzAzBz, ICtCp, CAM16UCS, DIN99d,
    IPTCanonical, JzAzBzCanonical, CAM16UCSCanonical, DIN99dCanonical,
)
from core.spaces_literature import (
    IPT as IPT_OLD, JzAzBz as JzAzBz_OLD, ICtCp as ICtCp_OLD,
    CAM16UCS as CAM16UCS_OLD, DIN99d as DIN99d_OLD,
)
from core.spaces_literature_canonical import (
    IPTCanonical as IPTCan_OLD, JzAzBzCanonical as JzCan_OLD,
    CAM16UCSCanonical as CAM16Can_OLD, DIN99dCanonical as DIN99Can_OLD,
)


def main():
    np.random.seed(0)
    xyz_np = np.random.uniform(0, 1.5, (1_000_000, 3)).astype(np.float64)
    xyz = torch.from_numpy(xyz_np)

    cases = [
        ("IPT", IPT, IPT_OLD),
        ("JzAzBz", JzAzBz, JzAzBz_OLD),
        ("ICtCp", ICtCp, ICtCp_OLD),
        ("CAM16UCS", CAM16UCS, CAM16UCS_OLD),
        ("DIN99d", DIN99d, DIN99d_OLD),
    ]
    cases_can = [
        ("IPTCanonical", IPTCanonical, IPTCan_OLD),
        ("JzAzBzCanonical", JzAzBzCanonical, JzCan_OLD),
        ("CAM16UCSCanonical", CAM16UCSCanonical, CAM16Can_OLD),
        ("DIN99dCanonical", DIN99dCanonical, DIN99Can_OLD),
    ]

    all_ok = True
    for name, NewCls, OldCls in cases + cases_can:
        # New constructor: dtype/device parametric
        try:
            sp_new = NewCls(device=torch.device("cpu"))
        except TypeError:
            sp_new = NewCls(torch.device("cpu"))
        sp_old = OldCls(torch.device("cpu"))

        try:
            lab_new = sp_new.forward(xyz)
            lab_old = sp_old.forward(xyz)
        except Exception as e:
            print(f"  ✗ {name}: forward error {type(e).__name__}: {e}")
            all_ok = False
            continue

        # NaN-aware compare: where both NaN, treat as equal
        finite_mask = lab_new.isfinite() & lab_old.isfinite()
        nan_match = lab_new.isnan() & lab_old.isnan()
        ok_mask = finite_mask | nan_match
        diff_t = (lab_new - lab_old).abs()
        diff_t = torch.where(finite_mask, diff_t, torch.zeros_like(diff_t))
        if not ok_mask.all():
            print(f"  ✗ {name}: NaN/Inf mismatch (new has NaN where old doesn't or vice versa)")
            all_ok = False
            continue
        diff = diff_t.max().item()
        # Round-trip
        rt_new = sp_new.inverse(lab_new)
        rt_diff = (rt_new - xyz).abs().max().item()
        if diff < 1e-13:
            print(f"  ✓ {name}: forward bit-exact ({diff:.2e}), round-trip {rt_diff:.2e}")
        else:
            print(f"  ✗ {name}: forward diff {diff:.2e}")
            all_ok = False

    if all_ok:
        print("\n✓ All Phase 6 spaces BIT-EXACT")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
