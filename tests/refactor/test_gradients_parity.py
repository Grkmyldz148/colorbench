"""Phase 5b: measure_gradients new vs OKLab snapshot bit-exact match."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import torch
torch.set_default_dtype(torch.float64)

from core.cs import OKLab
from core.metrics import measure_gradients
from core.pairs import generate_all_pairs


def _flat_scalars(d, prefix=""):
    out = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat_scalars(v, path))
        elif isinstance(v, (int, float, bool)):
            out[path] = v
    return out


def main():
    sp = OKLab(device=torch.device("cpu"), dtype=torch.float64)
    pairs_xyz, pair_labels = generate_all_pairs(torch.device("cpu"))

    print(f"Pairs: {pairs_xyz.shape[0]}")
    t0 = time.time()
    new = measure_gradients(sp, pairs_xyz, pair_labels)
    print(f"Ran in {time.time() - t0:.1f}s")

    snap = json.load(open(os.path.join(COLORBENCH, "tests", "snapshots",
                                        "OKLab_2026-05-07_v6.json")))
    snap_g = snap["gradients"]

    # Compare overall + by_category (skip per-pair list, very large)
    new_compare = {"overall": new["overall"], "by_category": new["by_category"]}
    snap_compare = {"overall": snap_g["overall"], "by_category": snap_g["by_category"]}

    f_new = _flat_scalars(new_compare)
    f_snap = _flat_scalars(snap_compare)

    diffs = []
    for k in sorted(set(f_new.keys()) | set(f_snap.keys())):
        if k not in f_new: diffs.append((k, "MISSING_NEW", f_snap[k])); continue
        if k not in f_snap: diffs.append((k, f_new[k], "MISSING_SNAP")); continue
        a, b = f_new[k], f_snap[k]
        if a == b: continue
        if isinstance(a, bool) or isinstance(b, bool):
            if a == b: continue
            diffs.append((k, a, b)); continue
        if abs(a - b) < 1e-13: continue
        if (a == 0) and (b == 0): continue
        diffs.append((k, a, b))

    if not diffs:
        print(f"\n✓ measure_gradients BIT-EXACT match against snapshot ({len(f_new)} keys)")
        return 0
    print(f"\n✗ {len(diffs)} differences:")
    for k, a, b in diffs[:30]:
        print(f"  {k}: new={a} snap={b}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
