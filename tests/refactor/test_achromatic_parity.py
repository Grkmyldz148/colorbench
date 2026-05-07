"""Phase 5a: measure_achromatic new vs OKLab snapshot bit-exact match."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import torch
torch.set_default_dtype(torch.float64)

from core.cs import OKLab
from core.metrics import measure_achromatic


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict): out.update(_flat(v, path))
        elif isinstance(v, (int, float)): out[path] = v
    return out


def main():
    sp = OKLab(device=torch.device("cpu"), dtype=torch.float64)
    new = measure_achromatic(sp)

    snap = json.load(open(os.path.join(COLORBENCH, "tests", "snapshots",
                                        "OKLab_2026-05-07_v6.json")))
    snap_ach = snap["achromatic"]

    diffs = []
    f_new = _flat(new); f_snap = _flat(snap_ach)
    for k in sorted(set(f_new.keys()) | set(f_snap.keys())):
        if k not in f_new: diffs.append((k, "MISSING_NEW", f_snap[k])); continue
        if k not in f_snap: diffs.append((k, f_new[k], "MISSING_SNAP")); continue
        a, b = f_new[k], f_snap[k]
        if a == b: continue
        if abs(a - b) < 1e-13: continue
        diffs.append((k, a, b))

    if not diffs:
        print(f"✓ measure_achromatic BIT-EXACT match against snapshot ({len(f_new)} keys)")
        return 0
    print(f"✗ {len(diffs)} differences:")
    for k, a, b in diffs[:20]:
        print(f"  {k}: new={a} snap={b}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
