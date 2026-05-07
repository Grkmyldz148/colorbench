"""Refactored OKLab + measure_roundtrip vs legacy snapshot.

Verifies that the new modular implementation produces bit-exact output
matching the existing OKLab_2026-05-07_v6.json snapshot.

This is the gate for shipping the C-refactor: no metric value may differ
from baseline (within float64 epsilon ~1e-13).
"""
import os
import sys
import json
import time

# Add colorbench/ to path so `from core.spaces import ...` works
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import torch
torch.set_default_dtype(torch.float64)

from core.cs import OKLab as OKLab_NEW
from core.metrics import measure_roundtrip as measure_NEW


def _key_diffs(new: dict, snap: dict, prefix: str = "") -> list:
    """Recursively compare scalar/int values; return list of (path, new, snap)."""
    diffs = []
    for k in sorted(set(new.keys()) | set(snap.keys())):
        path = f"{prefix}.{k}" if prefix else k
        if k not in new:
            diffs.append((path, "MISSING_IN_NEW", snap[k])); continue
        if k not in snap:
            diffs.append((path, new[k], "MISSING_IN_SNAP")); continue
        a, b = new[k], snap[k]
        if isinstance(a, dict) and isinstance(b, dict):
            diffs.extend(_key_diffs(a, b, path))
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if a == b: continue
            if abs(a - b) < 1e-13: continue
            if (a == 0) and (b == 0): continue
            diffs.append((path, a, b))
        elif a != b:
            diffs.append((path, a, b))
    return diffs


def main():
    print(f"colorbench: {COLORBENCH}")
    print("Building new OKLab + measure_roundtrip...")
    sp = OKLab_NEW(device=torch.device("cpu"), dtype=torch.float64)
    t0 = time.time()
    new = measure_NEW(sp)
    elapsed = time.time() - t0
    print(f"  ran in {elapsed:.1f}s")

    snap_path = os.path.join(COLORBENCH, "tests", "snapshots", "OKLab_2026-05-07_v6.json")
    snap_full = json.load(open(snap_path))
    snap = snap_full["roundtrip"]

    # Compare
    diffs = _key_diffs(new, snap)
    if not diffs:
        print(f"\n✓ BIT-EXACT MATCH against snapshot ({len(_flatten(new))} keys)")
        return 0

    print(f"\n✗ {len(diffs)} differences:")
    for path, a, b in diffs[:20]:
        print(f"  {path}: new={a} snap={b}")
    return 1


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict): out.update(_flatten(v, path))
        else: out[path] = v
    return out


if __name__ == "__main__":
    sys.exit(main())
