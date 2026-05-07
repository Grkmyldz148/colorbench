"""Phase 5c: gamut + gamut_mapping + hue + special_gradients + stability
new vs OKLab snapshot."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import time
import torch
torch.set_default_dtype(torch.float64)

from core.cs import OKLab
from core.metrics import (
    measure_gamut, measure_gamut_mapping, measure_hue,
    measure_special_gradients, measure_stability,
)


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, path))
        elif isinstance(v, (int, float, bool)):
            out[path] = v
    return out


def _compare(name, new, snap):
    f_new = _flat(new); f_snap = _flat(snap)
    diffs = []
    for k in sorted(set(f_new.keys()) | set(f_snap.keys())):
        if k not in f_new: diffs.append((k, "MISSING_NEW", f_snap[k])); continue
        if k not in f_snap: diffs.append((k, f_new[k], "MISSING_SNAP")); continue
        a, b = f_new[k], f_snap[k]
        if a == b: continue
        if isinstance(a, bool) or isinstance(b, bool):
            diffs.append((k, a, b)); continue
        if abs(a - b) < 1e-13: continue
        if (a == 0) and (b == 0): continue
        diffs.append((k, a, b))
    if not diffs:
        print(f"  ✓ {name}: BIT-EXACT match ({len(f_new)} keys)")
        return True
    print(f"  ✗ {name}: {len(diffs)} diffs")
    for k, a, b in diffs[:8]:
        print(f"    {k}: new={a} snap={b}")
    return False


def main():
    sp = OKLab(device=torch.device("cpu"), dtype=torch.float64)
    snap = json.load(open(os.path.join(COLORBENCH, "tests", "snapshots",
                                        "OKLab_2026-05-07_v6.json")))

    # gamut
    t0 = time.time(); g = measure_gamut(sp); print(f"  gamut ran in {time.time()-t0:.1f}s")
    ok1 = _compare("measure_gamut", g, snap["gamut"])

    # gamut_mapping
    t0 = time.time(); gm = measure_gamut_mapping(sp); print(f"  gamut_mapping ran in {time.time()-t0:.1f}s")
    ok2 = _compare("measure_gamut_mapping", gm, snap["gamut_mapping"])

    # hue
    h = measure_hue(sp)
    ok3 = _compare("measure_hue", h, snap["hue"])

    # special_gradients
    sg = measure_special_gradients(sp)
    ok4 = _compare("measure_special_gradients", sg, snap["specials"])

    # stability
    s = measure_stability(sp)
    ok5 = _compare("measure_stability", s, snap["stability"])

    if all([ok1, ok2, ok3, ok4, ok5]):
        print("\n✓ All Phase 5c metrics BIT-EXACT against snapshot")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
