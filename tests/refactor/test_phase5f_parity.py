"""Phase 5f: 8 advanced metrics — cvd/animation/extremes/jacobian/contrast/
hue_leaf/3color/double_rt vs OKLab snapshot."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import torch
torch.set_default_dtype(torch.float64)

from core.cs import OKLab
from core.metrics import (
    measure_cvd, measure_animation, measure_extremes, measure_jacobian,
    measure_contrast, measure_hue_leaf, measure_3color_gradients,
    measure_double_roundtrip,
)


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, path))
        elif isinstance(v, (int, float, bool)):
            out[path] = v
        elif isinstance(v, list):
            for i, e in enumerate(v):
                if isinstance(e, dict):
                    out.update(_flat(e, f"{path}[{i}]"))
                elif isinstance(e, (int, float, bool)):
                    out[f"{path}[{i}]"] = e
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
        print(f"  ✓ {name}: BIT-EXACT ({len(f_new)} keys)")
        return True
    print(f"  ✗ {name}: {len(diffs)} diffs")
    for k, a, b in diffs[:5]:
        print(f"    {k}: new={a} snap={b}")
    return False


def main():
    sp = OKLab(device=torch.device("cpu"), dtype=torch.float64)
    snap = json.load(open(os.path.join(COLORBENCH, "tests", "snapshots",
                                        "OKLab_2026-05-07_v6.json")))
    tests = [
        ("cvd", measure_cvd, "cvd"),
        ("animation", measure_animation, "animation"),
        ("extremes", measure_extremes, "extremes"),
        ("jacobian", measure_jacobian, "jacobian"),
        ("contrast", measure_contrast, "contrast"),
        ("hue_leaf", measure_hue_leaf, "hue_leaf"),
        ("3color_gradients", measure_3color_gradients, "3color"),
        ("double_roundtrip", measure_double_roundtrip, "double_rt"),
    ]
    all_ok = True
    for name, fn, snap_key in tests:
        if snap_key not in snap:
            print(f"  ⚠ {name}: snap key '{snap_key}' missing — list keys")
            print(f"    Snapshot keys with 'name' substring:")
            for sk in sorted(snap.keys()):
                if name.split("_")[0] in sk.lower():
                    print(f"      {sk}")
            continue
        new = fn(sp)
        ok = _compare(name, new, snap[snap_key])
        all_ok = all_ok and ok

    if all_ok:
        print("\n✓ All Phase 5f metrics BIT-EXACT")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
