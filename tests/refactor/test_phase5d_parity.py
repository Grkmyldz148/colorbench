"""Phase 5d: 7 perceptual metrics new vs OKLab snapshot."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import torch
torch.set_default_dtype(torch.float64)

from core.cs import OKLab
from core.metrics import (
    measure_munsell_value, measure_munsell_hue, measure_macadam_isotropy,
    measure_palette_uniformity, measure_tint_shade_hue,
    measure_dataviz_distinguishability, measure_multistop_gradient,
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
        ("munsell_value", measure_munsell_value, "munsell_value"),
        ("munsell_hue", measure_munsell_hue, "munsell_hue"),
        ("macadam_isotropy", measure_macadam_isotropy, "macadam_isotropy"),
        ("palette_uniformity", measure_palette_uniformity, "palette_uniformity"),
        ("tint_shade_hue", measure_tint_shade_hue, "tint_shade_hue"),
        ("dataviz_distinguishability", measure_dataviz_distinguishability,
         "dataviz_distinguish"),
        ("multistop_gradient", measure_multistop_gradient, "multistop_gradient"),
    ]
    all_ok = True
    for name, fn, snap_key in tests:
        new = fn(sp)
        ok = _compare(name, new, snap[snap_key])
        all_ok = all_ok and ok

    if all_ok:
        print("\n✓ All Phase 5d metrics BIT-EXACT")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
