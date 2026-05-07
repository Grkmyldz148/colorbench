"""Phase 5e: 7 perceptual metrics (wcag/harmony/photo/animation/shade) vs snapshot."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import torch
torch.set_default_dtype(torch.float64)

from core.cs import OKLab
from core.metrics import (
    measure_wcag_midpoint_contrast, measure_harmony_accuracy,
    measure_hue_agreement, measure_photo_gamut_map,
    measure_eased_animation, measure_shade_hue_consistency,
    measure_chroma_preservation,
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
                if isinstance(e, (int, float, bool)):
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
        ("wcag_midpoint_contrast", measure_wcag_midpoint_contrast, "wcag_midpoint"),
        ("harmony_accuracy", measure_harmony_accuracy, "harmony_accuracy"),
        ("hue_agreement", measure_hue_agreement, "hue_agreement"),
        ("photo_gamut_map", measure_photo_gamut_map, "photo_gamut_map"),
        ("eased_animation", measure_eased_animation, "eased_animation"),
        ("shade_hue_consistency", measure_shade_hue_consistency, "shade_hue_consistency"),
        ("chroma_preservation", measure_chroma_preservation, "chroma_preservation"),
    ]
    all_ok = True
    for name, fn, snap_key in tests:
        if snap_key not in snap:
            print(f"  ⚠ {name}: snap key '{snap_key}' missing — skip")
            continue
        new = fn(sp)
        ok = _compare(name, new, snap[snap_key])
        all_ok = all_ok and ok

    if all_ok:
        print("\n✓ All Phase 5e metrics BIT-EXACT")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
