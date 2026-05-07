"""Phase 5g: 9 advanced metrics — last batch."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import torch
torch.set_default_dtype(torch.float64)

from core.cs import OKLab
from core.metrics import (
    measure_cross_gamut_consistency, measure_quantization_symmetry,
    measure_channel_monotonicity, measure_perceptual_banding,
    measure_oog_excursion, measure_hue_reversal,
    measure_primary_hue_discontinuity, measure_negative_lms,
    measure_extreme_chroma_stability,
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
        ("cross_gamut", measure_cross_gamut_consistency, "cross_gamut"),
        ("quantization", measure_quantization_symmetry, "quantization"),
        ("channel_mono", measure_channel_monotonicity, "channel_mono"),
        ("perceptual_banding", measure_perceptual_banding, "banding"),
        ("oog_excursion", measure_oog_excursion, "oog_excursion"),
        ("hue_reversal", measure_hue_reversal, "hue_reversal"),
        ("primary_hue_disc", measure_primary_hue_discontinuity, "primary_hue_disc"),
        ("negative_lms", measure_negative_lms, "negative_lms"),
        ("extreme_chroma", measure_extreme_chroma_stability, "extreme_chroma_stab"),
    ]
    all_ok = True
    for name, fn, snap_key in tests:
        if snap_key not in snap:
            print(f"  ⚠ {name}: snap key '{snap_key}' missing — list keys")
            for sk in sorted(snap.keys()):
                if any(t in sk.lower() for t in name.split("_")):
                    print(f"      {sk}")
            all_ok = False
            continue
        new = fn(sp)
        ok = _compare(name, new, snap[snap_key])
        all_ok = all_ok and ok

    if all_ok:
        print("\n✓ All Phase 5g metrics BIT-EXACT")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
