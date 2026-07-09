"""Build the ColorBench leaderboard: every invertible colour-science colour
space + helmlab, ranked on the human_pool (46 real datasets), property by
property. Writes docs/leaderboard.json for the GitHub Pages site.

Run:  python3 research/leaderboard.py
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import colour  # noqa: E402
from core import human_pool as hp  # noqa: E402

_D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
_NEEDS_ILLUM = {"Lab", "Luv", "DIN99", "ProLab"}

# the invertible 3-vector perceptual/uniform models from colour-science
COLOUR_SPACES = ["Lab", "Luv", "IPT", "IPT_Ragoo2021", "Jzazbz", "ICtCp",
                 "ICaCb", "IgPgTg", "Oklab", "DIN99", "ProLab", "Yrg",
                 "hdr_CIELab", "CAM02UCS", "CAM16UCS", "CAM02LCD", "CAM16LCD",
                 "CAM02SCD", "CAM16SCD", "sUCS"]
# pretty display names
PRETTY = {"Lab": "CIELAB", "Luv": "CIELUV", "Oklab": "OKLab", "Jzazbz": "Jzazbz",
          "IPT_Ragoo2021": "IPT (Ragoo 2021)", "hdr_CIELab": "hdr-CIELAB",
          "CAM02UCS": "CAM02-UCS", "CAM16UCS": "CAM16-UCS", "CAM02LCD": "CAM02-LCD",
          "CAM16LCD": "CAM16-LCD", "CAM02SCD": "CAM02-SCD", "CAM16SCD": "CAM16-SCD"}


class ColourWrapper:
    """A colour-science model as a ColorBench forward-space (numpy forward)."""
    def __init__(self, name):
        self.name = PRETTY.get(name, name)
        self._f = getattr(colour, f"XYZ_to_{name}")
        self._illum = name in _NEEDS_ILLUM

    def forward(self, xyz):
        xyz = np.atleast_2d(np.asarray(xyz, float))
        out = self._f(xyz, _D65) if self._illum else self._f(xyz)
        return np.asarray(out, float)


def build_helmlab():
    from run import build_space, get_device
    d, dt, _ = get_device()
    ck = "/Volumes/harici_ssd/color-space/helmlab-main-repo/checkpoints/genspace_v0.11.1.json"
    sp = build_space("genspace", ck, d, dtype=dt)
    sp.name = "helmlab"
    return sp


# properties shown on the leaderboard (validated human-data judges; lower=better)
PROPS = ["difference", "hue", "discrimination", "3d_discrim", "tolerance", "spacing"]


def main():
    spaces = [ColourWrapper(n) for n in COLOUR_SPACES]
    try:
        spaces.append(build_helmlab())
    except Exception as e:
        print(f"  helmlab skipped: {e}")

    rows = {}
    for sp in spaces:
        try:
            panel = hp.evaluate_space_on_pool(sp, validated_only=True)["by_property"]
        except Exception as e:
            print(f"  {sp.name} failed: {e}"); continue
        prop_scores = {}
        for p in PROPS:
            vals = [v for v in panel.get(p, {}).values() if isinstance(v, (int, float))]
            if vals:
                prop_scores[p] = float(np.mean(vals))
        rows[sp.name] = prop_scores
        print(f"  {sp.name:20} " + "  ".join(f"{p}={prop_scores.get(p, float('nan')):.2f}"
                                             for p in PROPS if p in prop_scores))

    # per-property ranks (1 = best/lowest); overall = mean rank across properties
    prop_rank = {p: {} for p in PROPS}
    for p in PROPS:
        scored = sorted([(v[p], n) for n, v in rows.items() if p in v])
        for rank, (_, n) in enumerate(scored, 1):
            prop_rank[p][n] = rank
    overall = {}
    for n in rows:
        rs = [prop_rank[p][n] for p in PROPS if n in prop_rank[p]]
        overall[n] = round(sum(rs) / len(rs), 2) if rs else None
    order = sorted(rows, key=lambda n: overall[n] if overall[n] is not None else 99)

    winners = {p: min(prop_rank[p], key=prop_rank[p].get) for p in PROPS if prop_rank[p]}

    out = {
        "generated": "2026-07-09",
        "n_spaces": len(rows),
        "properties": PROPS,
        "property_winners": winners,
        "spaces": [
            {"name": n, "overall_rank": overall[n],
             "scores": rows[n],
             "ranks": {p: prop_rank[p].get(n) for p in PROPS if n in prop_rank[p]}}
            for n in order
        ],
    }
    dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "docs", "leaderboard.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n  wrote {dest} ({len(rows)} spaces)")
    print("  overall order:", " > ".join(order[:6]), "...")
    print("  property winners:", winners)


if __name__ == "__main__":
    main()
