#!/usr/bin/env python3
"""Regenerate the ColorBench-derived block of the site's claims.ts.

Reads the canonical ColorBench result JSONs (OKLab + GenSpace), computes the
head-to-head record and per-category breakdown via the SAME comparison engine
the benchmark uses, and rewrites the `record:` and `categories:` fields of
landing/landing-new/src/data/claims.ts in place. Run this after re-running
ColorBench so the whole site updates from one place.

Usage:  python scripts/generate_claims.py
        python scripts/generate_claims.py --check     # verify claims.ts matches, exit 1 on drift
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CB))
from core.comparison import compare_spaces  # noqa: E402

CLAIMS = CB.parent / "helmlab-main-repo" / "landing" / "landing-new" / "src" / "data" / "claims.ts"
OK = CB / "results" / "OKLab.json"
GS = CB / "results" / "HelmCT(genspace_v0.11.1.json).json"


def compute():
    ok = json.load(open(OK))
    gs = json.load(open(GS))
    cmp = compare_spaces({ok["space"]: ok, gs["space"]: gs})
    cat = defaultdict(lambda: [0, 0, 0])
    rec = [0, 0, 0]
    for t in cmp.tests:
        c = t.metric.category
        if t.is_tie:
            cat[c][2] += 1; rec[2] += 1
        elif t.winner and "OKLab" in t.winner:
            cat[c][1] += 1; rec[1] += 1
        else:
            cat[c][0] += 1; rec[0] += 1
    cats = sorted(cat.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return rec, cats


def render(rec, cats):
    record = f"{{ genspace: {rec[0]}, oklab: {rec[1]}, ties: {rec[2]}, total: {rec[0]+rec[1]+rec[2]} }}"
    lines = [
        f"    {{ name: '{c}',{' ' * max(1, 14 - len(c))}genspace: {g}, oklab: {o}, ties: {t} }},"
        for c, (g, o, t) in cats
    ]
    return record, "[\n" + "\n".join(lines) + "\n  ]"


def main() -> int:
    rec, cats = compute()
    record, categories = render(rec, cats)
    text = CLAIMS.read_text()
    new = re.sub(r"record:\s*\{[^}]*\}", f"record: {record}", text, count=1)
    new = re.sub(r"categories:\s*\[.*?\n  \]", f"categories: {categories}", new, count=1, flags=re.DOTALL)
    if "--check" in sys.argv:
        if new == text:
            print(f"OK — claims.ts matches ColorBench ({rec[0]}-{rec[1]}-{rec[2]})")
            return 0
        print(f"DRIFT — claims.ts record/categories differ from ColorBench ({rec[0]}-{rec[1]}-{rec[2]}). Run generate_claims.py.")
        return 1
    CLAIMS.write_text(new)
    print(f"wrote record {rec[0]}-{rec[1]}-{rec[2]} + {len(cats)} categories → {CLAIMS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
