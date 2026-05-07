"""Snapshot'tan iki uzayı kategorize karşılaştırır.

Çıktı:
  - Mathematical: X wins / Y losses / Z tied
  - Structural: ...
  - Perceptual_internal: ...
  - Perceptual_visible: ...

Bu paper iddialarını dürüstçe sunum için ana araç.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from colorbench.core.metric_categories import (
    CATEGORIES, CATEGORY_LABEL, GROUP_CATEGORY, category_of
)


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if k.startswith("_") or k in ("space","device","timestamp","total_time"):
            continue
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def lower_is_better(metric_id):
    """Heuristic: most metrics lower=better, except specifically named."""
    higher_better_keywords = [
        "fidelity", "preserved", "exact_count", "valid_cusps", "smoothness_min",
        "min_de", "min_step_de", "yellow_chroma", "_pct_invisible",
        "n_samples", "n_hues", "n_centers", "n_l_levels", "n_points",
        "n_mapped", "preserved_pct", "min_contrast", "mean_min_de",
        "invisible", "L_values",
    ]
    return not any(k in metric_id for k in higher_better_keywords)


def compare(snap_a, snap_b, label_a="A", label_b="B"):
    """Compare two flat metric dicts, return per-category win/loss/tied counts."""
    common = set(snap_a) & set(snap_b)
    by_cat = {c: {"wins_a": 0, "wins_b": 0, "tied": 0, "details": []} for c in CATEGORIES + ["unknown"]}
    for mid in common:
        a, b = snap_a[mid], snap_b[mid]
        cat = category_of(mid)
        diff = abs(a - b)
        rel = diff / max(abs(b), 1e-12)
        if rel < 0.02:
            by_cat[cat]["tied"] += 1
            verdict = "tied"
        else:
            lb = lower_is_better(mid)
            if (lb and a < b) or (not lb and a > b):
                by_cat[cat]["wins_a"] += 1
                verdict = label_a
            else:
                by_cat[cat]["wins_b"] += 1
                verdict = label_b
        by_cat[cat]["details"].append((mid, a, b, verdict))
    return by_cat


def main():
    import sys
    snap_dir = os.path.join(HERE, "..", "tests", "snapshots")
    # Latest snapshots (auto-pick newest by mtime)
    def newest(prefix):
        cands = [f for f in os.listdir(snap_dir) if f.startswith(prefix) and f.endswith(".json")]
        cands.sort(key=lambda f: os.path.getmtime(os.path.join(snap_dir, f)), reverse=True)
        return os.path.join(snap_dir, cands[0]) if cands else None

    h_path = newest("Helmgen_v0.11.1")
    o_path = newest("OKLab_2026")
    print(f"  Helmgen snapshot: {os.path.basename(h_path)}", file=sys.stderr)
    print(f"  OKLab snapshot:   {os.path.basename(o_path)}", file=sys.stderr)
    helmgen = flatten(json.load(open(h_path)))
    oklab = flatten(json.load(open(o_path)))
    by_cat = compare(helmgen, oklab, "Helmgen", "OKLab")

    print(f"\n{'═'*88}")
    print(f"  ColorBench KATEGORİZE — Helmgen vs OKLab (snapshot 2026-05-06)")
    print(f"{'═'*88}\n")

    print(f"  {'Kategori':<22} {'Helmgen':>10} {'OKLab':>10} {'Tied':>10} {'Toplam':>10}  Açıklama")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*10}  {'─'*40}")
    g_total = o_total = t_total = 0
    for cat in CATEGORIES:
        c = by_cat[cat]
        total = c["wins_a"] + c["wins_b"] + c["tied"]
        if total == 0:
            continue
        g_total += c["wins_a"]; o_total += c["wins_b"]; t_total += c["tied"]
        print(f"  {cat:<22} {c['wins_a']:>10} {c['wins_b']:>10} {c['tied']:>10} {total:>10}  {CATEGORY_LABEL.get(cat,'')[:50]}")

    if by_cat["unknown"]["wins_a"] + by_cat["unknown"]["wins_b"] + by_cat["unknown"]["tied"] > 0:
        c = by_cat["unknown"]
        print(f"  {'unknown':<22} {c['wins_a']:>10} {c['wins_b']:>10} {c['tied']:>10} "
              f"{c['wins_a']+c['wins_b']+c['tied']:>10}  (kategorize edilmemiş)")

    print(f"\n  {'TOPLAM':<22} {g_total:>10} {o_total:>10} {t_total:>10} {g_total+o_total+t_total:>10}\n")

    print(f"\n{'═'*88}")
    print(f"  DÜRÜST YORUM")
    print(f"{'═'*88}")
    pv = by_cat["perceptual_visible"]
    pi = by_cat["perceptual_internal"]
    s  = by_cat["structural"]
    m  = by_cat["mathematical"]

    pv_total = pv["wins_a"] + pv["wins_b"] + pv["tied"]
    print(f"\n  • PERCEPTUAL_VISIBLE (end-user gözle görür):")
    print(f"      Helmgen {pv['wins_a']} / OKLab {pv['wins_b']} / Tied {pv['tied']}  ({pv_total} metric)")
    if pv_total > 0:
        print(f"      Bu kategori paper'ın 'real-world görsel kazanım' iddiasının dayanağı.")
        if pv['wins_a'] > pv['wins_b']:
            print(f"      ⇒ Helmgen end-user görsel olarak {pv['wins_a']/pv_total*100:.0f}% metric'te öne.")

    pi_total = pi["wins_a"] + pi["wins_b"] + pi["tied"]
    print(f"\n  • PERCEPTUAL_INTERNAL (akademik perceptual uniformity):")
    print(f"      Helmgen {pi['wins_a']} / OKLab {pi['wins_b']} / Tied {pi['tied']}  ({pi_total} metric)")

    s_total = s["wins_a"] + s["wins_b"] + s["tied"]
    print(f"\n  • STRUCTURAL (sub-JND, gamut topology):")
    print(f"      Helmgen {s['wins_a']} / OKLab {s['wins_b']} / Tied {s['tied']}  ({s_total} metric)")
    print(f"      Bu sayılar 'uzay yapısal sağlığı' — kullanıcı görmez ama uzay tasarımı için önemli.")

    m_total = m["wins_a"] + m["wins_b"] + m["tied"]
    print(f"\n  • MATHEMATICAL (round-trip, NaN check):")
    print(f"      Helmgen {m['wins_a']} / OKLab {m['wins_b']} / Tied {m['tied']}  ({m_total} metric)")
    print(f"      Çoğu sıfır error — uzay matematiksel sağlığı.")

    print(f"\n  Önceki '64-9-17 / 90' özetinin kategorize hâli yukarıda.")


if __name__ == "__main__":
    main()
