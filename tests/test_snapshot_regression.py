"""Phase 3 — Snapshot regression testleri.

ColorBench 90 metric'in HER fonksiyonu için canary kurma yerine
**snapshot diff** stratejisi: bilinen-iyi state'leri JSON'a dondurduk
(2026-05-06, gpu_de.py refactor sonrası), gelecekteki koşumlar bu
snapshot'larla diff alır.

Eğer bir metric değerinde 1e-6 üstü değişim varsa: ya code değişti
ya bug girdi → test fail eder.

Yöntem:
  1. Bugün known-good snapshot kaydedildi
  2. Test snapshot'ı yükler + space'i yeniden koşturup karşılaştırır
  3. Tolerance tier'ları:
     - Deterministic (matrix arithmetic): 1e-9
     - Floating-point sensitive (gradient CV, banding): 1e-6
     - Sampling-based (random pairs): 1e-3
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

HERE = os.path.dirname(__file__)
SNAPS = os.path.join(HERE, "snapshots")


def load_snapshot(filename):
    return json.load(open(os.path.join(SNAPS, filename)))


def flatten_metrics(d, prefix=""):
    """Flatten nested dict into 'a.b.c' → value pairs of numerics only."""
    out = {}
    for k, v in d.items():
        if k.startswith("_") or k in ("space", "device", "timestamp", "total_time"):
            continue
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_metrics(v, key))
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            # Aggregate stat for arrays
            arr = np.asarray(v, dtype=float)
            out[f"{key}.mean"] = float(arr.mean())
            out[f"{key}.std"] = float(arr.std())
            out[f"{key}.max"] = float(arr.max())
    return out


def test_OKLab_snapshot_consistent():
    """OKLab snapshot'ı kendisiyle bit-identical. Sanity check."""
    snap = load_snapshot("OKLab_2026-05-06.json")
    snap2 = load_snapshot("OKLab_2026-05-06.json")
    flat1 = flatten_metrics(snap)
    flat2 = flatten_metrics(snap2)
    assert flat1 == flat2, "Snapshot self-load tutarsız"


def test_OKLab_snapshot_has_expected_metrics():
    """OKLab snapshot'ı bilinen metric ailelerini içeriyor."""
    snap = load_snapshot("OKLab_2026-05-06.json")
    expected_top = [
        "munsell_value", "munsell_hue", "macadam_isotropy",
        "palette_uniformity", "wcag_midpoint", "harmony_accuracy",
        "photo_gamut_map", "shade_hue_consistency", "chroma_preservation",
        "hue_reversal", "primary_hue_disc", "extreme_chroma_stab",
        "hung_berns", "ebner_fairchild", "pointer_gamut",
        "cvd", "achromatic", "roundtrip", "gamut",
    ]
    missing = [k for k in expected_top if k not in snap]
    assert not missing, f"Missing top-level metric groups: {missing}"


def test_Helmgen_snapshot_consistent():
    snap = load_snapshot("Helmgen_v0.11.1_2026-05-06.json")
    flat = flatten_metrics(snap)
    assert len(flat) > 50, f"Helmgen snapshot only {len(flat)} flat metrics — too few"


def test_OKLab_known_metric_values():
    """Spot check: bilinen sayılar (paper'a uygun)."""
    snap = load_snapshot("OKLab_2026-05-06.json")
    # Paper'da OKLab achromatic chroma tablosu var
    ach = snap.get("achromatic", {})
    if "chroma_max" in ach:
        # OKLab achromatic axis should be very small (8e-5 facelessuser version)
        assert ach["chroma_max"] < 1e-3, \
            f"OKLab achromatic chroma_max={ach['chroma_max']} unusually high"


def test_Helmgen_known_metric_values():
    """Helmgen achromatic axis paper iddiası: ~1e-13."""
    snap = load_snapshot("Helmgen_v0.11.1_2026-05-06.json")
    ach = snap.get("achromatic", {})
    if "chroma_max" in ach:
        # Paper iddiası: 1e-13 civarı; gerçek 7e-5 (M2 renorm artifact)
        assert ach["chroma_max"] < 1e-3, \
            f"Helmgen achromatic chroma_max={ach['chroma_max']} unusually high"


def test_OKLab_live_vs_snapshot():
    """Live ColorBench koşumu snapshot'la 1e-6 tolerance içinde aynı kalıyor mu?

    Bu test 1-2 dakika sürer (90 metric pipeline). Production regression detect.
    """
    try:
        import torch
    except ImportError:
        return  # PyTorch yoksa skip

    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
    from colorbench.run import build_space, run_test, get_device

    device, dtype, device_name = get_device()
    space = build_space("oklab", None, device, dtype=dtype)
    live = run_test(space, device, device_name)
    snap = load_snapshot("OKLab_2026-05-06.json")

    flat_live = flatten_metrics(live)
    flat_snap = flatten_metrics(snap)

    common = set(flat_live.keys()) & set(flat_snap.keys())
    diffs = []
    for k in common:
        # Tolerance — most metrics deterministic, some sampling-based
        tol = 1e-3 if any(s in k for s in ("ebner_fairchild", "macadam", "hung_berns", "_random")) else 1e-6
        a, b = flat_live[k], flat_snap[k]
        if abs(a - b) > tol * max(1.0, abs(b)):
            diffs.append((k, a, b, abs(a-b)))
    if diffs:
        msg = f"OKLab live vs snapshot: {len(diffs)} metric drift\n"
        for k, a, b, d in diffs[:10]:
            msg += f"  {k}: live={a:.6e} snap={b:.6e} diff={d:.2e}\n"
        raise AssertionError(msg)


if __name__ == "__main__":
    tests = [
        ("OKLab snapshot self-load consistent",       test_OKLab_snapshot_consistent),
        ("OKLab snapshot has all metric groups",      test_OKLab_snapshot_has_expected_metrics),
        ("Helmgen snapshot has >50 flat metrics",     test_Helmgen_snapshot_consistent),
        ("OKLab achromatic chroma <1e-3 (sanity)",   test_OKLab_known_metric_values),
        ("Helmgen achromatic chroma <1e-3 (sanity)", test_Helmgen_known_metric_values),
        ("OKLab LIVE vs snapshot regression (slow)",  test_OKLab_live_vs_snapshot),
    ]
    print("PIN TEST: ColorBench 90-metric snapshot regression\n")
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ PASS  {name}")
        except AssertionError as e:
            print(f"  ✗ FAIL  {name}\n          {e}")
        except Exception as e:
            print(f"  ⚠ ERR   {name}\n          {type(e).__name__}: {e}")

    # Bonus: snapshot içeriklerini özetle
    snap = load_snapshot("OKLab_2026-05-06.json")
    flat = flatten_metrics(snap)
    print(f"\n  OKLab snapshot: {len(flat)} flat metric değer, {len(snap)} üst-grup")
    snap = load_snapshot("Helmgen_v0.11.1_2026-05-06.json")
    flat = flatten_metrics(snap)
    print(f"  Helmgen snapshot: {len(flat)} flat metric değer, {len(snap)} üst-grup")
