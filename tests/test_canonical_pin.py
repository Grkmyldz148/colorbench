"""Phase 2 — Canonical wrapper'lar bit-identical mı?

`spaces_literature_canonical.py` colour-science'a delegate eden wrapper'lar.
Test eder: forward(xyz) == colour.XYZ_to_X(xyz) bit-identical (1e-12).
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import colour
try:
    import torch
    from colorbench.core.cs import (
        IPTCanonical, JzAzBzCanonical, CAM16UCSCanonical, DIN99dCanonical,
    )
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

D65_XY = np.array([0.95047, 1.0, 1.08883])[:2] / np.array([0.95047, 1.0, 1.08883]).sum()
RNG = np.random.default_rng(7)
N = 100


def _xyz_pairs():
    rgb = RNG.random((N, 3))
    return colour.sRGB_to_XYZ(rgb)


def _t(arr): return torch.tensor(arr, dtype=torch.float64)


def test_IPT_canonical_bit_identical():
    if not HAVE_TORCH: return
    sp = IPTCanonical(torch.device("cpu"))
    xyz = _xyz_pairs()
    ours = sp.forward(_t(xyz)).numpy()
    theirs = colour.XYZ_to_IPT(xyz)
    err = np.max(np.abs(ours - theirs))
    assert err < 1e-12, f"IPT canonical drift {err:.2e}"


def test_IPT_canonical_roundtrip():
    if not HAVE_TORCH: return
    sp = IPTCanonical(torch.device("cpu"))
    xyz = _xyz_pairs()
    xyz_back = sp.inverse(sp.forward(_t(xyz))).numpy()
    err = np.max(np.abs(xyz - xyz_back))
    assert err < 1e-9, f"IPT canonical round-trip {err:.2e}"


def test_JzAzBz_canonical_bit_identical():
    if not HAVE_TORCH: return
    sp = JzAzBzCanonical(torch.device("cpu"))
    xyz = _xyz_pairs()
    ours = sp.forward(_t(xyz)).numpy()
    theirs = colour.XYZ_to_Jzazbz(xyz)
    err = np.max(np.abs(ours - theirs))
    assert err < 1e-12, f"JzAzBz canonical drift {err:.2e}"


def test_JzAzBz_canonical_roundtrip():
    if not HAVE_TORCH: return
    sp = JzAzBzCanonical(torch.device("cpu"))
    xyz = _xyz_pairs()
    xyz_back = sp.inverse(sp.forward(_t(xyz))).numpy()
    err = np.max(np.abs(xyz - xyz_back))
    assert err < 1e-9, f"JzAzBz canonical round-trip {err:.2e}"


def test_CAM16UCS_canonical_bit_identical():
    if not HAVE_TORCH: return
    sp = CAM16UCSCanonical(torch.device("cpu"))
    xyz = _xyz_pairs()
    ours = sp.forward(_t(xyz)).numpy()
    theirs = colour.XYZ_to_CAM16UCS(xyz)
    err = np.max(np.abs(ours - theirs))
    assert err < 1e-12, f"CAM16UCS canonical drift {err:.2e}"


def test_DIN99d_canonical_bit_identical():
    if not HAVE_TORCH: return
    sp = DIN99dCanonical(torch.device("cpu"))
    xyz = _xyz_pairs()
    ours = sp.forward(_t(xyz)).numpy()
    lab = colour.XYZ_to_Lab(xyz, illuminant=D65_XY)
    theirs = colour.Lab_to_DIN99(lab, method="DIN99d")
    err = np.max(np.abs(ours - theirs))
    assert err < 1e-12, f"DIN99d canonical drift {err:.2e}"


if __name__ == "__main__":
    if not HAVE_TORCH: print("PyTorch yok"); sys.exit(0)
    tests = [
        ("IPTCanonical          forward bit-identical <1e-12",   test_IPT_canonical_bit_identical),
        ("IPTCanonical          round-trip <1e-9",                test_IPT_canonical_roundtrip),
        ("JzAzBzCanonical       forward bit-identical <1e-12",   test_JzAzBz_canonical_bit_identical),
        ("JzAzBzCanonical       round-trip <1e-9",                test_JzAzBz_canonical_roundtrip),
        ("CAM16UCSCanonical     forward bit-identical <1e-12",   test_CAM16UCS_canonical_bit_identical),
        ("DIN99dCanonical       forward bit-identical <1e-12",   test_DIN99d_canonical_bit_identical),
    ]
    print("PIN TEST: Canonical wrapper bit-identical (colour-science)\n")
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ PASS  {name}")
        except AssertionError as e:
            print(f"  ✗ FAIL  {name}\n          {e}")
        except Exception as e:
            print(f"  ⚠ ERR   {name}\n          {type(e).__name__}: {e}")
