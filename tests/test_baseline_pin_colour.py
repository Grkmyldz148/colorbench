"""Pin tests: ColorBench inline baselines must match colour-science to <1e-9.

If any of these fail, the inline implementation is wrong. The fix is to
either replace the implementation with a colour-science wrapper or fix the
math until bit-identik. Memory'de "CAM16-UCS bug 22.7 STRESS off" geçti —
o tarz silent bug'ları engellemek için bu dosya canary.

Pinned baselines:
  - _xyz_to_cielab               vs colour.XYZ_to_Lab
  - _cielab_de  (CIE 76)         vs colour.delta_E(method="CIE 1976")
  - _ciede2000                   vs colour.delta_E(method="CIE 2000")
  - _cie94_de                    vs colour.delta_E(method="CIE 1994")
  - _oklab_de                    vs colour.delta_E_Oklab (or XYZ_to_Oklab+L2)

Tolerans: 1e-9 (bit-identik). Failing test = silent baseline drift.
"""
import os, sys
import numpy as np
import colour
try:
    import pytest  # only used as decorator host; not required for __main__ runner
except ImportError:
    pytest = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from colorbench.core.metric_eval import (
    _xyz_to_cielab, _cielab_de, _ciede2000, _cie94_de, _oklab_de, _D65,
)

RNG = np.random.default_rng(0)
N = 100


def _random_xyz_pairs(n=N):
    """Random sRGB → XYZ pairs (D65) for stress testing."""
    rgb1 = RNG.random((n, 3))
    rgb2 = RNG.random((n, 3))
    xyz1 = colour.sRGB_to_XYZ(rgb1)
    xyz2 = colour.sRGB_to_XYZ(rgb2)
    return xyz1, xyz2


# --- Implementation pin tests (aynı illuminant her iki tarafta) ----------
# ColorBench _D65 → xy chromaticity, colour-science çağrısına explicit ver.
_CB_XY = np.array([_D65[0]/_D65.sum(), _D65[1]/_D65.sum()])


def test_xyz_to_cielab_matches_colour():
    xyz1, _ = _random_xyz_pairs()
    ours = _xyz_to_cielab(xyz1, _D65)
    theirs = colour.XYZ_to_Lab(xyz1, illuminant=_CB_XY)
    err = np.max(np.abs(ours - theirs))
    assert err < 1e-9, f"_xyz_to_cielab drift {err:.2e}"


def test_cielab_de_76_matches_colour():
    xyz1, xyz2 = _random_xyz_pairs()
    ours = _cielab_de(xyz1, xyz2, _D65)
    lab1 = colour.XYZ_to_Lab(xyz1, illuminant=_CB_XY)
    lab2 = colour.XYZ_to_Lab(xyz2, illuminant=_CB_XY)
    theirs = colour.delta_E(lab1, lab2, method="CIE 1976")
    err = np.max(np.abs(ours - theirs))
    assert err < 1e-9, f"_cielab_de drift {err:.2e}"


def test_ciede2000_matches_colour():
    xyz1, xyz2 = _random_xyz_pairs()
    ours = _ciede2000(xyz1, xyz2, _D65)
    lab1 = colour.XYZ_to_Lab(xyz1, illuminant=_CB_XY)
    lab2 = colour.XYZ_to_Lab(xyz2, illuminant=_CB_XY)
    theirs = colour.delta_E(lab1, lab2, method="CIE 2000")
    err = np.max(np.abs(ours - theirs))
    assert err < 1e-9, f"_ciede2000 drift {err:.2e}"


def test_cie94_de_matches_colour():
    xyz1, xyz2 = _random_xyz_pairs()
    ours = _cie94_de(xyz1, xyz2, _D65)
    lab1 = colour.XYZ_to_Lab(xyz1, illuminant=_CB_XY)
    lab2 = colour.XYZ_to_Lab(xyz2, illuminant=_CB_XY)
    theirs = colour.delta_E(lab1, lab2, method="CIE 1994")
    err = np.max(np.abs(ours - theirs))
    assert err < 1e-9, f"_cie94_de drift {err:.2e}"


# --- Convention pin test (ColorBench D65 vs CIE 1931 2° D65) -------------
def test_d65_convention_within_tolerance():
    """ColorBench D65 [0.95047, 1.0, 1.08883] is rounded; CIE 1931 2°
    standard is xy=(0.31270, 0.32900) → XYZ ≈ [0.95046, 1, 1.08906]."""
    sci_xy = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    diff_xy = np.max(np.abs(_CB_XY - sci_xy))
    # Tolerans: yuvarlama < 5e-5 (CIE 4-decimal precision)
    assert diff_xy < 5e-5, f"ColorBench D65 ↔ CIE 1931 2° xy diff {diff_xy:.2e}"


def test_oklab_de_matches_colour():
    xyz1, xyz2 = _random_xyz_pairs()
    ours = _oklab_de(xyz1, xyz2)
    lab1 = colour.XYZ_to_Oklab(xyz1)
    lab2 = colour.XYZ_to_Oklab(xyz2)
    theirs = np.linalg.norm(lab2 - lab1, axis=-1)
    err = np.max(np.abs(ours - theirs))
    assert err < 1e-9, f"_oklab_de drift {err:.2e}"


if __name__ == "__main__":
    # Pretty diagnostic mode (run as script, not pytest)
    tests = [
        ("_xyz_to_cielab    vs colour.XYZ_to_Lab (same illum)",  test_xyz_to_cielab_matches_colour),
        ("_cielab_de        vs colour.delta_E CIE 1976",         test_cielab_de_76_matches_colour),
        ("_ciede2000        vs colour.delta_E CIE 2000",         test_ciede2000_matches_colour),
        ("_cie94_de         vs colour.delta_E CIE 1994",         test_cie94_de_matches_colour),
        ("_oklab_de         vs colour.XYZ_to_Oklab + L2",        test_oklab_de_matches_colour),
        ("D65 convention    ColorBench ↔ CIE 1931 2° (≤5e-5)",   test_d65_convention_within_tolerance),
    ]
    print("PIN TEST: ColorBench inline baseline ↔ colour-science (1e-9 tolerance)\n")
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ PASS  {name}")
        except AssertionError as e:
            print(f"  ✗ FAIL  {name}")
            print(f"          {e}")
