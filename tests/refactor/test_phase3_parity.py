"""Phase 3: pwL, enrichment, neutral parity vs legacy HelmCT logic."""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
COLORBENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COLORBENCH)

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from core.cs import (
    PiecewiseLinearL, LGatedHueEnrichment,
    ChromaPreservingHueRotation, neutral_blend,
)


CKPT = "/Volumes/harici_ssd/color-space/helmlab-experimental/checkpoints/genspace_v0.11.1_api_legacy.json"


def test_pwL_parity():
    """PiecewiseLinearL forward+inverse vs legacy HelmCT._pw_forward / _pw_inverse."""
    d = json.load(open(CKPT))
    shifts = d["L_corr_pw"]
    step = d["L_corr_pw_step"]

    # Build NEW
    pw_new = PiecewiseLinearL.from_shifts(
        shifts, step, device=torch.device("cpu"), dtype=torch.float64
    )

    # Build LEGACY (matches colorbench/core/spaces.py:670-676)
    n = len(shifts)
    full_shifts = [0.0] + list(shifts) + [0.0]
    breakpoints = [i * step for i in range(n + 2)]
    breakpoints[-1] = 1.0
    L_in_old = torch.tensor(breakpoints, dtype=torch.float64)
    L_out_old = torch.tensor([b + s for b, s in zip(breakpoints, full_shifts)],
                             dtype=torch.float64)

    def fwd_old(L):
        idx = torch.searchsorted(L_in_old, L.clamp(0, 1), right=True) - 1
        idx = idx.clamp(0, len(L_in_old) - 2)
        L_lo = L_in_old[idx]
        L_hi = L_in_old[idx + 1]
        t = ((L - L_lo) / (L_hi - L_lo).clamp(min=1e-30)).clamp(0, 1)
        return L_out_old[idx] + t * (L_out_old[idx + 1] - L_out_old[idx])

    def inv_old(Lt):
        idx = torch.searchsorted(L_out_old, Lt.clamp(L_out_old[0], L_out_old[-1]),
                                 right=True) - 1
        idx = idx.clamp(0, len(L_out_old) - 2)
        Lo_lo = L_out_old[idx]
        Lo_hi = L_out_old[idx + 1]
        t = ((Lt - Lo_lo) / (Lo_hi - Lo_lo).clamp(min=1e-30)).clamp(0, 1)
        return L_in_old[idx] + t * (L_in_old[idx + 1] - L_in_old[idx])

    np.random.seed(0)
    L = torch.from_numpy(np.random.uniform(0, 1, 100_000).astype(np.float64))
    fwd_n = pw_new.forward(L)
    fwd_o = fwd_old(L)
    inv_n = pw_new.inverse(fwd_n)
    inv_o = inv_old(fwd_o)

    fwd_diff = (fwd_n - fwd_o).abs().max().item()
    inv_diff = (inv_n - inv_o).abs().max().item()
    rt_err = (inv_n - L).abs().max().item()
    print(f"  PWL forward diff: {fwd_diff:.2e}, inverse diff: {inv_diff:.2e}")
    print(f"  PWL round-trip: {rt_err:.2e}")
    assert fwd_diff < 1e-13 and inv_diff < 1e-13, "PWL mismatch"


def test_enrichment_parity():
    """LGatedHueEnrichment vs legacy HelmCT enrichment branch."""
    d = json.load(open(CKPT))
    enr_d = d["enrichment"]

    enr_new = LGatedHueEnrichment(
        amp=enr_d["amp"],
        center_deg=enr_d["center_deg"],
        sigma=enr_d["sigma"],
        L_lo=enr_d["L_lo"],
        L_hi=enr_d["L_hi"],
    )

    np.random.seed(0)
    N = 100_000
    L = torch.from_numpy(np.random.uniform(0, 1, N).astype(np.float64))
    a = torch.from_numpy(np.random.uniform(-0.3, 0.3, N).astype(np.float64))
    b = torch.from_numpy(np.random.uniform(-0.3, 0.3, N).astype(np.float64))

    a_new, b_new = enr_new.forward(L, a, b)
    a_rt, b_rt = enr_new.inverse(L, a_new, b_new)
    rt_a = (a_rt - a).abs().max().item()
    rt_b = (b_rt - b).abs().max().item()
    print(f"  Enrichment round-trip a: {rt_a:.2e}, b: {rt_b:.2e}")
    assert max(rt_a, rt_b) < 1e-13, "Enrichment round-trip too lossy"

    # Legacy parity
    PI = math.pi
    center = math.radians(enr_d["center_deg"])
    sigma = enr_d["sigma"]
    amp = enr_d["amp"]
    L_lo = enr_d["L_lo"]
    L_hi = enr_d["L_hi"]
    sig2 = sigma * sigma

    C = (a * a + b * b + 1e-30).sqrt()
    h = torch.atan2(b, a)
    t_g = ((L - L_lo) / (L_hi - L_lo)).clamp(0, 1)
    gate = torch.sin(PI * t_g).pow(2)
    dh = (h - center + PI) % (2 * PI) - PI
    gauss = torch.exp(-0.5 * (dh / sigma).pow(2))
    h_new = h + amp * gate * gauss
    a_old = C * torch.cos(h_new)
    b_old = C * torch.sin(h_new)

    diff_a = (a_new - a_old).abs().max().item()
    diff_b = (b_new - b_old).abs().max().item()
    print(f"  Enrichment forward parity a: {diff_a:.2e}, b: {diff_b:.2e}")
    assert max(diff_a, diff_b) < 1e-13, "Enrichment forward mismatch"


def test_hue_rotation_parity():
    """ChromaPreservingHueRotation vs legacy HelmCT hue_correction branch."""
    hc = [0.05, -0.03, 0.02, 0.01, 0.0, 0.0]  # arbitrary Fourier-3
    rot_new = ChromaPreservingHueRotation(hc, n_fixed_point=150)

    np.random.seed(1)
    N = 10_000
    a = torch.from_numpy(np.random.uniform(-0.3, 0.3, N).astype(np.float64))
    b = torch.from_numpy(np.random.uniform(-0.3, 0.3, N).astype(np.float64))

    a_new, b_new = rot_new.forward(a, b)
    a_rt, b_rt = rot_new.inverse(a_new, b_new)
    rt_a = (a_rt - a).abs().max().item()
    rt_b = (b_rt - b).abs().max().item()
    print(f"  HueRot round-trip a: {rt_a:.2e}, b: {rt_b:.2e}")
    assert max(rt_a, rt_b) < 1e-10, "HueRot round-trip too lossy"

    # Legacy forward
    C = (a * a + b * b + 1e-30).sqrt()
    h = torch.atan2(b, a)
    c1, s1, c2, s2, c3, s3 = hc
    dh = (c1 * torch.cos(h) + s1 * torch.sin(h)
          + c2 * torch.cos(2 * h) + s2 * torch.sin(2 * h)
          + c3 * torch.cos(3 * h) + s3 * torch.sin(3 * h))
    h_new = h + dh
    a_old = C * torch.cos(h_new)
    b_old = C * torch.sin(h_new)
    diff_a = (a_new - a_old).abs().max().item()
    diff_b = (b_new - b_old).abs().max().item()
    print(f"  HueRot forward parity a: {diff_a:.2e}, b: {diff_b:.2e}")
    assert max(diff_a, diff_b) < 1e-13, "HueRot forward mismatch"


def test_neutral_blend_parity():
    """neutral_blend vs legacy HelmCT inline branch."""
    np.random.seed(0)
    N = 10_000
    lms_c_np = np.random.uniform(0.0, 1.0, (N, 3)).astype(np.float64)
    # Inject some near-neutral samples
    lms_c_np[:50] = lms_c_np[:50, 0:1] + np.random.uniform(-1e-6, 1e-6, (50, 3))
    lms_c = torch.from_numpy(lms_c_np)

    out_new = neutral_blend(lms_c)

    # Legacy (matches colorbench/core/spaces.py:937-941)
    lms_mean = lms_c.mean(dim=1, keepdim=True)
    lms_spread = ((lms_c.max(dim=1).values - lms_c.min(dim=1).values)
                  / lms_mean.squeeze().abs().clamp(min=1e-30))
    blend_w = torch.exp(-(lms_spread / 1e-5).pow(2)).unsqueeze(1)
    out_old = lms_c + blend_w * (lms_mean.expand_as(lms_c) - lms_c)

    diff = (out_new - out_old).abs().max().item()
    print(f"  NeutralBlend max diff: {diff:.2e}")
    assert diff < 1e-13, "NeutralBlend mismatch"


if __name__ == "__main__":
    print("Test 1: PiecewiseLinearL parity")
    test_pwL_parity()

    print("\nTest 2: LGatedHueEnrichment parity")
    test_enrichment_parity()

    print("\nTest 3: ChromaPreservingHueRotation parity")
    test_hue_rotation_parity()

    print("\nTest 4: neutral_blend parity")
    test_neutral_blend_parity()

    print("\n✓ All Phase 3 parity tests PASS")
