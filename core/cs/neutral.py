"""Neutral correction utilities for HelmCT pipeline.

NeutralBlend
------------
Branchless smooth correction blending LMS_c toward channel mean for inputs
that are nearly achromatic (sRGB matrix rounding leaves small chromatic noise
on pure grays). Weight ≈ 1 at exact neutral, → 0 elsewhere. C∞-smooth.

NCLut
-----
Neutral Correction LUT: precomputes (a_err, b_err) at each L level by forwarding
sRGB grays through the pipeline (without NC) and recording the residual.
At inference: subtract interpolated error from (a, b) to drive achromatic
samples to exactly (a, b) = (0, 0).
"""
import torch


def neutral_blend(lms_c: torch.Tensor, sigma: float = 1e-5) -> torch.Tensor:
    """C∞-smooth correction toward channel mean for near-neutral inputs.

    weight = exp(-(spread/σ)²) where spread = (max-min)/|mean|.
    At exact neutral: weight ≈ 1 (full correction → all channels = mean).
    Chromatic input: weight ≈ 0 (no effect).
    """
    lms_mean = lms_c.mean(dim=1, keepdim=True)
    lms_max = lms_c.max(dim=1).values
    lms_min = lms_c.min(dim=1).values
    spread = (lms_max - lms_min) / lms_mean.squeeze(-1).abs().clamp(min=1e-30)
    w = torch.exp(-(spread / sigma).pow(2)).unsqueeze(-1)
    return lms_c + w * (lms_mean.expand_as(lms_c) - lms_c)


class NCLut:
    """Neutral Correction lookup. Built lazily on first use.

    The LUT stores (L, a_err, b_err) sampled along the achromatic axis.
    Forward pipeline subtracts a_err, b_err from (a, b); inverse adds them back.
    """

    def __init__(self, space, n_srgb: int = 512, n_hdr: int = 64):
        """Defer LUT build until first call (forward needs to be working first)."""
        self.space = space
        self.n_srgb = n_srgb
        self.n_hdr = n_hdr
        self._built = False
        self._L_keys = None
        self._a_err = None
        self._b_err = None

    def _build(self):
        sp = self.space
        device, dtype = sp.device, sp.dtype

        from .constants import M_SRGB
        ms = M_SRGB.to(device=device, dtype=dtype)

        # sRGB grays: capture matrix row-sum rounding noise
        v = torch.linspace(0.001, 0.999, self.n_srgb, device=device, dtype=dtype)
        threshold = 0.04045
        lin = torch.where(v <= threshold, v / 12.92, ((v + 0.055) / 1.055).pow(2.4))
        ones = torch.ones(3, device=device, dtype=dtype)
        row_sums = ms @ ones  # ≈ D65 with float rounding
        gray_xyz = lin.unsqueeze(1) * row_sums.unsqueeze(0)

        # HDR D65-proportional grays
        D65 = torch.tensor([0.95047, 1.0, 1.08883], device=device, dtype=dtype)
        Y_hdr = torch.linspace(1.01, 2.0, self.n_hdr, device=device, dtype=dtype)
        hdr_xyz = Y_hdr.unsqueeze(1) * D65.unsqueeze(0)
        all_xyz = torch.cat([gray_xyz, hdr_xyz], dim=0)

        # Forward without NC (caller must temporarily disable NC)
        lab = sp.forward(all_xyz)

        # Sort by L for searchsorted-based interpolation
        order = lab[:, 0].argsort()
        self._L_keys = lab[order, 0].contiguous()
        self._a_err = lab[order, 1].contiguous()
        self._b_err = lab[order, 2].contiguous()
        self._built = True

    def errors_at(self, L: torch.Tensor):
        """Return (a_err, b_err) interpolated at each L. Builds LUT on first call."""
        if not self._built:
            self._build()
        idx = (torch.searchsorted(self._L_keys, L, right=True) - 1).clamp(
            0, len(self._L_keys) - 2
        )
        L_lo = self._L_keys[idx]
        L_hi = self._L_keys[idx + 1]
        t = ((L - L_lo) / (L_hi - L_lo).clamp(min=1e-30)).clamp(0.0, 1.0)
        a_err = self._a_err[idx] + t * (self._a_err[idx + 1] - self._a_err[idx])
        b_err = self._b_err[idx] + t * (self._b_err[idx + 1] - self._b_err[idx])
        return a_err, b_err
