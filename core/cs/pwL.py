"""Piecewise-linear L correction — analytical forward and inverse.

Forward: L → L_out = L + interp(shifts, L) on [0, 1] piecewise grid
Inverse: exact (no Newton) — same algorithm with input/output breakpoints swapped.

Used by Helmgen GenSpace L_corr_pw stage.
"""
import torch


class PiecewiseLinearL:
    """Piecewise-linear L correction with exact analytical inverse.

    Args:
        L_in:  monotonic breakpoints in [0, 1]
        L_out: matching output breakpoints (= L_in + shifts)
        device, dtype: tensor placement
    """

    def __init__(self, L_in_list, L_out_list, device, dtype):
        self.device = device
        self.dtype = dtype
        self.L_in = torch.tensor(L_in_list, device=device, dtype=dtype)
        self.L_out = torch.tensor(L_out_list, device=device, dtype=dtype)

    @classmethod
    def from_shifts(cls, shifts, step, device, dtype):
        """Build from helmlab JSON 'L_corr_pw' format (interior shifts only).

        Pads endpoints with zero shift, ensures last breakpoint exactly 1.0.
        """
        n = len(shifts)
        full_shifts = [0.0] + list(shifts) + [0.0]
        breakpoints = [i * step for i in range(n + 2)]
        breakpoints[-1] = 1.0
        L_out = [b + s for b, s in zip(breakpoints, full_shifts)]
        return cls(breakpoints, L_out, device=device, dtype=dtype)

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        """L → L_out via piecewise-linear interpolation."""
        L_clamped = L.clamp(0.0, 1.0)
        idx = torch.searchsorted(self.L_in, L_clamped, right=True) - 1
        idx = idx.clamp(0, len(self.L_in) - 2)
        L_lo = self.L_in[idx]
        L_hi = self.L_in[idx + 1]
        t = ((L - L_lo) / (L_hi - L_lo).clamp(min=1e-30)).clamp(0.0, 1.0)
        return self.L_out[idx] + t * (self.L_out[idx + 1] - self.L_out[idx])

    def inverse(self, L_target: torch.Tensor) -> torch.Tensor:
        """Exact inverse — swap input/output breakpoints."""
        clamp_min = self.L_out[0].item()
        clamp_max = self.L_out[-1].item()
        L_clamped = L_target.clamp(clamp_min, clamp_max)
        idx = torch.searchsorted(self.L_out, L_clamped, right=True) - 1
        idx = idx.clamp(0, len(self.L_out) - 2)
        Lo_lo = self.L_out[idx]
        Lo_hi = self.L_out[idx + 1]
        t = ((L_target - Lo_lo) / (Lo_hi - Lo_lo).clamp(min=1e-30)).clamp(0.0, 1.0)
        return self.L_in[idx] + t * (self.L_in[idx + 1] - self.L_in[idx])
