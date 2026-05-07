"""Hue enrichment & rotation stages used by HelmCT.

LGatedHueEnrichment
-------------------
Forward: rotate hue by amp · gate(L) · gauss(h - center, σ)
  gate(L) = sin²(π·(L-L_lo)/(L_hi-L_lo)), 0 outside [L_lo, L_hi]
Inverse: Halley iteration (cubic convergence, 8 iter sufficient for float64).

ChromaPreservingHueRotation
---------------------------
Forward: h' = h + Σᵢ (cᵢ·cos(i·h) + sᵢ·sin(i·h)),  i ∈ {1,2,3}
Inverse: fixed-point iteration (~150 iter for stability across all hues).
"""
import math
import torch


class LGatedHueEnrichment:
    """L-gated hue enrichment, used in Helmgen pipeline.

    Args:
        amp: rotation amplitude (radians)
        center_deg: target hue (degrees, converted to radians internally)
        sigma: gaussian width (radians)
        L_lo, L_hi: gate edges
    """

    def __init__(self, amp: float, center_deg: float, sigma: float,
                 L_lo: float = 0.37, L_hi: float = 1.0):
        self.amp = float(amp)
        self.center = math.radians(center_deg)
        self.sigma = float(sigma)
        self.L_lo = float(L_lo)
        self.L_hi = float(L_hi)

    def _gate(self, L: torch.Tensor) -> torch.Tensor:
        t = ((L - self.L_lo) / (self.L_hi - self.L_lo)).clamp(0.0, 1.0)
        return torch.sin(math.pi * t).pow(2)

    def _wrap(self, dh: torch.Tensor) -> torch.Tensor:
        """Wrap angle difference to (-π, π]."""
        PI = math.pi
        return (dh + PI) % (2.0 * PI) - PI

    def forward(self, L: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        """Apply forward enrichment. Returns (a_new, b_new); L unchanged."""
        C = (a * a + b * b + 1e-30).sqrt()
        h = torch.atan2(b, a)
        gate = self._gate(L)
        dh = self._wrap(h - self.center)
        gauss = torch.exp(-0.5 * (dh / self.sigma).pow(2))
        h_new = h + self.amp * gate * gauss
        return C * torch.cos(h_new), C * torch.sin(h_new)

    def inverse(self, L: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                n_halley: int = 8):
        """Halley iteration to invert h_new → h. Returns (a, b) corrected."""
        C = (a * a + b * b + 1e-30).sqrt()
        h_target = torch.atan2(b, a)
        gate = self._gate(L)
        ag = self.amp * gate
        sig2 = self.sigma * self.sigma

        h = h_target.clone()
        for _ in range(n_halley):
            dh = self._wrap(h - self.center)
            gauss = torch.exp(-0.5 * (dh / self.sigma).pow(2))
            F = h + ag * gauss - h_target
            dg = gauss * (-dh / sig2)
            Fp = 1.0 + ag * dg
            ddg = gauss * (-1.0 / sig2 + dh * dh / (sig2 * sig2))
            Fpp = ag * ddg
            denom = 2.0 * Fp * Fp - F * Fpp
            denom = torch.where(denom.abs() < 1e-30, torch.ones_like(denom), denom)
            h = h - 2.0 * F * Fp / denom
        return C * torch.cos(h), C * torch.sin(h)


class ChromaPreservingHueRotation:
    """h' = h + Σᵢ (cᵢ·cos(i·h) + sᵢ·sin(i·h))  for i ∈ {1, 2, 3}.

    Up to 6 Fourier coefficients; missing ones default to 0. Chroma preserved.
    Inverse via fixed-point iteration.
    """

    def __init__(self, hc: list, n_fixed_point: int = 150):
        # Pad to 6 coefficients
        hc_padded = list(hc) + [0.0] * (6 - len(hc))
        self.c1, self.s1, self.c2, self.s2, self.c3, self.s3 = hc_padded[:6]
        self.is_active = any(abs(x) > 1e-10 for x in hc_padded[:6])
        self.n_fixed_point = n_fixed_point

    def _delta_h(self, h: torch.Tensor) -> torch.Tensor:
        return (
            self.c1 * torch.cos(h) + self.s1 * torch.sin(h)
            + self.c2 * torch.cos(2 * h) + self.s2 * torch.sin(2 * h)
            + self.c3 * torch.cos(3 * h) + self.s3 * torch.sin(3 * h)
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        if not self.is_active:
            return a, b
        C = (a * a + b * b + 1e-30).sqrt()
        h = torch.atan2(b, a)
        h_new = h + self._delta_h(h)
        return C * torch.cos(h_new), C * torch.sin(h_new)

    def inverse(self, a: torch.Tensor, b: torch.Tensor):
        if not self.is_active:
            return a, b
        a_orig, b_orig = a, b
        # Fixed-point: rotate (a_orig, b_orig) by -dh(h_current)
        for _ in range(self.n_fixed_point):
            h = torch.atan2(b, a)
            dh = self._delta_h(h)
            cd, sd = torch.cos(-dh), torch.sin(-dh)
            a = a_orig * cd - b_orig * sd
            b = a_orig * sd + b_orig * cd
        return a, b
