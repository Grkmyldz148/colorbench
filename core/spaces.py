"""Color space adapters — unified GPU tensor interface.

Any color space that implements forward(XYZ→Lab) and inverse(Lab→XYZ)
can be tested. The test engine only sees these two functions.

Built-in: OKLab, CIE Lab. External: GenSpace (via helmlab), Custom.
"""

from abc import ABC, abstractmethod
import os
import torch
import numpy as np

# CIE constants
D65 = torch.tensor([0.95047, 1.0, 1.08883])
M_SRGB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


class ColorSpace(ABC):
    """Protocol for any color space under test."""

    name: str

    @abstractmethod
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """XYZ (N,3) → Lab (N,3). Both float64 on same device."""
        ...

    @abstractmethod
    def inverse(self, lab: torch.Tensor) -> torch.Tensor:
        """Lab (N,3) → XYZ (N,3). Both float64 on same device."""
        ...


def _signed_cbrt(x: torch.Tensor) -> torch.Tensor:
    """Sign-preserving cube root: sign(x) * |x|^(1/3). Bijective, exact inverse."""
    return x.sign() * x.abs().pow(1.0 / 3.0)


class OKLab(ColorSpace):
    """Björn Ottosson's OKLab (2020). Bare M1→cbrt→M2, no enrichment."""

    name = "OKLab"

    def __init__(self, device: torch.device):
        M1_srgb = torch.tensor([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ], device=device, dtype=torch.float64)
        M_srgb_inv = torch.linalg.inv(M_SRGB.to(device))
        self.M1 = M1_srgb @ M_srgb_inv
        self.M2 = torch.tensor([
            [0.2104542553,  0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050,  0.4505937099],
            [0.0259040371,  0.7827717662, -0.8086757660],
        ], device=device, dtype=torch.float64)
        self.M1_inv = torch.linalg.inv(self.M1)
        self.M2_inv = torch.linalg.inv(self.M2)

    def forward(self, xyz):
        lms = xyz @ self.M1.T
        lms_c = _signed_cbrt(lms)
        return lms_c @ self.M2.T

    def inverse(self, lab):
        lms_c = lab @ self.M2_inv.T
        lms = torch.sign(lms_c) * lms_c.abs().pow(3.0)
        return lms @ self.M1_inv.T


class CIELab(ColorSpace):
    """CIE L*a*b* (1976). The classic perceptual space."""

    name = "CIELab"

    def __init__(self, device: torch.device):
        self.d65 = D65.to(device)
        self.delta = 6.0 / 29.0
        self.delta3 = self.delta ** 3

    def forward(self, xyz):
        r = xyz / self.d65
        f = torch.where(r > self.delta3,
                        r.pow(1.0 / 3.0),
                        r / (3 * self.delta ** 2) + 4.0 / 29.0)
        L = 116.0 * f[:, 1] - 16.0
        a = 500.0 * (f[:, 0] - f[:, 1])
        b = 200.0 * (f[:, 1] - f[:, 2])
        return torch.stack([L, a, b], dim=-1)

    def inverse(self, lab):
        fy = (lab[:, 0] + 16.0) / 116.0
        fx = lab[:, 1] / 500.0 + fy
        fz = fy - lab[:, 2] / 200.0
        f = torch.stack([fx, fy, fz], dim=-1)
        xyz = torch.where(f > self.delta,
                          f.pow(3.0),
                          3 * self.delta ** 2 * (f - 4.0 / 29.0))
        return xyz * self.d65


class GenSpaceAdapter(ColorSpace):
    """Adapter for GenSpace JSON (M1/M2/gamma). Pure GPU, no helmlab dependency."""

    def __init__(self, json_path: str, device: torch.device):
        import json as _json
        with open(json_path) as f:
            d = _json.load(f)
        self.name = f"GenSpace({os.path.basename(json_path)})"
        self.M1 = torch.tensor(d["M1"], device=device, dtype=torch.float64)
        self.M2 = torch.tensor(d["M2"], device=device, dtype=torch.float64)
        self.gamma = d.get("gamma", [1/3, 1/3, 1/3])
        self.M1_inv = torch.linalg.inv(self.M1)
        self.M2_inv = torch.linalg.inv(self.M2)
        self.device = device

    def forward(self, xyz):
        lms = xyz @ self.M1.T
        lms_c = _signed_cbrt(lms)
        return lms_c @ self.M2.T

    def inverse(self, lab):
        lms_c = lab @ self.M2_inv.T
        lms = torch.sign(lms_c) * lms_c.abs().pow(3.0)
        return lms @ self.M1_inv.T


class NakaRushtonEnriched(ColorSpace):
    """Naka-Rushton transfer + L correction + L-dep chroma + chroma power."""

    def __init__(self, json_path: str, device: torch.device):
        import json as _json
        with open(json_path) as f:
            d = _json.load(f)
        self.name = f"NR+Enrich({os.path.basename(json_path)})"
        self.M1 = torch.tensor(d["M1"], device=device, dtype=torch.float64)
        self.M2 = torch.tensor(d["M2"], device=device, dtype=torch.float64)
        self.M1_inv = torch.tensor(d["M1_inv"], device=device, dtype=torch.float64)
        self.M2_inv = torch.tensor(d["M2_inv"], device=device, dtype=torch.float64)
        self.n = d["n"]
        self.sigma = d["sigma"]
        self.s_gain = d["s_gain"]
        self.c1 = d["c1"]
        self.k = d["k"]
        self.cp = d["cp"]

    def forward(self, xyz):
        lms = (xyz @ self.M1.T).clamp(min=0)
        x_n = lms.pow(self.n)
        lms_c = self.s_gain * x_n / (x_n + self.sigma ** self.n)
        raw = lms_c @ self.M2.T
        L, a, b = raw[:, 0], raw[:, 1], raw[:, 2]
        L_out = L + self.c1 * L * (1.0 - L)
        C = torch.sqrt(a * a + b * b + 1e-30)
        f_L = torch.exp(self.k * (L - 0.5))
        C_out = f_L * C.pow(self.cp)
        a_out = a / C * C_out
        b_out = b / C * C_out
        return torch.stack([L_out, a_out, b_out], dim=-1)

    def inverse(self, lab):
        L_out, a_out, b_out = lab[:, 0], lab[:, 1], lab[:, 2]
        # Undo L correction
        L = L_out.clone()
        for _ in range(15):
            g = L + self.c1 * L * (1.0 - L) - L_out
            gp = 1.0 + self.c1 * (1.0 - 2.0 * L)
            L = L - g / gp.clamp(min=1e-10)
        # Undo chroma
        C_out = torch.sqrt(a_out ** 2 + b_out ** 2 + 1e-30)
        f_L = torch.exp(self.k * (L - 0.5))
        C_in = (C_out / f_L.clamp(min=1e-30)).clamp(min=0).pow(1.0 / self.cp)
        a_in = a_out / C_out * C_in
        b_in = b_out / C_out * C_in
        raw = torch.stack([L, a_in, b_in], dim=-1)
        lms_c = raw @ self.M2_inv.T
        # NR inverse: x = sigma * (y/(s-y))^(1/n)
        lms_c = lms_c.clamp(min=0)
        lms_c = torch.minimum(lms_c, torch.tensor(self.s_gain - 1e-10))
        ratio = (lms_c / (self.s_gain - lms_c).clamp(min=1e-30)).clamp(min=0)
        lms = self.sigma * ratio.pow(1.0 / self.n)
        return lms @ self.M1_inv.T


class GenSpaceEnriched(ColorSpace):
    """GenSpace with delta (piecewise-linear transfer) + L_corr (cubic L correction)."""

    def __init__(self, json_path: str, device: torch.device):
        import json as _json
        with open(json_path) as f:
            d = _json.load(f)
        self.name = f"GenSpace+Enrich({os.path.basename(json_path)})"
        self.M1 = torch.tensor(d["M1"], device=device, dtype=torch.float64)
        self.M2 = torch.tensor(d["M2"], device=device, dtype=torch.float64)
        self.M1_inv = torch.linalg.inv(self.M1)
        self.M2_inv = torch.linalg.inv(self.M2)
        self.delta = d.get("delta", 0.0)
        self.L_corr = d.get("L_corr", [0.0, 0.0, 0.0])
        self.c1, self.c2, self.c3 = self.L_corr
        self.device = device

    def _transfer(self, x):
        """Piecewise-linear + cbrt transfer (CIE Lab style delta)."""
        d = self.delta
        if d < 1e-10:
            return _signed_cbrt(x)
        d_cbrt = d ** (1.0 / 3.0)
        slope = 1.0 / (3.0 * d ** (2.0 / 3.0))
        offset = 2.0 / 3.0 * d_cbrt
        ax = x.abs()
        cbrt_part = ax.pow(1.0 / 3.0)
        lin_part = slope * ax + offset
        result = torch.where(ax >= d, cbrt_part, lin_part)
        return x.sign() * result

    def _transfer_inv(self, y):
        """Inverse of piecewise-linear + cbrt transfer."""
        d = self.delta
        if d < 1e-10:
            return torch.sign(y) * y.abs().pow(3.0)
        d_cbrt = d ** (1.0 / 3.0)
        slope = 1.0 / (3.0 * d ** (2.0 / 3.0))
        offset = 2.0 / 3.0 * d_cbrt
        ay = y.abs()
        cube_part = ay.pow(3.0)
        lin_part = (ay - offset) / slope
        lin_part = lin_part.clamp(min=0.0)
        result = torch.where(ay >= d_cbrt, cube_part, lin_part)
        return y.sign() * result

    def _L_corr_forward(self, L):
        """Cubic L correction: L' = L + c1*L*(1-L) + c2*L*(1-L)*(2L-1) + c3*L^2*(1-L)^2"""
        L1 = L * (1.0 - L)
        return L + self.c1 * L1 + self.c2 * L1 * (2.0 * L - 1.0) + self.c3 * L * L * (1.0 - L) * (1.0 - L)

    def _L_corr_inverse(self, L_prime, n_iter=50):
        """Newton iteration to invert L correction."""
        L = L_prime.clone()
        for _ in range(n_iter):
            L1 = L * (1.0 - L)
            f = L + self.c1 * L1 + self.c2 * L1 * (2.0 * L - 1.0) + self.c3 * L * L * (1.0 - L) * (1.0 - L) - L_prime
            df = 1.0 + self.c1 * (1.0 - 2.0 * L) + self.c2 * (6.0 * L * L - 6.0 * L + 1.0) + self.c3 * 2.0 * L * (1.0 - L) * (1.0 - 2.0 * L)
            L = L - f / df.clamp(min=1e-12)
        return L

    def forward(self, xyz):
        lms = xyz @ self.M1.T
        lms_c = self._transfer(lms)
        raw = lms_c @ self.M2.T
        if abs(self.c1) > 1e-15 or abs(self.c2) > 1e-15 or abs(self.c3) > 1e-15:
            L = raw[:, 0]
            L_out = self._L_corr_forward(L)
            return torch.stack([L_out, raw[:, 1], raw[:, 2]], dim=-1)
        return raw

    def inverse(self, lab):
        if abs(self.c1) > 1e-15 or abs(self.c2) > 1e-15 or abs(self.c3) > 1e-15:
            L_out = lab[:, 0]
            L = self._L_corr_inverse(L_out)
            raw = torch.stack([L, lab[:, 1], lab[:, 2]], dim=-1)
        else:
            raw = lab
        lms_c = raw @ self.M2_inv.T
        lms = self._transfer_inv(lms_c)
        return lms @ self.M1_inv.T


class GenSpaceBlueFix(ColorSpace):
    """GenSpace+Enrich with post-M2 C-proportional directional ab-offset for blue fix.

    Post-processing after L_corr:
      dir = (dir_a, dir_b) — optimal direction for sky-blue (from Jacobian analysis)
      f = k * C * max(1-L, 0) * gauss(hue - center, sigma)
      a' = a + f * dir_a
      b' = b + f * dir_b

    Properties: achromatic-safe (C=0 → no effect), L-dependent (white untouched),
    hue-targeted (only blue region), invertible (iterative).
    """

    def __init__(self, json_path: str, device: torch.device):
        import json as _json
        with open(json_path) as f:
            d = _json.load(f)
        self.name = f"GenSpace+BlueFix({os.path.basename(json_path)})"
        self.M1 = torch.tensor(d["M1"], device=device, dtype=torch.float64)
        self.M2 = torch.tensor(d["M2"], device=device, dtype=torch.float64)
        self.M1_inv = torch.linalg.inv(self.M1)
        self.M2_inv = torch.linalg.inv(self.M2)
        self.delta = d.get("delta", 0.0)
        self.L_corr = d.get("L_corr", [0.0, 0.0, 0.0])
        self.c1, self.c2, self.c3 = self.L_corr
        # Blue-fix params
        bf = d.get("blue_fix", {})
        self.bf_k = bf.get("k", 0.0)
        self.bf_sigma = bf.get("sigma", 30.0)
        self.bf_center = bf.get("center", 240.0)
        self.bf_dir_a = bf.get("dir_a", -0.8813)
        self.bf_dir_b = bf.get("dir_b", 0.4725)
        self.device = device

    def _transfer(self, x):
        d = self.delta
        if d < 1e-10:
            return _signed_cbrt(x)
        d_cbrt = d ** (1.0 / 3.0)
        slope = 1.0 / (3.0 * d ** (2.0 / 3.0))
        offset = 2.0 / 3.0 * d_cbrt
        ax = x.abs()
        cbrt_part = ax.pow(1.0 / 3.0)
        lin_part = slope * ax + offset
        result = torch.where(ax >= d, cbrt_part, lin_part)
        return x.sign() * result

    def _transfer_inv(self, y):
        d = self.delta
        if d < 1e-10:
            return torch.sign(y) * y.abs().pow(3.0)
        d_cbrt = d ** (1.0 / 3.0)
        slope = 1.0 / (3.0 * d ** (2.0 / 3.0))
        offset = 2.0 / 3.0 * d_cbrt
        ay = y.abs()
        cube_part = ay.pow(3.0)
        lin_part = (ay - offset) / slope
        lin_part = lin_part.clamp(min=0.0)
        result = torch.where(ay >= d_cbrt, cube_part, lin_part)
        return y.sign() * result

    def _L_corr_forward(self, L):
        L1 = L * (1.0 - L)
        return L + self.c1 * L1 + self.c2 * L1 * (2.0 * L - 1.0) + self.c3 * L * L * (1.0 - L) * (1.0 - L)

    def _L_corr_inverse(self, L_prime, n_iter=50):
        L = L_prime.clone()
        for _ in range(n_iter):
            L1 = L * (1.0 - L)
            f = L + self.c1 * L1 + self.c2 * L1 * (2.0 * L - 1.0) + self.c3 * L * L * (1.0 - L) * (1.0 - L) - L_prime
            df = 1.0 + self.c1 * (1.0 - 2.0 * L) + self.c2 * (6.0 * L * L - 6.0 * L + 1.0) + self.c3 * 2.0 * L * (1.0 - L) * (1.0 - 2.0 * L)
            L = L - f / df.clamp(min=1e-12)
        return L

    def _gauss_hue(self, theta_deg):
        d = (theta_deg - self.bf_center + 180.0) % 360.0 - 180.0
        return torch.exp(-d * d / (2.0 * self.bf_sigma ** 2))

    def _blue_fix_forward(self, lab):
        """Apply C-proportional directional ab-offset."""
        if self.bf_k < 1e-10:
            return lab
        L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
        C = torch.sqrt(a * a + b * b + 1e-30)
        theta = torch.atan2(b, a)
        theta_deg = (torch.rad2deg(theta)) % 360.0
        w = self._gauss_hue(theta_deg)
        f = self.bf_k * C * torch.clamp(1.0 - L, min=0.0) * w
        a_new = a + f * self.bf_dir_a
        b_new = b + f * self.bf_dir_b
        return torch.stack([L, a_new, b_new], dim=-1)

    def _blue_fix_inverse(self, lab):
        """Bisection on scalar f: solve g(f) = f - kL*C(f)*w(f) = 0."""
        if self.bf_k < 1e-10:
            return lab
        L = lab[:, 0]
        a_pp = lab[:, 1]
        b_pp = lab[:, 2]
        k = self.bf_k
        da_dir = self.bf_dir_a
        db_dir = self.bf_dir_b
        one_m_L = torch.clamp(1.0 - L, min=0.0)
        kL = k * one_m_L

        def eval_g(fv):
            a = a_pp - fv * da_dir
            b = b_pp - fv * db_dir
            C = torch.sqrt(a * a + b * b + 1e-30)
            theta_deg = (torch.rad2deg(torch.atan2(b, a))) % 360.0
            w = self._gauss_hue(theta_deg)
            return fv - kL * C * w

        # Bounds: f_lo=0 (g<0 for active pixels), f_hi = kL * C_input (generous upper)
        C_input = torch.sqrt(a_pp * a_pp + b_pp * b_pp + 1e-30)
        f_lo = torch.zeros_like(L)
        f_hi = kL * C_input * 2.0 + 0.01  # generous upper bound

        # Bisection: 60 iterations → 2^-60 ≈ 1e-18 precision
        for _ in range(60):
            f_mid = 0.5 * (f_lo + f_hi)
            g_mid = eval_g(f_mid)
            # g is monotonically increasing: g(f_lo) < 0, g(f_hi) > 0
            f_lo = torch.where(g_mid < 0, f_mid, f_lo)
            f_hi = torch.where(g_mid >= 0, f_mid, f_hi)

        f = 0.5 * (f_lo + f_hi)
        a = a_pp - f * da_dir
        b = b_pp - f * db_dir
        return torch.stack([L, a, b], dim=-1)

    def forward(self, xyz):
        lms = xyz @ self.M1.T
        lms_c = self._transfer(lms)
        raw = lms_c @ self.M2.T
        if abs(self.c1) > 1e-15 or abs(self.c2) > 1e-15 or abs(self.c3) > 1e-15:
            L = raw[:, 0]
            L_out = self._L_corr_forward(L)
            lab = torch.stack([L_out, raw[:, 1], raw[:, 2]], dim=-1)
        else:
            lab = raw
        return self._blue_fix_forward(lab)

    def inverse(self, lab):
        lab = self._blue_fix_inverse(lab)
        if abs(self.c1) > 1e-15 or abs(self.c2) > 1e-15 or abs(self.c3) > 1e-15:
            L_out = lab[:, 0]
            L = self._L_corr_inverse(L_out)
            raw = torch.stack([L, lab[:, 1], lab[:, 2]], dim=-1)
        else:
            raw = lab
        lms_c = raw @ self.M2_inv.T
        lms = self._transfer_inv(lms_c)
        return lms @ self.M1_inv.T


class CustomSpace(ColorSpace):
    """Wrap any forward/inverse callables as a ColorSpace."""

    def __init__(self, name: str, forward_fn, inverse_fn):
        self.name = name
        self._fwd = forward_fn
        self._inv = inverse_fn

    def forward(self, xyz):
        return self._fwd(xyz)

    def inverse(self, lab):
        return self._inv(lab)
