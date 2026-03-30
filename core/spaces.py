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
D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)
M_SRGB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float64)


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


class NonlinearM1(ColorSpace):
    """Nonlinear M1 with blue-selective cross term: lms[0] += d*(1-Y)*Z.

    Keeps v7b's gamut geometry while fixing blue-white gradient.
    Inverse: Newton iteration with analytical Jacobian.
    """

    def __init__(self, json_path, device, label=None):
        import json as _json
        with open(json_path) as f:
            d = _json.load(f)
        self.name = label or f"NonlinearM1({os.path.basename(json_path)})"
        dev = device
        self._M1 = torch.tensor(d["M1"], dtype=torch.float64, device=dev)
        self._M2 = torch.tensor(d["M2"], dtype=torch.float64, device=dev)
        self._M1_inv = torch.linalg.inv(self._M1)
        self._M2_inv = torch.linalg.inv(self._M2)
        self._d = d.get("cross_term_d", 0.0)
        self._k_ach = d.get("cross_term_k", 0.0)
        lc = d.get("L_corr", [0, 0, 0])
        self._lc = torch.tensor(lc, dtype=torch.float64, device=dev)
        self._has_lc = any(abs(x) > 1e-10 for x in lc)

    def _fwd_lms(self, xyz):
        """XYZ → LMS with cross term."""
        lms = xyz @ self._M1.T
        if self._d != 0:
            if self._k_ach != 0:
                cross = self._d * (xyz[:, 2] - self._k_ach * xyz[:, 1])
            else:
                cross = self._d * (1.0 - xyz[:, 1]) * xyz[:, 2]
            lms = lms.clone()
            lms[:, 0] = lms[:, 0] + cross
        return lms

    def _inv_lms(self, lms_target):
        """LMS → XYZ via Newton iteration."""
        xyz = lms_target @ self._M1_inv.T  # initial guess
        for _ in range(50):
            lms = xyz @ self._M1.T
            if self._d != 0:
                if self._k_ach != 0:
                    cross = self._d * (xyz[:, 2] - self._k_ach * xyz[:, 1])
                else:
                    cross = self._d * (1.0 - xyz[:, 1]) * xyz[:, 2]
                lms = lms.clone()
                lms[:, 0] = lms[:, 0] + cross
            err = lms - lms_target  # (N, 3)

            # Jacobian per sample (batch Newton)
            J = self._M1.unsqueeze(0).expand(xyz.shape[0], -1, -1).clone()
            if self._d != 0:
                if self._k_ach != 0:
                    J[:, 0, 1] = J[:, 0, 1] + (-self._d * self._k_ach)
                    J[:, 0, 2] = J[:, 0, 2] + self._d
                else:
                    J[:, 0, 1] = J[:, 0, 1] + (-self._d * xyz[:, 2])
                    J[:, 0, 2] = J[:, 0, 2] + (self._d * (1.0 - xyz[:, 1]))

            # Solve J @ dx = err → dx = J^-1 @ err
            dx = torch.linalg.solve(J, err.unsqueeze(-1)).squeeze(-1)
            xyz = xyz - dx
        return xyz

    def forward(self, xyz):
        lms = self._fwd_lms(xyz)
        lms_c = torch.sign(lms) * lms.abs().clamp(min=1e-30).pow(1.0 / 3.0)
        lab = lms_c @ self._M2.T
        if self._has_lc:
            L = lab[:, 0:1]
            c1, c2, c3 = self._lc[0], self._lc[1], self._lc[2]
            t = L * (1.0 - L)
            L_new = L + c1*t + c2*t*(2.0*L - 1.0) + c3*L**2*(1.0 - L)**2
            lab = torch.cat([L_new, lab[:, 1:2], lab[:, 2:3]], dim=1)
        return lab

    def inverse(self, lab):
        lab = lab.clone()
        if self._has_lc:
            L1 = lab[:, 0:1]
            L = L1.clone()
            c1, c2, c3 = self._lc[0], self._lc[1], self._lc[2]
            for _ in range(15):
                t = L * (1.0 - L)
                f = L + c1*t + c2*t*(2*L-1) + c3*L**2*(1-L)**2 - L1
                df = 1 + c1*(1-2*L) + c2*(6*L**2-6*L+1) + c3*2*L*(1-L)*(1-2*L)
                L = L - f / df.clamp(min=1e-12)
            lab = torch.cat([L, lab[:, 1:2], lab[:, 2:3]], dim=1)
        lms_c = lab @ self._M2_inv.T
        lms = torch.sign(lms_c) * lms_c.abs().pow(3.0)
        return self._inv_lms(lms)


class HelmCT(ColorSpace):
    """Helmlab cross-term pipeline with chroma-preserving hue rotation + L_corr7.

    Pipeline: M1 → cross_term(d,k) → cbrt → M2 → chroma_hue_rot → L_corr7 → Lab
    Inverse: L_corr7(Newton) → hue_rot(fixed-point) → M2_inv → cube → cross_term_inv(analytical) → M1_inv
    """

    def __init__(self, json_path, device, label=None):
        import json as _json
        with open(json_path) as f:
            d = _json.load(f)
        self.name = label or f"HelmCT({os.path.basename(json_path)})"
        dev = device
        self._M1 = torch.tensor(d["M1"], dtype=torch.float64, device=dev)
        self._M2 = torch.tensor(d["M2"], dtype=torch.float64, device=dev)
        self._M2_inv = torch.linalg.inv(self._M2)

        # Cross-term L: lms[0] += d*(Z - k*Y)
        self._d = d.get("cross_d", 0.0)
        self._k = d.get("cross_k", 0.0)
        # Cross-term M: lms[1] += d2*(Z - k2*X)
        self._d2 = d.get("cross_d2", 0.0)
        self._k2 = d.get("cross_k2", 0.0)
        # Cross-term S: lms[2] += d3*(X - k3*Y)
        self._d3 = d.get("cross_d3", 0.0)
        self._k3 = d.get("cross_k3", 0.0)

        # Analytical inverse for cross-terms: modify M1
        self._M1_mod = self._M1.clone()
        if self._d != 0:
            self._M1_mod[0, 1] = self._M1_mod[0, 1] - self._d * self._k
            self._M1_mod[0, 2] = self._M1_mod[0, 2] + self._d
        if self._d2 != 0:
            self._M1_mod[1, 0] = self._M1_mod[1, 0] - self._d2 * self._k2
            self._M1_mod[1, 2] = self._M1_mod[1, 2] + self._d2
        if self._d3 != 0:
            self._M1_mod[2, 0] = self._M1_mod[2, 0] + self._d3
            self._M1_mod[2, 1] = self._M1_mod[2, 1] - self._d3 * self._k3
        self._M1_mod_inv = torch.linalg.inv(self._M1_mod)

        # Transfer function (cbrt default, or naka_rushton, power, cielab_delta)
        self._transfer = d.get("transfer", "cbrt")
        if self._transfer == "naka_rushton":
            self._nr_n = d.get("nr_n", 0.76)
            self._nr_sigma = d.get("nr_sigma", 0.33)
            self._nr_s = d.get("nr_s", 0.71)
        elif self._transfer == "power":
            self._gamma_val = d.get("gamma_val", 1.0/3.0)
        elif self._transfer == "cielab_delta":
            self._cielab_delta = d.get("cielab_delta", 0.008856)
            self._cielab_kappa = d.get("cielab_kappa", 903.3)
        elif self._transfer == "softcbrt":
            self._softcbrt_eps = d.get("softcbrt_eps", 0.001)

        # Chroma-preserving hue rotation (Fourier 4 or 6)
        # Support both formats: "hue_correction" list OR individual "hue_cos1/sin1" fields
        hc = d.get("hue_correction", None)
        if hc is None:
            hc = [d.get("hue_cos1", 0), d.get("hue_sin1", 0),
                  d.get("hue_cos2", 0), d.get("hue_sin2", 0),
                  d.get("hue_cos3", 0), d.get("hue_sin3", 0)]
        while len(hc) < 6:
            hc.append(0.0)
        self._hc = hc
        self._has_hc = any(abs(x) > 1e-10 for x in hc)

        # L_corr degree 7
        lc7 = d.get("L_corr_7", None)
        if lc7 is not None:
            self._lc7 = torch.tensor(lc7, dtype=torch.float64, device=dev)
            self._has_lc7 = True
        else:
            self._lc7 = None
            self._has_lc7 = False

        # L_corr degree 5
        lc5 = d.get("L_corr_5", None)
        if lc5 is not None and not self._has_lc7:
            self._lc5 = torch.tensor(lc5, dtype=torch.float64, device=dev)
            self._has_lc5 = True
        else:
            self._lc5 = None
            self._has_lc5 = False

        # Piecewise-linear L_corr (analytically invertible!)
        lcpw = d.get("L_corr_pw", None)
        if lcpw is not None and not self._has_lc7 and not self._has_lc5:
            # lcpw = shifts at interior breakpoints
            # step size from JSON or default 1/(len+1)
            n = len(lcpw)
            step = d.get("L_corr_pw_step", 1.0 / (n + 1))
            shifts = [0.0] + list(lcpw) + [0.0]
            breakpoints = [i * step for i in range(n + 2)]
            # Ensure last breakpoint is 1.0
            breakpoints[-1] = 1.0
            # L_out breakpoints = L + shift
            self._pw_L_in = torch.tensor(breakpoints, dtype=torch.float64, device=dev)
            self._pw_L_out = torch.tensor([b + s for b, s in zip(breakpoints, shifts)],
                                           dtype=torch.float64, device=dev)
            self._has_pw = True
        else:
            self._has_pw = False

        # L_corr degree 3 fallback
        lc = d.get("L_corr", [0, 0, 0])
        self._lc = torch.tensor(lc, dtype=torch.float64, device=dev)
        self._has_lc = (any(abs(x) > 1e-10 for x in lc) and
                        not self._has_lc7 and not self._has_lc5 and not self._has_pw)

        # Chroma power + L-dependent chroma scaling
        self._cp = d.get("chroma_power", 1.0)
        self._ck = d.get("chroma_k", 0.0)
        # Chroma-aware cp: smooth transition cp=1.0 at C=0, cp=_cp at high C
        # cp_delta controls transition sharpness (higher = sharper)
        self._cp_delta = d.get("chroma_power_delta", 0.0)  # 0 = classic (uniform cp)
        self._has_cp = (abs(self._cp - 1.0) > 1e-10 or abs(self._ck) > 1e-10
                        or abs(self._cp_delta) > 1e-10)

        # Ab-axis scaling: a *= s_a, b *= s_b (anisotropic)
        self._sa = d.get("scale_a", 1.0)
        self._sb = d.get("scale_b", 1.0)
        self._has_ab_scale = abs(self._sa - 1.0) > 1e-10 or abs(self._sb - 1.0) > 1e-10

        # Hue-dependent L correction: L -= delta * max(0, cos(h - center))^width
        self._hue_L_delta = d.get("hue_L_delta", 0.0)
        self._hue_L_center = d.get("hue_L_center", -1.5708)  # -π/2 = blue
        self._hue_L_width = d.get("hue_L_width", 2.0)
        self._has_hue_L = abs(self._hue_L_delta) > 1e-10

    def _pw_forward(self, L):
        """Piecewise-linear L correction: L_out = L + interp(shifts, L)."""
        # For each input L, find segment and interpolate
        L_in = self._pw_L_in   # [0, 0.1, 0.2, ..., 1.0]
        L_out = self._pw_L_out  # [0+s0, 0.1+s1, ...]
        # Use torch searchsorted for vectorized segment finding
        idx = torch.searchsorted(L_in, L.clamp(0, 1), right=True) - 1
        idx = idx.clamp(0, len(L_in) - 2)
        # Linear interpolation within segment
        L_lo = L_in[idx]
        L_hi = L_in[idx + 1]
        t = (L - L_lo) / (L_hi - L_lo).clamp(min=1e-30)
        t = t.clamp(0, 1)
        return L_out[idx] + t * (L_out[idx + 1] - L_out[idx])

    def _pw_inverse(self, L_target):
        """Exact inverse of piecewise-linear L correction."""
        L_out = self._pw_L_out  # output breakpoints (monotonically increasing)
        L_in = self._pw_L_in    # input breakpoints
        # Find segment in OUTPUT space
        idx = torch.searchsorted(L_out, L_target.clamp(L_out[0], L_out[-1]), right=True) - 1
        idx = idx.clamp(0, len(L_out) - 2)
        # Linear interpolation in reverse
        Lo_lo = L_out[idx]
        Lo_hi = L_out[idx + 1]
        t = (L_target - Lo_lo) / (Lo_hi - Lo_lo).clamp(min=1e-30)
        t = t.clamp(0, 1)
        return L_in[idx] + t * (L_in[idx + 1] - L_in[idx])

    def _apply_lc7(self, L):
        p = self._lc7
        t = L * (1.0 - L)
        h = 0.5 - L
        return L + p[0]*t + p[1]*t*h + p[2]*t*t + p[3]*t*t*h + p[4]*t*t*t + p[5]*t*t*t*h + p[6]*t*t*t*t

    def _apply_lc7_deriv(self, L):
        """Derivative of L_corr7 for Newton iteration."""
        p = self._lc7
        t = L * (1.0 - L)
        dt = 1.0 - 2.0 * L
        h = 0.5 - L
        dh = -1.0
        t2 = t * t
        t3 = t2 * t
        return (1.0
                + p[0] * dt
                + p[1] * (dt * h + t * dh)
                + p[2] * 2 * t * dt
                + p[3] * (2 * t * dt * h + t2 * dh)
                + p[4] * 3 * t2 * dt
                + p[5] * (3 * t2 * dt * h + t3 * dh)
                + p[6] * 4 * t3 * dt)

    def forward(self, xyz):
        # 1. M1
        lms = xyz @ self._M1.T
        # 2. Cross-terms
        if self._d != 0 or self._d2 != 0 or self._d3 != 0:
            lms = lms.clone()
            if self._d != 0:
                lms[:, 0] = lms[:, 0] + self._d * (xyz[:, 2] - self._k * xyz[:, 1])
            if self._d2 != 0:
                lms[:, 1] = lms[:, 1] + self._d2 * (xyz[:, 2] - self._k2 * xyz[:, 0])
            if self._d3 != 0:
                lms[:, 2] = lms[:, 2] + self._d3 * (xyz[:, 0] - self._k3 * xyz[:, 1])
        # 3. Transfer function
        if self._transfer == "naka_rushton":
            ax = torch.abs(lms).clamp(min=1e-30)
            lms_c = torch.sign(lms) * self._nr_s * ax.pow(self._nr_n) / (ax.pow(self._nr_n) + self._nr_sigma ** self._nr_n)
        elif self._transfer == "power":
            gamma = self._gamma_val
            lms_c = torch.sign(lms) * torch.abs(lms).pow(gamma)
        elif self._transfer == "cielab_delta":
            delta = self._cielab_delta
            kappa = self._cielab_kappa
            ax = torch.abs(lms)
            cbrt_val = ax.pow(1.0 / 3.0)
            lin_val = (kappa * ax + 16.0) / 116.0
            lms_c = torch.sign(lms) * torch.where(ax > delta, cbrt_val, lin_val)
        elif self._transfer == "softcbrt":
            eps = self._softcbrt_eps
            ax = torch.abs(lms)
            lms_c = torch.sign(lms) * ((ax + eps).pow(1.0 / 3.0) - eps ** (1.0 / 3.0))
        else:
            lms_c = torch.sign(lms) * torch.abs(lms).pow(1.0 / 3.0)
        # 4. M2
        raw = lms_c @ self._M2.T
        L, a, b = raw[:, 0], raw[:, 1], raw[:, 2]
        # 5. Chroma-preserving hue rotation
        if self._has_hc:
            C = torch.sqrt(a * a + b * b + 1e-30)
            h = torch.atan2(b, a)
            c1, s1, c2, s2, c3, s3 = self._hc
            dh = (c1 * torch.cos(h) + s1 * torch.sin(h) +
                  c2 * torch.cos(2*h) + s2 * torch.sin(2*h) +
                  c3 * torch.cos(3*h) + s3 * torch.sin(3*h))
            h_new = h + dh
            a = C * torch.cos(h_new)
            b = C * torch.sin(h_new)
        # 5b. Chroma power + L-dependent scaling
        if self._has_cp:
            C = torch.sqrt(a * a + b * b + 1e-30)
            scale = torch.ones_like(C)
            if abs(self._cp - 1.0) > 1e-10:
                if abs(self._cp_delta) > 1e-10:
                    # Chroma-aware: effective_cp smoothly transitions from 1.0 (at C=0) to cp (at high C)
                    # blend = C^2 / (C^2 + delta^2) → 0 at C=0, 1 at C>>delta
                    blend = C * C / (C * C + self._cp_delta * self._cp_delta)
                    effective_cp = 1.0 + blend * (self._cp - 1.0)
                    scale = scale * C.pow(effective_cp - 1.0)
                else:
                    scale = scale * C.pow(self._cp - 1.0)
            if abs(self._ck) > 1e-10:
                scale = scale * torch.exp(self._ck * (L - 0.5))
            a = a * scale
            b = b * scale
        # 5d. Ab-axis scaling
        if self._has_ab_scale:
            a = a * self._sa
            b = b * self._sb
        # 5e. Hue-dependent L correction
        if self._has_hue_L:
            h_cur = torch.atan2(b, a)
            cos_diff = torch.cos(h_cur - self._hue_L_center)
            weight = cos_diff.clamp(min=0).pow(self._hue_L_width)
            L = L - self._hue_L_delta * weight * L * (1.0 - L)
        # 6. L_corr
        if self._has_pw:
            # Piecewise-linear: exact forward
            L = self._pw_forward(L)
        elif self._has_lc7:
            L = self._apply_lc7(L)
        elif self._has_lc5:
            p = self._lc5
            t = L * (1.0 - L)
            h5 = 0.5 - L
            L = L + p[0]*t + p[1]*t*h5 + p[2]*t*t + p[3]*t*t*h5 + p[4]*t*t*t
        elif self._has_lc:
            t = L * (1.0 - L)
            L = L + self._lc[0]*t + self._lc[1]*t*(2*L-1) + self._lc[2]*t*t
        return torch.stack([L, a, b], dim=-1)

    def inverse(self, lab):
        L_out, a, b = lab[:, 0].clone(), lab[:, 1].clone(), lab[:, 2].clone()

        # 6. Undo L_corr
        L = L_out.clone()
        if self._has_pw:
            # Piecewise-linear: exact inverse (no Newton!)
            L = self._pw_inverse(L_out)
        elif self._has_lc7:
            for _ in range(30):
                f = self._apply_lc7(L) - L_out
                df = self._apply_lc7_deriv(L)
                df = torch.where(df.abs() < 1e-12, torch.ones_like(df), df)
                L = L - f / df
        elif self._has_lc5:
            p = self._lc5
            for _ in range(20):
                t = L * (1.0 - L)
                h5 = 0.5 - L
                dt = 1.0 - 2.0 * L
                dh5 = -1.0
                f = L + p[0]*t + p[1]*t*h5 + p[2]*t*t + p[3]*t*t*h5 + p[4]*t*t*t - L_out
                t2 = t * t
                df = (1.0 + p[0]*dt + p[1]*(dt*h5+t*dh5) + p[2]*2*t*dt +
                      p[3]*(2*t*dt*h5+t2*dh5) + p[4]*3*t2*dt)
                df = torch.where(df.abs() < 1e-12, torch.ones_like(df), df)
                L = L - f / df
        elif self._has_lc:
            c1, c2, c3 = self._lc[0], self._lc[1], self._lc[2]
            for _ in range(15):
                t = L * (1.0 - L)
                f = L + c1*t + c2*t*(2*L-1) + c3*t*t - L_out
                dt = 1.0 - 2.0 * L
                df = 1 + c1*dt + c2*(dt*(2*L-1)+t*2) + c3*2*t*dt
                df = torch.where(df.abs() < 1e-12, torch.ones_like(df), df)
                L = L - f / df

        # 5e. Undo hue-dependent L correction (Newton)
        if self._has_hue_L:
            h_cur = torch.atan2(b, a)
            cos_diff = torch.cos(h_cur - self._hue_L_center)
            weight = cos_diff.clamp(min=0).pow(self._hue_L_width)
            L_target = L.clone()
            for _ in range(20):
                f = L - self._hue_L_delta * weight * L * (1.0 - L) - L_target
                df = 1.0 - self._hue_L_delta * weight * (1.0 - 2.0 * L)
                df = torch.where(df.abs() < 1e-12, torch.ones_like(df), df)
                L = L - f / df
        # 5c. Undo ab-axis scaling
        if self._has_ab_scale:
            a = a / self._sa
            b = b / self._sb
        # 5b. Undo chroma power + L-dependent scaling
        if self._has_cp:
            C = torch.sqrt(a * a + b * b + 1e-30)
            inv_scale = torch.ones_like(C)
            if abs(self._ck) > 1e-10:
                inv_scale = inv_scale * torch.exp(-self._ck * (L - 0.5))
            if abs(self._cp - 1.0) > 1e-10:
                C_after_k = C * inv_scale
                if abs(self._cp_delta) > 1e-10:
                    # Chroma-aware inverse: Newton iteration
                    # Forward: C_new = C_orig * C_orig^(blend(C_orig)*(cp-1))
                    # where blend(x) = x^2/(x^2+delta^2)
                    C_orig = C_after_k.pow(1.0 / self._cp)  # initial guess
                    delta2 = self._cp_delta * self._cp_delta
                    for _ in range(20):
                        bl = C_orig * C_orig / (C_orig * C_orig + delta2)
                        eff_cp = 1.0 + bl * (self._cp - 1.0)
                        f_val = C_orig.pow(eff_cp) - C_after_k
                        # derivative: d/dC [C^(eff_cp)] = eff_cp * C^(eff_cp-1) + C^eff_cp * ln(C) * d(eff_cp)/dC
                        # d(bl)/dC = 2*C*delta^2 / (C^2+delta^2)^2
                        dbl = 2.0 * C_orig * delta2 / (C_orig * C_orig + delta2).pow(2)
                        deff = dbl * (self._cp - 1.0)
                        log_C = torch.log(C_orig.clamp(min=1e-30))
                        df_val = eff_cp * C_orig.pow(eff_cp - 1.0) + C_orig.pow(eff_cp) * log_C * deff
                        df_val = torch.where(df_val.abs() < 1e-15, torch.ones_like(df_val), df_val)
                        C_orig = C_orig - f_val / df_val
                        C_orig = C_orig.clamp(min=0)
                    inv_scale = C_orig / C.clamp(min=1e-30)
                else:
                    C_orig = C_after_k.pow(1.0 / self._cp)
                    inv_scale = C_orig / C.clamp(min=1e-30)
            a = a * inv_scale
            b = b * inv_scale

        # 5. Undo chroma-preserving hue rotation (fixed-point iteration)
        if self._has_hc:
            c1, s1, c2, s2, c3, s3 = self._hc
            a_orig, b_orig = a.clone(), b.clone()
            for _ in range(150):
                h = torch.atan2(b, a)
                dh = (c1 * torch.cos(h) + s1 * torch.sin(h) +
                      c2 * torch.cos(2*h) + s2 * torch.sin(2*h) +
                      c3 * torch.cos(3*h) + s3 * torch.sin(3*h))
                cd, sd = torch.cos(-dh), torch.sin(-dh)
                a = a_orig * cd - b_orig * sd
                b = a_orig * sd + b_orig * cd

        # 4. Undo M2
        raw = torch.stack([L, a, b], dim=-1)
        lms_c = raw @ self._M2_inv.T
        # 3. Undo transfer function
        if self._transfer == "naka_rushton":
            ax = torch.abs(lms_c).clamp(min=1e-30)
            ratio = ax / (self._nr_s - ax).clamp(min=1e-30)
            lms = torch.sign(lms_c) * self._nr_sigma * ratio.pow(1.0 / self._nr_n)
        elif self._transfer == "power":
            gamma = self._gamma_val
            lms = torch.sign(lms_c) * lms_c.abs().pow(1.0 / gamma)
        elif self._transfer == "cielab_delta":
            delta = self._cielab_delta
            kappa = self._cielab_kappa
            ax = torch.abs(lms_c)
            cube_val = ax.pow(3.0)
            lin_val = (116.0 * ax - 16.0) / kappa
            f_delta = (kappa * delta + 16.0) / 116.0
            lms = torch.sign(lms_c) * torch.where(ax > f_delta, cube_val, lin_val)
        elif self._transfer == "softcbrt":
            eps = self._softcbrt_eps
            eps_cbrt = eps ** (1.0 / 3.0)
            ax = torch.abs(lms_c)
            # inverse: x = (y + eps^(1/3))^3 - eps
            lms = torch.sign(lms_c) * ((ax + eps_cbrt).pow(3.0) - eps)
        else:
            lms = torch.sign(lms_c) * lms_c.abs().pow(3.0)
        # 2+1. Undo cross-term + M1 (analytical)
        xyz = lms @ self._M1_mod_inv.T
        return xyz


class HueDep(ColorSpace):
    """Hue-dependent M2: fixed L row + Fourier-rotated a,b rows + L_corr."""

    def __init__(self, json_path, device, label=None):
        import json as _json, math
        with open(json_path) as f:
            d = _json.load(f)
        self.name = label or f"HueDep({os.path.basename(json_path)})"
        dev = device
        self._M1 = torch.tensor(d["M1"], dtype=torch.float64, device=dev)
        self._M1_inv = torch.linalg.inv(self._M1)
        # M2 rows
        m2f = d["M2_full"]
        self._M2_L = torch.tensor(m2f[0], dtype=torch.float64, device=dev)
        self._M2_a = torch.tensor(m2f[1], dtype=torch.float64, device=dev)
        self._M2_b = torch.tensor(m2f[2], dtype=torch.float64, device=dev)
        # Full M2 and inverse for round-trip
        self._M2 = torch.tensor(m2f, dtype=torch.float64, device=dev)
        self._M2_inv = torch.linalg.inv(self._M2)
        # Fourier rotation
        rf = d["rotation_fourier"]
        self._rc1 = rf["c1"]
        self._rs1 = rf["s1"]
        self._rc2 = rf["c2"]
        self._rs2 = rf["s2"]
        # L_corr
        lc = d.get("L_corr", [0, 0, 0])
        self._lc = torch.tensor(lc, dtype=torch.float64, device=dev)
        self._has_lc = any(abs(x) > 1e-10 for x in lc)

    def _rotation_angle(self, h):
        return (self._rc1 * torch.cos(h) + self._rs1 * torch.sin(h) +
                self._rc2 * torch.cos(2 * h) + self._rs2 * torch.sin(2 * h))

    def forward(self, xyz):
        lms = (xyz @ self._M1.T).clamp(min=0)
        lms_c = torch.sign(lms) * lms.abs().pow(1.0 / 3.0)
        L = lms_c @ self._M2_L
        a_raw = lms_c @ self._M2_a
        b_raw = lms_c @ self._M2_b
        # Hue-dependent rotation
        h = torch.atan2(b_raw, a_raw)
        theta = self._rotation_angle(h)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        a = a_raw * cos_t - b_raw * sin_t
        b = a_raw * sin_t + b_raw * cos_t
        lab = torch.stack([L, a, b], dim=-1)
        # L_corr
        if self._has_lc:
            Lv = lab[:, 0:1]
            c1, c2, c3 = self._lc[0], self._lc[1], self._lc[2]
            t = Lv * (1.0 - Lv)
            L_new = Lv + c1 * t + c2 * t * (2.0 * Lv - 1.0) + c3 * Lv ** 2 * (1.0 - Lv) ** 2
            lab = torch.cat([L_new, lab[:, 1:2], lab[:, 2:3]], dim=1)
        return lab

    def inverse(self, lab):
        lab = lab.clone()
        # Undo L_corr
        if self._has_lc:
            L1 = lab[:, 0:1]
            L = L1.clone()
            c1, c2, c3 = self._lc[0], self._lc[1], self._lc[2]
            for _ in range(15):
                t = L * (1.0 - L)
                f = L + c1 * t + c2 * t * (2 * L - 1) + c3 * L ** 2 * (1 - L) ** 2 - L1
                df = 1.0 + c1 * (1 - 2 * L) + c2 * (6 * L ** 2 - 6 * L + 1) + c3 * 2 * L * (1 - L) * (1 - 2 * L)
                L = L - f / df.clamp(min=1e-12)
            lab = torch.cat([L, lab[:, 1:2], lab[:, 2:3]], dim=1)
        # Undo hue rotation: solve h_raw + theta(h_raw) = h_out
        a_out, b_out = lab[:, 1], lab[:, 2]
        h_out = torch.atan2(b_out, a_out)
        C = (a_out ** 2 + b_out ** 2).sqrt()
        h_raw = h_out.clone()
        for _ in range(10):
            theta = self._rotation_angle(h_raw)
            h_raw = h_out - theta
        theta_final = self._rotation_angle(h_raw)
        cos_t = torch.cos(theta_final)
        sin_t = torch.sin(theta_final)
        # Reverse rotation: [a_raw, b_raw] = R(-theta) @ [a, b]
        a_raw = a_out * cos_t + b_out * sin_t
        b_raw = -a_out * sin_t + b_out * cos_t
        raw = torch.stack([lab[:, 0], a_raw, b_raw], dim=-1)
        lms_c = raw @ self._M2_inv.T
        lms = torch.sign(lms_c) * lms_c.abs().pow(3.0)
        return lms @ self._M1_inv.T


class NativePolar(ColorSpace):
    """Native polar color space: outputs (L, C, h) instead of (L, a, b).

    Linear interpolation in this space = chroma-preserving polar interpolation.
    No muddy midpoints by construction.

    Handles:
    - h wrap-around: shortest arc via modular arithmetic in interpolate()
    - Achromatic (C=0): h defaults to 0, blends smoothly
    - Invertible: (L,C,h) → (L,a,b) → XYZ is exact

    The trick: forward() returns (L, C, h_scaled) where h_scaled = h/(2π).
    This keeps h in [0,1] range similar to L and C.
    """

    def __init__(self, base_space, label=None):
        self._base = base_space
        self.name = label or f"NativePolar({base_space.name})"
        self._PI = 3.141592653589793
        self._MSi = torch.linalg.inv(M_SRGB).to(dtype=torch.float64)

    def forward(self, xyz):
        """XYZ → (L, C, h_scaled)"""
        lab = self._base.forward(xyz)
        L = lab[:, 0]
        a = lab[:, 1]
        b = lab[:, 2]
        C = (a**2 + b**2).sqrt()
        h = torch.atan2(b, a)  # [-π, π]
        h_scaled = (h / (2 * self._PI)) % 1.0  # [0, 1]
        return torch.stack([L, C, h_scaled], dim=-1)

    def inverse(self, lch):
        """(L, C, h_scaled) → XYZ"""
        L = lch[:, 0]
        C = lch[:, 1].clamp(min=0)
        h = lch[:, 2] * 2 * self._PI  # back to radians
        a = C * torch.cos(h)
        b = C * torch.sin(h)
        lab = torch.stack([L, a, b], dim=-1)
        return self._base.inverse(lab)

    def interpolate(self, xyz1, xyz2, n_steps=26):
        """Chroma-boosted linear interpolation.

        Follows the LINEAR path in (a,b) plane (no rainbow) but BOOSTS
        chroma at each step to match the linearly interpolated C from endpoints.

        Result: hue = linear (direct, no rainbow), chroma = preserved (no mud).
        Best of both worlds.
        """
        lab1 = self._base.forward(xyz1.unsqueeze(0) if xyz1.dim() == 1 else xyz1)[0]
        lab2 = self._base.forward(xyz2.unsqueeze(0) if xyz2.dim() == 1 else xyz2)[0]

        L1, a1, b1 = lab1[0], lab1[1], lab1[2]
        L2, a2, b2 = lab2[0], lab2[1], lab2[2]
        C1 = (a1**2 + b1**2).sqrt()
        C2 = (a2**2 + b2**2).sqrt()

        # Quadratic Bezier in (a,b) plane with control point pushed away from origin.
        # This curves the path AWAY from gray → preserves chroma.
        # Smooth by construction (no hue discontinuity, no sRGB clipping artifacts).

        # Linear midpoint
        mx = 0.5 * (a1 + a2)
        my = 0.5 * (b1 + b2)
        M_norm = (mx**2 + my**2).sqrt()

        # Target midpoint chroma (average of endpoints)
        C_mid_target = 0.5 * (C1 + C2)

        # Push strength: how much to push control point away from origin
        # Proportional to chroma deficit at midpoint
        if M_norm > 0.001:
            # Push direction: from origin toward midpoint
            dx = mx / M_norm
            dy = my / M_norm
            # Push amount: fill the gap between linear C and target C
            k = (C_mid_target - M_norm).clamp(min=0) * 0.8
        else:
            # Midpoint is at origin (complementary colors) — can't determine push direction
            # Fall back to linear (no improvement possible)
            dx = torch.tensor(0.0, device=xyz1.device)
            dy = torch.tensor(0.0, device=xyz1.device)
            k = torch.tensor(0.0, device=xyz1.device)

        # Bezier control point
        qx = mx + k * dx
        qy = my + k * dy

        t = torch.linspace(0, 1, n_steps, device=xyz1.device, dtype=xyz1.dtype)
        results = []

        for i in range(n_steps):
            ti = t[i]
            L_i = L1 + ti * (L2 - L1)

            # Quadratic Bezier: B(t) = (1-t)²P1 + 2t(1-t)Q + t²P2
            a_i = (1-ti)**2 * a1 + 2*ti*(1-ti) * qx + ti**2 * a2
            b_i = (1-ti)**2 * b1 + 2*ti*(1-ti) * qy + ti**2 * b2

            lab_i = torch.stack([L_i, a_i, b_i])
            xyz_i = self._base.inverse(lab_i.unsqueeze(0))[0]
            results.append(xyz_i)

        return torch.stack(results)


class PolarBlend(ColorSpace):
    """Any base space + polar-aware interpolation.

    Wraps an existing space. forward/inverse are identical.
    The difference: interpolate() uses polar (LCh) blending instead of
    linear Lab, preserving chroma through the gradient.

    Achromatic-safe: blends smoothly between polar and linear based on
    endpoint chromas, avoiding h discontinuity at C=0.
    """

    def __init__(self, base_space, label=None):
        self._base = base_space
        self.name = label or f"Polar({base_space.name})"

    def forward(self, xyz):
        return self._base.forward(xyz)

    def inverse(self, lab):
        return self._base.inverse(lab)

    def interpolate(self, xyz1, xyz2, n_steps=26):
        """Polar-aware interpolation: preserves chroma, rotates hue."""
        lab1 = self.forward(xyz1.unsqueeze(0) if xyz1.dim() == 1 else xyz1)[0]
        lab2 = self.forward(xyz2.unsqueeze(0) if xyz2.dim() == 1 else xyz2)[0]

        PI = 3.141592653589793
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]
        L2, a2, b2 = lab2[0], lab2[1], lab2[2]

        C1 = (a1**2 + b1**2).sqrt()
        C2 = (a2**2 + b2**2).sqrt()
        h1 = torch.atan2(b1, a1)
        h2 = torch.atan2(b2, a2)

        # Shortest hue arc
        dh = h2 - h1
        dh = torch.where(dh > PI, dh - 2*PI, dh)
        dh = torch.where(dh < -PI, dh + 2*PI, dh)

        # Polar weight: high when both endpoints are chromatic
        # Smoothly transitions to linear when either endpoint is achromatic
        C_min = torch.minimum(C1, C2)
        C_max = torch.maximum(C1, C2)
        w_polar = torch.clamp(C_min / (C_max + 1e-10), 0, 1)
        # Additional: scale by absolute chroma (very low C → linear)
        w_polar = w_polar * torch.clamp(C_min * 20, 0, 1)

        t = torch.linspace(0, 1, n_steps, device=lab1.device, dtype=lab1.dtype)

        labs = []
        for i in range(n_steps):
            ti = t[i]
            L_i = L1 + ti * (L2 - L1)

            # Polar path
            C_polar = C1 + ti * (C2 - C1)
            h_polar = h1 + ti * dh
            a_polar = C_polar * torch.cos(h_polar)
            b_polar = C_polar * torch.sin(h_polar)

            # Linear path
            a_linear = a1 + ti * (a2 - a1)
            b_linear = b1 + ti * (b2 - b1)

            # Blend
            a_i = w_polar * a_polar + (1 - w_polar) * a_linear
            b_i = w_polar * b_polar + (1 - w_polar) * b_linear

            labs.append(torch.stack([L_i, a_i, b_i]))

        lab_interp = torch.stack(labs)
        return self.inverse(lab_interp)


class TwoStage(ColorSpace):
    """Two-stage pipeline: XYZ -> M1a -> cbrt -> M1b -> cbrt -> M2 -> L_corr -> Lab."""

    def __init__(self, json_path, device, label=None):
        import json as _json
        with open(json_path) as f:
            d = _json.load(f)
        self.name = label or f"TwoStage({os.path.basename(json_path)})"
        dev = device
        self._M1a = torch.tensor(d["M1a"], dtype=torch.float64, device=dev)
        self._M1b = torch.tensor(d["M1b"], dtype=torch.float64, device=dev)
        self._M2 = torch.tensor(d["M2"], dtype=torch.float64, device=dev)
        self._M1a_inv = torch.linalg.inv(self._M1a)
        self._M1b_inv = torch.linalg.inv(self._M1b)
        self._M2_inv = torch.linalg.inv(self._M2)
        lc = d.get("L_corr", [0, 0, 0])
        self._lc = torch.tensor(lc, dtype=torch.float64, device=dev)
        self._has_lc = any(abs(x) > 1e-10 for x in lc)

    def forward(self, xyz):
        lms1 = (xyz @ self._M1a.T).clamp(min=0)
        inter = torch.sign(lms1) * lms1.abs().pow(1.0 / 3.0)
        lms2 = (inter @ self._M1b.T).clamp(min=0)
        opp = torch.sign(lms2) * lms2.abs().pow(1.0 / 3.0)
        lab = opp @ self._M2.T
        if self._has_lc:
            L = lab[:, 0:1]
            c1, c2, c3 = self._lc[0], self._lc[1], self._lc[2]
            t = L * (1.0 - L)
            lab = torch.cat([L + c1 * t + c2 * t * (2.0 * L - 1.0) + c3 * L ** 2 * (1.0 - L) ** 2,
                             lab[:, 1:2], lab[:, 2:3]], dim=1)
        return lab

    def inverse(self, lab):
        lab = lab.clone()
        if self._has_lc:
            L1 = lab[:, 0:1]
            L = L1.clone()
            c1, c2, c3 = self._lc[0], self._lc[1], self._lc[2]
            for _ in range(15):
                t = L * (1.0 - L)
                f = L + c1 * t + c2 * t * (2.0 * L - 1.0) + c3 * L ** 2 * (1.0 - L) ** 2 - L1
                df = 1.0 + c1 * (1.0 - 2.0 * L) + c2 * (6.0 * L ** 2 - 6.0 * L + 1.0) + c3 * 2.0 * L * (1.0 - L) * (1.0 - 2.0 * L)
                L = L - f / df.clamp(min=1e-12)
            lab = torch.cat([L, lab[:, 1:2], lab[:, 2:3]], dim=1)
        opp = lab @ self._M2_inv.T
        lms2 = torch.sign(opp) * opp.abs().pow(3.0)
        inter = lms2 @ self._M1b_inv.T
        lms1 = torch.sign(inter) * inter.abs().pow(3.0)
        return lms1 @ self._M1a_inv.T
