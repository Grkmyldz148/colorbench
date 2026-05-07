"""GenSpace adapter family — drop-in port from legacy core/spaces.py.

These spaces predate the modular ColorSpace architecture and are kept here
as cohesive ports until each family member can be split into its own focused
module. They all subclass the new `ColorSpace` base from `.base` and produce
bit-exact output identical to the legacy implementations in `core.spaces`.

Family members
--------------
GenSpaceAdapter      — bare M1 → cbrt → M2 (no enrichment)
NakaRushtonEnriched  — Naka-Rushton transfer + L correction + L-dep chroma
GenSpaceEnriched     — piecewise-linear (CIE Lab style δ) + L_corr cubic
GenSpaceBlueFix      — Enriched + post-M2 directional ab-offset for blue
NonlinearM1          — Nonlinear M1 with blue-selective cross term
CustomSpace          — wrap arbitrary forward/inverse callables
"""
from __future__ import annotations

import json as _json
import os

import torch

from .base import ColorSpace, signed_cbrt


# ─── GenSpaceAdapter ─────────────────────────────────────────────────────────

class GenSpaceAdapter(ColorSpace):
    """Adapter for GenSpace JSON (M1/M2/gamma). Pure GPU, no helmlab dependency."""

    def __init__(self, json_path: str, device: torch.device,
                 dtype: torch.dtype = torch.float64):
        with open(json_path) as f:
            d = _json.load(f)
        self.name = f"GenSpace({os.path.basename(json_path)})"
        self.device = device
        self.dtype = dtype
        self.M1 = torch.tensor(d["M1"], device=device, dtype=dtype)
        self.M2 = torch.tensor(d["M2"], device=device, dtype=dtype)
        self.gamma = d.get("gamma", [1 / 3, 1 / 3, 1 / 3])
        self.M1_inv = torch.linalg.inv(self.M1)
        self.M2_inv = torch.linalg.inv(self.M2)

    def forward(self, xyz):
        lms = xyz @ self.M1.T
        return signed_cbrt(lms) @ self.M2.T

    def inverse(self, lab):
        lms_c = lab @ self.M2_inv.T
        lms = torch.sign(lms_c) * lms_c.abs().pow(3.0)
        return lms @ self.M1_inv.T


# ─── NakaRushtonEnriched ────────────────────────────────────────────────────

class NakaRushtonEnriched(ColorSpace):
    """Naka-Rushton transfer + L correction + L-dep chroma + chroma power."""

    def __init__(self, json_path: str, device: torch.device,
                 dtype: torch.dtype = torch.float64):
        with open(json_path) as f:
            d = _json.load(f)
        self.name = f"NR+Enrich({os.path.basename(json_path)})"
        self.device = device
        self.dtype = dtype
        self.M1 = torch.tensor(d["M1"], device=device, dtype=dtype)
        self.M2 = torch.tensor(d["M2"], device=device, dtype=dtype)
        self.M1_inv = torch.tensor(d["M1_inv"], device=device, dtype=dtype)
        self.M2_inv = torch.tensor(d["M2_inv"], device=device, dtype=dtype)
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
        L = L_out.clone()
        for _ in range(15):
            g = L + self.c1 * L * (1.0 - L) - L_out
            gp = 1.0 + self.c1 * (1.0 - 2.0 * L)
            L = L - g / gp.clamp(min=1e-10)
        C_out = torch.sqrt(a_out ** 2 + b_out ** 2 + 1e-30)
        f_L = torch.exp(self.k * (L - 0.5))
        C_in = (C_out / f_L.clamp(min=1e-30)).clamp(min=0).pow(1.0 / self.cp)
        a_in = a_out / C_out * C_in
        b_in = b_out / C_out * C_in
        raw = torch.stack([L, a_in, b_in], dim=-1)
        lms_c = raw @ self.M2_inv.T
        lms_c = lms_c.clamp(min=0)
        lms_c = torch.minimum(lms_c, torch.tensor(self.s_gain - 1e-10))
        ratio = (lms_c / (self.s_gain - lms_c).clamp(min=1e-30)).clamp(min=0)
        lms = self.sigma * ratio.pow(1.0 / self.n)
        return lms @ self.M1_inv.T


# ─── GenSpaceEnriched ───────────────────────────────────────────────────────

class GenSpaceEnriched(ColorSpace):
    """GenSpace with delta (piecewise-linear transfer) + L_corr (cubic L correction)."""

    def __init__(self, json_path: str, device: torch.device,
                 dtype: torch.dtype = torch.float64):
        with open(json_path) as f:
            d = _json.load(f)
        self.name = f"GenSpace+Enrich({os.path.basename(json_path)})"
        self.device = device
        self.dtype = dtype
        self.M1 = torch.tensor(d["M1"], device=device, dtype=dtype)
        self.M2 = torch.tensor(d["M2"], device=device, dtype=dtype)
        self.M1_inv = torch.linalg.inv(self.M1)
        self.M2_inv = torch.linalg.inv(self.M2)
        self.delta = d.get("delta", 0.0)
        self.L_corr = d.get("L_corr", [0.0, 0.0, 0.0])
        self.c1, self.c2, self.c3 = self.L_corr

    def _transfer(self, x):
        d = self.delta
        if d < 1e-10:
            return signed_cbrt(x)
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
        return (L + self.c1 * L1 + self.c2 * L1 * (2.0 * L - 1.0)
                + self.c3 * L * L * (1.0 - L) * (1.0 - L))

    def _L_corr_inverse(self, L_prime, n_iter=50):
        L = L_prime.clone()
        for _ in range(n_iter):
            L1 = L * (1.0 - L)
            f = (L + self.c1 * L1 + self.c2 * L1 * (2.0 * L - 1.0)
                 + self.c3 * L * L * (1.0 - L) * (1.0 - L) - L_prime)
            df = (1.0 + self.c1 * (1.0 - 2.0 * L)
                  + self.c2 * (6.0 * L * L - 6.0 * L + 1.0)
                  + self.c3 * 2.0 * L * (1.0 - L) * (1.0 - 2.0 * L))
            L = L - f / df.clamp(min=1e-12)
        return L

    def forward(self, xyz):
        lms_c = self._transfer(xyz @ self.M1.T)
        raw = lms_c @ self.M2.T
        if abs(self.c1) > 1e-15 or abs(self.c2) > 1e-15 or abs(self.c3) > 1e-15:
            L_out = self._L_corr_forward(raw[:, 0])
            return torch.stack([L_out, raw[:, 1], raw[:, 2]], dim=-1)
        return raw

    def inverse(self, lab):
        if abs(self.c1) > 1e-15 or abs(self.c2) > 1e-15 or abs(self.c3) > 1e-15:
            L = self._L_corr_inverse(lab[:, 0])
            raw = torch.stack([L, lab[:, 1], lab[:, 2]], dim=-1)
        else:
            raw = lab
        lms_c = raw @ self.M2_inv.T
        return self._transfer_inv(lms_c) @ self.M1_inv.T


# ─── GenSpaceBlueFix ────────────────────────────────────────────────────────

class GenSpaceBlueFix(ColorSpace):
    """GenSpace+Enrich with post-M2 C-proportional directional ab-offset for blue fix."""

    def __init__(self, json_path: str, device: torch.device,
                 dtype: torch.dtype = torch.float64):
        with open(json_path) as f:
            d = _json.load(f)
        self.name = f"GenSpace+BlueFix({os.path.basename(json_path)})"
        self.device = device
        self.dtype = dtype
        self.M1 = torch.tensor(d["M1"], device=device, dtype=dtype)
        self.M2 = torch.tensor(d["M2"], device=device, dtype=dtype)
        self.M1_inv = torch.linalg.inv(self.M1)
        self.M2_inv = torch.linalg.inv(self.M2)
        self.delta = d.get("delta", 0.0)
        self.L_corr = d.get("L_corr", [0.0, 0.0, 0.0])
        self.c1, self.c2, self.c3 = self.L_corr
        bf = d.get("blue_fix", {})
        self.bf_k = bf.get("k", 0.0)
        self.bf_sigma = bf.get("sigma", 30.0)
        self.bf_center = bf.get("center", 240.0)
        self.bf_dir_a = bf.get("dir_a", -0.8813)
        self.bf_dir_b = bf.get("dir_b", 0.4725)

    def _transfer(self, x):
        d = self.delta
        if d < 1e-10:
            return signed_cbrt(x)
        d_cbrt = d ** (1.0 / 3.0)
        slope = 1.0 / (3.0 * d ** (2.0 / 3.0))
        offset = 2.0 / 3.0 * d_cbrt
        ax = x.abs()
        result = torch.where(ax >= d, ax.pow(1.0 / 3.0), slope * ax + offset)
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
        lin_part = ((ay - offset) / slope).clamp(min=0.0)
        return y.sign() * torch.where(ay >= d_cbrt, cube_part, lin_part)

    def _L_corr_forward(self, L):
        L1 = L * (1.0 - L)
        return (L + self.c1 * L1 + self.c2 * L1 * (2.0 * L - 1.0)
                + self.c3 * L * L * (1.0 - L) * (1.0 - L))

    def _L_corr_inverse(self, L_prime, n_iter=50):
        L = L_prime.clone()
        for _ in range(n_iter):
            L1 = L * (1.0 - L)
            f = (L + self.c1 * L1 + self.c2 * L1 * (2.0 * L - 1.0)
                 + self.c3 * L * L * (1.0 - L) * (1.0 - L) - L_prime)
            df = (1.0 + self.c1 * (1.0 - 2.0 * L)
                  + self.c2 * (6.0 * L * L - 6.0 * L + 1.0)
                  + self.c3 * 2.0 * L * (1.0 - L) * (1.0 - 2.0 * L))
            L = L - f / df.clamp(min=1e-12)
        return L

    def _gauss_hue(self, theta_deg):
        d = (theta_deg - self.bf_center + 180.0) % 360.0 - 180.0
        return torch.exp(-d * d / (2.0 * self.bf_sigma ** 2))

    def _blue_fix_forward(self, lab):
        if self.bf_k < 1e-10:
            return lab
        L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
        C = torch.sqrt(a * a + b * b + 1e-30)
        theta_deg = (torch.rad2deg(torch.atan2(b, a))) % 360.0
        w = self._gauss_hue(theta_deg)
        f = self.bf_k * C * torch.clamp(1.0 - L, min=0.0) * w
        return torch.stack(
            [L, a + f * self.bf_dir_a, b + f * self.bf_dir_b], dim=-1,
        )

    def _blue_fix_inverse(self, lab):
        if self.bf_k < 1e-10:
            return lab
        L = lab[:, 0]
        a_pp = lab[:, 1]
        b_pp = lab[:, 2]
        kL = self.bf_k * torch.clamp(1.0 - L, min=0.0)

        def eval_g(fv):
            a = a_pp - fv * self.bf_dir_a
            b = b_pp - fv * self.bf_dir_b
            C = torch.sqrt(a * a + b * b + 1e-30)
            theta_deg = (torch.rad2deg(torch.atan2(b, a))) % 360.0
            w = self._gauss_hue(theta_deg)
            return fv - kL * C * w

        C_input = torch.sqrt(a_pp * a_pp + b_pp * b_pp + 1e-30)
        f_lo = torch.zeros_like(L)
        f_hi = kL * C_input * 2.0 + 0.01
        for _ in range(60):
            f_mid = 0.5 * (f_lo + f_hi)
            g_mid = eval_g(f_mid)
            f_lo = torch.where(g_mid < 0, f_mid, f_lo)
            f_hi = torch.where(g_mid >= 0, f_mid, f_hi)

        f = 0.5 * (f_lo + f_hi)
        return torch.stack(
            [L, a_pp - f * self.bf_dir_a, b_pp - f * self.bf_dir_b], dim=-1,
        )

    def forward(self, xyz):
        lms_c = self._transfer(xyz @ self.M1.T)
        raw = lms_c @ self.M2.T
        if abs(self.c1) > 1e-15 or abs(self.c2) > 1e-15 or abs(self.c3) > 1e-15:
            L_out = self._L_corr_forward(raw[:, 0])
            lab = torch.stack([L_out, raw[:, 1], raw[:, 2]], dim=-1)
        else:
            lab = raw
        return self._blue_fix_forward(lab)

    def inverse(self, lab):
        lab = self._blue_fix_inverse(lab)
        if abs(self.c1) > 1e-15 or abs(self.c2) > 1e-15 or abs(self.c3) > 1e-15:
            L = self._L_corr_inverse(lab[:, 0])
            raw = torch.stack([L, lab[:, 1], lab[:, 2]], dim=-1)
        else:
            raw = lab
        return self._transfer_inv(raw @ self.M2_inv.T) @ self.M1_inv.T


# ─── NonlinearM1 ────────────────────────────────────────────────────────────

class NonlinearM1(ColorSpace):
    """Nonlinear M1 with blue-selective cross term: lms[0] += d*(1-Y)*Z."""

    def __init__(self, json_path: str, device: torch.device,
                 dtype: torch.dtype = torch.float64, label: str = None):
        with open(json_path) as f:
            d = _json.load(f)
        self.name = label or f"NonlinearM1({os.path.basename(json_path)})"
        self.device = device
        self.dtype = dtype
        self._M1 = torch.tensor(d["M1"], dtype=dtype, device=device)
        self._M2 = torch.tensor(d["M2"], dtype=dtype, device=device)
        self._M1_inv = torch.linalg.inv(self._M1)
        self._M2_inv = torch.linalg.inv(self._M2)
        self._d = d.get("cross_term_d", 0.0)
        self._k_ach = d.get("cross_term_k", 0.0)
        lc = d.get("L_corr", [0, 0, 0])
        self._lc = torch.tensor(lc, dtype=dtype, device=device)
        self._has_lc = any(abs(x) > 1e-10 for x in lc)

    def _fwd_lms(self, xyz):
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
        xyz = lms_target @ self._M1_inv.T
        for _ in range(50):
            lms = xyz @ self._M1.T
            if self._d != 0:
                if self._k_ach != 0:
                    cross = self._d * (xyz[:, 2] - self._k_ach * xyz[:, 1])
                else:
                    cross = self._d * (1.0 - xyz[:, 1]) * xyz[:, 2]
                lms = lms.clone()
                lms[:, 0] = lms[:, 0] + cross
            err = lms - lms_target

            J = self._M1.unsqueeze(0).expand(xyz.shape[0], -1, -1).clone()
            if self._d != 0:
                if self._k_ach != 0:
                    J[:, 0, 1] = J[:, 0, 1] + (-self._d * self._k_ach)
                    J[:, 0, 2] = J[:, 0, 2] + self._d
                else:
                    J[:, 0, 1] = J[:, 0, 1] + (-self._d * xyz[:, 2])
                    J[:, 0, 2] = J[:, 0, 2] + (self._d * (1.0 - xyz[:, 1]))

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
            L_new = L + c1 * t + c2 * t * (2.0 * L - 1.0) + c3 * L ** 2 * (1.0 - L) ** 2
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
                f = L + c1 * t + c2 * t * (2 * L - 1) + c3 * L ** 2 * (1 - L) ** 2 - L1
                df = (1 + c1 * (1 - 2 * L) + c2 * (6 * L ** 2 - 6 * L + 1)
                      + c3 * 2 * L * (1 - L) * (1 - 2 * L))
                L = L - f / df.clamp(min=1e-12)
            lab = torch.cat([L, lab[:, 1:2], lab[:, 2:3]], dim=1)
        lms_c = lab @ self._M2_inv.T
        lms = torch.sign(lms_c) * lms_c.abs().pow(3.0)
        return self._inv_lms(lms)


# ─── CustomSpace ────────────────────────────────────────────────────────────

class CustomSpace(ColorSpace):
    """Wrap any forward/inverse callables as a ColorSpace."""

    def __init__(self, name: str, forward_fn, inverse_fn,
                 device: torch.device = None, dtype: torch.dtype = torch.float64):
        self.name = name
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self._fwd = forward_fn
        self._inv = inverse_fn

    def forward(self, xyz):
        return self._fwd(xyz)

    def inverse(self, lab):
        return self._inv(lab)
