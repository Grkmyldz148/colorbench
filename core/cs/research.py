"""Research-only color spaces — drop-in port from legacy core/spaces.py.

These four spaces are not used by ColorBench production runs but appear in
research scripts. Ported here so that legacy core/spaces.py can eventually
be deleted entirely.

  HueDep       — Fourier-rotated M2 a/b rows + L_corr; Newton inverse for hue
  NativePolar  — wraps a base space, exposes (L, C, h_scaled), Bezier interp
  PolarBlend   — wraps a base space, polar/linear-blend interpolate()
  TwoStage     — XYZ → M1a → cbrt → M1b → cbrt → M2 (double-cbrt pipeline)

All four bit-exact ports of the legacy implementations.
"""
from __future__ import annotations

import json as _json
import os

import torch

from .base import ColorSpace
from .constants import M_SRGB


# ─── HueDep ──────────────────────────────────────────────────────────────────

class HueDep(ColorSpace):
    """Hue-dependent M2: fixed L row + Fourier-rotated a,b rows + L_corr."""

    def __init__(self, json_path: str, device: torch.device,
                 dtype: torch.dtype = torch.float64, label: str = None):
        with open(json_path) as f:
            d = _json.load(f)
        self.name = label or f"HueDep({os.path.basename(json_path)})"
        self.device = device
        self.dtype = dtype
        self._M1 = torch.tensor(d["M1"], dtype=dtype, device=device)
        self._M1_inv = torch.linalg.inv(self._M1)

        m2f = d["M2_full"]
        self._M2_L = torch.tensor(m2f[0], dtype=dtype, device=device)
        self._M2_a = torch.tensor(m2f[1], dtype=dtype, device=device)
        self._M2_b = torch.tensor(m2f[2], dtype=dtype, device=device)
        self._M2 = torch.tensor(m2f, dtype=dtype, device=device)
        self._M2_inv = torch.linalg.inv(self._M2)

        rf = d["rotation_fourier"]
        self._rc1 = rf["c1"]
        self._rs1 = rf["s1"]
        self._rc2 = rf["c2"]
        self._rs2 = rf["s2"]

        lc = d.get("L_corr", [0, 0, 0])
        self._lc = torch.tensor(lc, dtype=dtype, device=device)
        self._has_lc = any(abs(x) > 1e-10 for x in lc)

    def _rotation_angle(self, h):
        return (self._rc1 * torch.cos(h) + self._rs1 * torch.sin(h)
                + self._rc2 * torch.cos(2 * h) + self._rs2 * torch.sin(2 * h))

    def forward(self, xyz):
        lms = (xyz @ self._M1.T).clamp(min=0)
        lms_c = torch.sign(lms) * lms.abs().pow(1.0 / 3.0)
        L = lms_c @ self._M2_L
        a_raw = lms_c @ self._M2_a
        b_raw = lms_c @ self._M2_b
        h = torch.atan2(b_raw, a_raw)
        theta = self._rotation_angle(h)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        a = a_raw * cos_t - b_raw * sin_t
        b = a_raw * sin_t + b_raw * cos_t
        lab = torch.stack([L, a, b], dim=-1)
        if self._has_lc:
            Lv = lab[:, 0:1]
            c1, c2, c3 = self._lc[0], self._lc[1], self._lc[2]
            t = Lv * (1.0 - Lv)
            L_new = (Lv + c1 * t + c2 * t * (2.0 * Lv - 1.0)
                     + c3 * Lv ** 2 * (1.0 - Lv) ** 2)
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
                f = (L + c1 * t + c2 * t * (2 * L - 1)
                     + c3 * L ** 2 * (1 - L) ** 2 - L1)
                df = (1.0 + c1 * (1 - 2 * L)
                      + c2 * (6 * L ** 2 - 6 * L + 1)
                      + c3 * 2 * L * (1 - L) * (1 - 2 * L))
                L = L - f / df.clamp(min=1e-12)
            lab = torch.cat([L, lab[:, 1:2], lab[:, 2:3]], dim=1)

        a_out, b_out = lab[:, 1], lab[:, 2]
        h_out = torch.atan2(b_out, a_out)
        h_raw = h_out.clone()
        for _ in range(10):
            h_raw = h_out - self._rotation_angle(h_raw)
        theta_final = self._rotation_angle(h_raw)
        cos_t, sin_t = torch.cos(theta_final), torch.sin(theta_final)
        a_raw = a_out * cos_t + b_out * sin_t
        b_raw = -a_out * sin_t + b_out * cos_t
        raw = torch.stack([lab[:, 0], a_raw, b_raw], dim=-1)
        lms_c = raw @ self._M2_inv.T
        lms = torch.sign(lms_c) * lms_c.abs().pow(3.0)
        return lms @ self._M1_inv.T


# ─── NativePolar ─────────────────────────────────────────────────────────────

class NativePolar(ColorSpace):
    """Native polar wrapper — outputs (L, C, h_scaled). Linear interp = polar interp.

    Linear interpolation in this space = chroma-preserving polar interpolation.
    No muddy midpoints by construction.
    """

    def __init__(self, base_space: ColorSpace, label: str = None):
        self._base = base_space
        self.name = label or f"NativePolar({base_space.name})"
        self.device = base_space.device
        self.dtype = base_space.dtype
        self._PI = 3.141592653589793

    def forward(self, xyz):
        lab = self._base.forward(xyz)
        L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
        C = (a ** 2 + b ** 2).sqrt()
        h = torch.atan2(b, a)
        h_scaled = (h / (2 * self._PI)) % 1.0
        return torch.stack([L, C, h_scaled], dim=-1)

    def inverse(self, lch):
        L = lch[:, 0]
        C = lch[:, 1].clamp(min=0)
        h = lch[:, 2] * 2 * self._PI
        a = C * torch.cos(h)
        b = C * torch.sin(h)
        return self._base.inverse(torch.stack([L, a, b], dim=-1))

    def interpolate(self, xyz1, xyz2, n_steps=26):
        """Quadratic Bezier in (a,b): linear hue path with chroma preservation."""
        lab1 = self._base.forward(xyz1.unsqueeze(0) if xyz1.dim() == 1 else xyz1)[0]
        lab2 = self._base.forward(xyz2.unsqueeze(0) if xyz2.dim() == 1 else xyz2)[0]

        L1, a1, b1 = lab1[0], lab1[1], lab1[2]
        L2, a2, b2 = lab2[0], lab2[1], lab2[2]
        C1 = (a1 ** 2 + b1 ** 2).sqrt()
        C2 = (a2 ** 2 + b2 ** 2).sqrt()

        mx = 0.5 * (a1 + a2)
        my = 0.5 * (b1 + b2)
        M_norm = (mx ** 2 + my ** 2).sqrt()
        C_mid_target = 0.5 * (C1 + C2)

        if M_norm > 0.001:
            dx = mx / M_norm
            dy = my / M_norm
            k = (C_mid_target - M_norm).clamp(min=0) * 0.8
        else:
            dx = torch.tensor(0.0, device=xyz1.device, dtype=xyz1.dtype)
            dy = torch.tensor(0.0, device=xyz1.device, dtype=xyz1.dtype)
            k = torch.tensor(0.0, device=xyz1.device, dtype=xyz1.dtype)

        qx = mx + k * dx
        qy = my + k * dy

        t = torch.linspace(0, 1, n_steps, device=xyz1.device, dtype=xyz1.dtype)
        results = []
        for i in range(n_steps):
            ti = t[i]
            L_i = L1 + ti * (L2 - L1)
            a_i = (1 - ti) ** 2 * a1 + 2 * ti * (1 - ti) * qx + ti ** 2 * a2
            b_i = (1 - ti) ** 2 * b1 + 2 * ti * (1 - ti) * qy + ti ** 2 * b2
            xyz_i = self._base.inverse(torch.stack([L_i, a_i, b_i]).unsqueeze(0))[0]
            results.append(xyz_i)
        return torch.stack(results)


# ─── PolarBlend ──────────────────────────────────────────────────────────────

class PolarBlend(ColorSpace):
    """Wraps a base space; interpolate() blends polar (LCh) with linear (Lab)."""

    def __init__(self, base_space: ColorSpace, label: str = None):
        self._base = base_space
        self.name = label or f"Polar({base_space.name})"
        self.device = base_space.device
        self.dtype = base_space.dtype

    def forward(self, xyz):
        return self._base.forward(xyz)

    def inverse(self, lab):
        return self._base.inverse(lab)

    def interpolate(self, xyz1, xyz2, n_steps=26):
        """Polar/linear blend: weight polar high when both endpoints chromatic."""
        lab1 = self.forward(xyz1.unsqueeze(0) if xyz1.dim() == 1 else xyz1)[0]
        lab2 = self.forward(xyz2.unsqueeze(0) if xyz2.dim() == 1 else xyz2)[0]

        PI = 3.141592653589793
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]
        L2, a2, b2 = lab2[0], lab2[1], lab2[2]
        C1 = (a1 ** 2 + b1 ** 2).sqrt()
        C2 = (a2 ** 2 + b2 ** 2).sqrt()
        h1 = torch.atan2(b1, a1)
        h2 = torch.atan2(b2, a2)

        dh = h2 - h1
        dh = torch.where(dh > PI, dh - 2 * PI, dh)
        dh = torch.where(dh < -PI, dh + 2 * PI, dh)

        C_min = torch.minimum(C1, C2)
        C_max = torch.maximum(C1, C2)
        w_polar = torch.clamp(C_min / (C_max + 1e-10), 0, 1)
        w_polar = w_polar * torch.clamp(C_min * 20, 0, 1)

        t = torch.linspace(0, 1, n_steps, device=lab1.device, dtype=lab1.dtype)
        labs = []
        for i in range(n_steps):
            ti = t[i]
            L_i = L1 + ti * (L2 - L1)
            C_polar = C1 + ti * (C2 - C1)
            h_polar = h1 + ti * dh
            a_polar = C_polar * torch.cos(h_polar)
            b_polar = C_polar * torch.sin(h_polar)
            a_linear = a1 + ti * (a2 - a1)
            b_linear = b1 + ti * (b2 - b1)
            a_i = w_polar * a_polar + (1 - w_polar) * a_linear
            b_i = w_polar * b_polar + (1 - w_polar) * b_linear
            labs.append(torch.stack([L_i, a_i, b_i]))

        return self.inverse(torch.stack(labs))


# ─── TwoStage ────────────────────────────────────────────────────────────────

class TwoStage(ColorSpace):
    """Two-stage pipeline: XYZ → M1a → cbrt → M1b → cbrt → M2 → L_corr → Lab."""

    def __init__(self, json_path: str, device: torch.device,
                 dtype: torch.dtype = torch.float64, label: str = None):
        with open(json_path) as f:
            d = _json.load(f)
        self.name = label or f"TwoStage({os.path.basename(json_path)})"
        self.device = device
        self.dtype = dtype
        self._M1a = torch.tensor(d["M1a"], dtype=dtype, device=device)
        self._M1b = torch.tensor(d["M1b"], dtype=dtype, device=device)
        self._M2 = torch.tensor(d["M2"], dtype=dtype, device=device)
        self._M1a_inv = torch.linalg.inv(self._M1a)
        self._M1b_inv = torch.linalg.inv(self._M1b)
        self._M2_inv = torch.linalg.inv(self._M2)
        lc = d.get("L_corr", [0, 0, 0])
        self._lc = torch.tensor(lc, dtype=dtype, device=device)
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
            lab = torch.cat([
                L + c1 * t + c2 * t * (2.0 * L - 1.0)
                + c3 * L ** 2 * (1.0 - L) ** 2,
                lab[:, 1:2], lab[:, 2:3],
            ], dim=1)
        return lab

    def inverse(self, lab):
        lab = lab.clone()
        if self._has_lc:
            L1 = lab[:, 0:1]
            L = L1.clone()
            c1, c2, c3 = self._lc[0], self._lc[1], self._lc[2]
            for _ in range(15):
                t = L * (1.0 - L)
                f = (L + c1 * t + c2 * t * (2.0 * L - 1.0)
                     + c3 * L ** 2 * (1.0 - L) ** 2 - L1)
                df = (1.0 + c1 * (1.0 - 2.0 * L)
                      + c2 * (6.0 * L ** 2 - 6.0 * L + 1.0)
                      + c3 * 2.0 * L * (1.0 - L) * (1.0 - 2.0 * L))
                L = L - f / df.clamp(min=1e-12)
            lab = torch.cat([L, lab[:, 1:2], lab[:, 2:3]], dim=1)
        opp = lab @ self._M2_inv.T
        lms2 = torch.sign(opp) * opp.abs().pow(3.0)
        inter = lms2 @ self._M1b_inv.T
        lms1 = torch.sign(inter) * inter.abs().pow(3.0)
        return lms1 @ self._M1a_inv.T
