"""HelmCT — Helmlab cross-term pipeline color space.

Pipeline (forward):
  XYZ → [pre-warp]                      optional Y-shift based on X/Z ratio
      → M1 (3×3 matrix)
      → [cross-terms]                   modify L,M,S with cross-channel terms
      → transfer (cbrt | depcubic | ...)
      → [neutral blend]                 default ON, branchless smooth
      → M2 (3×3 matrix)
      → [chroma hue rotation]           Fourier-3 hue mapping
      → [chroma power + L-dependent]    radial chroma scaling
      → [ab scale]                      anisotropic axis scaling
      → [hue-dependent L correction]    L-= δ·max(0, cos(h-c))^w · L(1-L)
      → [L correction]                  PWL | LC7 | LC5 | LC3
      → [L-gated hue enrichment]        L-gated Gaussian hue rotation
      → [neutral correction]            LUT subtraction
  → Lab

All optional stages are disabled cleanly when their JSON keys are absent.
Inverse runs the pipeline in reverse with the appropriate analytical or
iterative inverse for each stage.

This module composes stages defined in:
  core/cs/transfer.py    — transfer functions
  core/cs/pwL.py         — piecewise-linear L
  core/cs/enrichment.py  — hue rotation, L-gated enrichment
  core/cs/neutral.py     — neutral blend, NC LUT

For monolithic legacy implementation see colorbench/core/spaces.py:HelmCT.
"""
import json
import math
import os

import torch

from .base import ColorSpace
from .transfer import from_json_spec as build_transfer
from .pwL import PiecewiseLinearL
from .enrichment import LGatedHueEnrichment, ChromaPreservingHueRotation
from .neutral import neutral_blend, NCLut


class HelmCT(ColorSpace):
    """Helmlab cross-term pipeline. Loaded from JSON checkpoint."""

    def __init__(self, json_path: str, device: torch.device = None,
                 dtype: torch.dtype = torch.float64, label: str = None):
        with open(json_path) as f:
            d = json.load(f)
        self._cfg = d
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.name = label or f"HelmCT({os.path.basename(json_path)})"

        self._init_matrices(d)
        self._init_cross_terms(d)
        self._transfer = build_transfer(d)
        self._init_hue_rotation(d)
        self._init_chroma(d)
        self._init_ab_scale(d)
        self._init_hue_L_correction(d)
        self._init_L_correction(d)
        self._init_enrichment(d)
        self._init_pre_warp(d)
        self._init_neutral_correction(d)

    # ─── stage initializers ────────────────────────────────────────

    def _init_matrices(self, d):
        dev, dt = self.device, self.dtype
        self._M1 = torch.tensor(d["M1"], device=dev, dtype=dt)
        self._M2 = torch.tensor(d["M2"], device=dev, dtype=dt)
        self._M2_inv = torch.linalg.inv(self._M2)

    def _init_cross_terms(self, d):
        # Three optional cross-channel additions on LMS:
        #   L += d  · (Z - k  · Y)
        #   M += d2 · (Z - k2 · X)
        #   S += d3 · (X - k3 · Y)
        self._d, self._k = d.get("cross_d", 0.0), d.get("cross_k", 0.0)
        self._d2, self._k2 = d.get("cross_d2", 0.0), d.get("cross_k2", 0.0)
        self._d3, self._k3 = d.get("cross_d3", 0.0), d.get("cross_k3", 0.0)
        self._has_cross = any(abs(x) > 1e-10 for x in (self._d, self._d2, self._d3))

        # Analytical inverse: fold cross-terms into M1
        M1_mod = self._M1.clone()
        if self._d:
            M1_mod[0, 1] = M1_mod[0, 1] - self._d * self._k
            M1_mod[0, 2] = M1_mod[0, 2] + self._d
        if self._d2:
            M1_mod[1, 0] = M1_mod[1, 0] - self._d2 * self._k2
            M1_mod[1, 2] = M1_mod[1, 2] + self._d2
        if self._d3:
            M1_mod[2, 0] = M1_mod[2, 0] + self._d3
            M1_mod[2, 1] = M1_mod[2, 1] - self._d3 * self._k3
        self._M1_mod_inv = torch.linalg.inv(M1_mod)

    def _init_hue_rotation(self, d):
        hc = d.get("hue_correction", None)
        if hc is None:
            hc = [d.get(f"hue_{p}{i}", 0.0) for i in (1, 2, 3) for p in ("cos", "sin")]
            # Reorder to [c1, s1, c2, s2, c3, s3]
            hc = [hc[0], hc[1], hc[2], hc[3], hc[4], hc[5]] if len(hc) >= 6 else hc
        self._hue_rot = ChromaPreservingHueRotation(hc, n_fixed_point=150)

    def _init_chroma(self, d):
        self._cp = d.get("chroma_power", 1.0)
        self._ck = d.get("chroma_k", 0.0)
        self._cp_delta = d.get("chroma_power_delta", 0.0)
        self._has_chroma = (
            abs(self._cp - 1.0) > 1e-10
            or abs(self._ck) > 1e-10
            or abs(self._cp_delta) > 1e-10
        )

    def _init_ab_scale(self, d):
        self._sa = d.get("scale_a", 1.0)
        self._sb = d.get("scale_b", 1.0)
        self._has_ab = abs(self._sa - 1.0) > 1e-10 or abs(self._sb - 1.0) > 1e-10

    def _init_hue_L_correction(self, d):
        self._hue_L_delta = d.get("hue_L_delta", 0.0)
        self._hue_L_center = d.get("hue_L_center", -math.pi / 2)
        self._hue_L_width = d.get("hue_L_width", 2.0)
        self._has_hue_L = abs(self._hue_L_delta) > 1e-10

    def _init_L_correction(self, d):
        # Priority: PWL > LC7 > LC5 > LC3
        self._L_pwl = None
        self._lc7 = None
        self._lc5 = None
        self._lc3 = None

        if (lcpw := d.get("L_corr_pw")) is not None:
            n = len(lcpw)
            step = d.get("L_corr_pw_step", 1.0 / (n + 1))
            self._L_pwl = PiecewiseLinearL.from_shifts(
                lcpw, step, device=self.device, dtype=self.dtype
            )
        elif (lc7 := d.get("L_corr_7")) is not None:
            self._lc7 = torch.tensor(lc7, device=self.device, dtype=self.dtype)
        elif (lc5 := d.get("L_corr_5")) is not None:
            self._lc5 = torch.tensor(lc5, device=self.device, dtype=self.dtype)
        else:
            lc = d.get("L_corr", [0, 0, 0])
            if any(abs(x) > 1e-10 for x in lc):
                self._lc3 = torch.tensor(lc, device=self.device, dtype=self.dtype)

    def _init_enrichment(self, d):
        enr = d.get("enrichment")
        if enr and enr.get("type") == "L_gated_hue" and abs(enr.get("amp", 0)) > 1e-10:
            self._enrichment = LGatedHueEnrichment(
                amp=enr["amp"],
                center_deg=enr.get("center_deg", 240),
                sigma=enr.get("sigma", 0.7),
                L_lo=enr.get("L_lo", 0.37),
                L_hi=enr.get("L_hi", 1.0),
            )
        else:
            self._enrichment = None

    def _init_pre_warp(self, d):
        pw = d.get("pre_warp")
        if pw and abs(pw.get("eps", 0.0)) > 1e-10:
            self._pw_eps = pw["eps"]
            self._pw_xz_ratio = pw.get("xz_ratio", 21.4)
            self._pw_sigma = pw.get("sigma", 5.0)
            self._has_prewarp = True
        else:
            self._has_prewarp = False

    def _init_neutral_correction(self, d):
        self._has_nc = bool(d.get("neutral_correction") or d.get("nc"))
        self._neutral_blend_on = bool(d.get("neutral_blend", True))
        self._nc_lut = NCLut(self) if self._has_nc else None

    # ─── stage helpers ─────────────────────────────────────────────

    def _pre_m1_warp(self, xyz, sign):
        """Pre-M1 Y-shift (sign=+1 forward, -1 inverse). Exactly invertible."""
        if not self._has_prewarp:
            return xyz
        X, Z = xyz[:, 0:1], xyz[:, 2:3]
        ratio = X / Z.clamp(min=1e-30)
        dist2 = (ratio - self._pw_xz_ratio) ** 2
        w = self._pw_eps * torch.exp(-dist2 / (2 * self._pw_sigma ** 2))
        result = xyz.clone()
        result[:, 1:2] = xyz[:, 1:2] + sign * w
        return result

    def _apply_cross_forward(self, lms, xyz):
        if not self._has_cross:
            return lms
        out = lms.clone()
        if self._d:
            out[:, 0] = out[:, 0] + self._d * (xyz[:, 2] - self._k * xyz[:, 1])
        if self._d2:
            out[:, 1] = out[:, 1] + self._d2 * (xyz[:, 2] - self._k2 * xyz[:, 0])
        if self._d3:
            out[:, 2] = out[:, 2] + self._d3 * (xyz[:, 0] - self._k3 * xyz[:, 1])
        return out

    def _apply_chroma_forward(self, L, a, b):
        C = (a * a + b * b + 1e-30).sqrt()
        scale = torch.ones_like(C)
        if abs(self._cp - 1.0) > 1e-10:
            if abs(self._cp_delta) > 1e-10:
                blend = C * C / (C * C + self._cp_delta ** 2)
                eff_cp = 1.0 + blend * (self._cp - 1.0)
                scale = scale * C.pow(eff_cp - 1.0)
            else:
                scale = scale * C.pow(self._cp - 1.0)
        if abs(self._ck) > 1e-10:
            scale = scale * torch.exp(self._ck * (L - 0.5))
        return a * scale, b * scale

    def _apply_chroma_inverse(self, L, a, b):
        C = (a * a + b * b + 1e-30).sqrt()
        inv_scale = torch.ones_like(C)
        if abs(self._ck) > 1e-10:
            inv_scale = inv_scale * torch.exp(-self._ck * (L - 0.5))
        if abs(self._cp - 1.0) > 1e-10:
            C_after_k = C * inv_scale
            if abs(self._cp_delta) > 1e-10:
                # Newton iter for chroma-aware inverse
                C_orig = C_after_k.pow(1.0 / self._cp)
                d2 = self._cp_delta ** 2
                for _ in range(20):
                    bl = C_orig * C_orig / (C_orig * C_orig + d2)
                    eff_cp = 1.0 + bl * (self._cp - 1.0)
                    f = C_orig.pow(eff_cp) - C_after_k
                    dbl = 2.0 * C_orig * d2 / (C_orig * C_orig + d2).pow(2)
                    deff = dbl * (self._cp - 1.0)
                    log_C = torch.log(C_orig.clamp(min=1e-30))
                    df = (eff_cp * C_orig.pow(eff_cp - 1.0)
                          + C_orig.pow(eff_cp) * log_C * deff)
                    df = torch.where(df.abs() < 1e-15, torch.ones_like(df), df)
                    C_orig = (C_orig - f / df).clamp(min=0)
                inv_scale = C_orig / C.clamp(min=1e-30)
            else:
                C_orig = C_after_k.pow(1.0 / self._cp)
                inv_scale = C_orig / C.clamp(min=1e-30)
        return a * inv_scale, b * inv_scale

    def _apply_hue_L_forward(self, L, a, b):
        h = torch.atan2(b, a)
        cos_diff = torch.cos(h - self._hue_L_center)
        weight = cos_diff.clamp(min=0).pow(self._hue_L_width)
        return L - self._hue_L_delta * weight * L * (1.0 - L)

    def _apply_hue_L_inverse(self, L_target, a, b):
        h = torch.atan2(b, a)
        cos_diff = torch.cos(h - self._hue_L_center)
        weight = cos_diff.clamp(min=0).pow(self._hue_L_width)
        L = L_target.clone()
        for _ in range(20):
            f = L - self._hue_L_delta * weight * L * (1.0 - L) - L_target
            df = 1.0 - self._hue_L_delta * weight * (1.0 - 2.0 * L)
            df = torch.where(df.abs() < 1e-12, torch.ones_like(df), df)
            L = L - f / df
        return L

    def _apply_lc7_forward(self, L):
        p = self._lc7
        t = L * (1.0 - L)
        h = 0.5 - L
        return (L
                + p[0] * t + p[1] * t * h + p[2] * t * t + p[3] * t * t * h
                + p[4] * t * t * t + p[5] * t * t * t * h + p[6] * t * t * t * t)

    def _apply_lc7_deriv(self, L):
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

    def _apply_lc7_inverse(self, L_target):
        L = L_target.clone()
        for _ in range(30):
            f = self._apply_lc7_forward(L) - L_target
            df = self._apply_lc7_deriv(L)
            df = torch.where(df.abs() < 1e-12, torch.ones_like(df), df)
            L = L - f / df
        return L

    def _apply_lc5_forward(self, L):
        p = self._lc5
        t = L * (1.0 - L)
        h5 = 0.5 - L
        return L + p[0]*t + p[1]*t*h5 + p[2]*t*t + p[3]*t*t*h5 + p[4]*t*t*t

    def _apply_lc5_inverse(self, L_target):
        p = self._lc5
        L = L_target.clone()
        for _ in range(20):
            t = L * (1.0 - L)
            h5 = 0.5 - L
            dt = 1.0 - 2.0 * L
            dh5 = -1.0
            f = self._apply_lc5_forward(L) - L_target
            t2 = t * t
            df = (1.0 + p[0] * dt + p[1] * (dt * h5 + t * dh5) + p[2] * 2 * t * dt
                  + p[3] * (2 * t * dt * h5 + t2 * dh5) + p[4] * 3 * t2 * dt)
            df = torch.where(df.abs() < 1e-12, torch.ones_like(df), df)
            L = L - f / df
        return L

    def _apply_lc3_forward(self, L):
        c = self._lc3
        t = L * (1.0 - L)
        return L + c[0] * t + c[1] * t * (2 * L - 1) + c[2] * t * t

    def _apply_lc3_inverse(self, L_target):
        c = self._lc3
        L = L_target.clone()
        for _ in range(15):
            t = L * (1.0 - L)
            f = self._apply_lc3_forward(L) - L_target
            dt = 1.0 - 2.0 * L
            df = 1.0 + c[0] * dt + c[1] * (dt * (2 * L - 1) + t * 2) + c[2] * 2 * t * dt
            df = torch.where(df.abs() < 1e-12, torch.ones_like(df), df)
            L = L - f / df
        return L

    def _apply_L_forward(self, L):
        if self._L_pwl is not None:
            return self._L_pwl.forward(L)
        if self._lc7 is not None:
            return self._apply_lc7_forward(L)
        if self._lc5 is not None:
            return self._apply_lc5_forward(L)
        if self._lc3 is not None:
            return self._apply_lc3_forward(L)
        return L

    def _apply_L_inverse(self, L):
        if self._L_pwl is not None:
            return self._L_pwl.inverse(L)
        if self._lc7 is not None:
            return self._apply_lc7_inverse(L)
        if self._lc5 is not None:
            return self._apply_lc5_inverse(L)
        if self._lc3 is not None:
            return self._apply_lc3_inverse(L)
        return L

    # ─── pipeline ──────────────────────────────────────────────────

    def forward(self, xyz):
        # 0. Pre-M1 warp
        xyz = self._pre_m1_warp(xyz, +1)
        # 1. M1
        lms = xyz @ self._M1.T
        # 2. Cross-terms
        lms = self._apply_cross_forward(lms, xyz)
        # 3. Transfer
        lms_c = self._transfer.forward(lms)
        # 3.5 Neutral blend (default ON)
        if self._neutral_blend_on:
            lms_c = neutral_blend(lms_c)
        # 4. M2
        raw = lms_c @ self._M2.T
        L, a, b = raw[:, 0], raw[:, 1], raw[:, 2]
        # 5. Hue rotation
        a, b = self._hue_rot.forward(a, b)
        # 5b. Chroma power + L-dependent
        if self._has_chroma:
            a, b = self._apply_chroma_forward(L, a, b)
        # 5c. Ab axis scaling
        if self._has_ab:
            a, b = a * self._sa, b * self._sb
        # 5e. Hue-dep L
        if self._has_hue_L:
            L = self._apply_hue_L_forward(L, a, b)
        # 6. L correction
        L = self._apply_L_forward(L)
        # 7. L-gated hue enrichment
        if self._enrichment is not None:
            a, b = self._enrichment.forward(L, a, b)
        # 8. Neutral correction (LUT)
        if self._has_nc:
            a_err, b_err = self._nc_lut.errors_at(L)
            a, b = a - a_err, b - b_err
        return torch.stack([L, a, b], dim=-1)

    def inverse(self, lab):
        L_out = lab[:, 0].clone()
        a, b = lab[:, 1].clone(), lab[:, 2].clone()
        # 8. Undo NC
        if self._has_nc:
            a_err, b_err = self._nc_lut.errors_at(L_out)
            a, b = a + a_err, b + b_err
        # 7. Undo enrichment
        if self._enrichment is not None:
            a, b = self._enrichment.inverse(L_out, a, b)
        # 6. Undo L correction
        L = self._apply_L_inverse(L_out)
        # 5e. Undo hue-dep L
        if self._has_hue_L:
            L = self._apply_hue_L_inverse(L, a, b)
        # 5c. Undo ab scaling
        if self._has_ab:
            a, b = a / self._sa, b / self._sb
        # 5b. Undo chroma
        if self._has_chroma:
            a, b = self._apply_chroma_inverse(L, a, b)
        # 5. Undo hue rotation
        a, b = self._hue_rot.inverse(a, b)
        # 4. Undo M2
        raw = torch.stack([L, a, b], dim=-1)
        lms_c = raw @ self._M2_inv.T
        # 3.5 Neutral blend (matching forward)
        if self._neutral_blend_on:
            lms_c = neutral_blend(lms_c)
        # 3. Undo transfer
        lms = self._transfer.inverse(lms_c)
        # 2+1. Cross-term + M1 inverse (analytical, fused)
        xyz = lms @ self._M1_mod_inv.T
        # 0. Undo pre-warp
        return self._pre_m1_warp(xyz, -1)
