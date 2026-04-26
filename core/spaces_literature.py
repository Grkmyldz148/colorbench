"""Literature color spaces — benchmarking scan.

Five published color spaces implemented to the ColorBench `ColorSpace` interface:
  1. IPT        (Ebner & Fairchild 1998)
  2. JzAzBz     (Safdar, Cui, Kim, Luo 2017)
  3. ICtCp      (Dolby / ITU-R BT.2100)
  4. CAM16-UCS  (Li, Li, Wang, Luo 2017; Jab form)  — may fall back to DIN99d
  5. DIN99d     (Cui et al. 2002; German standard extension)

Each forward: XYZ (N,3) → Lab-like (N,3), float64.
Each inverse: Lab-like (N,3) → XYZ  (N,3), float64.
XYZ convention: D65 white at Y=1 (i.e. [0.95047, 1.0, 1.08883]).
"""

from __future__ import annotations

import math
import torch

from .spaces import ColorSpace, D65


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _signed_pow(x: torch.Tensor, p: float) -> torch.Tensor:
    """sign(x) * |x|^p — bijective sign-preserving power (for odd-symmetric transfers)."""
    return x.sign() * x.abs().pow(p)


# ═════════════════════════════════════════════════════════════════════════════
# 1. IPT  (Ebner & Fairchild 1998)
# ═════════════════════════════════════════════════════════════════════════════

class IPT(ColorSpace):
    """IPT — Ebner & Fairchild (1998) 'Development and testing of a color space (IPT)…'.

    Hunt-Pointer-Estevez D65 LMS → whitepoint normalization → |x|^0.43 → opponent I,P,T.
    Input XYZ is D65-normalized (Y=1).

    The whitepoint LMS_w normalization (LMS' = LMS / LMS_w) is the canonical Ebner&Fairchild
    formulation — with published 4-digit matrix it drops achromatic error from ~1e-4 to ~1e-16.
    """

    name = "IPT"

    def __init__(self, device: torch.device):
        self.device = device
        # Hunt-Pointer-Estevez for D65 (Ebner & Fairchild)
        self.M_LMS = torch.tensor([
            [ 0.4002,  0.7075, -0.0807],
            [-0.2280,  1.1500,  0.0612],
            [ 0.0000,  0.0000,  0.9184],
        ], device=device, dtype=torch.float64)

        # LMS' → IPT
        self.M_IPT = torch.tensor([
            [ 0.4000,  0.4000,  0.2000],
            [ 4.4550, -4.8510,  0.3960],
            [ 0.8056,  0.3572, -1.1628],
        ], device=device, dtype=torch.float64)

        self.M_LMS_inv = torch.linalg.inv(self.M_LMS)
        self.M_IPT_inv = torch.linalg.inv(self.M_IPT)

        # Whitepoint LMS for D65 @ Y=1
        d65 = D65.to(device)
        self.LMS_w = d65 @ self.M_LMS.T  # (3,)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        lms = xyz @ self.M_LMS.T
        lms_n = lms / self.LMS_w           # whitepoint-normalize
        lms_p = _signed_pow(lms_n, 0.43)
        return lms_p @ self.M_IPT.T

    def inverse(self, ipt: torch.Tensor) -> torch.Tensor:
        lms_p = ipt @ self.M_IPT_inv.T
        lms_n = _signed_pow(lms_p, 1.0 / 0.43)
        lms = lms_n * self.LMS_w           # undo whitepoint-normalize
        return lms @ self.M_LMS_inv.T


# ═════════════════════════════════════════════════════════════════════════════
# 2. JzAzBz  (Safdar, Cui, Kim, Luo 2017)
# ═════════════════════════════════════════════════════════════════════════════

class JzAzBz(ColorSpace):
    """JzAzBz — Safdar, Cui, Kim, Luo (2017).

    HDR-ready perceptual space based on PQ. Input XYZ is D65-normalized (Y=1);
    internally scaled to 100 cd/m² reference.

    PQ constants (verified against Safdar 2017 & Colour-Science reference):
        c1 = 3424/4096  = 0.8359375
        c2 = 2413/128   = 18.8515625
        c3 = 2392/128   = 18.6875
        n  = 2610/16384 = 0.15930175781
        p  = 1.7 * 2523/32 = 134.034375
    """

    name = "JzAzBz"

    # PQ constants
    _c1 = 3424.0 / 4096.0
    _c2 = 2413.0 / 128.0
    _c3 = 2392.0 / 128.0
    _n  = 2610.0 / 16384.0
    _p  = 1.7 * 2523.0 / 32.0
    _n_inv = 1.0 / _n
    _p_inv = 1.0 / _p

    _b = 1.15
    _g = 0.66
    _d = -0.56
    _d0 = 1.6295499532821566e-11

    def __init__(self, device: torch.device):
        self.device = device

        # Pre-conditioning:  X' = b*X - (b-1)*Z;  Y' = g*Y + (1-g)*X;  Z' = Z
        # In matrix form on [X, Y, Z]:
        self.M_pre = torch.tensor([
            [self._b, 0.0,   -(self._b - 1.0)],
            [1.0 - self._g, self._g, 0.0],
            [0.0, 0.0, 1.0],
        ], device=device, dtype=torch.float64)

        self.M_JZ = torch.tensor([
            [ 0.41478972,  0.579999,   0.0146480],
            [-0.2015100,   1.120649,   0.0531008],
            [-0.0166008,   0.264800,   0.6684799],
        ], device=device, dtype=torch.float64)

        self.M_IZ = torch.tensor([
            [0.5,       0.5,        0.0],
            [3.524000, -4.066708,   0.542708],
            [0.199076,  1.096799,  -1.295875],
        ], device=device, dtype=torch.float64)

        self.M_pre_inv = torch.linalg.inv(self.M_pre)
        self.M_JZ_inv = torch.linalg.inv(self.M_JZ)
        self.M_IZ_inv = torch.linalg.inv(self.M_IZ)

        # Whitepoint LMS for D65 at Y=100 scale.  We normalize LMS by LMS_w before PQ
        # so that D65-grey inputs give exactly zero chromatic signal.  This is a
        # benchmarking-friendly modification of JzAzBz (the original published form
        # leaves ~1e-4 chroma at D65).  Jz values are scaled accordingly: peak white
        # corresponds to LMS/LMS_w = 1 → PQ(1*100/10000) = PQ(0.01).
        d65 = D65.to(device) * 100.0
        XYZp_w = d65 @ self.M_pre.T
        self.LMS_w = XYZp_w @ self.M_JZ.T         # (3,)

    # ── PQ forward and inverse (operates on linear light) ────────────────
    def _pq(self, x: torch.Tensor) -> torch.Tensor:
        # Safe for nonneg or near-zero negative values (use signed pow to stay stable)
        xn = _signed_pow(x, self._n)
        return ((self._c1 + self._c2 * xn) / (1.0 + self._c3 * xn)).clamp(min=0).pow(self._p)

    def _pq_inv(self, y: torch.Tensor) -> torch.Tensor:
        # y = ((c1 + c2*xn) / (1 + c3*xn))^p
        # → z = y^(1/p); xn = (z - c1) / (c2 - c3*z)
        z = y.clamp(min=0).pow(self._p_inv)
        num = z - self._c1
        den = self._c2 - self._c3 * z
        xn = num / den
        # x = xn^(1/n), sign-preserving (xn may be slightly negative for blacks)
        x = _signed_pow(xn, self._n_inv)
        return x

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # Scale to 100 cd/m² reference (Y=1 becomes Y=100)
        XYZ = xyz * 100.0
        # Pre-conditioning
        XYZp = XYZ @ self.M_pre.T
        # LMS (linear)
        LMS = XYZp @ self.M_JZ.T
        # Whitepoint-normalize so D65 gives LMS_n = (1,1,1) exactly
        LMS_n = LMS / self.LMS_w
        # PQ; scale LMS_n by 100/10000 = 0.01 so D65 input → PQ(0.01), matching original
        LMSp = self._pq(LMS_n * 0.01)
        # Opponent
        IZ = LMSp @ self.M_IZ.T
        Iz = IZ[:, 0]
        # Jz compression
        Jz = (1.0 + self._d) * Iz / (1.0 + self._d * Iz) - self._d0
        return torch.stack([Jz, IZ[:, 1], IZ[:, 2]], dim=-1)

    def inverse(self, jab: torch.Tensor) -> torch.Tensor:
        Jz = jab[:, 0]
        az = jab[:, 1]
        bz = jab[:, 2]
        # Invert Jz: Iz = (Jz + d0) / (1 + d - d*(Jz + d0))
        J_d0 = Jz + self._d0
        Iz = J_d0 / (1.0 + self._d - self._d * J_d0)
        IZ = torch.stack([Iz, az, bz], dim=-1)
        # LMSp
        LMSp = IZ @ self.M_IZ_inv.T
        # Invert PQ → LMS_n * 0.01, then undo scale and whitepoint
        LMS_n_scaled = self._pq_inv(LMSp)
        LMS_n = LMS_n_scaled / 0.01
        LMS = LMS_n * self.LMS_w
        # Invert M_JZ
        XYZp = LMS @ self.M_JZ_inv.T
        # Invert pre-conditioning
        XYZ = XYZp @ self.M_pre_inv.T
        # Unscale from 100 cd/m²
        return XYZ / 100.0


# ═════════════════════════════════════════════════════════════════════════════
# 3. ICtCp  (Dolby / ITU-R BT.2100)
# ═════════════════════════════════════════════════════════════════════════════

class ICtCp(ColorSpace):
    """ICtCp — ITU-R BT.2100 (Dolby). PQ variant.

    XYZ → Rec.2020 linear RGB → LMS → PQ(LMS, BT.2100) → ICtCp opponent.
    Input XYZ is D65-normalized (Y=1); diffuse white is mapped to ~203 cd/m²
    (a common HDR reference).
    """

    name = "ICtCp"

    # Same PQ constants as JzAzBz (BT.2100 PQ)
    _c1 = 3424.0 / 4096.0
    _c2 = 2413.0 / 128.0
    _c3 = 2392.0 / 128.0
    _n  = 2610.0 / 16384.0
    _p  = 2523.0 / 32.0          # Note: BT.2100 is 2523/32 ≈ 78.84375
    _n_inv = 1.0 / _n
    _p_inv = 1.0 / _p

    # Peak mapping: diffuse white -> 203 cd/m² relative to 10000 cd/m² peak
    _peak_scale = 203.0 / 10000.0

    def __init__(self, device: torch.device):
        self.device = device

        # XYZ (D65) → Rec.2020 linear RGB
        self.M_XYZ_to_R2020 = torch.tensor([
            [ 1.7166511,  -0.3556708,  -0.2533663],
            [-0.6666844,   1.6164812,   0.0157685],
            [ 0.0176399,  -0.0427706,   0.9421031],
        ], device=device, dtype=torch.float64)

        # Rec.2020 → LMS (BT.2100 ICtCp primary matrix, dyadic /4096)
        self.M_ICT = torch.tensor([
            [1688.0/4096.0, 2146.0/4096.0,  262.0/4096.0],
            [ 683.0/4096.0, 2951.0/4096.0,  462.0/4096.0],
            [  99.0/4096.0,  309.0/4096.0, 3688.0/4096.0],
        ], device=device, dtype=torch.float64)

        # LMS' → ICtCp
        self.M_ICT2 = torch.tensor([
            [ 2048.0/4096.0,   2048.0/4096.0,     0.0         ],
            [ 6610.0/4096.0, -13613.0/4096.0,  7003.0/4096.0],
            [17933.0/4096.0, -17390.0/4096.0,  -543.0/4096.0],
        ], device=device, dtype=torch.float64)

        self.M_XYZ_to_R2020_inv = torch.linalg.inv(self.M_XYZ_to_R2020)
        self.M_ICT_inv = torch.linalg.inv(self.M_ICT)
        self.M_ICT2_inv = torch.linalg.inv(self.M_ICT2)

        # Whitepoint LMS for D65 (Y=1).  We normalize LMS by LMS_w before PQ
        # so that D65-grey inputs produce exactly zero Ct, Cp.  (Benchmark-friendly mod;
        # the original ITU BT.2100 form leaves ~1e-5 chroma at D65.)
        d65 = D65.to(device)
        RGB_w = d65 @ self.M_XYZ_to_R2020.T
        self.LMS_w = RGB_w @ self.M_ICT.T         # (3,)

    def _pq(self, x: torch.Tensor) -> torch.Tensor:
        xn = _signed_pow(x, self._n)
        return ((self._c1 + self._c2 * xn) / (1.0 + self._c3 * xn)).clamp(min=0).pow(self._p)

    def _pq_inv(self, y: torch.Tensor) -> torch.Tensor:
        z = y.clamp(min=0).pow(self._p_inv)
        num = z - self._c1
        den = self._c2 - self._c3 * z
        xn = num / den
        return _signed_pow(xn, self._n_inv)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        RGB = xyz @ self.M_XYZ_to_R2020.T
        LMS = RGB @ self.M_ICT.T
        # Whitepoint-normalize LMS (benchmark mod — clean achromatic at D65)
        LMS_n = LMS / self.LMS_w
        # Apply PQ with diffuse white -> 203 cd/m² (peak 10000)
        LMSp = self._pq(LMS_n * self._peak_scale)
        return LMSp @ self.M_ICT2.T

    def inverse(self, ictcp: torch.Tensor) -> torch.Tensor:
        LMSp = ictcp @ self.M_ICT2_inv.T
        LMS_scaled = self._pq_inv(LMSp)
        LMS_n = LMS_scaled / self._peak_scale
        LMS = LMS_n * self.LMS_w
        RGB = LMS @ self.M_ICT_inv.T
        XYZ = RGB @ self.M_XYZ_to_R2020_inv.T
        return XYZ


# ═════════════════════════════════════════════════════════════════════════════
# 4. CAM16-UCS  (Li, Li, Wang, Luo 2017) — Jab form
# ═════════════════════════════════════════════════════════════════════════════

class CAM16UCS(ColorSpace):
    """CAM16-UCS — Li, Li, Wang, Luo (2017). Jab form.

    Fixed D65 viewing conditions (La=40, Yb=20, average surround F=1, c=0.69, Nc=1).
    Output: [J', a', b']  where a' = M' cos h, b' = M' sin h.

    For benchmark purposes we use FULL chromatic adaptation (D=1). The natural
    CAM16 degree-of-adaptation factor D ≈ 0.886 leaves small nonzero chroma at
    the white point (~1.6 UCS units); setting D=1 makes achromatic exactly
    zero, allowing fair comparison with zero-bias spaces like OKLab.
    """

    name = "CAM16-UCS"

    # UCS coefficients
    _c1 = 0.007
    _c2 = 0.0228

    def __init__(self, device: torch.device):
        self.device = device

        # Viewing conditions (D65, La=40, Yb=20, average surround)
        La = 40.0
        Yb = 20.0
        Yw = 100.0
        c_surr = 0.69
        Nc = 1.0

        k = 1.0 / (5.0 * La + 1.0)
        F_L = 0.2 * (k ** 4) * (5.0 * La) + 0.1 * ((1.0 - k ** 4) ** 2) * ((5.0 * La) ** (1.0 / 3.0))
        n_ratio = Yb / Yw
        z = 1.48 + math.sqrt(n_ratio)
        Nbb = 0.725 * (1.0 / n_ratio) ** 0.2
        Ncb = Nbb
        D = 1.0  # Full chromatic adaptation (benchmark mode)

        self.F_L = F_L
        self.c_surr = c_surr
        self.Nc = Nc
        self.Nbb = Nbb
        self.Ncb = Ncb
        self.z = z
        self.D = D
        self.n_ratio = n_ratio

        # CAT16 matrix (XYZ → LMS cone-like)
        self.M16 = torch.tensor([
            [ 0.401288,  0.650173, -0.051461],
            [-0.250268,  1.204414,  0.045854],
            [-0.002079,  0.048952,  0.953127],
        ], device=device, dtype=torch.float64)
        self.M16_inv = torch.linalg.inv(self.M16)

        # Compute D-weighted white for adaptation (D=1 gives RGBwc = Yw for all channels)
        XYZw = torch.tensor([95.047, 100.0, 108.883], device=device, dtype=torch.float64)
        RGBw = XYZw @ self.M16.T
        D_RGBw = D * (Yw / RGBw) + (1.0 - D)
        self.D_RGBw = D_RGBw
        # Post-adaptation white cone responses — should be (Yw, Yw, Yw) with D=1
        RGBwc = D_RGBw * RGBw
        a_w = self._cone_compress(RGBwc)
        A_w = (2.0 * a_w[0] + a_w[1] + (1.0 / 20.0) * a_w[2]) * Nbb
        self.A_w = A_w.item() if hasattr(A_w, "item") else A_w

        # Precompute alpha scale factor (chroma normalization from paper, Eq. 11)
        self._K_alpha = (1.64 - 0.29 ** self.n_ratio) ** 0.73

    # ── Cone compression (forward + inverse) ────────────────────────────
    def _cone_compress(self, R: torch.Tensor) -> torch.Tensor:
        """Post-adaptation cone response.  Sign-preserving for safety on negative R."""
        x = (self.F_L * R.abs() / 100.0).pow(0.42)
        # a = sign(R) * 400*x / (x+27.13) + 0.1   (per CAM16 spec; +0.1 offset unsigned)
        return torch.sign(R) * (400.0 * x / (x + 27.13)) + 0.1

    def _cone_decompress(self, a: torch.Tensor) -> torch.Tensor:
        """Inverse of _cone_compress.  a → R (closed form)."""
        # r = a - 0.1;  core = sign(R) * 400*x/(x+27.13) = r
        r = a - 0.1
        rabs = r.abs().clamp(max=400.0 - 1e-12)
        # rabs = 400*x/(x+27.13) → x = 27.13*rabs / (400 - rabs)
        x = 27.13 * rabs / (400.0 - rabs)
        R_abs = 100.0 / self.F_L * x.pow(1.0 / 0.42)
        return torch.sign(r) * R_abs

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # CAM16 expects XYZ on [0,100] scale — our input is Y=1 normalized.
        XYZ = xyz * 100.0
        RGB = XYZ @ self.M16.T
        RGBc = RGB * self.D_RGBw                       # chromatic adaptation (D=1 here)
        a = self._cone_compress(RGBc)
        Ra, Ga, Ba = a[:, 0], a[:, 1], a[:, 2]

        # Opponent signals
        a_opp = Ra - (12.0 / 11.0) * Ga + (1.0 / 11.0) * Ba
        b_opp = (1.0 / 9.0) * (Ra + Ga - 2.0 * Ba)

        # Hue
        h_rad = torch.atan2(b_opp, a_opp)

        # Lightness J from achromatic signal A
        A = (2.0 * Ra + Ga + (1.0 / 20.0) * Ba) * self.Nbb
        J_ratio = (A / self.A_w).clamp(min=0)
        J = 100.0 * J_ratio.pow(self.c_surr * self.z)

        # Chroma C via t (Eq. 10–11 of Li et al. 2017)
        e_t = 0.25 * (torch.cos(h_rad + 2.0) + 3.8)
        t_num = (50000.0 / 13.0) * self.Nc * self.Ncb * e_t * torch.sqrt(a_opp ** 2 + b_opp ** 2)
        t_den = Ra + Ga + (21.0 / 20.0) * Ba            # never negative for real colors
        t = t_num / t_den.clamp(min=1e-30)
        alpha = t.clamp(min=0).pow(0.9) * self._K_alpha
        C = alpha * (J / 100.0).clamp(min=0).sqrt()

        # Colorfulness M, UCS scaling
        M = C * (self.F_L ** 0.25)
        Jp = (1.0 + 100.0 * self._c1) * J / (1.0 + self._c1 * J)
        Mp = torch.log1p(self._c2 * M) / self._c2       # (1/c2) * ln(1 + c2*M)
        ap = Mp * torch.cos(h_rad)
        bp = Mp * torch.sin(h_rad)
        return torch.stack([Jp, ap, bp], dim=-1)

    def inverse(self, jab: torch.Tensor) -> torch.Tensor:
        Jp = jab[:, 0]
        ap = jab[:, 1]
        bp = jab[:, 2]

        # Invert UCS J'  →  J
        J = Jp / ((1.0 + 100.0 * self._c1) - self._c1 * Jp)
        # Hue + colorfulness magnitude
        Mp = torch.sqrt(ap ** 2 + bp ** 2)
        h_rad = torch.atan2(bp, ap)
        M = torch.expm1(self._c2 * Mp) / self._c2
        C = M / (self.F_L ** 0.25)

        # Recover t from C, J via C = alpha * sqrt(J/100);  alpha = t^0.9 * K
        sqrtJ = (J / 100.0).clamp(min=0).sqrt()
        alpha = C / sqrtJ.clamp(min=1e-30)
        t = (alpha / self._K_alpha).clamp(min=0).pow(1.0 / 0.9)

        # Recover achromatic signal A (hence p2)
        A = self.A_w * (J / 100.0).clamp(min=0).pow(1.0 / (self.c_surr * self.z))
        p2 = A / self.Nbb

        # Closed-form recovery of a_opp, b_opp.  Derivation:
        #   t = p1_hue * |ab| / t_den         (forward)
        #   where p1_hue = (50000/13)*Nc*Ncb*e_t,  t_den = Ra + Ga + (21/20)*Ba.
        #   Using Ra,Ga,Ba = linear(p2, a_opp, b_opp), t_den reduces to
        #       t_den = p2 - (671*cos(h) + 6588*sin(h))/1403 * |ab|
        #   Substituting and solving for |ab|:
        #       |ab| = t * p2 / ( p1_hue + t * (671*cos(h) + 6588*sin(h))/1403 )
        hs = torch.sin(h_rad)
        hc = torch.cos(h_rad)
        e_t = 0.25 * (torch.cos(h_rad + 2.0) + 3.8)
        p1_hue = (50000.0 / 13.0) * self.Nc * self.Ncb * e_t
        K_hue = (671.0 * hc + 6588.0 * hs) / 1403.0
        denom = p1_hue + t * K_hue
        magnitude = t * p2 / denom
        # When t → 0, chroma is zero
        magnitude = torch.where(t < 1e-20, torch.zeros_like(magnitude), magnitude)
        a_opp = magnitude * hc
        b_opp = magnitude * hs

        # Recover Ra, Ga, Ba from p2, a_opp, b_opp  (inverse of the CAM16 opponent matrix)
        Ra = (460.0 / 1403.0) * p2 + (451.0 / 1403.0) * a_opp + (288.0 / 1403.0) * b_opp
        Ga = (460.0 / 1403.0) * p2 - (891.0 / 1403.0) * a_opp - (261.0 / 1403.0) * b_opp
        Ba = (460.0 / 1403.0) * p2 - (220.0 / 1403.0) * a_opp - (6300.0 / 1403.0) * b_opp

        a = torch.stack([Ra, Ga, Ba], dim=-1)
        RGBc = self._cone_decompress(a)
        RGB = RGBc / self.D_RGBw
        XYZ = RGB @ self.M16_inv.T
        return XYZ / 100.0


# ═════════════════════════════════════════════════════════════════════════════
# 5. DIN99d  (Cui, Luo, Rigg, Roesler, Witt 2002 — German standard)
# ═════════════════════════════════════════════════════════════════════════════

class DIN99d(ColorSpace):
    """DIN99d — Cui et al. (2002). Improved DIN99 with XYZ pre-rotation.

    Pipeline: XYZ → (X', Y, Z) with X'=1.12*X - 0.12*Z → CIELab →
              L99 = 325.22*ln(1 + 0.0036*L*)
              rotate (a*,b*) by 50° → (e, f); f *= 1.14
              G = sqrt(e² + f²)
              C99 = 22.5 * ln(1 + 0.06*G)
              h99 = atan2(f, e) + 50°
              a99 = C99*cos(h99);  b99 = C99*sin(h99)
    """

    name = "DIN99d"

    # DIN99d constants
    _kL = 325.22
    _kA = 0.0036
    _kC = 22.5
    _kG = 0.06
    _theta = 50.0       # degrees
    _f_scale = 1.14     # f-axis compression

    def __init__(self, device: torch.device):
        self.device = device
        self.d65 = D65.to(device)
        self.delta = 6.0 / 29.0
        self.delta3 = self.delta ** 3
        # Pre-rotation matrix: X' = 1.12*X - 0.12*Z, Y, Z
        self.M_pre = torch.tensor([
            [ 1.12, 0.0, -0.12],
            [ 0.0,  1.0,  0.0 ],
            [ 0.0,  0.0,  1.0 ],
        ], device=device, dtype=torch.float64)
        self.M_pre_inv = torch.linalg.inv(self.M_pre)
        self._theta_rad = math.radians(self._theta)
        self._cos_t = math.cos(self._theta_rad)
        self._sin_t = math.sin(self._theta_rad)

        # Effective whitepoint for CIELab step = pre-rotated D65
        # (ensures XYZ = s * D65 produces exactly a*=b*=0)
        self.w_eff = self.d65 @ self.M_pre.T          # (3,)

    # ── CIELab forward/inverse using the pre-rotated whitepoint ─────────
    def _cielab_forward(self, XYZ_pre: torch.Tensor) -> torch.Tensor:
        r = XYZ_pre / self.w_eff
        f = torch.where(r > self.delta3,
                        r.abs().pow(1.0 / 3.0),
                        r / (3.0 * self.delta ** 2) + 4.0 / 29.0)
        L = 116.0 * f[:, 1] - 16.0
        a = 500.0 * (f[:, 0] - f[:, 1])
        b = 200.0 * (f[:, 1] - f[:, 2])
        return torch.stack([L, a, b], dim=-1)

    def _cielab_inverse(self, lab: torch.Tensor) -> torch.Tensor:
        fy = (lab[:, 0] + 16.0) / 116.0
        fx = lab[:, 1] / 500.0 + fy
        fz = fy - lab[:, 2] / 200.0
        f = torch.stack([fx, fy, fz], dim=-1)
        xyz = torch.where(f > self.delta,
                          f.pow(3.0),
                          3.0 * self.delta ** 2 * (f - 4.0 / 29.0))
        return xyz * self.w_eff

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # Pre-rotate XYZ
        XYZp = xyz @ self.M_pre.T
        # CIELab on pre-rotated XYZ
        Lab = self._cielab_forward(XYZp)
        L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]

        # DIN99d L compression
        # Use log1p for numerical stability (L may be 0)
        L99 = self._kL * torch.log1p(self._kA * L)

        # Rotate (a, b) by +50°; then scale f by 1.14
        e = a * self._cos_t + b * self._sin_t
        f_raw = -a * self._sin_t + b * self._cos_t
        f = self._f_scale * f_raw

        G = torch.sqrt(e * e + f * f + 1e-30)
        # Avoid sqrt(0) derivative problem; we'll zero out chroma below when G≈0

        C99 = self._kC * torch.log1p(self._kG * G)
        h99 = torch.atan2(f, e) + self._theta_rad

        # Safe handling at G==0: cos/sin * 0 is 0, fine
        a99 = C99 * torch.cos(h99)
        b99 = C99 * torch.sin(h99)

        # When G is machine-zero, result is exactly 0
        zero_mask = G < 1e-20
        a99 = torch.where(zero_mask, torch.zeros_like(a99), a99)
        b99 = torch.where(zero_mask, torch.zeros_like(b99), b99)
        return torch.stack([L99, a99, b99], dim=-1)

    def inverse(self, lab99: torch.Tensor) -> torch.Tensor:
        L99 = lab99[:, 0]
        a99 = lab99[:, 1]
        b99 = lab99[:, 2]

        # Invert L99
        L = torch.expm1(L99 / self._kL) / self._kA

        # Invert chroma compression
        C99 = torch.sqrt(a99 * a99 + b99 * b99 + 1e-30)
        G = torch.expm1(C99 / self._kC) / self._kG
        # Hue in DIN99d space
        h99 = torch.atan2(b99, a99)
        # Un-rotate by -theta
        h_rot = h99 - self._theta_rad
        e = G * torch.cos(h_rot)
        f = G * torch.sin(h_rot)
        # Undo f scaling
        f_raw = f / self._f_scale
        # Un-rotate by -theta: (a,b) = rot(-θ) * (e,f_raw)
        # Rotation inverse: a = e*cos(θ) - f_raw*sin(θ); b = e*sin(θ) + f_raw*cos(θ)
        a = e * self._cos_t - f_raw * self._sin_t
        b = e * self._sin_t + f_raw * self._cos_t

        # Zero-mask for achromatic
        zero_mask = C99 < 1e-20
        a = torch.where(zero_mask, torch.zeros_like(a), a)
        b = torch.where(zero_mask, torch.zeros_like(b), b)

        Lab = torch.stack([L, a, b], dim=-1)
        XYZp = self._cielab_inverse(Lab)
        XYZ = XYZp @ self.M_pre_inv.T
        return XYZ
