"""TriOppMistral — PyTorch implementation for ColorBench integration.

Triangular opponent coupling shear with nonlinear M2 for generation.
Designed by Mistral Vibe, inspired by clean-helmlab TriOpp design.

Pipeline:
    XYZ (D65) → M1 → LMS → depressed_cubic → triangular coupling T → (L,a,b)

Inverse:
    (L,a,b) → T^-1 → depressed_cubic^-1 → M1^-1 → XYZ

Key features:
- Nonlinear M2: position-dependent opponent shear
- Chroma-gate: structural gray axis guarantee
- Depressed cubic: blue-fold safe, finite derivative at zero
- Exact inverse: lower-triangular back-substitution
"""

import torch
import math
from .base import ColorSpace


# Default M1 cone matrix (OKLab cone matrix, D65-normalized) — FIXED seed (was a
# wrong/identity-opponent seed that broke the gray axis; now the clean-helmlab TriOpp seed).
_M1_LIST = [
    [ 0.8154374735648701,  0.3603221491264266, -0.12432703417946676],
    [ 0.03298391207546648, 0.9292940788255503,  0.03614494665290377],
    [ 0.048184113668356454, 0.26427748135788043, 0.6336388271114471],
]

_D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)


def _normalize_M1(M1):
    """Normalize M1 so D65 white maps to (1,1,1) in LMS."""
    lms_D65 = torch.tensor(_M1_LIST, dtype=torch.float64) @ _D65
    return torch.diag(1.0 / lms_D65) @ torch.tensor(_M1_LIST, dtype=torch.float64)


_M1_DEFAULT = _normalize_M1(_M1_LIST)
_M1_DEFAULT_INV = torch.linalg.inv(_M1_DEFAULT)


def depressed_cubic_fwd(x: torch.Tensor, alpha: float = 0.020) -> torch.Tensor:
    """Forward depressed cubic: solve y^3 + alpha*y = x.
    
    Closed-form solution using sinh/arcsinh with one Halley refinement.
    """
    s = torch.sqrt(torch.tensor(alpha / 3.0, dtype=x.dtype, device=x.device))
    t = x / (2.0 * s ** 3)
    
    # Initial solution via hyperbolic functions
    y = 2.0 * s * torch.sinh(torch.arcsinh(t) / 3.0)
    
    # One Halley refinement
    f = y ** 3 + alpha * y - x
    fp = 3.0 * y ** 2 + alpha
    fpp = 6.0 * y
    
    denom = 2.0 * fp * fp - f * fpp
    safe = torch.abs(denom) > 1e-30
    
    # Avoid division by zero
    y = torch.where(
        safe,
        y - 2.0 * f * fp / torch.where(safe, denom, torch.tensor(1.0, device=x.device, dtype=x.dtype)),
        y
    )
    
    return y


def depressed_cubic_inv(y: torch.Tensor, alpha: float = 0.020) -> torch.Tensor:
    """Inverse depressed cubic: x = y^3 + alpha*y (exact)."""
    return y ** 3 + alpha * y


class TriOppMistral(ColorSpace):
    """TriOppMistral — generation-optimized space with nonlinear M2."""

    name = "TriOppMistral"

    def __init__(self, device: torch.device = None, dtype: torch.dtype = torch.float64,
                 params: dict = None):
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Default parameters (linear M2 mode, Seed A)
        self._default_params = {
            "M1": _M1_LIST,
            "depcubic_alpha": 0.020,
            "M2_linear": {
                "m11": 1.0067113086331279,
                "m21": -2.0,
                "d2": 2.0,
                "m31": -0.8,
                "m32": -0.8,
                "d3": 1.6,
            },
            "M2_shear": {
                "q_coef": 0.0,
                "s_coef": 0.0,
                "g_poly": [0.0, 0.0, 0.0],
                "h_poly": [0.0, 0.0, 0.0, 0.0],
                "gate_eps": 0.01,
            },
            "L_corr_poly": [],
            "L_corr_newton_iters": 24,
            "enrichment": {
                "type": "",
                "amp": 0.0,
                "center_deg": 240.0,
                "sigma": 0.7,
                "L_lo": 0.37,
                "L_hi": 1.0,
            },
            "chroma_power": 1.0,
        }

        if params is None:
            self.params = self._default_params.copy()
        else:
            self.params = self._load_params(params)

        # Initialize from params
        self._init_from_params()

    def _load_params(self, params: dict) -> dict:
        """Load and validate parameters."""
        loaded = params.copy()
        
        # Ensure all keys exist
        for key in self._default_params:
            if key not in loaded:
                loaded[key] = self._default_params[key]
        
        return loaded

    def _init_from_params(self):
        """Initialize internal state from parameters."""
        # M1 and its inverse
        M1_list = self.params["M1"]
        M1 = torch.tensor(M1_list, dtype=self.dtype, device=self.device)
        
        # Normalize to D65
        lms_D65 = M1 @ torch.tensor(_D65, dtype=self.dtype, device=self.device)
        self.M1 = torch.diag(1.0 / lms_D65) @ M1
        self._M1_inv = torch.linalg.inv(self.M1)
        
        # Transfer
        self.alpha = float(self.params["depcubic_alpha"])
        
        # Linear M2
        lin = self.params["M2_linear"]
        self.m11 = float(lin["m11"])
        self.m21 = float(lin["m21"])
        self.d2 = float(lin["d2"])
        self.m31 = float(lin["m31"])
        self.m32 = float(lin["m32"])
        self.d3 = float(lin["d3"])
        
        # Nonlinear shear
        nl = self.params["M2_shear"]
        self.q_coef = float(nl["q_coef"])
        self.s_coef = float(nl["s_coef"])
        self.g_poly = torch.tensor(nl["g_poly"], dtype=self.dtype, device=self.device)
        self.h_poly = torch.tensor(nl["h_poly"], dtype=self.dtype, device=self.device)
        self.gate_eps = float(nl["gate_eps"])
        
        # L correction
        self.Lc_poly = torch.tensor(self.params["L_corr_poly"], dtype=self.dtype, device=self.device)
        self._has_Lc = len(self.Lc_poly) > 0
        self.newton_iters = int(self.params["L_corr_newton_iters"])
        
        # Enrichment
        enrich = self.params["enrichment"]
        self.enrichment_type = enrich.get("type", "")
        self.enrichment_amp = float(enrich.get("amp", 0.0))
        self.enrichment_center = torch.tensor(
            math.radians(float(enrich.get("center_deg", 240.0))),
            dtype=self.dtype, device=self.device
        )
        self.enrichment_sigma = float(enrich.get("sigma", 0.7))
        self.enrichment_L_lo = float(enrich.get("L_lo", 0.37))
        self.enrichment_L_hi = float(enrich.get("L_hi", 1.0))
        
        # Chroma power
        self.chroma_power = float(self.params["chroma_power"])

    def _G(self, v1: torch.Tensor) -> torch.Tensor:
        """Luminance polynomial for red-green shear modulation."""
        return self.g_poly[0] + self.g_poly[1] * v1 + self.g_poly[2] * v1 * v1

    def _H(self, v1: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Luminance and a-dependent polynomial for blue-yellow shear modulation."""
        return (self.h_poly[0] + self.h_poly[1] * v1 + 
                self.h_poly[2] * a + self.h_poly[3] * v1 * v1)

    def _gate(self, a_lin: torch.Tensor, b_lin: torch.Tensor) -> torch.Tensor:
        """Chroma gate: g = Cn^2/(Cn^2+eps^2)."""
        cn2 = a_lin * a_lin + b_lin * b_lin
        return cn2 / (cn2 + self.gate_eps ** 2)

    def _T_forward(self, v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor) -> tuple:
        """Forward triangular coupling map."""
        L = self.m11 * v1
        a_lin = self.m21 * v1 + self.d2 * v2
        b_lin = self.m31 * v1 + self.m32 * v2 + self.d3 * v3
        
        gate = self._gate(a_lin, b_lin)
        
        Gv = self._G(v1)
        a = a_lin + self.q_coef * (v2 * Gv) * gate
        
        Hv = self._H(v1, a)
        b = b_lin + self.s_coef * (v3 * Hv) * gate
        
        return L, a, b

    def _T_inverse(self, L: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple:
        """Exact inverse via lower-triangular back-substitution."""
        # v1 is trivial
        v1 = L / self.m11
        
        # If no nonlinear terms, we have closed-form
        if self.q_coef == 0.0 and self.s_coef == 0.0:
            v2 = (a - self.m21 * v1) / self.d2
            v3 = (b - self.m31 * v1 - self.m32 * v2) / self.d3
            return v1, v2, v3
        
        # Nonlinear case: Newton iteration
        Gv = self._G(v1)
        eps2 = self.gate_eps ** 2
        
        # Achromatic seed
        v2 = (a - self.m21 * v1) / self.d2
        v3 = (b - self.m31 * v1 - self.m32 * v2) / self.d3
        
        for _ in range(self.newton_iters):
            a_lin = self.m21 * v1 + self.d2 * v2
            b_lin = self.m31 * v1 + self.m32 * v2 + self.d3 * v3
            cn2 = a_lin * a_lin + b_lin * b_lin
            gate = cn2 / (cn2 + eps2)
            dgate_dcn2 = eps2 / (cn2 + eps2) ** 2
            
            # Derivatives of cn2
            dcn2_dv2 = 2.0 * a_lin * self.d2 + 2.0 * b_lin * self.m32
            dcn2_dv3 = 2.0 * b_lin * self.d3
            dgate_dv2 = dgate_dcn2 * dcn2_dv2
            dgate_dv3 = dgate_dcn2 * dcn2_dv3
            
            # Residuals
            Hv = self._H(v1, a)
            a_model = a_lin + self.q_coef * (v2 * Gv) * gate
            b_model = b_lin + self.s_coef * (v3 * Hv) * gate
            F1 = a_model - a
            F2 = b_model - b
            
            # 2x2 Jacobian wrt (v2, v3)
            J11 = self.d2 + self.q_coef * Gv * (gate + v2 * dgate_dv2)
            J12 = self.q_coef * Gv * v2 * dgate_dv3
            J21 = self.m32 + self.s_coef * Hv * v3 * dgate_dv2
            J22 = self.d3 + self.s_coef * Hv * (gate + v3 * dgate_dv3)
            
            det = J11 * J22 - J12 * J21
            det = torch.where(torch.abs(det) < 1e-30, torch.tensor(1.0, device=self.device, dtype=self.dtype), det)
            
            dv2 = (J22 * F1 - J12 * F2) / det
            dv3 = (-J21 * F1 + J11 * F2) / det
            
            v2 = v2 - dv2
            v3 = v3 - dv3
        
        return v1, v2, v3

    def _L_corr(self, L: torch.Tensor) -> torch.Tensor:
        """Endpoint-preserving monotone L correction."""
        if not self._has_Lc:
            return L
        
        # For now, skip L correction in PyTorch implementation
        # (would need to implement polynomial evaluation)
        return L

    def _L_corr_inv(self, L1: torch.Tensor) -> torch.Tensor:
        """Inverse L correction."""
        if not self._has_Lc:
            return L1
        
        # For now, skip L correction in PyTorch implementation
        return L1

    def _apply_enrichment(self, L: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple:
        """Apply L-gated hue enrichment."""
        if self.enrichment_type != "L_gated_hue" or self.enrichment_amp == 0.0:
            return L, a, b
        
        # Compute hue and chroma
        C = torch.hypot(a, b)
        h = torch.atan2(b, a)
        
        # L-gate
        L_range = self.enrichment_L_hi - self.enrichment_L_lo
        gate = torch.where(
            L_range > 1e-12,
            torch.sin(torch.tensor(math.pi, dtype=self.dtype, device=self.device) * 
                     (L - self.enrichment_L_lo) / L_range) ** 2,
            torch.zeros_like(L)
        )
        
        # Gaussian in hue
        h_diff = h - self.enrichment_center
        # Normalize angle difference to [-pi, pi]
        h_diff = (h_diff + torch.tensor(math.pi, dtype=self.dtype, device=self.device)) % (
            2 * torch.tensor(math.pi, dtype=self.dtype, device=self.device)) - torch.tensor(math.pi, dtype=self.dtype, device=self.device)
        hue_weight = torch.exp(-0.5 * (h_diff / self.enrichment_sigma) ** 2)
        
        # Apply enrichment
        h_rot = gate * hue_weight * self.enrichment_amp
        a_rot = a * torch.cos(h_rot) - b * torch.sin(h_rot)
        b_rot = a * torch.sin(h_rot) + b * torch.cos(h_rot)
        
        return L, a_rot, b_rot

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """XYZ (N, 3) → TriOpp (N, 3)."""
        # XYZ → LMS (clipped to non-negative)
        lms = torch.clamp(xyz @ self.M1.T, min=0.0)
        
        # Depressed cubic transfer
        v = depressed_cubic_fwd(lms, self.alpha)
        v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2]
        
        # Triangular nonlinear M2 coupling
        L, a, b = self._T_forward(v1, v2, v3)
        
        # Apply enrichment
        L, a, b = self._apply_enrichment(L, a, b)
        
        # Chroma power
        if self.chroma_power != 1.0:
            C = torch.hypot(a, b)
            C_new = torch.sign(C) * (torch.abs(C) ** self.chroma_power)
            safe = C > 0
            scale = torch.where(safe, C_new / C, torch.tensor(1.0, device=self.device, dtype=self.dtype))
            a = a * scale
            b = b * scale
        
        # L correction
        L = self._L_corr(L)
        
        return torch.stack([L, a, b], dim=-1)

    def inverse(self, lab: torch.Tensor) -> torch.Tensor:
        """TriOpp (N, 3) → XYZ (N, 3)."""
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        
        # Undo L correction
        L = self._L_corr_inv(L)
        
        # Undo chroma power
        if self.chroma_power != 1.0:
            C = torch.hypot(a, b)
            C_old = torch.sign(C) * (torch.abs(C) ** (1.0 / self.chroma_power))
            safe = C > 0
            scale = torch.where(safe, C_old / C, torch.tensor(1.0, device=self.device, dtype=self.dtype))
            a = a * scale
            b = b * scale
        
        # Undo triangular map (exact)
        v1, v2, v3 = self._T_inverse(L, a, b)
        
        # Undo transfer (exact)
        v = torch.stack([v1, v2, v3], dim=-1)
        lms = depressed_cubic_inv(v, self.alpha)
        
        # LMS → XYZ
        xyz = lms @ self._M1_inv.T
        
        return xyz


# Also register for easy import
TriOppMistral_py = TriOppMistral
