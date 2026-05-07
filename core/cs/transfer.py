"""Color-space transfer functions — forward + inverse, sign-preserving.

Each transfer is a small dataclass-like object exposing .forward(x) and .inverse(y).
All preserve dtype and device of input tensors.

Available transfers:
  CbrtTransfer       — cbrt forward, cube inverse (analytic)
  DepCubicTransfer   — x³ + α·x = y; forward via Cardano + Halley refine,
                       inverse analytic
  NakaRushtonTransfer — n·s · |x|^n / (|x|^n + σ^n) saturation
  SoftcbrtTransfer   — (|x|+ε)^(1/3) - ε^(1/3) (smooth near zero)
  CielabDeltaTransfer — CIE 1976 piecewise (delta=0.008856 default)
  PowerTransfer      — |x|^γ, sign-preserving
  RationalTransfer   — x·(a+b·x)/(1+c·x) (smooth saturation)

Factory helper `from_json_spec` materializes the right transfer from a
HelmCT JSON checkpoint's `transfer` field + auxiliary parameters.
"""
import torch

from .base import signed_cbrt, signed_cube


class CbrtTransfer:
    """y = sign(x)·|x|^(1/3); inverse = sign(y)·|y|³. Bit-exact bijective."""
    name = "cbrt"

    def forward(self, x):
        return signed_cbrt(x)

    def inverse(self, y):
        return signed_cube(y)


class DepCubicTransfer:
    """Helmgen's depressed-cubic: y³ + α·y = x.

    Forward solves for y given x (Cardano via sinh substitution + Halley refine).
    Inverse is analytic (exact, no iteration).
    """
    name = "depcubic"

    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def forward(self, x):
        a = self.alpha
        if a == 0.0:
            return signed_cbrt(x)
        s = (a / 3.0) ** 0.5
        s3_2 = 2.0 * s ** 3
        t = x / s3_2
        # Cardano initial guess via sinh
        y = 2.0 * s * torch.sinh(torch.arcsinh(t) / 3.0)
        # Halley refinement (cubic convergence; matches PyTorch baseline)
        for _ in range(1):
            y2 = y * y
            y3 = y * y2
            f = y3 + a * y - x
            fp = 3.0 * y2 + a
            fpp = 6.0 * y
            denom = 2.0 * fp * fp - f * fpp
            safe = denom.abs() > 1e-30
            denom = torch.where(safe, denom, torch.ones_like(denom))
            y = torch.where(safe, y - 2.0 * f * fp / denom, y)
        return y

    def inverse(self, y):
        return y * y * y + self.alpha * y


class SoftcbrtTransfer:
    """y = sign(x) · ((|x|+ε)^(1/3) - ε^(1/3)). Smooth near zero."""
    name = "softcbrt"

    def __init__(self, eps: float):
        self.eps = float(eps)

    def forward(self, x):
        eps = self.eps
        ax = x.abs()
        return x.sign() * ((ax + eps).pow(1.0 / 3.0) - eps ** (1.0 / 3.0))

    def inverse(self, y):
        eps = self.eps
        eps_cbrt = eps ** (1.0 / 3.0)
        ay = y.abs()
        return y.sign() * ((ay + eps_cbrt).pow(3.0) - eps)


class CielabDeltaTransfer:
    """CIE L*a*b* 1976 piecewise transfer.

    Forward: f(t) = t^(1/3) if t > δ³, else (κ·t + 16)/116
    Inverse: f^-1(s) = s³ if s > δ, else (116·s - 16)/κ
    """
    name = "cielab_delta"

    def __init__(self, delta: float = 0.008856, kappa: float = 903.3):
        self.delta = float(delta)
        self.kappa = float(kappa)

    def forward(self, x):
        ax = x.abs()
        cbrt = ax.pow(1.0 / 3.0)
        lin = (self.kappa * ax + 16.0) / 116.0
        return x.sign() * torch.where(ax > self.delta, cbrt, lin)

    def inverse(self, y):
        ay = y.abs()
        cube = ay.pow(3.0)
        lin = (116.0 * ay - 16.0) / self.kappa
        f_delta = (self.kappa * self.delta + 16.0) / 116.0
        return y.sign() * torch.where(ay > f_delta, cube, lin)


class PowerTransfer:
    """y = sign(x) · |x|^γ. Inverse: |y|^(1/γ)."""
    name = "power"

    def __init__(self, gamma: float = 1.0 / 3.0):
        self.gamma = float(gamma)

    def forward(self, x):
        return x.sign() * x.abs().pow(self.gamma)

    def inverse(self, y):
        return y.sign() * y.abs().pow(1.0 / self.gamma)


class NakaRushtonTransfer:
    """y = sign(x) · s · |x|^n / (|x|^n + σ^n). Photoreceptor-style saturation."""
    name = "naka_rushton"

    def __init__(self, n: float = 0.76, sigma: float = 0.33, s: float = 0.71):
        self.n = float(n)
        self.sigma = float(sigma)
        self.s = float(s)

    def forward(self, x):
        ax = x.abs().clamp(min=1e-30)
        axn = ax.pow(self.n)
        return x.sign() * self.s * axn / (axn + self.sigma ** self.n)

    def inverse(self, y):
        ay = y.abs().clamp(min=1e-30)
        ratio = ay / (self.s - ay).clamp(min=1e-30)
        return y.sign() * self.sigma * ratio.pow(1.0 / self.n)


class RationalTransfer:
    """y = sign(x) · |x| · (a + b·|x|) / (1 + c·|x|).

    Inverse via quadratic formula: b·t² + (a - c·|y|)·t - |y| = 0, t = |x|.
    """
    name = "rational"

    def __init__(self, a: float = 3.8, b: float = 2.2, c: float = 5.0):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def forward(self, x):
        ax = x.abs()
        return x.sign() * ax * (self.a + self.b * ax) / (1.0 + self.c * ax)

    def inverse(self, y):
        ay = y.abs()
        disc = (self.a - self.c * ay) ** 2 + 4.0 * self.b * ay
        disc = disc.clamp(min=0.0)
        return y.sign() * (-(self.a - self.c * ay) + torch.sqrt(disc)) / (2.0 * self.b)


def from_json_spec(d: dict):
    """Build a transfer from a HelmCT JSON checkpoint's parameters.

    Args:
        d: full JSON dict (must contain 'transfer' key, plus auxiliaries)

    Returns: a transfer object with .forward / .inverse methods.
    """
    t = d.get("transfer", "cbrt")
    if t == "cbrt":
        return CbrtTransfer()
    if t == "depcubic":
        return DepCubicTransfer(d.get("depcubic_alpha", 0.015))
    if t == "naka_rushton":
        return NakaRushtonTransfer(
            n=d.get("nr_n", 0.76),
            sigma=d.get("nr_sigma", 0.33),
            s=d.get("nr_s", 0.71),
        )
    if t == "softcbrt":
        return SoftcbrtTransfer(d.get("softcbrt_eps", 0.001))
    if t == "cielab_delta":
        return CielabDeltaTransfer(
            delta=d.get("cielab_delta", 0.008856),
            kappa=d.get("cielab_kappa", 903.3),
        )
    if t == "power":
        return PowerTransfer(d.get("gamma_val", 1.0 / 3.0))
    if t == "rational":
        return RationalTransfer(
            a=d.get("rational_a", 3.8),
            b=d.get("rational_b", 2.2),
            c=d.get("rational_c", 5.0),
        )
    raise ValueError(f"Unknown transfer '{t}'")
