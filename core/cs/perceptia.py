"""Perceptia-Spacing — the chroma/hue-priority GENERATION specialist (research output
2026-05-30). A flat uniform embedding fit to human Munsell spacing (held-out),
generalizes to independent OSA-UCS. Wins chroma/hue uniformity; the value axis is the
value-priority specialist's job (regime-split family) — so expect strong chroma/
gradient metrics, weaker value/lightness ones. NOT a universal champion by design.

Embedding (on CIELAB base):
  L' = L*^q ;  a' = w(h)·C*^p·cos h ;  b' = w(h)·C*^p·sin h ;  w(h)=exp(hue harmonics)
"""
import torch
from .base import ColorSpace

_D65 = [0.95047, 1.0, 1.08883]
_DELTA = 6.0 / 29.0
_DELTA3 = _DELTA ** 3
# frozen (perceptia_spacing_params.json, Munsell held-out)
_Q = 1.024
_P = 0.984
_HUE = [-0.004, 0.065, -0.222, 0.081, 0.043, 0.005, 0.026]


class Perceptia(ColorSpace):
    name = "Perceptia-Spacing"

    def __init__(self, device=None, dtype=torch.float64):
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.d65 = torch.tensor(_D65, device=self.device, dtype=self.dtype)
        self.hue = torch.tensor(_HUE, device=self.device, dtype=self.dtype)

    def _cielab(self, xyz):
        r = xyz / self.d65
        f = torch.where(r > _DELTA3, r.pow(1/3.), r/(3*_DELTA**2) + 4/29.)
        L = 116*f[:,1] - 16; a = 500*(f[:,0]-f[:,1]); b = 200*(f[:,1]-f[:,2])
        return L, a, b

    def _cielab_inv(self, L, a, b):
        fy = (L+16)/116.; fx = a/500.+fy; fz = fy-b/200.
        f = torch.stack([fx,fy,fz], -1)
        xyz = torch.where(f > _DELTA, f.pow(3.), 3*_DELTA**2*(f-4/29.))
        return xyz * self.d65

    def _w(self, h):
        c = self.hue
        return torch.exp(c[0] + c[1]*torch.cos(h) + c[2]*torch.sin(h)
                         + c[3]*torch.cos(2*h) + c[4]*torch.sin(2*h)
                         + c[5]*torch.cos(3*h) + c[6]*torch.sin(3*h))

    def forward(self, xyz):
        L, a, b = self._cielab(xyz)
        C = torch.sqrt(a*a + b*b).clamp(min=1e-12); h = torch.atan2(b, a)
        Lp = L.clamp(min=0).pow(_Q)
        Cp = self._w(h) * C.pow(_P)
        return torch.stack([Lp, Cp*torch.cos(h), Cp*torch.sin(h)], -1)

    def inverse(self, coords):
        Lp = coords[:,0].clamp(min=0); ap = coords[:,1]; bp = coords[:,2]
        Cp = torch.sqrt(ap*ap + bp*bp).clamp(min=1e-12); h = torch.atan2(bp, ap)
        L = Lp.pow(1.0/_Q)
        C = (Cp / self._w(h)).clamp(min=0).pow(1.0/_P)
        return self._cielab_inv(L, C*torch.cos(h), C*torch.sin(h))
