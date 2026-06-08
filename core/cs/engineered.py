"""Engineered substrate (research 2026-05-30) — new opponent M1,M2 + nonlinearity gamma,
fit end-to-end to Munsell value+chroma, OSA-UCS, gray-axis, hue-linearity, and a direct
blue/red->white purple-shift penalty. Beats OKLab on all axes + clean gradients.
Forward: lms=M1@XYZ ; lms'=sign|lms|^g ; Lab=M2@lms'.
"""
import json, os
import torch
from .base import ColorSpace

_P = json.load(open(os.path.join(os.path.dirname(__file__), "substrate_params.json")))


class Engineered(ColorSpace):
    name = "Perceptia-Substrate"

    def __init__(self, device=None, dtype=torch.float64):
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.M1 = torch.tensor(_P["M1"], device=self.device, dtype=dtype)
        self.M2 = torch.tensor(_P["M2"], device=self.device, dtype=dtype)
        self.M1i = torch.linalg.inv(self.M1)
        self.M2i = torch.linalg.inv(self.M2)
        self.g = float(_P["gamma"])

    def forward(self, xyz):
        lms = xyz @ self.M1.T
        lmsp = torch.sign(lms) * lms.abs().pow(self.g)
        return lmsp @ self.M2.T

    def inverse(self, lab):
        lmsp = lab @ self.M2i.T
        lms = torch.sign(lmsp) * lmsp.abs().pow(1.0 / self.g)
        return lms @ self.M1i.T
