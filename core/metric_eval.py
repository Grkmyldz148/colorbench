"""MetricSpace evaluation — STRESS scoring on perceptual color difference datasets.

Completely separate from GenSpace benchmark. Does not touch run_test() or any
existing gpu_metrics modules.

Usage:
    from core.metric_eval import run_metric_evaluation
    results = run_metric_evaluation(
        metric_json="research/checkpoints/metricspace_v21.json",
        datasets_dir="datasets"
    )

STRESS: lower = better. CIEDE2000 baseline ~29, CIELab ~30.
Datasets:
  - COMBVD   : 3813 pairs (BFD-P, LEEDS, RIT-DuPont, WITT)
  - MacAdam  : 128 pairs (MacAdam 1974 uniform color scales, D65)
  - Human FB : 3552 observer judgements (71 observers, sRGB hex)
"""

import json
import os
import re
import sys
import numpy as np

# ── Bradford CAT (arbitrary white → D65) ─────────────────────────────────────

_BRADFORD = np.array([
    [ 0.8951,  0.2664, -0.1614],
    [-0.7502,  1.7135,  0.0367],
    [ 0.0389, -0.0685,  1.0296],
], dtype=np.float64)
_BRADFORD_INV = np.linalg.inv(_BRADFORD)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _cat_to_d65(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    """Bradford chromatic adaptation: arbitrary white → D65."""
    if np.allclose(white, _D65, atol=1e-4):
        return xyz
    cone_src = _BRADFORD @ white
    cone_dst = _BRADFORD @ _D65
    scale = cone_dst / cone_src
    M = _BRADFORD_INV @ (np.diag(scale) @ _BRADFORD)
    return xyz @ M.T


# ── Baseline metrics (standalone numpy) ───────────────────────────────────────

def _xyz_to_cielab(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    r = xyz / white
    delta3 = (6.0 / 29.0) ** 3
    f = np.where(r > delta3, r ** (1.0 / 3.0), r / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _cielab_de(xyz1: np.ndarray, xyz2: np.ndarray, white: np.ndarray) -> np.ndarray:
    lab1 = _xyz_to_cielab(xyz1, white)
    lab2 = _xyz_to_cielab(xyz2, white)
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))


def _ciede2000(xyz1: np.ndarray, xyz2: np.ndarray, white: np.ndarray) -> np.ndarray:
    lab1 = _xyz_to_cielab(xyz1, white)
    lab2 = _xyz_to_cielab(xyz2, white)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_avg = (C1 + C2) / 2
    C_avg7 = C_avg ** 7
    G = 0.5 * (1 - np.sqrt(C_avg7 / (C_avg7 + 25 ** 7)))
    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = np.where(
        np.abs(h2p - h1p) <= 180, h2p - h1p,
        np.where(h2p - h1p > 180, h2p - h1p - 360, h2p - h1p + 360)
    )
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    Lp_avg = (L1 + L2) / 2
    Cp_avg = (C1p + C2p) / 2
    hp_avg = np.where(
        np.abs(h1p - h2p) <= 180, (h1p + h2p) / 2,
        np.where(h1p + h2p < 360, (h1p + h2p + 360) / 2, (h1p + h2p - 360) / 2)
    )

    T = (1
         - 0.17 * np.cos(np.radians(hp_avg - 30))
         + 0.24 * np.cos(np.radians(2 * hp_avg))
         + 0.32 * np.cos(np.radians(3 * hp_avg + 6))
         - 0.20 * np.cos(np.radians(4 * hp_avg - 63)))

    SL = 1 + 0.015 * (Lp_avg - 50) ** 2 / np.sqrt(20 + (Lp_avg - 50) ** 2)
    SC = 1 + 0.045 * Cp_avg
    SH = 1 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg ** 7
    RC = 2 * np.sqrt(Cp_avg7 / (Cp_avg7 + 25 ** 7))
    d_theta = 30 * np.exp(-((hp_avg - 275) / 25) ** 2)
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    return np.sqrt(
        (dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )


def _oklab_de(xyz1: np.ndarray, xyz2: np.ndarray) -> np.ndarray:
    """OKLab ΔE (uses D65 natively)."""
    M1 = np.array([
        [0.8189330101, 0.3618667424, -0.1288597137],
        [0.0329845436, 0.9293118715, 0.0361456387],
        [0.0482003018, 0.2643662691, 0.6338517070],
    ])
    M2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ])
    lms1 = xyz1 @ M1.T
    lms2 = xyz2 @ M1.T
    lms1 = np.cbrt(np.maximum(lms1, 0))
    lms2 = np.cbrt(np.maximum(lms2, 0))
    lab1 = lms1 @ M2.T
    lab2 = lms2 @ M2.T
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))


def _cam16_ucs_de(xyz1: np.ndarray, xyz2: np.ndarray) -> np.ndarray:
    """CAM16-UCS ΔE (Li et al. 2017). Average surround, D65, LA=64 cd/m², Yb=20%."""
    LA, Yb = 64.0, 20.0
    F, c, Nc = 1.0, 0.69, 1.0

    D = float(F * (1 - (1 / 3.6) * np.exp((-LA - 42) / 92)))
    D = float(np.clip(D, 0.0, 1.0))
    k = 1.0 / (5 * LA + 1)
    FL = float(0.2 * k ** 4 * (5 * LA) + 0.1 * (1 - k ** 4) ** 2 * (5 * LA) ** (1.0 / 3))
    n = Yb / 100.0
    Nbb = 0.725 * n ** (-0.2)
    Ncb = Nbb
    z = 1.48 + np.sqrt(50 * n)

    MCAT16 = np.array([
        [ 0.401288,  0.650173, -0.051461],
        [-0.250268,  1.204414,  0.045854],
        [-0.002079,  0.048952,  0.953127],
    ], dtype=np.float64)
    MCAT16_INV = np.linalg.inv(MCAT16)
    MHPE = np.array([
        [ 0.38971,  0.68898, -0.07868],
        [-0.22981,  1.18340,  0.04641],
        [ 0.00000,  0.00000,  1.00000],
    ], dtype=np.float64)

    # White D65 (normalised Y=1)
    w = np.array([0.95047, 1.0, 1.08883])
    lms_w = MCAT16 @ w
    rw, gw, bw = float(lms_w[0]), float(lms_w[1]), float(lms_w[2])

    # Adaptation scale factors (Yw=1 normalised)
    sr = D / rw + (1 - D)
    sg = D / gw + (1 - D)
    sb = D / bw + (1 - D)

    # Compress: x_norm in [0,1] (Ywhite=1); equivalent to (FL*x*100/100)^0.42
    def _compress(x):
        t = (FL * np.abs(x)) ** 0.42
        return np.sign(x) * 400.0 * t / (t + 27.13) + 0.1

    # Adapted white → HPE → compress → achromatic response
    lms_w_ad = MCAT16_INV @ np.array([sr * rw, sg * gw, sb * bw])
    lms_w_hpe = MHPE @ lms_w_ad
    rw_p = _compress(lms_w_hpe[0])
    gw_p = _compress(lms_w_hpe[1])
    bw_p = _compress(lms_w_hpe[2])
    Aw = (2 * rw_p + gw_p + 0.05 * bw_p - 0.305) * Nbb

    def _to_jm(xyz_arr):
        lms = xyz_arr @ MCAT16.T
        rc = sr * lms[:, 0];  gc = sg * lms[:, 1];  bc = sb * lms[:, 2]
        xyz_ad = np.stack([rc, gc, bc], axis=-1) @ MCAT16_INV.T
        lms_h = xyz_ad @ MHPE.T
        rp = _compress(lms_h[:, 0])
        gp = _compress(lms_h[:, 1])
        bp = _compress(lms_h[:, 2])

        a = rp - 12 * gp / 11 + bp / 11
        b_op = (rp + gp - 2 * bp) / 9
        h_rad = np.arctan2(b_op, a)

        A = (2 * rp + gp + 0.05 * bp - 0.305) * Nbb
        J = 100.0 * (np.maximum(A, 0.0) / Aw) ** (c * z)

        et = 0.25 * (np.cos(h_rad + 2.0) + 3.8)
        denom = np.maximum(rp + gp + 21 / 20 * bp, 1e-10)
        t_val = 50000.0 / 13 * Nc * Ncb * et * np.sqrt(a ** 2 + b_op ** 2) / denom
        C = (np.maximum(t_val, 0.0) ** 0.9 * np.sqrt(np.maximum(J, 0.0) / 100.0)
             * (1.64 - 0.29 ** n) ** 0.73)
        M = C * FL ** 0.25

        h_deg = np.degrees(h_rad) % 360
        return J, M, h_deg

    J1, M1, h1 = _to_jm(xyz1)
    J2, M2, h2 = _to_jm(xyz2)

    c1, c2 = 0.007, 0.0228
    Jp1 = (1 + 100 * c1) * J1 / (1 + c1 * J1)
    Jp2 = (1 + 100 * c1) * J2 / (1 + c1 * J2)
    Mp1 = np.log(np.maximum(1 + c2 * M1, 1e-30)) / c2
    Mp2 = np.log(np.maximum(1 + c2 * M2, 1e-30)) / c2

    aM1 = Mp1 * np.cos(np.radians(h1));  bM1 = Mp1 * np.sin(np.radians(h1))
    aM2 = Mp2 * np.cos(np.radians(h2));  bM2 = Mp2 * np.sin(np.radians(h2))

    return np.sqrt((Jp1 - Jp2) ** 2 + (aM1 - aM2) ** 2 + (bM1 - bM2) ** 2)


def _cie94_de(xyz1: np.ndarray, xyz2: np.ndarray, white: np.ndarray) -> np.ndarray:
    """CIE94 ΔE (CIE TC 1-29, 1994). Graphic arts parametric factors."""
    lab1 = _xyz_to_cielab(xyz1, white)
    lab2 = _xyz_to_cielab(xyz2, white)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    dL = L1 - L2
    dC = C1 - C2
    dH2 = (a1 - a2) ** 2 + (b1 - b2) ** 2 - dC ** 2
    SL = 1.0
    SC = 1 + 0.045 * C1
    SH = 1 + 0.015 * C1
    return np.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + np.maximum(dH2, 0) / SH ** 2)


def _ciecam02_ucs_de(xyz1: np.ndarray, xyz2: np.ndarray) -> np.ndarray:
    """CIECAM02-UCS ΔE (Luo et al. 2006). Same conditions as CAM16-UCS."""
    LA, Yb = 64.0, 20.0
    F, c, Nc = 1.0, 0.69, 1.0

    D = float(F * (1 - (1 / 3.6) * np.exp((-LA - 42) / 92)))
    D = float(np.clip(D, 0.0, 1.0))
    k = 1.0 / (5 * LA + 1)
    FL = float(0.2 * k ** 4 * (5 * LA) + 0.1 * (1 - k ** 4) ** 2 * (5 * LA) ** (1.0 / 3))
    n = Yb / 100.0
    Nbb = 0.725 * n ** (-0.2)
    Ncb = Nbb
    z = 1.48 + np.sqrt(50 * n)

    # CIECAM02 uses M_CAT02 (not CAT16)
    MCAT02 = np.array([
        [ 0.7328,  0.4296, -0.1624],
        [-0.7036,  1.6975,  0.0061],
        [ 0.0030,  0.0136,  0.9834],
    ], dtype=np.float64)
    MCAT02_INV = np.linalg.inv(MCAT02)
    MHPE = np.array([
        [ 0.38971,  0.68898, -0.07868],
        [-0.22981,  1.18340,  0.04641],
        [ 0.00000,  0.00000,  1.00000],
    ], dtype=np.float64)

    w = np.array([0.95047, 1.0, 1.08883])
    lms_w = MCAT02 @ w
    rw, gw, bw = float(lms_w[0]), float(lms_w[1]), float(lms_w[2])

    sr = D / rw + (1 - D)
    sg = D / gw + (1 - D)
    sb = D / bw + (1 - D)

    def _compress(x):
        t = (FL * np.abs(x)) ** 0.42
        return np.sign(x) * 400.0 * t / (t + 27.13) + 0.1

    lms_w_ad = MCAT02_INV @ np.array([sr * rw, sg * gw, sb * bw])
    lms_w_hpe = MHPE @ lms_w_ad
    rw_p = _compress(lms_w_hpe[0])
    gw_p = _compress(lms_w_hpe[1])
    bw_p = _compress(lms_w_hpe[2])
    Aw = (2 * rw_p + gw_p + 0.05 * bw_p - 0.305) * Nbb

    def _to_jm(xyz_arr):
        lms = xyz_arr @ MCAT02.T
        rc = sr * lms[:, 0];  gc = sg * lms[:, 1];  bc = sb * lms[:, 2]
        xyz_ad = np.stack([rc, gc, bc], axis=-1) @ MCAT02_INV.T
        lms_h = xyz_ad @ MHPE.T
        rp = _compress(lms_h[:, 0])
        gp = _compress(lms_h[:, 1])
        bp = _compress(lms_h[:, 2])

        a = rp - 12 * gp / 11 + bp / 11
        b_op = (rp + gp - 2 * bp) / 9
        h_rad = np.arctan2(b_op, a)

        A = (2 * rp + gp + 0.05 * bp - 0.305) * Nbb
        J = 100.0 * (np.maximum(A, 0.0) / Aw) ** (c * z)

        et = 0.25 * (np.cos(h_rad + 2.0) + 3.8)
        denom = np.maximum(rp + gp + 21 / 20 * bp, 1e-10)
        t_val = 50000.0 / 13 * Nc * Ncb * et * np.sqrt(a ** 2 + b_op ** 2) / denom
        C = (np.maximum(t_val, 0.0) ** 0.9 * np.sqrt(np.maximum(J, 0.0) / 100.0)
             * (1.64 - 0.29 ** n) ** 0.73)
        M = C * FL ** 0.25
        h_deg = np.degrees(h_rad) % 360
        return J, M, h_deg

    J1, M1, h1 = _to_jm(xyz1)
    J2, M2, h2 = _to_jm(xyz2)

    # CAM02-UCS: c1=0.007, c2=0.0228 (same as CAM16-UCS)
    c1, c2 = 0.007, 0.0228
    Jp1 = (1 + 100 * c1) * J1 / (1 + c1 * J1)
    Jp2 = (1 + 100 * c1) * J2 / (1 + c1 * J2)
    Mp1 = np.log(np.maximum(1 + c2 * M1, 1e-30)) / c2
    Mp2 = np.log(np.maximum(1 + c2 * M2, 1e-30)) / c2

    aM1 = Mp1 * np.cos(np.radians(h1));  bM1 = Mp1 * np.sin(np.radians(h1))
    aM2 = Mp2 * np.cos(np.radians(h2));  bM2 = Mp2 * np.sin(np.radians(h2))

    return np.sqrt((Jp1 - Jp2) ** 2 + (aM1 - aM2) ** 2 + (bM1 - bM2) ** 2)


def _din99_de(xyz1: np.ndarray, xyz2: np.ndarray, white: np.ndarray) -> np.ndarray:
    """DIN99 ΔE (DIN 6176:2001). Rotation of CIELab with logarithmic chroma compression."""
    lab1 = _xyz_to_cielab(xyz1, white)
    lab2 = _xyz_to_cielab(xyz2, white)

    cos16, sin16 = np.cos(np.radians(16)), np.sin(np.radians(16))

    def _to_din99(lab):
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        L99 = 105.51 * np.log(1 + 0.0158 * np.maximum(L, 0))
        eo = a * cos16 + b * sin16
        fo = 0.7 * (-a * sin16 + b * cos16)
        G = np.sqrt(eo ** 2 + fo ** 2)
        hef = np.arctan2(fo, eo)
        G99 = 22.5 * np.log(1 + 0.0435 * G)
        return np.stack([L99, G99 * np.cos(hef), G99 * np.sin(hef)], axis=-1)

    d1 = _to_din99(lab1)
    d2 = _to_din99(lab2)
    return np.sqrt(np.sum((d1 - d2) ** 2, axis=-1))


def _jzazbz_de(xyz1: np.ndarray, xyz2: np.ndarray, L_sdr: float = 203.0) -> np.ndarray:
    """Jzazbz ΔEz (Safdar et al. 2021). Designed for D65 input.

    L_sdr: reference display peak luminance in cd/m² (203 = SDR reference).
    """
    b = 1.15
    g = 0.66
    n = 0.15930175664
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    m2 = 78.84375
    d = -0.56
    d0 = 1.6295499532821566e-11

    M_abs = np.array([
        [ 0.41478972,  0.579999,  0.014648 ],
        [-0.20151000,  1.120649,  0.053101 ],
        [-0.01660080,  0.264800,  0.668480 ],
    ])
    M_lms_to_izab = np.array([
        [ 0.500000,  0.500000,  0.000000],
        [ 3.524000, -4.066708,  0.542708],
        [ 0.199076,  1.096799, -1.295875],
    ])

    def _to_jzazbz(xyz):
        # Scale to absolute cd/m²
        xyz_abs = xyz * L_sdr
        X, Y, Z = xyz_abs[..., 0], xyz_abs[..., 1], xyz_abs[..., 2]
        # Crosstalk correction
        Xp = b * X - (b - 1) * Z
        Yp = g * Y - (g - 1) * X
        Zp = Z
        xyz_ct = np.stack([Xp, Yp, Zp], axis=-1)
        # Absolute LMS
        lms = xyz_ct @ M_abs.T
        # PQ transfer
        lms_c = np.clip(lms, 0, None)
        xp = (lms_c / 10000.0) ** n
        lms_pq = ((c1 + c2 * xp) / (1 + c3 * xp)) ** m2
        # Opponent channels
        izab = lms_pq @ M_lms_to_izab.T
        Lhat = izab[..., 0]
        az = izab[..., 1]
        bz = izab[..., 2]
        Jz = (1 + d) * Lhat / (1 + d * Lhat) - d0
        return np.stack([Jz, az, bz], axis=-1)

    jab1 = _to_jzazbz(xyz1)
    jab2 = _to_jzazbz(xyz2)
    return np.sqrt(np.sum((jab1 - jab2) ** 2, axis=-1))


# ── STRESS ─────────────────────────────────────────────────────────────────────

def stress(de_pred: np.ndarray, dv_obs: np.ndarray) -> float:
    """STRESS score. Lower = better. CIEDE2000 baseline ~29."""
    de = np.asarray(de_pred, dtype=np.float64)
    dv = np.asarray(dv_obs, dtype=np.float64)
    # Optimal linear scaling: F = Σ(ΔE·ΔV) / Σ(ΔE²)
    F = np.dot(de, dv) / np.dot(de, de)
    residuals = F * de - dv
    return 100.0 * np.sqrt(np.sum(residuals ** 2) / np.sum(dv ** 2))


# ── Dataset loaders ─────────────────────────────────────────────────────────────

def load_combvd(datasets_dir: str):
    """3813 XYZ color pairs with visual difference scores (multi-illuminant).

    Returns (xyz1, xyz2, whites, dv, sub_datasets).
    """
    path = os.path.join(datasets_dir, "combvd_pairs.json")
    data = json.load(open(path))
    xyz1 = np.array([d["xyz1"] for d in data])
    xyz2 = np.array([d["xyz2"] for d in data])
    whites = np.array([d["white"] for d in data])
    dv = np.array([d["dv"] for d in data])
    return xyz1, xyz2, whites, dv


def load_combvd_from_xlsx(datasets_dir: str):
    """Load COMBVD from xlsx with sub-dataset labels (requires openpyxl)."""
    try:
        import openpyxl
    except ImportError:
        return None

    path = os.path.join(datasets_dir, "combvd.xlsx")
    if not os.path.exists(path):
        return None

    wb = openpyxl.load_workbook(path)
    ws = wb["COM_Corrected_UNWEIGHTED"]

    records = []
    current_dataset = None
    for r in range(4, ws.max_row + 1):
        label = ws.cell(r, 1).value
        dv_val = ws.cell(r, 2).value
        if label:
            current_dataset = label
        if dv_val is None:
            continue
        row = [ws.cell(r, c).value for c in range(2, 12)]
        if None in row:
            continue
        records.append({
            "dataset": current_dataset,
            "dv": row[0],
            "white": [row[1]/100, row[2]/100, row[3]/100],
            "xyz1": [row[4]/100, row[5]/100, row[6]/100],
            "xyz2": [row[7]/100, row[8]/100, row[9]/100],
        })

    return records


def load_macadam1974(datasets_dir: str):
    """MacAdam (1974) uniform color scales — 128 pairs, D65, CIE 1964.

    Returns (xyz1, xyz2, white_d65, dv).
    Reference: MacAdam, JOSA 64(12), 1691-1702 (1974).
    """
    table1_path = os.path.join(datasets_dir, "macadam1974", "table1.yaml")
    table2_path = os.path.join(datasets_dir, "macadam1974", "table2.yaml")

    # Parse tile colorimetry (xyY, D65, 10° observer)
    tiles = {}
    for line in open(table2_path):
        line = line.strip()
        m = re.match(r'"(\w+)":\s*\[(.+)\]', line)
        if m:
            vals = [float(x.strip()) for x in m.group(2).split(",")]
            tiles[m.group(1)] = vals  # [x, y, Y, g, j, L]

    # D65 10° observer white (xyY → XYZ)
    xw, yw = 0.31272, 0.32903
    white = np.array([xw / yw, 1.0, (1.0 - xw - yw) / yw])  # ≈ [0.9504, 1.0, 1.0888]

    def xyyToXYZ(x, y, Y_pct):
        Y = Y_pct / 100.0
        return np.array([x * Y / y, Y, (1.0 - x - y) * Y / y])

    # Parse pairs [pair_num, "tile1", "tile2", dv_obs, ...]
    pairs = []
    for line in open(table1_path):
        line = line.strip()
        m = re.match(r"-\s*\[(\d+),\s*\"(\w+)\",\s*\"(\w+)\",\s*([\d.]+)", line)
        if m:
            t1, t2 = m.group(2), m.group(3)
            dv_obs = float(m.group(4))
            if t1 in tiles and t2 in tiles:
                x1, y1, Y1 = tiles[t1][0], tiles[t1][1], tiles[t1][2]
                x2, y2, Y2 = tiles[t2][0], tiles[t2][1], tiles[t2][2]
                xyz1 = xyyToXYZ(x1, y1, Y1)
                xyz2 = xyyToXYZ(x2, y2, Y2)
                pairs.append((xyz1, xyz2, dv_obs))

    if not pairs:
        return None, None, None, None

    xyz1 = np.array([p[0] for p in pairs])
    xyz2 = np.array([p[1] for p in pairs])
    dv = np.array([p[2] for p in pairs])
    return xyz1, xyz2, white, dv


def load_human_feedback(datasets_dir: str):
    """3552 observer judgements (71 observers, sRGB hex pairs)."""
    path = os.path.join(datasets_dir, "human_feedback.json")
    data = json.load(open(path))
    judgements = data["judgements"]

    _M_SRGB = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    def hex_to_xyz(h):
        h = h.lstrip("#")
        rgb = np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return _M_SRGB @ linear

    xyz1, xyz2, dv = [], [], []
    for j in judgements:
        xyz1.append(hex_to_xyz(j["hex1"]))
        xyz2.append(hex_to_xyz(j["hex2"]))
        dv.append(float(j["perceived_dv"]))

    return np.array(xyz1), np.array(xyz2), np.array(dv)


# ── MetricSpace wrapper ─────────────────────────────────────────────────────────

def _load_metric_space(metric_json: str, repo_root: str):
    """Load MetricSpace from params JSON via helmlab src."""
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from helmlab.spaces.metric import MetricSpace, MetricParams
    params = MetricParams.from_dict(json.load(open(metric_json)))
    return MetricSpace(params)


# ── Main evaluation ─────────────────────────────────────────────────────────────

def run_metric_evaluation(metric_json: str, datasets_dir: str, repo_root: str) -> dict:
    """
    Evaluate MetricSpace against COMBVD, MacAdam 1974, and Human Feedback.
    Returns dict with STRESS scores for all metrics and datasets.
    """
    print("\n══════════════════════════════════════════════════════")
    print("  MetricSpace Evaluation — STRESS Scores")
    print("══════════════════════════════════════════════════════\n")

    print("Loading MetricSpace...", end=" ", flush=True)
    metric = _load_metric_space(metric_json, repo_root)
    print("done")

    results = {}

    # Common baseline set (applied per-dataset with appropriate white)
    def run_baselines(xyz1, xyz2, white_scalar, dv, is_d65=True):
        """Run all baselines. white_scalar: single white point for CIELab/CIEDE2000."""
        scores = {}
        for name, fn in [
            ("MetricSpace", lambda: metric.distance(xyz1, xyz2)),
            ("CIEDE2000",   lambda: _ciede2000(xyz1, xyz2, white_scalar)),
            ("CIE Lab",     lambda: _cielab_de(xyz1, xyz2, white_scalar)),
            ("OKLab",       lambda: _oklab_de(xyz1, xyz2)),
            ("Jzazbz",      lambda: _jzazbz_de(xyz1, xyz2)),
        ]:
            print(f"  {name}...", end=" ", flush=True)
            de = fn()
            s = stress(de, dv)
            scores[name] = round(s, 2)
            print(f"STRESS = {s:.2f}")
        return scores

    # ── COMBVD (3813 pairs, multi-illuminant) ──────────────────────────────
    print("\n[1/3] COMBVD (3813 pairs, multi-illuminant)")
    xyz1_raw, xyz2_raw, whites_raw, dv = load_combvd(datasets_dir)

    # Adapt all pairs to D65 for MetricSpace, OKLab, Jzazbz
    xyz1_d65 = np.array([_cat_to_d65(xyz1_raw[i], whites_raw[i]) for i in range(len(xyz1_raw))])
    xyz2_d65 = np.array([_cat_to_d65(xyz2_raw[i], whites_raw[i]) for i in range(len(xyz2_raw))])

    # For CIELab / CIEDE2000 / CIE94 / DIN99: use per-pair white (correct for multi-illuminant)
    scores_combvd = {}
    for name, fn in [
        ("MetricSpace",   lambda: metric.distance(xyz1_d65, xyz2_d65)),
        ("CIEDE2000",     lambda: np.array([
            _ciede2000(xyz1_raw[i:i+1], xyz2_raw[i:i+1], whites_raw[i])
            for i in range(len(dv))]).ravel()),
        ("CIE94",         lambda: np.array([
            _cie94_de(xyz1_raw[i:i+1], xyz2_raw[i:i+1], whites_raw[i])
            for i in range(len(dv))]).ravel()),
        ("CIE Lab",       lambda: np.array([
            _cielab_de(xyz1_raw[i:i+1], xyz2_raw[i:i+1], whites_raw[i])
            for i in range(len(dv))]).ravel()),
        ("DIN99",         lambda: np.array([
            _din99_de(xyz1_raw[i:i+1], xyz2_raw[i:i+1], whites_raw[i])
            for i in range(len(dv))]).ravel()),
        ("OKLab",         lambda: _oklab_de(xyz1_d65, xyz2_d65)),
        ("CAM16-UCS",     lambda: _cam16_ucs_de(xyz1_d65, xyz2_d65)),
        ("CIECAM02-UCS",  lambda: _ciecam02_ucs_de(xyz1_d65, xyz2_d65)),
        ("Jzazbz",        lambda: _jzazbz_de(xyz1_d65, xyz2_d65)),
    ]:
        print(f"  {name}...", end=" ", flush=True)
        de = fn()
        s = stress(de, dv)
        scores_combvd[name] = round(s, 2)
        print(f"STRESS = {s:.2f}")

    results["COMBVD"] = scores_combvd

    # Per-sub-dataset COMBVD breakdown (if xlsx available)
    combvd_records = load_combvd_from_xlsx(datasets_dir)
    if combvd_records:
        print("\n  COMBVD sub-dataset breakdown:")
        sub_datasets = {}
        for rec in combvd_records:
            sd = rec["dataset"]
            if sd not in sub_datasets:
                sub_datasets[sd] = []
            sub_datasets[sd].append(rec)

        sub_results = {}
        for sd_name, recs in sorted(sub_datasets.items()):
            x1 = np.array([r["xyz1"] for r in recs])
            x2 = np.array([r["xyz2"] for r in recs])
            wh = np.array([r["white"] for r in recs])
            dv_sub = np.array([r["dv"] for r in recs])
            x1_d65 = np.array([_cat_to_d65(x1[i], wh[i]) for i in range(len(x1))])
            x2_d65 = np.array([_cat_to_d65(x2[i], wh[i]) for i in range(len(x2))])
            de_ms = metric.distance(x1_d65, x2_d65)
            de_00 = np.array([_ciede2000(x1[i:i+1], x2[i:i+1], wh[i]) for i in range(len(dv_sub))]).ravel()
            s_ms = stress(de_ms, dv_sub)
            s_00 = stress(de_00, dv_sub)
            n = len(recs)
            marker = " ★" if s_ms < s_00 else ""
            print(f"    {sd_name:<18} n={n:>4}  MetricSpace={s_ms:.2f}{marker}  CIEDE2000={s_00:.2f}")
            sub_results[sd_name] = {"MetricSpace": round(s_ms, 2), "CIEDE2000": round(s_00, 2), "n": n}
        results["COMBVD_subsets"] = sub_results

    # ── MacAdam 1974 (D65, CIE 1964) ──────────────────────────────────────
    print("\n[2/3] MacAdam 1974 (D65, CIE 1964, uniform color scales)")
    xyz1_mac, xyz2_mac, white_mac, dv_mac = load_macadam1974(datasets_dir)

    if xyz1_mac is not None:
        # MacAdam white ≈ D65, apply CAT for exactness
        xyz1_mac_d65 = _cat_to_d65(xyz1_mac, white_mac)
        xyz2_mac_d65 = _cat_to_d65(xyz2_mac, white_mac)

        scores_mac = {}
        for name, fn in [
            ("MetricSpace",   lambda: metric.distance(xyz1_mac_d65, xyz2_mac_d65)),
            ("CIEDE2000",     lambda: _ciede2000(xyz1_mac, xyz2_mac, white_mac)),
            ("CIE94",         lambda: _cie94_de(xyz1_mac, xyz2_mac, white_mac)),
            ("CIE Lab",       lambda: _cielab_de(xyz1_mac, xyz2_mac, white_mac)),
            ("DIN99",         lambda: _din99_de(xyz1_mac, xyz2_mac, white_mac)),
            ("OKLab",         lambda: _oklab_de(xyz1_mac_d65, xyz2_mac_d65)),
            ("CAM16-UCS",     lambda: _cam16_ucs_de(xyz1_mac_d65, xyz2_mac_d65)),
            ("CIECAM02-UCS",  lambda: _ciecam02_ucs_de(xyz1_mac_d65, xyz2_mac_d65)),
            ("Jzazbz",        lambda: _jzazbz_de(xyz1_mac_d65, xyz2_mac_d65)),
        ]:
            print(f"  {name}...", end=" ", flush=True)
            de = fn()
            s = stress(de, dv_mac)
            scores_mac[name] = round(s, 2)
            print(f"STRESS = {s:.2f}")
        results["MacAdam1974"] = scores_mac
        print(f"  ({len(dv_mac)} pairs)")
    else:
        print("  [skipped — macadam1974/ not found]")

    # ── Human Feedback (3552 judgements) ──────────────────────────────────
    print("\n[3/3] Human Feedback (3552 judgements, 71 observers, sRGB)")
    hxyz1, hxyz2, hdv = load_human_feedback(datasets_dir)

    hscores = {}
    for name, fn in [
        ("MetricSpace",   lambda: metric.distance(hxyz1, hxyz2)),
        ("CIEDE2000",     lambda: _ciede2000(hxyz1, hxyz2, _D65)),
        ("CIE94",         lambda: _cie94_de(hxyz1, hxyz2, _D65)),
        ("CIE Lab",       lambda: _cielab_de(hxyz1, hxyz2, _D65)),
        ("DIN99",         lambda: _din99_de(hxyz1, hxyz2, _D65)),
        ("OKLab",         lambda: _oklab_de(hxyz1, hxyz2)),
        ("CAM16-UCS",     lambda: _cam16_ucs_de(hxyz1, hxyz2)),
        ("CIECAM02-UCS",  lambda: _ciecam02_ucs_de(hxyz1, hxyz2)),
        ("Jzazbz",        lambda: _jzazbz_de(hxyz1, hxyz2)),
    ]:
        print(f"  {name}...", end=" ", flush=True)
        de = fn()
        s = stress(de, hdv)
        hscores[name] = round(s, 2)
        print(f"STRESS = {s:.2f}")

    results["HumanFeedback"] = hscores

    # ── Summary ────────────────────────────────────────────────────────────
    _print_summary(results)
    return results


def _print_summary(results: dict):
    # Only show main datasets (not sub-dataset breakdown)
    main_keys = [k for k in results.keys() if not k.endswith("_subsets")]
    if not main_keys:
        return

    metrics = list(results[main_keys[0]].keys())
    col = 16

    print("\n" + "─" * (20 + col * len(main_keys)))
    header = f"  {'Metric':<18}" + "".join(f"{d:>{col}}" for d in main_keys)
    print(header)
    print("─" * (20 + col * len(main_keys)))

    for m in metrics:
        row = f"  {m:<18}"
        for d in main_keys:
            v = results[d].get(m)
            if v is None:
                row += f"{'—':>{col}}"
                continue
            # ★ = best in this dataset column
            best = min(results[d].values())
            marker = " ★" if v == best else "  "
            row += f"{v:>{col-2}.2f}{marker}"
        print(row)

    print("─" * (20 + col * len(main_keys)))
    print("  ★ = best in column  |  lower STRESS = better\n")

    # Rank by average STRESS across available datasets
    print("  Average STRESS across datasets:")
    avgs = {}
    for m in metrics:
        vals = [results[d][m] for d in main_keys if m in results.get(d, {})]
        avgs[m] = round(sum(vals) / len(vals), 2) if vals else None

    ranked = sorted([(v, k) for k, v in avgs.items() if v is not None])
    for rank, (v, name) in enumerate(ranked, 1):
        marker = " ← BEST" if rank == 1 else ""
        print(f"    {rank}. {name:<18} avg={v:.2f}{marker}")
    print()
