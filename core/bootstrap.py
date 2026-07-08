"""Paired-bootstrap tie decision — statistical ties instead of an arbitrary
relative threshold.

Motivation: "A beat B by 0.8%" is meaningless if the metric's sampling noise
is 2%. For metrics that aggregate PER-ITEM values over a shared item set
(gradient pairs, MacAdam ellipses, constant-hue loci), we resample the SAME
item indices for both spaces (paired bootstrap — removes between-item variance,
so even small item sets give a usable difference CI) and call the metric a TIE
when the 95% CI of the aggregate difference contains 0.

Metrics without per-item structure (exhaustive-grid max errors, single-ladder
CVs with <MIN_ITEMS steps) keep the relative-threshold tie rule; comparison.py
records which rule decided each metric so no decision is silent.

Determinism: fixed seed — same inputs, same verdict, every run.
"""
import numpy as np

SEED = 42
N_RESAMPLES = 2000
CI_LEVEL = 0.95
MIN_ITEMS = 8


def _stat_rows(arr: np.ndarray, stat: str) -> np.ndarray:
    """Aggregate each resample row (n_resamples, n_items) → (n_resamples,)."""
    if stat == "mean":
        return arr.mean(axis=1)
    if stat == "mean_pos":
        # mean over strictly-positive items — mirrors the cv>0 validity filter
        pos = arr > 0
        cnt = np.maximum(pos.sum(axis=1), 1)
        return np.where(pos, arr, 0.0).sum(axis=1) / cnt
    if stat == "p95_pos":
        out = np.empty(arr.shape[0])
        for i in range(arr.shape[0]):
            row = arr[i]
            row = row[row > 0]
            out[i] = np.quantile(row, 0.95) if row.size else 0.0
        return out
    if stat == "cv":
        m = arr.mean(axis=1)
        return np.where(m > 1e-12, arr.std(axis=1) / np.maximum(m, 1e-12), 0.0)
    raise ValueError(f"unknown bootstrap stat: {stat}")


def paired_decision(items_a, items_b, stat: str, lower_is_better: bool = True,
                    n_resamples: int = N_RESAMPLES, seed: int = SEED,
                    ci: float = CI_LEVEL):
    """Paired bootstrap over a SHARED item set.

    Returns None when the item sets can't be paired (different lengths or too
    few items) — caller falls back to the threshold rule. Otherwise:
      {"outcome": "a"|"b"|"tie", "diff_ci": [lo, hi], "n_items": N}
    """
    a = np.asarray(items_a, dtype=np.float64).ravel()
    b = np.asarray(items_b, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size < MIN_ITEMS:
        return None
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_resamples, a.size))
    diff = _stat_rows(a[idx], stat) - _stat_rows(b[idx], stat)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(diff, [alpha, 1.0 - alpha])

    if lo <= 0.0 <= hi:
        outcome = "tie"
    else:
        a_smaller = hi < 0.0
        outcome = ("a" if a_smaller else "b") if lower_is_better \
            else ("b" if a_smaller else "a")
    return {"outcome": outcome, "diff_ci": [float(lo), float(hi)],
            "n_items": int(a.size)}


def stress_ci(de, dv, n_resamples: int = N_RESAMPLES, seed: int = SEED,
              ci: float = CI_LEVEL):
    """Bootstrap CI for a STRESS score (resampling pairs). For display —
    tells the reader whether a 1-point STRESS gap is signal or noise."""
    de = np.asarray(de, dtype=np.float64).ravel()
    dv = np.asarray(dv, dtype=np.float64).ravel()
    n = de.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    d, v = de[idx], dv[idx]
    F = (d * v).sum(axis=1) / np.maximum((d * d).sum(axis=1), 1e-30)
    resid = F[:, None] * d - v
    s = 100.0 * np.sqrt((resid ** 2).sum(axis=1) /
                        np.maximum((v ** 2).sum(axis=1), 1e-30))
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(s, [alpha, 1.0 - alpha])
    return float(lo), float(hi)
