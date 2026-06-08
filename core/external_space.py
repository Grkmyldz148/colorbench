"""Black-box external-space adapter for ColorBench metric mode.

A benchmark must test ANY space, not a fixed parametric family. ColorBench's
canonical, frozen parts — datasets, STRESS formula, Bradford CAT — stay UNTOUCHED.
A space only needs to provide one thing: distance(xyz1, xyz2) -> de (numpy in/out).

This adapter lets architectures that outgrew the bundled MetricSpace JSON format
(e.g. the regime_family_v2 / alpha_v5_A2 substrate with Fourier-8 + dist-modulation
+ lp_dark) plug in WITHOUT modifying ColorBench again. Add a space once via its
distance callable; never touch the benchmark per-space.
"""
import os
import sys
import numpy as np


class ExternalMetricSpace:
    """Wraps any distance callable as a ColorBench-compatible metric space.

    The callable must accept (xyz1, xyz2) numpy arrays of shape (N,3) and return
    a length-N numpy array of per-pair distances. ColorBench only ever calls
    .distance() on this object — STRESS / datasets / CAT are handled by ColorBench.
    """
    def __init__(self, distance_fn, name="ExternalSpace"):
        self._dfn = distance_fn
        self.name = name

    def distance(self, xyz1, xyz2):
        de = self._dfn(np.asarray(xyz1, dtype=np.float64),
                       np.asarray(xyz2, dtype=np.float64))
        de = np.asarray(de, dtype=np.float64).ravel()
        return de


def load_regime_champion(npy_path):
    """Load an alpha_v5_A2 / regime_family_v2 substrate (.npy param vector) as an
    ExternalMetricSpace. The architecture code lives in the research tree."""
    import torch
    torch.set_default_dtype(torch.float64)

    # locate the regime_family_v2 architecture code
    # __file__ = .../color-space/colorbench/core/external_space.py  -> 3 dirnames up
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base = os.path.join(repo_root, "research", "perflab", "families",
                        "perceptialab_family_v1", "measurement_v22")
    for p in (os.path.join(base, "alpha_v5_2026-05-10"),
              os.path.join(base, "regime_family_v2_2026-05-10")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from regime_family_v2 import _build_frozen_alpha_v5_a2

    m = _build_frozen_alpha_v5_a2()
    # load param vector into the model
    vec = np.load(npy_path)
    ks, ss, idx, new = [], [], 0, {}
    for k, v in m.state_dict().items():
        ks.append(k); ss.append(v.shape)
    for k, s in zip(ks, ss):
        n = int(np.prod(s))
        new[k] = torch.tensor(vec[idx:idx + n].reshape(s), dtype=torch.float64)
        idx += n
    m.load_state_dict(new, strict=False)
    if idx != len(vec):
        raise ValueError(f"param vector length {len(vec)} != model param count {idx}")

    def dfn(x1, x2):
        with torch.no_grad():
            de = m.distance(torch.tensor(x1, dtype=torch.float64),
                            torch.tensor(x2, dtype=torch.float64))
        return de.detach().numpy()

    return ExternalMetricSpace(dfn, name=f"regime[{os.path.basename(npy_path)}]")
