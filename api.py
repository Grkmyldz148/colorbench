"""ColorBench public API — evaluate ANY color space end-to-end in one call.

    import colorbench as cb

    space = cb.load("oklab")                       # built-in
    space = cb.from_json("params.json")            # checkpoint space
    space = cb.wrap(fwd, inv, name="myspace",      # ANY space: two callables
                    trained_on=["munsell"])        #   + fit-data declaration

    profile = cb.evaluate(space)                   # full scan vs cached baselines
    print(profile.verdict())                       # tiered + fair verdict
    print(profile.scorecard())                     # property × space karne
    profile.html("report.html")

`wrap` is what makes ColorBench everyone's benchmark: any space enters with a
forward (XYZ→coords) and inverse (coords→XYZ) over torch (N,3) tensors —
no subclassing, no registration. Baseline reports (OKLab, CIELab, …) are
cached under results/baselines keyed by METHODOLOGY_VERSION, so evaluating a
candidate does not re-run the competitors.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core.constants import METHODOLOGY_VERSION

BASELINE_DIR = os.path.join(_HERE, "results", "baselines")


# ── space constructors ────────────────────────────────────────────────────────

def load(name: str, json_path: str = None, device: str = "cpu"):
    """Built-in space by name (oklab, cielab, helmct, genspace --json, …)."""
    from run import build_space
    return build_space(name, json_path, device)


def from_json(path: str, kind: str = "genspace", device: str = "cpu"):
    """Checkpoint space from a params JSON. Reads its optional
    "trained_on": [...] declaration for the contamination guard."""
    space = load(kind, path, device)
    try:
        decl = json.load(open(path)).get("trained_on")
        if decl and not getattr(space, "trained_on", None):
            space.trained_on = decl
    except Exception:
        pass
    return space


class _WrappedSpace:
    """Adapter: two callables → ColorBench space."""

    def __init__(self, forward, inverse, name, trained_on=None,
                 device="cpu", dtype=None):
        import torch
        self._fwd = forward
        self._inv = inverse
        self.name = name
        self.device = torch.device(device)
        self.dtype = dtype or torch.float64
        self.trained_on = list(trained_on or [])

    def forward(self, xyz):
        return self._fwd(xyz)

    def inverse(self, lab):
        return self._inv(lab)


def wrap(forward, inverse, name: str = "custom", trained_on=None,
         device: str = "cpu"):
    """Enter ANY color space into the benchmark: forward (XYZ→coords) and
    inverse (coords→XYZ), both over torch (N, 3) tensors. Declare the human
    datasets the space was fit on via trained_on — judges built on those
    datasets are flagged in-sample for it (three-way holdout guard)."""
    return _WrappedSpace(forward, inverse, name, trained_on, device)


# ── baseline cache ────────────────────────────────────────────────────────────

def _baseline_report(name: str, device, device_name: str, use_cache: bool = True):
    from run import build_space, run_test
    from core.report import save_json
    os.makedirs(BASELINE_DIR, exist_ok=True)
    path = os.path.join(BASELINE_DIR, f"{name}_v{METHODOLOGY_VERSION}.json")
    if use_cache and os.path.exists(path):
        return json.load(open(path))
    space = build_space(name, None, device)
    report = run_test(space, device, device_name)
    report["trained_on"] = list(getattr(space, "trained_on", []) or [])
    save_json(report, path)
    return report


# ── profile ───────────────────────────────────────────────────────────────────

class Profile:
    """Everything one evaluate() run produced."""

    def __init__(self, space, report, comparison, baseline_names, human):
        self.space = space
        self.report = report              # full 94-metric results dict
        self.comparison = comparison      # core.comparison.Comparison
        self.baseline_names = baseline_names
        self.human_panel = human          # human_pool by_property (or None)

    def verdict(self) -> str:
        """Tiered (headline) + weighted fair verdict vs every baseline."""
        from core.judge_provenance import tiered_winhist, format_tiered_verdict
        from core.fair_verdict import fair_winhist
        from core.contamination import summarize
        me = self.report["space"]
        blocks = []
        for b in self.baseline_names:
            wh = tiered_winhist(self.comparison.tests, me, b)
            fw = fair_winhist(self.comparison, me, b)
            blocks.append(format_tiered_verdict(wh, me, b))
            blocks.append(f"  AĞIRLIKLI (gamut×1/3, CIELab-ref×0): "
                          f"{me} {fw['a']} – {fw['b']} {b}  (tie {fw['tie']})")
        cs = summarize(self.comparison)
        if cs:
            blocks.append(cs)
        return "\n".join(blocks)

    def scorecard(self, extra_spaces: dict = None) -> str:
        """Property × space karne (candidate + baselines + any extras)."""
        from run import build_space, get_device
        from core.scorecard import scorecard as _sc
        device, _, _ = get_device()
        spaces = {self.report["space"]: self.space}
        for b in self.baseline_names:
            spaces[b] = build_space(b.lower().replace(" ", ""), None, device)
        spaces.update(extra_spaces or {})
        return _sc(spaces)

    def html(self, path: str) -> str:
        from core.html_report import generate
        generate(self.comparison, path)
        return path


def evaluate(space, baselines=("oklab", "cielab"), use_cache: bool = True,
             human_panel: bool = True) -> Profile:
    """Full end-to-end scan of one space: the 94 generation metrics, the
    comparison against cached baselines (tiered/fair verdict, bootstrap ties,
    ruler-sensitivity, contamination guard) and the human-data panel."""
    from run import build_space, run_test, get_device
    if isinstance(space, str):
        space = load(space)
    device, _, device_name = get_device()

    report = run_test(space, device, device_name)
    report["trained_on"] = list(getattr(space, "trained_on", []) or [])

    results = {report["space"]: report}
    baseline_names = []
    for b in baselines:
        bl = _baseline_report(b, device, device_name, use_cache)
        if bl["space"] != report["space"]:
            results[bl["space"]] = bl
            baseline_names.append(bl["space"])

    from core.comparison import compare_spaces
    comparison = compare_spaces(results)

    human = None
    if human_panel:
        try:
            from core import human_pool as hp
            human = hp.evaluate_space_on_pool(space, validated_only=True)["by_property"]
        except Exception:
            human = None

    return Profile(space, report, comparison, baseline_names, human)
