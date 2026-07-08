"""ColorBench — a rigorous, self-auditing benchmark for perceptual color spaces.

    import colorbench as cb
    profile = cb.evaluate(cb.wrap(fwd, inv, name="myspace"))
    print(profile.verdict())
    print(profile.scorecard())
"""
from .api import load, from_json, wrap, evaluate, Profile  # noqa: F401
from .core.constants import METHODOLOGY_VERSION  # noqa: F401

__all__ = ["load", "from_json", "wrap", "evaluate", "Profile",
           "METHODOLOGY_VERSION"]
