"""Dataset location + on-demand fetch — the one place that answers "where is
the data?" so ColorBench runs from a pip install, not just the dev checkout.

Two data roots:
  - BASELINE data (compare/metric modes): combvd_pairs.json, hung_berns/,
    ebner_fairchild/, pointer_gamut/, macadam1974/, human_feedback.json.
    Lives in <repo-parent>/datasets in the dev layout.
  - The human-perception POOL (color-perception-datasets, 46 datasets): used by
    core.human_pool.

Resolution order for each (first hit wins):
  1. explicit env var (COLORBENCH_DATA / COLOR_PERCEPTION_POOL)
  2. the dev-checkout sibling directory
  3. the user cache (~/.cache/colorbench/...), auto-downloaded from GitHub on
     first use when absent.

No path is silently wrong: a miss raises with the exact env var + download URL.
"""
import os
import sys
import tarfile
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../colorbench/core
_COLORBENCH = os.path.dirname(_HERE)                          # .../colorbench
_REPO_PARENT = os.path.dirname(_COLORBENCH)                   # .../color-space
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "colorbench")

POOL_REPO = "Grkmyldz148/color-perception-datasets"
POOL_TARBALL = f"https://github.com/{POOL_REPO}/archive/refs/heads/main.tar.gz"


def _first_existing(candidates, marker=None):
    for c in candidates:
        if c and os.path.isdir(c) and (marker is None or os.path.exists(os.path.join(c, marker))):
            return c
    return None


def _download_tarball(url: str, dest_dir: str, strip_top: bool = True) -> str:
    """Download+extract a GitHub tarball into dest_dir; return the extracted
    root (top-level dir stripped when strip_top)."""
    os.makedirs(dest_dir, exist_ok=True)
    tmp = os.path.join(dest_dir, "_download.tar.gz")
    print(f"  downloading {url} → {dest_dir} ...", file=sys.stderr, flush=True)
    urllib.request.urlretrieve(url, tmp)
    with tarfile.open(tmp) as tf:
        members = tf.getmembers()
        top = members[0].name.split("/")[0] if members else ""
        tf.extractall(dest_dir)
    os.remove(tmp)
    return os.path.join(dest_dir, top) if strip_top else dest_dir


# ── the human-perception pool (color-perception-datasets) ─────────────────────

def pool_dir(auto_fetch: bool = True) -> str:
    """Directory holding the 46 pool datasets (each <name>/canonical.csv)."""
    env = os.environ.get("COLOR_PERCEPTION_POOL")
    sibling = os.path.normpath(os.path.join(_REPO_PARENT, "..",
                                            "color-perception-datasets", "datasets"))
    cache = os.path.join(_CACHE, "color-perception-datasets", "datasets")
    hit = _first_existing([env, sibling, cache], marker="combvd")
    if hit:
        return hit
    if auto_fetch:
        root = _download_tarball(POOL_TARBALL, os.path.join(_CACHE, "_pool_dl"))
        target = os.path.join(root, "datasets")
        if os.path.isdir(target):
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            if not os.path.isdir(cache):
                os.replace(target, cache)
            return cache if os.path.isdir(cache) else target
    raise FileNotFoundError(
        "color-perception-datasets pool not found. Set COLOR_PERCEPTION_POOL to "
        "its datasets/ directory, clone "
        f"https://github.com/{POOL_REPO} next to the repo, or allow the "
        "auto-download (needs network).")


# ── baseline data for compare/metric modes ───────────────────────────────────

def baseline_dir(auto_fetch: bool = True) -> str:
    """Directory holding compare/metric baseline data (combvd_pairs.json etc.).
    In the dev layout this is <repo-parent>/datasets; several files there are
    also mirrored in the pool, so we fall back to the pool when the standalone
    baseline dir is absent."""
    env = os.environ.get("COLORBENCH_DATA")
    dev = os.path.join(_REPO_PARENT, "datasets")
    hit = _first_existing([env, dev], marker="combvd_pairs.json")
    if hit:
        return hit
    # fall back to the pool (it carries combvd, macadam1974, hung_berns, ...)
    try:
        return pool_dir(auto_fetch=auto_fetch)
    except FileNotFoundError:
        raise FileNotFoundError(
            "ColorBench baseline data not found. Set COLORBENCH_DATA to a "
            "directory containing combvd_pairs.json + macadam1974/ + "
            "human_feedback.json, or place them in <repo>/datasets.")


def resolve(relpath: str, auto_fetch: bool = True) -> str:
    """Absolute path to a baseline-data file, fetching the data root if needed."""
    return os.path.join(baseline_dir(auto_fetch=auto_fetch), relpath)
