"""Judge provenance — WHO judges each metric, and is that judge ceiling-bound?

Motivation (Görkem, 2026-06-08): "if we measure WITH CIELab we can only ever
find something as good as CIELab. No single space is best at everything; judge
each property with whatever is genuinely best at THAT property."

A metric's verdict is only as trustworthy as its JUDGE. ColorBench already has a
modular human-fit ruler layer (rulers.py: Perceptia-Spacing for uniformity,
CIEDE2000 for difference — both grounded in human data). But a subset of metrics
still judge HUE/GEOMETRY against CIE Lab as if CIE Lab were ground truth. CIE Lab
has a known blue→purple hue bend; any space that CORRECTS it is penalized by
those metrics. So a high W-L-T can hide the fact that the wins came from a
ceiling-bound judge.

This module tags every metric group by the PROVENANCE of its judge into 5 tiers,
and splits a comparison's W-L-T by tier so the trustworthy verdict is readable
separately from the ceiling-bound one.

Tiers (best → worst trust for ranking a candidate space):
  1. human_psychophysical — judged DIRECTLY against published human data
     (Hung-Berns/Ebner-Fairchild constant-hue loci, Munsell, MacAdam, Pointer).
     Ceiling = human perception itself. The gold verdict.
  2. structural — coordinate-free objective facts (round-trip error, gamut cusp
     validity, channel monotonicity, OOG). No reference space involved → cannot
     be ceiling-bound. Trustworthy.
  3. human_fit_ruler — judged by a fixed instrument FIT TO / VALIDATED ON human
     data (Perceptia-Spacing = Munsell equal-step held-out; CIEDE2000 = CIE
     224:2017 human difference). Ceiling = that ruler (high, but a model).
     Trustworthy for what it measures (spacing / difference), not for hue truth.
  4. cielab_reference — judges HUE or GEOMETRY against CIE Lab AS GROUND TRUTH.
     CEILING-BOUND at CIE Lab's quality. A hue-correcting space (Abney-fixed,
     post-hue-warp) can look WORSE here precisely because it is BETTER. Read with
     skepticism; for hue-correcting candidates, defer to tier 1 (human loci).
  5. heuristic_proxy — designed targets with no human or CIE grounding (even-60°
     primary hue grid, blue-white/red-white midpoint heuristics). Diagnostic only.

Headline verdict for re-ranking = tiers 1 + 2 + 3 (trustworthy).
Tiers 4-5 are reported separately and flagged, never folded into the headline.

Tier 1 (human psychophysical) is realized DIRECTLY by core.human_pool, which
grounds each property (difference / hue / discrimination / 3D) in the full
color-perception-datasets pool (real JND ellipses, constant-hue loci, Koenderink
3D), not just the handful wired into the 94 ColorBench metrics. Use
human_pool.compare_on_pool(a, b) for the best-of-breed human verdict.
"""

# ── group → tier ─────────────────────────────────────────────────────────────
# Group names match core.metric_categories.GROUP_CATEGORY (the MetricDef group
# field / metric-id prefix before the first '.').

TIER_HUMAN  = "human_psychophysical"
TIER_STRUCT = "structural"
TIER_RULER  = "human_fit_ruler"
TIER_CIELAB = "cielab_reference"
TIER_PROXY  = "heuristic_proxy"   # arbitrary designed target — no human/CIE grounding

GROUP_PROVENANCE = {
    # 1. human_psychophysical — direct published human datasets
    "hung_berns":           TIER_HUMAN,   # constant-hue loci, human
    "ebner_fairchild":      TIER_HUMAN,   # constant perceived-hue surfaces, human
    # munsell_*: human-data INPUT (ASTM D1535 chips) but the judge is a CV in
    # the candidate's OWN coordinates — legitimate for a spacing property, yet
    # any space DESIGNED to linearize Munsell (CIELab) scores well by
    # construction. Read with that caveat.
    "munsell_value":        TIER_HUMAN,   # Munsell value scale, human
    "munsell_hue":          TIER_HUMAN,   # Munsell hue spacing, human
    "macadam_isotropy":     TIER_HUMAN,   # real 1942 JND ellipse geometry (fixed 2026-07)
    "pointer_gamut":        TIER_HUMAN,   # Pointer real-surface gamut, human

    # 2. structural — coordinate-free objective facts, no reference space
    "roundtrip":            TIER_STRUCT,
    "double_rt":            TIER_STRUCT,
    "cross_gamut":          TIER_STRUCT,
    "channel_mono":         TIER_STRUCT,
    "negative_lms":         TIER_STRUCT,
    "quantization":         TIER_STRUCT,
    "achromatic":           TIER_STRUCT,
    "extremes":             TIER_STRUCT,
    "primary_hue_disc":     TIER_STRUCT,
    "oog_excursion":        TIER_STRUCT,
    "extreme_chroma_stab":  TIER_STRUCT,
    "stability":            TIER_STRUCT,
    "gamut":                TIER_STRUCT,   # cusp count / smoothness — gamut topology

    # 3. human_fit_ruler — Perceptia-Spacing (Munsell-fit) or CIEDE2000 (CIE std).
    #    These measure SPACING / DIFFERENCE / DISTINGUISHABILITY, judged by a
    #    human-validated instrument. Trustworthy for that question.
    "gradients":            TIER_RULER,   # spacing ruler (Perceptia-Spacing)
    "multistop_gradient":   TIER_RULER,   # spacing ruler
    "banding":              TIER_RULER,   # spacing ruler
    "animation":            TIER_RULER,   # spacing ruler
    "3color":               TIER_RULER,   # ΔE2000 distinguishability
    "photo_gamut_map":      TIER_RULER,   # ΔE2000 difference
    "gamut_mapping":        TIER_RULER,   # ΔE2000 difference
    "palette_uniformity":   TIER_RULER,   # ΔE2000 spacing
    "chroma_preservation":  TIER_RULER,   # ΔE2000 magnitude
    "dataviz_distinguish":  TIER_RULER,   # min pairwise ΔE2000
    "eased_animation":      TIER_RULER,
    "wcag_midpoint":        TIER_RULER,   # contrast = luminance (CIE std)
    "contrast":             TIER_RULER,
    "cvd":                  TIER_RULER,   # ΔE2000 under CVD simulation

    # 4. cielab_reference — CEILING-BOUND: hue/geometry judged vs CIE Lab as truth
    "hue_agreement":        TIER_CIELAB,  # angular deviation from CIE Lab hue
    "hue_leaf":             TIER_CIELAB,  # constant-hue-in-space vs CIE Lab hue
    "hue_reversal":         TIER_CIELAB,  # CIE Lab hue direction reversals
    "tint_shade_hue":       TIER_CIELAB,  # tint/shade hue drift vs CIE Lab
    "shade_hue_consistency":TIER_CIELAB,  # shade palette hue vs CIE Lab
    "harmony_accuracy":     TIER_CIELAB,  # rotation accuracy vs CIE Lab
    "jacobian":             TIER_CIELAB,  # local uniformity vs CIE Lab geometry

    # 5. heuristic_proxy — designed targets with NO human or CIE grounding.
    #    Reported for diagnostics, never in the headline.
    "hue":                  TIER_PROXY,   # even-60° primary grid — arbitrary target
                                          # (real yellow ≈ 85-100°, no space places
                                          # sRGB primaries on a 60° lattice)
    "specials":             TIER_PROXY,   # blue-white G/R, red-white G−B, yellow-C:
                                          # directional appearance heuristics
}

# Phase 9-11 user_* tests all measure difference/distinguishability via CIEDE2000
# → human_fit_ruler (CIEDE2000 is CIE-standard human-validated difference).
# Listed by prefix so the ~37 user_* groups need no per-name entry.
_USER_PREFIX = "user_"

TIER_INFO = {
    TIER_HUMAN: {
        "label": "İnsan psikofiziği (doğrudan insan verisi)",
        "ceiling": "human perception itself",
        "trustworthy": True,
        "rank": 1,
    },
    TIER_STRUCT: {
        "label": "Yapısal (koordinat-bağımsız olgu)",
        "ceiling": "none — objective fact",
        "trustworthy": True,
        "rank": 2,
    },
    TIER_RULER: {
        "label": "İnsan-fit cetvel (Perceptia-Spacing / CIEDE2000)",
        "ceiling": "the human-fit ruler (high, but a model)",
        "trustworthy": True,
        "rank": 3,
    },
    TIER_CIELAB: {
        "label": "CIE Lab-referanslı (TAVAN-BAĞIMLI — hue düzeltene haksız)",
        "ceiling": "CIE Lab quality (penalizes hue-correcting spaces)",
        "trustworthy": False,
        "rank": 4,
    },
    TIER_PROXY: {
        "label": "Sezgisel proxy (keyfi hedef — insan/CIE dayanağı yok)",
        "ceiling": "the designed target itself",
        "trustworthy": False,
        "rank": 5,
    },
}

TRUSTWORTHY_TIERS = [TIER_HUMAN, TIER_STRUCT, TIER_RULER]


def provenance_of(metric_id: str) -> str:
    """Flat metric id ('group.metric' or 'group') → provenance tier.

    Unknown groups default to human_fit_ruler (most metrics measure difference
    via a human-validated ruler); explicitly-listed cielab_reference groups are
    the only ones flagged ceiling-bound.
    """
    group = metric_id.split(".")[0] if "." in metric_id else metric_id
    if group.startswith(_USER_PREFIX):
        return TIER_RULER
    return GROUP_PROVENANCE.get(group, TIER_RULER)


def tiered_winhist(test_results, space_a: str, space_b: str):
    """Split a head-to-head W-L-T between two spaces by judge provenance tier.

    test_results: iterable of objects with .metric (MetricDef with .result_key
    + .lower_is_better + .abs_tie), .scores {space: float|None}, .ref_spaces.

    Returns: {tier: {"a": wins_a, "b": wins_b, "tie": ties, "n": total}} plus a
    "_headline" key aggregating the trustworthy tiers (1+2+3).
    """
    import math
    from .comparison import decide_outcome
    out = {t: {"a": 0, "b": 0, "tie": 0, "n": 0} for t in TIER_INFO}

    for tr in test_results:
        mdef = tr.metric
        gid = getattr(mdef, "result_key", "") or getattr(mdef, "score_path", "")
        tier = provenance_of(gid)
        sc_a = tr.scores.get(space_a)
        sc_b = tr.scores.get(space_b)
        if sc_a is None or sc_b is None:
            continue
        if space_a in tr.ref_spaces or space_b in tr.ref_spaces:
            continue
        cont = getattr(tr, "contaminated", {}) or {}
        if cont.get(space_a) == "full" or cont.get(space_b) == "full":
            continue  # in-sample (fit-data overlap) — uninformative pair
        # NaN/inf = computed but failed → loses to a finite opponent;
        # two failures carry no information (skip).
        fin_a = math.isfinite(sc_a)
        fin_b = math.isfinite(sc_b)
        if not fin_a and not fin_b:
            continue
        if fin_a != fin_b:
            out[tier]["n"] += 1
            out[tier]["a" if fin_a else "b"] += 1
            continue
        out[tier]["n"] += 1
        tr_items = getattr(tr, "items", {}) or {}
        outcome = decide_outcome(mdef, sc_a, sc_b,
                                 tr_items.get(space_a), tr_items.get(space_b))
        out[tier]["a" if outcome == "a" else "b" if outcome == "b" else "tie"] += 1

    headline = {"a": 0, "b": 0, "tie": 0, "n": 0}
    for t in TRUSTWORTHY_TIERS:
        for k in ("a", "b", "tie", "n"):
            headline[k] += out[t][k]
    out["_headline"] = headline
    return out


def verdict_vs_baseline(comparison, baseline: str = "OKLab", candidates=None):
    """Convenience: tiered W-L-T of each candidate vs a baseline, from a
    core.comparison.Comparison object (comparison.tests + comparison.space_names).

    Returns {candidate: winhist}. Print with format_tiered_verdict.
    """
    names = candidates if candidates is not None else [
        s for s in comparison.space_names if s != baseline]
    return {c: tiered_winhist(comparison.tests, c, baseline) for c in names}


def format_tiered_verdict(winhist, space_a: str, space_b: str) -> str:
    """Pretty-print the tiered verdict; headline (trustworthy) first, then the
    ceiling-bound CIE Lab tier flagged separately."""
    lines = []
    h = winhist["_headline"]
    lines.append(f"GÜVENİLİR VERDİCT (tier 1+2+3, {h['n']} metrik): "
                 f"{space_a} {h['a']} – {h['b']} {space_b}  (tie {h['tie']})")
    lines.append("  kırılım:")
    for tier in sorted(TIER_INFO, key=lambda t: TIER_INFO[t]["rank"]):
        d = winhist[tier]
        if d["n"] == 0:
            continue
        flag = "" if TIER_INFO[tier]["trustworthy"] else "  ⚠ TAVAN-BAĞIMLI"
        lines.append(f"    [{TIER_INFO[tier]['rank']}] {TIER_INFO[tier]['label']}: "
                     f"{d['a']}-{d['b']}-{d['tie']} (n={d['n']}){flag}")
    c = winhist[TIER_CIELAB]
    if c["n"]:
        lines.append(f"  NOT: CIE Lab-referanslı {c['n']} metrik headline'a DAHİL DEĞİL. "
                     f"Hue düzelten aday burada haksız düşük görünebilir — tier 1 (insan) esas alınır.")
    return "\n".join(lines)
