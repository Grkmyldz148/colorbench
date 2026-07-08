"""Fairness-corrected verdict (2026-06-08) — turns ColorBench's raw 94-metric
W-L-T into a fair weighted verdict by applying the three deductions found in the
fairness audit:

  Fix 1 — DE-WEIGHT GAMUT. 31/94 metrics are 'gamut', but they are ~10 sub-metrics
          measured across 3 gamuts (sRGB/P3/Rec2020). Counting each separately
          gives gamut 3× weight and skews the aggregate toward gamut-optimized
          generation spaces. Each gamut metric gets weight 1/3 → gamut contributes
          ~10 effective, balanced with the rest.

  Fix 3 — DROP CEILING-BOUND CIELab-REFERENCE METRICS. The 11 tier-4 metrics judge
          hue/geometry against CIELab AS truth, penalizing hue-correcting spaces.
          Weight 0 in the fair aggregate. The HUE property is instead judged by the
          real-human-data panel (core.human_pool: Hung-Berns / Ebner-Fairchild /
          Munsell) when space objects are supplied.

  (Fix 2 — neutral spacing ruler — is applied upstream in core.rulers: the
   'spacing' ruler is now a 3-uniform-space consensus, so the ruler-tier metrics
   already arrive de-biased. No verdict-layer action needed.)

Use fair_winhist(comparison, a, b) for the weighted ColorBench verdict, or
fair_verdict_full(space_a, space_b, comparison) to also fold in the human pool.
"""
import math

from .judge_provenance import provenance_of, TIER_CIELAB, TIER_PROXY, TIER_INFO


def metric_weight(result_key: str) -> float:
    """Fairness weight for one metric group."""
    if provenance_of(result_key) in (TIER_CIELAB, TIER_PROXY):
        return 0.0                      # Fix 3: drop ceiling-bound + arbitrary-target metrics
    if result_key == "gamut":
        return 1.0 / 3.0                # Fix 1: 3-gamut repeats share one vote
    return 1.0


def fair_winhist(comparison, space_a: str, space_b: str):
    """Weighted head-to-head between two spaces over comparison.tests.
    Outcomes come from comparison.decide_outcome (abs_tie → bootstrap CI →
    relative threshold), so this path always agrees with solo/h2h/tiered.
    Returns {'a','b','tie','weight_total', plus 'raw_a','raw_b','raw_tie' (unweighted
    for comparison), and 'dropped_cielab','gamut_collapsed'}."""
    from .comparison import decide_outcome
    a = b = tie = 0.0
    raw_a = raw_b = raw_tie = 0
    dropped = gamut_n = 0
    for tr in comparison.tests:
        mdef = tr.metric
        rk = getattr(mdef, "result_key", "")
        w = metric_weight(rk)
        sc_a = tr.scores.get(space_a); sc_b = tr.scores.get(space_b)
        if sc_a is None or sc_b is None:
            continue
        if space_a in tr.ref_spaces or space_b in tr.ref_spaces:
            continue
        cont = getattr(tr, "contaminated", {}) or {}
        if cont.get(space_a) == "full" or cont.get(space_b) == "full":
            continue  # in-sample (fit-data overlap) — uninformative pair
        # NaN/inf = computed but failed → loses to a finite opponent;
        # two failures carry no information (skip).
        fin_a = math.isfinite(sc_a); fin_b = math.isfinite(sc_b)
        if not fin_a and not fin_b:
            continue
        if fin_a != fin_b:
            outcome = "a" if fin_a else "b"
        else:
            tr_items = getattr(tr, "items", {}) or {}
            outcome = decide_outcome(mdef, sc_a, sc_b,
                                     tr_items.get(space_a), tr_items.get(space_b))
        # raw (true ColorBench, every metric counts 1) for honest before/after
        raw_a += outcome == "a"; raw_b += outcome == "b"; raw_tie += outcome == "tie"
        if w == 0:
            dropped += 1
        if rk == "gamut":
            gamut_n += 1
        # weighted tally
        if outcome == "a":
            a += w
        elif outcome == "b":
            b += w
        else:
            tie += w
    return {"a": round(a, 2), "b": round(b, 2), "tie": round(tie, 2),
            "raw_a": raw_a, "raw_b": raw_b, "raw_tie": raw_tie,
            "dropped_cielab": dropped, "gamut_metrics": gamut_n}


def fair_verdict_full(space_a, space_b, comparison, name_a="A", name_b="B"):
    """Full fair verdict: weighted ColorBench W-L-T (Fix 1+3) PLUS the real-human
    hue/discrimination panel (core.human_pool) replacing the dropped CIELab hue
    metrics. space_a/space_b are the space OBJECTS; comparison is their Comparison.
    Returns a printable summary string."""
    wh = fair_winhist(comparison, name_a, name_b)
    lines = [f"FAIR VERDICT — {name_a} vs {name_b}",
             f"  ColorBench ağırlıklı W-L-T (gamut 1/3, CIELab-ref dışlandı): "
             f"{name_a} {wh['a']} – {wh['b']} {name_b} (tie {wh['tie']})",
             f"    ham (düzeltmesiz): {wh['raw_a']}-{wh['raw_b']}-{wh['raw_tie']}  | "
             f"dışlanan CIELab-ref: {wh['dropped_cielab']}  | gamut metrik: {wh['gamut_metrics']} (×1/3)"]
    try:
        from . import human_pool as hp
        lines.append("  + İnsan paneli (gerçek veri, CIELab-hue yerine):")
        panel = hp.compare_on_pool(space_a, space_b, name_a, name_b, validated_only=True)
        for ln in panel.splitlines()[1:]:
            lines.append("  " + ln)
    except Exception as e:
        lines.append(f"  (insan paneli atlandı: {repr(e)[:80]})")
    return "\n".join(lines)
