"""Fit-data contamination guard — the three-way holdout rule, enforced by
machine instead of discipline.

Rule (Görkem): ruler-fit / candidate-fit / benchmark-test data must be
DISJOINT. A candidate space that was fit on a human dataset must not be
scored by a judge built on that same dataset — the score would be in-sample.

A candidate declares its fit data with a "trained_on" list (dataset names as
in color-perception-datasets) — in its params JSON for --json spaces, or as a
`.trained_on` attribute. Undeclared = empty (nothing flagged); the report says
so explicitly, because "no declaration" must never silently read as "clean".

Contamination levels per metric group:
  "full"    — the judge IS the declared dataset (e.g. munsell_value judge on a
              Munsell-fit space). The space is excluded from winning that
              metric and the pair is skipped in head-to-heads (uninformative).
  "partial" — the judge's RULER has 1/3 exposure to a declared dataset (the
              spacing consensus contains Munsell-fit Perceptia-Spacing).
              Reported as a caveat, not excluded — 2/3 of the consensus is
              independent.
"""

# metric group → human datasets its judge is built on (direct judges)
METRIC_DATA_SOURCES = {
    "munsell_value":    {"munsell"},
    "munsell_hue":      {"munsell"},
    "macadam_isotropy": {"macadam1942"},
    "hung_berns":       {"hung_berns"},
    "ebner_fairchild":  {"ebner_fairchild"},
    "pointer_gamut":    {"pointer"},
}

# groups judged by the spacing consensus (1/3 Perceptia-Spacing = Munsell-fit)
SPACING_RULER_GROUPS = {"gradients", "multistop_gradient", "banding",
                        "animation", "eased_animation"}
SPACING_RULER_SOURCES = {"munsell"}


def trained_on_of(obj) -> set:
    """Declared fit datasets of a space object or a results/report dict."""
    if isinstance(obj, dict):
        raw = obj.get("trained_on") or []
    else:
        raw = getattr(obj, "trained_on", None) or []
    return {str(d).strip().lower() for d in raw}


def contamination_of(result_key: str, trained_on: set) -> str | None:
    """None | 'full' | 'partial' for one metric group vs one declaration."""
    if not trained_on:
        return None
    direct = METRIC_DATA_SOURCES.get(result_key)
    if direct and direct & trained_on:
        return "full"
    if result_key in SPACING_RULER_GROUPS and SPACING_RULER_SOURCES & trained_on:
        return "partial"
    return None


def summarize(comp) -> str | None:
    """One-paragraph contamination summary for a Comparison, or None."""
    full = {}
    partial = {}
    for tr in comp.tests:
        cont = getattr(tr, "contaminated", None) or {}
        for sname, level in cont.items():
            (full if level == "full" else partial).setdefault(sname, set()).add(
                tr.metric.result_key)
    if not full and not partial:
        return None
    lines = ["KONTAMİNASYON (fit-verisi ∩ yargıç-verisi):"]
    for sname, groups in sorted(full.items()):
        lines.append(f"  {sname}: {len(groups)} metrik grubu IN-SAMPLE → "
                     f"kazanamaz/h2h dışı: {', '.join(sorted(groups))}")
    for sname, groups in sorted(partial.items()):
        lines.append(f"  {sname}: kısmi (spacing cetvelinin 1/3'ü Munsell-fit): "
                     f"{', '.join(sorted(groups))} — dahil ama dikkatle oku")
    return "\n".join(lines)
