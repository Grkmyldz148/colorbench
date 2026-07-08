"""Property × space scorecard — the benchmark's primary output format.

The project's own headline finding (T11) is that NO space is best at
everything: CIELab wins difference, OKLab wins hue/discrimination, etc.
A single aggregate number hides exactly that, so the scorecard shows the
per-property picture: every validated human-data judge as a row, every
candidate as a column, best-in-row marked.
"""
from . import human_pool as hp
from .contamination import trained_on_of

# stable row order
_PROP_ORDER = ["difference", "difference_rank", "hue", "discrimination",
               "3d_discrim", "tolerance", "hk_mechanism", "adaptation"]
_HIGHER_BETTER_PROPS = {"difference_rank", "hk_mechanism"}


def scorecard(spaces: dict, validated_only: bool = True) -> str:
    """Render the karne for {display_name: space_object}.

    Per dataset row: each space's score, best marked ★ (fit-data-contaminated
    cells marked ⚠ and excluded from best). Ends with a per-property winner
    summary — the honest replacement for a single overall score.
    """
    names = list(spaces.keys())
    pools = {n: hp.evaluate_space_on_pool(s, validated_only=validated_only)
             ["by_property"] for n, s in spaces.items()}
    fits = {n: trained_on_of(s) for n, s in spaces.items()}

    colw = max(10, max(len(n) for n in names) + 2)
    lines = ["KARNE — özellik × uzay (her satırda ★ = en iyi; ⚠ = in-sample, yarış dışı)",
             "  " + " " * 38 + "".join(f"{n:>{colw}}" for n in names)]
    prop_wins = {}

    for prop in _PROP_ORDER:
        datasets = sorted({ds for n in names for ds in pools[n].get(prop, {})})
        if not datasets:
            continue
        lower = prop not in _HIGHER_BETTER_PROPS
        arrow = "↓" if lower else "↑"
        lines.append(f"\n  [{prop}] ({arrow} iyi)")
        row_wins = {n: 0 for n in names}
        for ds in datasets:
            vals = {}
            for n in names:
                v = pools[n].get(prop, {}).get(ds)
                if isinstance(v, (int, float)):
                    vals[n] = v
            eligible = {n: v for n, v in vals.items() if ds.lower() not in fits[n]}
            best = None
            if eligible:
                best = (min if lower else max)(eligible, key=eligible.get)
            row = f"    {ds:36}"
            for n in names:
                v = vals.get(n)
                if v is None:
                    row += f"{'—':>{colw}}"
                    continue
                mark = "★" if n == best else ("⚠" if ds.lower() in fits[n] else " ")
                row += f"{v:>{colw - 1}.3f}{mark}"
            lines.append(row)
            if best is not None and prop in hp._LOWER_BETTER:
                row_wins[best] += 1
        if prop in hp._LOWER_BETTER and any(row_wins.values()):
            winner = max(row_wins, key=row_wins.get)
            prop_wins[prop] = (winner, row_wins)

    if prop_wins:
        lines.append("\n  ÖZELLİK KAZANANLARI (tek toplam skor YOK — bilinçli):")
        for prop, (winner, wins) in prop_wins.items():
            detail = ", ".join(f"{n}:{w}" for n, w in wins.items())
            top = wins[winner]
            if sum(1 for w in wins.values() if w == top) > 1:
                winner = "berabere"
            lines.append(f"    {prop:16} → {winner}   ({detail})")
    return "\n".join(lines)
