"""HTML report generator for space-test-project comparison results.

Takes a Comparison dataclass (from comparison.py) and produces a standalone
HTML report with scorecard, radar chart, per-category tables, and h2h matrix.
"""

import math
from datetime import datetime
from .comparison import Comparison, TestResult, MetricDef


CSS = """
:root { --bg: #ffffff; --fg: #111827; --muted: #6b7280; --border: #e5e7eb;
        --best: #dcfce7; --worst: #fee2e2; --accent: #6366f1; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       color: var(--fg); background: var(--bg); line-height: 1.6;
       max-width: 1200px; margin: 0 auto; padding: 20px; }
h1 { font-size: 1.8rem; margin-bottom: 8px; }
h2 { font-size: 1.3rem; margin-top: 32px; margin-bottom: 12px;
     border-bottom: 2px solid var(--accent); padding-bottom: 4px; }
h3 { font-size: 1.05rem; margin-top: 16px; margin-bottom: 6px; color: var(--muted); }
.subtitle { color: var(--muted); margin-bottom: 24px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 16px; font-size: 0.9rem; }
th, td { padding: 8px 12px; border: 1px solid var(--border); text-align: right; }
th { background: #f9fafb; text-align: left; font-weight: 600; }
td:first-child { text-align: left; font-weight: 500; }
td.best { background: var(--best); font-weight: 700; }
td.worst { background: var(--worst); }
td.ref { background: #f3f4f6; color: #9ca3af; font-style: italic; }
.scorecard { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
             gap: 12px; margin-bottom: 24px; }
.score-card { border: 1px solid var(--border); border-radius: 8px; padding: 16px;
              text-align: center; }
.score-card .value { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
.score-card .label { font-size: 0.85rem; color: var(--muted); }
.radar { width: 100%; max-width: 500px; display: block; margin: 16px auto; }
.section { margin-bottom: 32px; }
.h2h-table td { text-align: center; }
.h2h-win { background: var(--best); font-weight: 700; }
.h2h-lose { background: var(--worst); }
"""


def _fmt_score(val: float | None, mdef: MetricDef) -> str:
    """Format a metric score for display."""
    if val is None:
        return "N/A"
    # Only multiply by 100 for base-metric CVs stored as 0.xx (gradient overall)
    display_val = val
    if mdef.unit == "%" and mdef.result_key == "gradients":
        display_val = val * 100
    fmt = mdef.format_str or ".4f"
    try:
        formatted = f"{display_val:{fmt}}"
    except (ValueError, TypeError):
        formatted = str(val)
    if mdef.unit and mdef.unit not in ("", "/360"):
        formatted += mdef.unit
    elif mdef.unit == "/360":
        formatted += "/360"
    return formatted


def _score_class(space: str, tr: TestResult) -> str:
    """CSS class for a score cell."""
    if space in tr.ref_spaces:
        return "ref"
    fair = {k: v for k, v in tr.scores.items()
            if v is not None and k not in tr.ref_spaces}
    if not fair:
        return ""
    vals = list(fair.values())
    if tr.metric.lower_is_better:
        best_val, worst_val = min(vals), max(vals)
    else:
        best_val, worst_val = max(vals), min(vals)
    v = tr.scores.get(space)
    if v is None:
        return ""
    if v == best_val:
        return "best"
    if v == worst_val and len(vals) > 2:
        return "worst"
    return ""


def _radar_svg(spaces, test_results, width=480, height=480):
    """SVG radar chart from first 12 test results."""
    tests = test_results[:12]
    n = len(tests)
    if n < 3:
        return ""

    cx, cy = width / 2, height / 2
    r = min(cx, cy) - 60
    colors = ['#6366f1', '#ec4899', '#f59e0b', '#10b981', '#3b82f6',
              '#8b5cf6', '#ef4444', '#14b8a6']
    angle_step = 2 * math.pi / n

    svg = f'<svg viewBox="0 0 {width} {height}" class="radar">\n'

    # Grid rings
    for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
        pts = []
        for i in range(n):
            a = -math.pi / 2 + i * angle_step
            pts.append(f"{cx + r * level * math.cos(a):.1f},{cy + r * level * math.sin(a):.1f}")
        svg += f'  <polygon points="{" ".join(pts)}" fill="none" stroke="#e5e7eb"/>\n'

    # Axes + labels
    for i, tr in enumerate(tests):
        a = -math.pi / 2 + i * angle_step
        px, py = cx + r * math.cos(a), cy + r * math.sin(a)
        svg += f'  <line x1="{cx}" y1="{cy}" x2="{px:.1f}" y2="{py:.1f}" stroke="#d1d5db"/>\n'
        lx, ly = cx + (r + 30) * math.cos(a), cy + (r + 30) * math.sin(a)
        anchor = "middle"
        if abs(math.cos(a)) > 0.3:
            anchor = "start" if math.cos(a) > 0 else "end"
        label = tr.metric.name[:22]
        svg += f'  <text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="10" fill="#6b7280">{label}</text>\n'

    # Normalize scores per metric
    for si, space in enumerate(spaces):
        color = colors[si % len(colors)]
        pts = []
        for i, tr in enumerate(tests):
            a = -math.pi / 2 + i * angle_step
            fair = {k: v for k, v in tr.scores.items() if v is not None and k not in tr.ref_spaces}
            if not fair:
                pts.append(f"{cx:.1f},{cy:.1f}")
                continue
            vals = list(fair.values())
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx > mn else 1
            v = tr.scores.get(space, mn)
            if v is None:
                v = mn
            if space in tr.ref_spaces:
                norm = 1.0
            elif tr.metric.lower_is_better:
                norm = 1.0 - (v - mn) / rng
            else:
                norm = (v - mn) / rng
            norm = max(0.05, min(1.0, norm))
            pts.append(f"{cx + r * norm * math.cos(a):.1f},{cy + r * norm * math.sin(a):.1f}")
        svg += f'  <polygon points="{" ".join(pts)}" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="2"/>\n'

    # Legend
    for si, space in enumerate(spaces):
        color = colors[si % len(colors)]
        svg += f'  <rect x="{20 + si * 130}" y="{height - 25}" width="12" height="12" fill="{color}"/>\n'
        svg += f'  <text x="{36 + si * 130}" y="{height - 14}" font-size="12" fill="#374151">{space}</text>\n'

    svg += '</svg>\n'
    return svg


def generate(comp: Comparison, output_path: str, title: str = "Color Space Benchmark"):
    """Generate standalone HTML report from Comparison."""
    n_tests = len(comp.tests)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">Generated {ts} | Spaces: {', '.join(comp.space_names)} | {n_tests} metrics</p>
"""

    # ── Scorecard ──
    html += '<h2>Scorecard</h2>\n<div class="scorecard">\n'
    for s in comp.space_names:
        solo = comp.solo_wins.get(s, 0)
        shared = comp.shared_wins.get(s, 0)
        html += f'<div class="score-card"><div class="value">{solo}</div>'
        html += f'<div class="label">{s} solo wins'
        if shared > 0:
            html += f' (+{shared} shared)'
        html += '</div></div>\n'
    html += '</div>\n'

    # ── Head-to-Head Matrix ──
    if len(comp.space_names) >= 2:
        html += '<h2>Head-to-Head</h2>\n'
        html += '<table class="h2h-table"><tr><th></th>'
        for s in comp.space_names:
            html += f'<th>{s}</th>'
        html += '</tr>\n'
        for s1 in comp.space_names:
            html += f'<tr><td><strong>{s1}</strong></td>'
            for s2 in comp.space_names:
                if s1 == s2:
                    html += '<td>-</td>'
                else:
                    key = (s1, s2) if (s1, s2) in comp.head_to_head else (s2, s1)
                    h = comp.head_to_head.get(key)
                    if h:
                        if key == (s1, s2):
                            w, l = h['w1'], h['w2']
                        else:
                            w, l = h['w2'], h['w1']
                        t = h['tie']
                        cls = "h2h-win" if w > l else ("h2h-lose" if l > w else "")
                        html += f'<td class="{cls}">{w}-{l} ({t}t)</td>'
                    else:
                        html += '<td>?</td>'
            html += '</tr>\n'
        html += '</table>\n'

    # ── Radar ──
    svg = _radar_svg(comp.space_names, comp.tests)
    if svg:
        html += f'<h3>Performance Radar (outer = better)</h3>\n{svg}\n'

    # ── Per-category metric tables ──
    categories = []
    for tr in comp.tests:
        if tr.metric.category not in categories:
            categories.append(tr.metric.category)

    for cat in categories:
        cat_tests = [tr for tr in comp.tests if tr.metric.category == cat]
        html += f'<div class="section">\n<h2>{cat}</h2>\n'
        html += '<table><tr><th>Metric</th>'
        for s in comp.space_names:
            html += f'<th>{s}</th>'
        html += '<th>Winner</th></tr>\n'

        for tr in cat_tests:
            html += '<tr>'
            html += f'<td>{tr.metric.name}'
            if tr.metric.unit:
                html += f' <small>({tr.metric.unit})</small>'
            html += '</td>'
            for s in comp.space_names:
                cls = _score_class(s, tr)
                val = _fmt_score(tr.scores.get(s), tr.metric)
                if s in tr.ref_spaces:
                    val += ' <small>(ref)</small>'
                html += f'<td class="{cls}">{val}</td>'
            w = tr.winner or ("TIE" if tr.is_tie else "?")
            html += f'<td>{w}</td></tr>\n'

        html += '</table>\n</div>\n'

    # ── Footer ──
    html += f"""
<hr style="margin-top:40px;border:none;border-top:1px solid var(--border)">
<p style="font-size:0.8rem;color:var(--muted);text-align:center;margin-top:12px">
Generated by space-test-project | {n_tests} metrics | {ts}
</p>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
