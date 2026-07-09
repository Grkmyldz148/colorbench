# ColorBench

A rigorous, GPU-accelerated benchmark for comparing perceptual color spaces — and
for auditing *its own* fairness. ColorBench measures everything, hides nothing, and
ships a tiered-verdict layer so a single win-count never misleads you.

## Two evaluation modes

Both modes apply identical pre-processing to every space, are deterministic, and
ship documented fairness notes.

| Mode | Question | What it scores | Engine |
|------|----------|----------------|--------|
| **`compare`** (gradient / generation) | "Is this space good for *generating* color?" | 94 metrics across sRGB / P3 / Rec.2020 — gamut geometry & cusps, gradient/banding quality, palette uniformity, hue stability, round-trip precision | PyTorch (`core/cs/`, `core/metrics/`) |
| **`metric`** (STRESS / difference) | "Is this space good for *predicting color difference*?" | STRESS on human pair datasets (COMBVD 3813, MacAdam 1974 128) + Human-Feedback 3552 as **rank-only** (Spearman ρ — its 5-level ordinal DV makes STRESS an artefact; excluded from the pooled avg) | NumPy (`core/metric_eval.py`), CAM16-UCS/Jzazbz via **colour-science** |

## Quick start

```bash
# Python 3.11+, PyTorch, NumPy, colour-science (≥0.4)
pip install torch numpy colour-science

# compare mode — head-to-head over 94 generation metrics
python run.py oklab cielab
python run.py oklab genspace --json path/to/params.json

# metric mode — STRESS (perceptual-difference prediction)
python run.py metric --json path/to/metricspace.json
```

Or from Python — **any** color space enters with two callables:

```python
import colorbench as cb

space = cb.wrap(fwd, inv, name="myspace",     # XYZ→coords, coords→XYZ (torch N×3)
                trained_on=["munsell"])       # fit-data declaration (holdout guard)
profile = cb.evaluate(space)                  # 94 metrics vs cached baselines
print(profile.verdict())                      # tiered + fair verdict
print(profile.scorecard())                    # property × space karne
profile.html("report.html")
```

Output: terminal summary + JSON reports + an HTML comparison, all written to
`results/` (which is **git-ignored** — it holds run artifacts, not source).

## The three runners (what each one is for)

| Script | Purpose |
|--------|---------|
| **`run.py`** | The main entry point. `compare` mode (pass ≥1 built-in space name or `--json`) and `metric` mode (`run.py metric …`). Everything below is a niche side-tool. |
| **`run_gma_benchmark.py`** | A **gamut-mapping** benchmark (Display-P3 → sRGB): scores 7 mapping algorithms (clip, OKLab/CIELab LCh, etc.) on 1000 P3-extreme colors. Not a space test — an algorithm test. |
| **`run_near_mono.py`** | A **near-monochrome palette** diagnostic: in space *S*'s coordinates, how 1-dimensional does palette *P* look (PCA variance on the dominant axis)? Used for the "near-mono" landing claim. |

## Fairness — and how it's enforced

ColorBench's design goal is to be a *trustworthy ruler*, so its own fairness is
audited and corrected, not assumed. Full audit:
`research/notebook/02-space-pool/COLORBENCH_FAIRNESS_AUDIT.md`.

**Mechanical integrity (verified, ~clean):**
- **Third-party ground truth** — CIEDE2000 (CIE 224:2017), CIELab, CIE94, CAM16-UCS,
  Jzazbz come from `colour-science`, not hand-rolled. STRESS formula is scale-invariant
  (verified end-to-end). Deterministic (fixed seeds).
- **Reference spaces can't win** — a space used as a metric's reference is excluded
  from win-counting.
- **Self-referential scores excluded** — when CIELab trivially scores 0 on a "deviation
  from CIELab" metric, that 0 is detected and dropped (`core/judge_provenance.py`).
- **Bradford CAT** applied per-pair to spaces that don't model multi-illuminant whites.

**Structural skews — found by audit, corrected:**
1. **Gamut was over-weighted** (31/94 metrics = ~10 sub-metrics × 3 gamuts → 3× weight).
   `core/fair_verdict.py` gives gamut metrics weight 1/3.
2. **The spacing ruler wasn't neutral** (Perceptia-Spacing ≈ CIELab; gradient rankings
   flipped by ruler choice). `core/rulers.py` now makes the `spacing` ruler a **consensus**
   of three uniform spaces {Perceptia-Spacing, CAM16-UCS, Jzazbz}, removing the lever.
3. **11 metrics judged hue against CIELab as truth** (penalizing hue-correcting spaces).
   `fair_verdict.py` drops them; hue is instead judged by real human data.
4. **MacAdam isotropy measured the wrong thing** (2026-07): it perturbed centers by a
   fixed *xy circle* and rewarded ratio→1 — i.e. rewarded *ignoring* MacAdam anisotropy.
   Now it samples each real 1942 JND ellipse perimeter (a/b/θ, Bradford C→D65), so
   ratio 1.0 genuinely means matching human thresholds.
5. **Gradient subset CVs bucketed pairs in each space's own coordinates** (2026-07) —
   different spaces were scored on different pair populations. Bucketing is now in
   fixed CIE Lab (D65): identical populations for every candidate.
6. **NaN/inf scores could corrupt the verdict** (2026-07): a space that failed to
   compute a metric could cancel the winner or count as a tie, order-dependently.
   Non-finite scores now deterministically lose to any finite opponent.
7. **HumanFB scored with STRESS despite its ordinal 5-level DV** (2026-07): now
   rank-only (Spearman ρ), separated from every pooled/headline number.

**Methodology v2 (2026-07-08) — statistical rigor layer:**
- **Statistical ties.** For metrics with per-item structure (gradient pairs,
  ellipses, hue loci), ties are decided by a **paired bootstrap**: same resample
  indices for both spaces, tie iff the 95% CI of the aggregate difference
  contains 0 (`core/bootstrap.py`, fixed seed — deterministic). The arbitrary
  1% threshold survives only for metrics without item structure, and every run
  reports which rule decided how many metrics. STRESS scores print with CI95.
- **Ruler-sensitivity flag.** Spacing-ruler metrics are additionally computed
  under EACH consensus member (Perceptia-Spacing / CAM16-UCS / Jzazbz); a
  verdict that flips with the ruler is flagged `SENSITIVE` — a property of the
  ruler, not the space.
- **Contamination guard.** Candidates declare fit data (`"trained_on"` in the
  params JSON / `cb.wrap(..., trained_on=[...])`). Judges built on a declared
  dataset are in-sample for that space: it cannot win them, head-to-head pairs
  are skipped, and the report prints the contamination summary. This is the
  three-way holdout rule (ruler-fit / candidate-fit / test disjoint) enforced
  by machine.
- **Scorecard output.** The primary result is the property × space karne with
  per-property winners — deliberately **no overall score**, because no space is
  best at everything (the project's central finding).

## The fairness / verdict layer (`core/`)

These modules turn a raw comparison into a verdict you can trust:

| Module | Role |
|--------|------|
| `judge_provenance.py` | Tags each of the 94 metrics by **who judges it** — human-psychophysical / structural / human-fit-ruler / CIELab-reference / heuristic-proxy — and reports W-L-T split by tier (ceiling-bound and arbitrary-target tiers flagged, never in the headline). |
| `human_pool.py` | **Best-of-breed human panel.** Grounds each property in the curated [`color-perception-datasets`](https://github.com/Grkmyldz148/color-perception-datasets) pool — 24 datasets wired as of 2026-07: JND ellipses (MacAdam 1942, Luo-Rigg, Alder, Hong 2025 with measured-primaries colorimetry), 3D ellipsoids (Koenderink 2026, Brown 1957, Brown-MacAdam 1949, Wyszecki-Fielder 1971), Lab-ellipsoids (Huang 2012), tolerance vectors (RIT-DuPont/Berns 1991), constant-hue loci (Hung-Berns, Ebner-Fairchild, Munsell), unique hues (Xiao 2011), **OSA-UCS uniform spacing (558-sample committee atlas — the independent, non-Munsell spacing anchor)**, H-K (Sanders-Wyszecki, Wyszecki 1967, Zhang aperture + Fairchild-Pirrotta 1991 object-colour), CAT, WCS naming & Asano observer-metamerism (diagnostic). 15 schema-aware judges, each validated (gray-ramp + degenerate-space direction check) before entering the headline tier. |
| `fair_verdict.py` | Weighted W-L-T applying the 3 fixes above + folds in the human panel. `fair_verdict_full(space_a, space_b, comparison)`. |
| `rulers.py` | Modular human-fit rulers (difference / threshold / spacing-consensus) — each property measured with the right instrument. |

## Architecture

```
colorbench/
  run.py                  # main runner — compare + metric modes
  run_gma_benchmark.py    # side-tool: gamut-mapping algorithm benchmark
  run_near_mono.py        # side-tool: near-monochrome palette diagnostic
  core/
    cs/                   # color-space implementations (OKLab, CIELab, HelmCT, …) + literature canon
    metrics/              # the 94 compare-mode metric implementations
    comparison.py         # METRIC_DEFS (94) + winner/tie logic + head-to-head
    metric_eval.py        # metric mode: STRESS on human datasets (colour-science baselines)
    rulers.py             # modular human-fit rulers (incl. spacing consensus)
    judge_provenance.py   # per-metric judge tiering + tiered verdict
    human_pool.py         # 43-dataset best-of-breed human panel
    fair_verdict.py       # gamut-deweighted, CIELab-ceiling-free weighted verdict
    gpu_de.py             # GPU CIEDE2000
    report.py / html_report.py   # terminal / JSON / HTML output
    pairs.py              # deterministic gradient-pair generator
  tests/                  # pin tests vs colour-science + snapshot regression
  scripts/                # category report, claims generator
  results/                # run artifacts — GIT-IGNORED (not source)
```

- Runs on CUDA GPU or CPU (automatic fallback), float64 throughout.
- All random tests use fixed seeds — fully deterministic.
- The `metric` mode reads datasets from `../datasets`; the human pool reads the
  separate `color-perception-datasets` repo (path via `COLOR_PERCEPTION_POOL`).

## Reading a result

Never read the raw 94-metric win-count as the verdict — it over-weights gamut and
includes CIELab-ceiling metrics. Since 2026-07 `run.py` prints the corrected verdict
automatically after every multi-space comparison: the tiered W-L-T (headline = tiers
1-3), the weighted fair verdict (gamut ×1/3, ceiling-bound/proxy ×0), and — for
2-space runs with the dataset pool available — the real human-data panel.

Programmatic access:

```python
from core.comparison import compare_spaces
from core.fair_verdict import fair_verdict_full
cmp = compare_spaces(results_by_space)
print(fair_verdict_full(space_a_obj, space_b_obj, cmp, "A", "B"))
# → gamut-balanced, CIELab-ceiling-free weighted W-L-T + the real human-data panel
```

## Tests

```bash
python -m pytest tests/      # pins vs colour-science + snapshot regression
```

## Citation

```
ColorBench: A rigorous, self-auditing benchmark for perceptual color spaces.
https://github.com/Grkmyldz148/colorbench
```

## License

MIT — [Görkem Yıldız](https://gorkemyildiz.com)
