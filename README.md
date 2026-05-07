# ColorBench

A rigorous, GPU-accelerated benchmark for comparing perceptual color spaces.

ColorBench has **two evaluation modes**, used for different scientific
questions. Both apply identical pre-processing to every space, are
deterministic, and ship with documented fairness notes.

| Mode | What it tests | Tooling | Used by |
|------|---------------|---------|---------|
| **`compare` (gradient/palette)** | 90 metrics × 3038 gradient pairs across sRGB/P3/Rec.2020 — gamut geometry, gradient quality, palette uniformity, hue stability, cusp behaviour | PyTorch (`core/spaces.py`, `core/spaces_literature.py`) | "Is this space good for color *generation*?" |
| **`metric` (STRESS)** | 4 perceptual datasets (COMBVD 3813, MacAdam 1974 128, He~2022 82, Human Feedback 3552) scored with the STRESS estimator | NumPy (`core/metric_eval.py`) — delegates CAM16-UCS, CIECAM02-UCS, $J_z a_z b_z$ to **`colour-science`** for canonical reference values | "Is this space good for color *difference prediction*?" |

The two modes share dataset loaders, the STRESS formula, and the
Bradford CAT pre-processing convention; they differ only in what they
measure (gradient quality vs ΔE prediction). See [§ Fairness](#fairness)
below for what each mode does and does not control for.

## Why

Every color space claims to be "perceptually uniform" but there's no standard way to verify this. Existing comparisons cherry-pick metrics, use inconsistent test conditions, or hide CIE Lab's structural advantages.

ColorBench measures everything and hides nothing. Each result includes fairness notes explaining which metrics favor which spaces and why.

## Quick Start

```bash
# Requirements: Python 3.11+, PyTorch, NumPy, colour-science (≥0.4)
pip install torch numpy colour-science

# Mode 1 — gradient/palette benchmark
python run.py oklab cielab                        # head-to-head 90 metrics
python run.py oklab genspace --json params.json   # custom JSON space

# Mode 2 — STRESS evaluation (perceptual distance prediction)
python run.py metric --json metricspace_v21.json  # COMBVD/MacAdam/HumFB STRESS

# Output: terminal summary + JSON reports + HTML comparison in results/
```

## Fairness

We use third-party reference implementations wherever possible to remove
"author-implements-its-own-baselines" bias:

- **CAM16-UCS** (Li et al. 2017): `colour.XYZ_to_CAM16UCS`. Default
  conditions (`L_A=64 cd/m²`, `Y_b=20%`, average surround). The earlier
  ColorBench shipped a hand-rolled NumPy implementation that produced
  ~22 STRESS points high on COMBVD; replaced 2026-05-06.
- **CIECAM02-UCS** (Luo et al. 2006): `colour.XYZ_to_CIECAM02` + JMh →
  CAM02-UCS via `colour.JMh_CIECAM02_to_CAM02UCS`. Same conditions as
  CAM16-UCS.
- **Jzazbz** (Safdar et al. 2017): `colour.XYZ_to_Jzazbz`. HDR-aware
  PQ transfer with the published 10,000 cd/m² peak constant.
- **CIEDE2000**, **CIE94**, **CIE Lab**, **DIN99**, **OKLab**: own
  NumPy implementations, validated against `colour-science` to within
  ±0.5 STRESS points across all three datasets.
- **Bradford CAT** is applied per-pair to all spaces that don't model
  multi-illuminant whites internally.

PyTorch reference classes in `core/spaces_literature.py` (used by the
`compare` mode) are **separately implemented** because that mode requires
batched GPU evaluation and gradient/palette tests need full-D adaptation
to make achromatic axis exactly zero (D=1 by design). They are
intentionally not bit-identical to `metric_eval`'s reference values, but
both are documented and reproducible.

## What It Measures

### 83 Metrics in 12 Categories

| Category | Count | What It Tests |
|----------|-------|---------------|
| **Gamut Geometry** | 27 | Cusps, monotonicity, cliff, smoothness, boundary continuity, invalid cusps, bad hues — across sRGB, P3, and Rec.2020 |
| **Gradient Quality** | 10 | CV mean/p95/max, hue drift, banding, 3-color CV, subset CVs (bright, dark, high-chroma, cross-lightness, near-achromatic) |
| **Application** | 12 | Palette L* spacing, tint/shade hue, data viz, multi-stop gradient, WCAG contrast, harmony accuracy, photo gamut mapping, eased animation, shade palette |
| **Perceptual Accuracy** | 5 | Munsell Value/Hue spacing, MacAdam ellipse isotropy, hue leaf constancy, CIE Lab hue agreement |
| **Numerical Stability** | 3 | Round-trip precision across 16.7M colors (sRGB, P3, Rec.2020) |
| **Structural** | 8 | OOG excursion, hue reversals, primary hue discontinuity, negative LMS, extreme chroma amplification |
| **Advanced** | 6 | 1000-trip RT accumulation, 8-bit quantization, channel monotonicity, Jacobian condition, cross-gamut consistency |
| **Hue** | 2 | Hue RMS, primary L range |
| **Achromatic** | 2 | Gray ramp chroma (sRGB + pure D65) |
| **Special** | 3 | Blue-white midpoint G/R, red-white midpoint, yellow chroma |
| **Banding** | 2 | Invisible gradient steps, duplicate 8-bit steps |
| **Accessibility** | 2 | CVD simulation (protan/deutan) gradient dE |

### 3038 Gradient Pairs

- **sRGB** (1552): primaries, complementary, hue sweep, saturation, lightness, near-achromatic, dark/bright extremes, gamut boundary, skin tones, earth tones, warm/cool transitions, very similar colors, neon, high luminance, UI shade palettes, 1000 random
- **Display P3** (749): primaries, cross-gamut, hue sweep, near-achromatic, boundary, lightness, 500 random
- **Rec.2020** (743): primaries, cross-gamut, hue sweep, near-achromatic, boundary, lightness, 500 random

## Supported Spaces

Built-in:
- **OKLab** — Bjorn Ottosson (2020), CSS Color Level 4 standard
- **CIE Lab** — CIE (1976), legacy standard

From JSON checkpoint:
- **GenSpace** — M1/gamma/M2 pipeline with optional L correction
- **GenSpace+BlueFix** — GenSpace with blue channel post-processing
- **Naka-Rushton** — Neurophysiological cone response + enrichment
- **Custom** — Any forward/inverse function pair

## Fairness

ColorBench documents its own biases. Every JSON report includes a `_methodology` section with fairness notes:

**CIEDE2000 structural bias** (medium): Gradient CV and related metrics use CIEDE2000, which is built on CIE Lab. This gives CIE Lab-adjacent spaces a structural advantage. No independent alternative exists.

**Munsell data favors CIE Lab** (medium): CIE Lab was designed to linearize Munsell Value. High Munsell Value scores mean agreement with CIE Lab, not necessarily perceptual accuracy.

**MacAdam ellipses are 1942 data** (medium): Local isotropy is measured at MacAdam's original chromaticity points. Spaces optimized for different regions may score poorly despite being perceptually superior.

**Self-referential detection**: When a space trivially scores zero on a test because it IS the reference frame (e.g., CIE Lab on "hue agreement with CIE Lab"), the score is marked `(ref)` and excluded from win counting.

## Output

**Terminal**: Full metric breakdown per space + comparison table with winners.

**JSON**: Complete raw data for every metric, gradient pair detail, cusp geometry, and methodology notes. Machine-readable for further analysis.

**HTML**: Visual comparison report with scorecard, head-to-head matrix, radar chart, and per-category tables.

## Architecture

```
colorbench/
  run.py                          # CLI runner (39 test functions + compare)
  core/
    spaces.py                     # 8 space implementations (single source of truth)
    pairs.py                      # 3038 gradient pair generator (deterministic)
    gpu_metrics.py                # Core metrics (round-trip, gradient, gamut, etc.)
    gpu_metrics_advanced.py       # Advanced metrics (CVD, animation, Jacobian, etc.)
    gpu_metrics_perceptual.py     # Perceptual/application metrics (Munsell, MacAdam, etc.)
    constants.py                  # Hardcoded data (Munsell, MacAdam, WCAG — zero file deps)
    comparison.py                 # 83 METRIC_DEFS + winner logic + head-to-head
    html_report.py                # HTML report generator
    report.py                     # JSON + terminal output
```

- Pure PyTorch — runs on CUDA GPU or CPU (automatic fallback)
- float64 precision throughout
- All random tests use fixed seeds — fully deterministic
- Zero external data dependencies — Munsell/MacAdam constants are hardcoded

## Adding a Custom Space

```python
from core.cs import CustomSpace

def my_forward(xyz):  # (N, 3) tensor → (N, 3) tensor
    ...

def my_inverse(lab):  # (N, 3) tensor → (N, 3) tensor
    ...

space = CustomSpace("My Space", my_forward, my_inverse)
```

## Citation

If you use ColorBench in research or tooling:

```
ColorBench: A rigorous benchmark for perceptual color spaces.
https://github.com/Grkmyldz148/colorbench
```

## License

MIT

## Author

[Gorkem Yildiz](https://gorkemyildiz.com)
