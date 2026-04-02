# ColorBench

A rigorous, GPU-accelerated benchmark for comparing perceptual color spaces.

83 metrics across 12 categories. 3038 gradient pairs spanning sRGB, Display P3, and Rec.2020 gamuts. Fully deterministic, reproducible, and documented — including its own limitations.

## Why

Every color space claims to be "perceptually uniform" but there's no standard way to verify this. Existing comparisons cherry-pick metrics, use inconsistent test conditions, or hide CIE Lab's structural advantages.

ColorBench measures everything and hides nothing. Each result includes fairness notes explaining which metrics favor which spaces and why.

## Quick Start

```bash
# Requirements: Python 3.11+, PyTorch, NumPy
pip install torch numpy

# Compare OKLab vs CIE Lab
python run.py oklab cielab

# Compare a custom space from JSON
python run.py oklab genspace --json path/to/params.json

# Output: terminal summary + JSON reports + HTML comparison in results/
```

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
from core.spaces import CustomSpace

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
