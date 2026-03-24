# ColorBench

A rigorous, GPU-accelerated benchmark for comparing perceptual color spaces.

46 metrics across 8 categories. 2903 gradient pairs spanning sRGB, Display P3, and Rec.2020 gamuts. Fully deterministic, reproducible, and documented — including its own limitations.

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

### 46 Metrics in 8 Categories

| Category | Metrics | What It Tests |
|----------|---------|---------------|
| **Gradient Quality** (6) | CV mean/p95/max, hue drift, banding, 3-color CV | How uniform are color transitions? |
| **Numerical Stability** (5) | Round-trip (16.7M colors), 1000-trip accumulation, 8-bit quantization, channel monotonicity, Jacobian condition | Does the math work? |
| **Achromatic Fidelity** (2) | Gray ramp chroma (sRGB + pure D65) | Do grays stay gray? |
| **Gamut Geometry** (7) | Cusp positions, monotonicity, cliff steepness, smoothness, volume fill (sRGB/P3/Rec.2020) | Is the gamut boundary well-behaved? |
| **Hue Properties** (4) | Hue RMS, primary L range, hue leaf constancy, hue agreement with CIE Lab | Are hue angles correct? |
| **Perceptual Uniformity** (4) | Munsell Value/Hue spacing, MacAdam ellipse isotropy, blue-white/red-white midpoints | Does equal distance mean equal perception? |
| **Application Scenarios** (8) | Palette L* spacing, tint/shade hue, data viz distinguishability, multi-stop gradient, WCAG contrast, harmony accuracy, photo gamut mapping, eased animation | Does it work in real design tasks? |
| **Accessibility** (2) | CVD simulation (protan/deutan) gradient dE | Is it usable for color blind users? |

### 2903 Gradient Pairs

- **sRGB** (1417): primaries, complementary, hue sweep, saturation, lightness, near-achromatic, dark/bright extremes, gamut boundary, 1000 random
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
  run.py                          # CLI runner (32 test functions + compare)
  core/
    spaces.py                     # 7 space implementations (single source of truth)
    pairs.py                      # 2903 gradient pair generator (deterministic)
    gpu_metrics.py                # 8 core metrics (round-trip, gradient, gamut, etc.)
    gpu_metrics_advanced.py       # 12 advanced metrics (CVD, animation, Jacobian, etc.)
    gpu_metrics_perceptual.py     # 12 perceptual/application metrics (Munsell, MacAdam, etc.)
    constants.py                  # Hardcoded data (Munsell, MacAdam, WCAG — zero file deps)
    comparison.py                 # 46 METRIC_DEFS + winner logic + head-to-head
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
