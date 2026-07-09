# Human-predictivity of the engineering metrics (2026-07-09)

Method: `research/metric_predictivity.py`. Across N=9 built-in literature spaces
(OKLab, CIELab, IPT, JzAzBz, ICtCp, CAM16-UCS, DIN99d, Perceptia, Engineered),
Spearman-correlate each engineering compare-mode metric against the human_pool
property it claims to measure. No new observers — the 46 human datasets are the
ground truth. rho>0 = the engineering metric ranks spaces the way the human
data does (both lower = better).

| engineering metric              | human property  | rho   | verdict |
|---------------------------------|-----------------|-------|---------|
| gradients.cv_high_chroma        | discrimination  | +0.62 | STRONG human-predictive |
| banding.total_duplicate_pct     | spacing         | +0.57 | weak |
| gradients.cv_mean               | spacing         | +0.42 | weak |
| palette_uniformity.mean_cv      | spacing         | +0.40 | weak |
| tint_shade_hue.mean_max_drift   | hue             | **-0.92** | **ANTI-correlated** |

## Headline finding — empirical vindication of the CIELab-reference tier

`tint_shade_hue` (a TIER_CIELAB metric — it judges hue drift against CIELab as
truth) ANTI-correlates with the human hue verdict at rho = -0.92. A space that
"wins" on that engineering metric tends to LOSE on real human constant-hue data.
This is exactly the failure the tier system predicts: CIELab-reference metrics
penalise hue-correcting spaces. The fair verdict already drops these (weight 0);
this study shows empirically WHY that was right — the metric ranks spaces
opposite to humans.

## Honest reading of the rest

- The chromatic-gradient-CV metric IS human-predictive for discrimination
  (rho 0.62): a real, validated engineering proxy.
- The spacing engineering metrics (gradient CV, banding, palette CV) are only
  WEAKLY human-predictive (0.4-0.57). They are noisy proxies for human spacing
  — which is precisely why the human_pool carries direct spacing judges
  (OSA-UCS, MacAdam) rather than trusting the engineering CV alone.
- hue_leaf: insufficient data (some spaces returned no value).

## Caveat

N=9 is small; treat rho as directional, not precise. The tint_shade_hue result
is strong enough (|rho|=0.92) to act on; the weak positives (0.4-0.6) say
"loosely tracks, don't trust alone" — consistent with the whole design of
routing perceptual questions to the human_pool.
