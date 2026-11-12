# Supplemental LightGBM checks

All methods use the same 150-round cap, grouped validation, class weights,
imputation and 15 shared outer folds. References reuse identical main fits.
Scores include balanced accuracy, macro-F1 and accuracy. Paired tables use
Nadeau-Bengio corrected tests and intervals, secondary Wilcoxon, and Holm
within each task/metric table. Intervals are individual, not simultaneous.

| Check | Predictors without/with log scale | Animal | Bird | Holm tests/metric |
|---|---|---|---|---:|
| H=1, H=4 and a=2, converged IRMF | 360/361 or 200/201 | [Scores](sensitivity/TABLE_animal.md), [pairs](sensitivity/PAIRS_animal.md) | [Scores](sensitivity/TABLE_bird.md), [pairs](sensitivity/PAIRS_bird.md) | 56 |
| a=2, ten slots, GD60 IRMF and linear IMF | 200/201 | [Scores](matched/TABLE_animal.md), [pairs](matched/PAIRS_animal.md) | [Scores](matched/TABLE_bird.md), [pairs](matched/PAIRS_bird.md) | 56 |
| Nine evenly spaced scale slots | 180/181 | [Scores](slots9/TABLE_animal.md), [pairs](slots9/PAIRS_animal.md) | [Scores](slots9/TABLE_bird.md), [pairs](slots9/PAIRS_bird.md) | 20 |
| EMD+IRMF feature concatenation | 540/541 | [Scores](concat/TABLE_animal.md), [pairs](concat/PAIRS_animal.md) | [Scores](concat/TABLE_bird.md), [pairs](concat/PAIRS_bird.md) | 42 |

The H table includes all five main references and the three extra converged
tags. The matched table also includes converged a=2. The nine-slot references
are unchanged raw and EMD inputs. Fusion uses default or converged H=2 IRMF,
includes shared log scale once, and retains all five main references.

[Validation](sensitivity_validation.json), [clip scores](supplemental_clip_scores.csv),
[training cost](supplemental_costs.csv), [source decomposition costs](all_decomposition_costs.csv),
and [exact stage commands](supplemental_manifest.json) are saved separately.
