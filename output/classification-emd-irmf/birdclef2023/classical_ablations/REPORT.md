# birdclef2023: classical ablations

## species, fixed logreg

Convergence: -0.12 to +0.07 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 2.
H sensitivity: -0.19 to +0.36 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.
Matched-count contrasts against EMD: +3.81 to +4.43 pp; corrected-t/Holm rejects 4 positive and 0 negative contrasts of 4.
IRMF contrasts against linear IMF: +0.38 to +0.82 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.

| method             | without log_scale BA/F1/Acc   | with log_scale BA/F1/Acc   |
|:-------------------|:------------------------------|:---------------------------|
| raw                | 37.61/36.88/37.61             | 37.57/36.78/37.56          |
| EMD                | 43.45/43.18/43.45             | 43.34/43.09/43.34          |
| linear sqrt2       | 48.86/48.61/48.86             | 48.57/48.35/48.57          |
| IRMF GD60 sqrt2    | 49.50/49.26/49.50             | 49.39/49.16/49.38          |
| IRMF H2 conv sqrt2 | 49.57/49.29/49.57             | 49.27/49.00/49.27          |
| IRMF H1 conv sqrt2 | 49.45/49.21/49.44             | 49.44/49.21/49.43          |
| IRMF H4 conv sqrt2 | 49.53/49.25/49.52             | 49.08/48.86/49.08          |
| IRMF H2 conv a2    | 47.88/47.65/47.88             | 47.53/47.34/47.53          |
| linear a2          | 47.34/47.13/47.34             | 47.15/46.94/47.15          |

Scores are percentages. Paired differences below are percentage points, 95% Nadeau-Bengio CI; corrected-t Holm p / secondary Wilcoxon Holm p.

| contrast             | without log_scale                          | with log_scale                            |
|:---------------------|:-------------------------------------------|:------------------------------------------|
| phase2 IRMF-EMD      | +6.05 [+4.72, +7.39]; 2.74e-06 / 0.00159   | +6.05 [+4.49, +7.60]; 1.58e-05 / 0.00159  |
| phase2 linear-EMD    | +5.41 [+4.24, +6.57]; 2.25e-06 / 0.00159   | +5.23 [+4.02, +6.44]; 4.63e-06 / 0.00159  |
| phase2 IRMF-raw      | +11.89 [+9.84, +13.94]; 1.39e-07 / 0.00159 | +11.82 [+9.98, +13.66]; 3.9e-08 / 0.00159 |
| Q1 conv-GD60         | +0.07 [-0.72, +0.85]; 1 / 0.851            | -0.12 [-0.95, +0.72]; 1 / 1               |
| Q2 H1-H2             | -0.12 [-1.29, +1.05]; 1 / 1                | +0.17 [-1.16, +1.50]; 1 / 1               |
| Q2 H4-H2             | -0.04 [-1.10, +1.02]; 1 / 1                | -0.19 [-1.11, +0.73]; 1 / 1               |
| Q2 H1-H4             | -0.07 [-1.75, +1.60]; 1 / 1                | +0.36 [-1.07, +1.80]; 1 / 1               |
| Q3 IRMF a2-EMD       | +4.43 [+2.96, +5.91]; 0.000263 / 0.00159   | +4.20 [+2.43, +5.96]; 0.00239 / 0.00159   |
| Q3 linear a2-EMD     | +3.89 [+3.32, +4.46]; 1.77e-08 / 0.00159   | +3.81 [+2.53, +5.09]; 0.000267 / 0.00159  |
| Q4 GD60-linear sqrt2 | +0.65 [-0.60, +1.89]; 1 / 0.281            | +0.82 [-0.41, +2.05]; 1 / 0.117           |
| Q4 conv-linear sqrt2 | +0.71 [-0.46, +1.88]; 1 / 0.259            | +0.70 [-0.41, +1.81]; 1 / 0.133           |
| Q4 conv-linear a2    | +0.54 [-0.68, +1.76]; 1 / 0.73             | +0.38 [-0.83, +1.60]; 1 / 1               |
| conv sqrt2-EMD       | +6.12 [+4.84, +7.39]; 1.51e-06 / 0.00159   | +5.93 [+4.32, +7.55]; 2.98e-05 / 0.00159  |

Holm covers all 26 cells above, including both feature sets. The all-classifier table corrects 78 cells per metric. CIs are marginal, not multiplicity-adjusted. A nonsignificant result does not establish equivalence.

[species macro-F1, accuracy and all classifiers](paired_all_metrics_species.csv). [Fixed-classifier tests for all three metrics](paired_fixed_classifier_species.csv). [Means and fold SDs](metric_summary_species.csv).

Runtime 27.5 min; 450 new fits and 360 reused phase-2 fits, four single-thread workers.

0 new fits recorded warnings; [messages](fit_warnings_species.csv) are retained rather than suppressed. These counts include inner-grid candidates when present.

The run log records 1 worker-replacement warning(s). Final coverage and unique checkpoint counts are checked separately; no missing fit is silently dropped.

## passerine, fixed logreg

Convergence: -0.19 to -0.15 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 2.
H sensitivity: -0.59 to +0.15 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.
Matched-count contrasts against EMD: +1.52 to +1.60 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 4.
IRMF contrasts against linear IMF: -0.57 to +0.08 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.

| method             | without log_scale BA/F1/Acc   | with log_scale BA/F1/Acc   |
|:-------------------|:------------------------------|:---------------------------|
| raw                | 70.61/71.23/74.37             | 70.54/71.15/74.29          |
| EMD                | 75.25/75.45/77.20             | 75.17/75.36/77.09          |
| linear sqrt2       | 77.98/78.22/79.79             | 77.83/78.08/79.65          |
| IRMF GD60 sqrt2    | 77.56/77.80/79.40             | 77.56/77.79/79.39          |
| IRMF H2 conv sqrt2 | 77.41/77.67/79.30             | 77.37/77.64/79.27          |
| IRMF H1 conv sqrt2 | 76.97/77.29/79.00             | 76.96/77.28/78.99          |
| IRMF H4 conv sqrt2 | 77.56/77.79/79.38             | 77.44/77.65/79.24          |
| IRMF H2 conv a2    | 76.79/77.09/78.80             | 76.77/77.06/78.78          |
| linear a2          | 76.82/76.95/78.53             | 76.69/76.82/78.40          |

Scores are percentages. Paired differences below are percentage points, 95% Nadeau-Bengio CI; corrected-t Holm p / secondary Wilcoxon Holm p.

| contrast             | without log_scale                        | with log_scale                           |
|:---------------------|:-----------------------------------------|:-----------------------------------------|
| phase2 IRMF-EMD      | +2.31 [+0.88, +3.74]; 0.0783 / 0.00232   | +2.39 [+1.32, +3.45]; 0.00662 / 0.00159  |
| phase2 linear-EMD    | +2.73 [+1.36, +4.09]; 0.0166 / 0.00159   | +2.66 [+1.34, +3.98]; 0.0157 / 0.00159   |
| phase2 IRMF-raw      | +6.95 [+4.55, +9.35]; 0.000566 / 0.00159 | +7.02 [+4.68, +9.37]; 0.000417 / 0.00159 |
| Q1 conv-GD60         | -0.15 [-0.71, +0.41]; 1 / 1              | -0.19 [-0.95, +0.57]; 1 / 1              |
| Q2 H1-H2             | -0.44 [-1.29, +0.42]; 1 / 0.536          | -0.41 [-1.44, +0.63]; 1 / 1              |
| Q2 H4-H2             | +0.15 [-0.70, +1.00]; 1 / 1              | +0.07 [-0.72, +0.86]; 1 / 1              |
| Q2 H1-H4             | -0.59 [-1.75, +0.58]; 1 / 0.414          | -0.48 [-1.73, +0.78]; 1 / 1              |
| Q3 IRMF a2-EMD       | +1.55 [+0.45, +2.64]; 0.173 / 0.00232    | +1.60 [+0.48, +2.73]; 0.173 / 0.00159    |
| Q3 linear a2-EMD     | +1.57 [+0.19, +2.94]; 0.45 / 0.00159     | +1.52 [+0.18, +2.86]; 0.45 / 0.00391     |
| Q4 GD60-linear sqrt2 | -0.42 [-1.86, +1.02]; 1 / 1              | -0.27 [-1.60, +1.06]; 1 / 1              |
| Q4 conv-linear sqrt2 | -0.57 [-1.84, +0.70]; 1 / 0.765          | -0.46 [-1.81, +0.89]; 1 / 1              |
| Q4 conv-linear a2    | -0.02 [-0.98, +0.94]; 1 / 1              | +0.08 [-0.85, +1.01]; 1 / 1              |
| conv sqrt2-EMD       | +2.16 [+0.58, +3.74]; 0.184 / 0.00916    | +2.20 [+0.62, +3.77]; 0.174 / 0.00311    |

Holm covers all 26 cells above, including both feature sets. The all-classifier table corrects 78 cells per metric. CIs are marginal, not multiplicity-adjusted. A nonsignificant result does not establish equivalence.

[passerine macro-F1, accuracy and all classifiers](paired_all_metrics_passerine.csv). [Fixed-classifier tests for all three metrics](paired_fixed_classifier_passerine.csv). [Means and fold SDs](metric_summary_passerine.csv).

Runtime 5.8 min; 450 new fits and 360 reused phase-2 fits, four single-thread workers.

0 new fits recorded warnings; [messages](fit_warnings_passerine.csv) are retained rather than suppressed. These counts include inner-grid candidates when present.

The run log records 1 worker-replacement warning(s). Final coverage and unique checkpoint counts are checked separately; no missing fit is silently dropped.

## Reproducibility and limits

LDA, logistic regression and linear SVM retain their phase-2 estimators, grids, grouped tuning where used and all 15 saved splits. No baseline was refitted. The reporting classifier maximizes phase-2 mean BA across the four main methods and both feature sets, with alphabetical tie breaking, and stays fixed across all contrasts. This outer-score selection makes the inference exploratory. [The original six-classifier phase-2 report](../classical/REPORT.md) retains its original comparison families and verdicts.

| tag                |   slots |   cpu_seconds_per_frame |
|:-------------------|--------:|------------------------:|
| irmf_h2_a1.41_conv |      18 |                 0.05792 |
| irmf_h1_a1.41_conv |      18 |                 0.06230 |
| irmf_h4_a1.41_conv |      18 |                 0.05422 |
| irmf_h2_a2_conv    |      10 |                 0.03355 |
| imf_linear_a2      |      10 |                 0.00427 |

Slot counts include residuals and EMD's zero padding. EMD has 9 slots; a2 has 9 slots on UAS and 10 on the L=2048 datasets. Thus a2 is a near-matched, not identical, component count. No dimension-matched feature selection was performed. A2 also changes the bandwidth schedule, so this is not a pure feature-count intervention.

All converged-tag channel-stages passed their stopping rule; all selected feature tables are finite, aligned and complete. Shared log_scale columns match raw exactly. The Newton tags solve pimf's lookup score to root tolerance 1e-12, with a 100-iteration cap; they preserve normalization, kernel, boundary and schedule. Reuse checks recompute BA/F1/accuracy from saved OOF predictions, match every held-out frame and label to the saved splits, and check fold sizes, seeds and IDs. Input hashes, estimator signatures, grids and readiness checks are in audit_*.json.

The optional GD60 a2 tag requires completed decomposition and feature caches and explicit inclusion within the budget; its readiness/skip reason is recorded in the audit. Known phase-2 driver limitations are missing native OOF probabilities and prediction filenames shared across batches. The isolated wrapper records probabilities when supported and writes only phase2b checkpoints. Linear SVM has no native probability output. The new wrapper's early object-dtype y_true serialization was fixed; a one-time repair reconstructs Unicode labels from verified frame IDs and prepared metadata, requiring every stored metric to reproduce. [Repair audit](../../uas_birds_60ghz/classical_ablations/checkpoint_label_repair.json). No new shared numerical bug was found.

```bash
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/baselines/ablations/run_ablations.py --dataset birdclef2023 --task species
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/baselines/ablations/run_ablations.py --dataset birdclef2023 --task passerine --allow-secondary
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/baselines/ablations/analyse_ablations.py
```

Code changes are confined to baselines/ablations/. Large matrices, predictions and per-fit caches remain under DATA_ROOT/<dataset>/results/phase2b/. No commit was made.
