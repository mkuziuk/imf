# esc50_animals: classical ablations

## animal, fixed linsvm

Convergence: -0.31 to +0.16 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 2.
H sensitivity: -1.22 to +1.31 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.
Matched-count contrasts against EMD: +9.75 to +11.00 pp; corrected-t/Holm rejects 4 positive and 0 negative contrasts of 4.
IRMF contrasts against linear IMF: -1.00 to +0.37 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.

| method             | without log_scale BA/F1/Acc   | with log_scale BA/F1/Acc   |
|:-------------------|:------------------------------|:---------------------------|
| raw                | 44.34/41.25/43.99             | 45.92/43.33/45.55          |
| EMD                | 54.32/52.69/54.11             | 56.55/55.15/56.37          |
| linear sqrt2       | 65.27/64.19/65.18             | 67.44/66.58/67.67          |
| IRMF GD60 sqrt2    | 65.29/64.36/65.32             | 66.44/65.70/66.63          |
| IRMF H2 conv sqrt2 | 64.98/63.96/65.04             | 66.61/65.84/66.90          |
| IRMF H1 conv sqrt2 | 64.43/63.31/64.43             | 66.69/65.89/66.98          |
| IRMF H4 conv sqrt2 | 64.69/63.74/64.76             | 65.38/64.53/65.58          |
| IRMF H2 conv a2    | 65.32/64.36/65.54             | 66.30/65.38/66.63          |
| linear a2          | 64.95/64.04/65.07             | 66.72/65.88/67.03          |

Scores are percentages. Paired differences below are percentage points, 95% Nadeau-Bengio CI; corrected-t Holm p / secondary Wilcoxon Holm p.

| contrast             | without log_scale                           | with log_scale                             |
|:---------------------|:--------------------------------------------|:-------------------------------------------|
| phase2 IRMF-EMD      | +10.97 [+7.75, +14.19]; 8.96e-05 / 0.00159  | +9.89 [+6.44, +13.35]; 0.00046 / 0.00159   |
| phase2 linear-EMD    | +10.95 [+7.71, +14.19]; 9.22e-05 / 0.00159  | +10.89 [+8.22, +13.55]; 1.2e-05 / 0.00159  |
| phase2 IRMF-raw      | +20.95 [+14.36, +27.54]; 0.000176 / 0.00159 | +20.53 [+12.43, +28.62]; 0.00131 / 0.00159 |
| Q1 conv-GD60         | -0.31 [-2.55, +1.93]; 1 / 1                 | +0.16 [-1.09, +1.41]; 1 / 1                |
| Q2 H1-H2             | -0.55 [-4.21, +3.12]; 1 / 1                 | +0.08 [-1.28, +1.44]; 1 / 1                |
| Q2 H4-H2             | -0.28 [-3.01, +2.44]; 1 / 1                 | -1.22 [-3.87, +1.43]; 1 / 0.302            |
| Q2 H1-H4             | -0.26 [-5.40, +4.87]; 1 / 1                 | +1.31 [-1.72, +4.33]; 1 / 1                |
| Q3 IRMF a2-EMD       | +11.00 [+7.40, +14.60]; 0.000245 / 0.00159  | +9.75 [+5.94, +13.56]; 0.00126 / 0.00159   |
| Q3 linear a2-EMD     | +10.63 [+7.54, +13.72]; 8.33e-05 / 0.00159  | +10.17 [+7.37, +12.98]; 4.78e-05 / 0.00159 |
| Q4 GD60-linear sqrt2 | +0.02 [-3.21, +3.24]; 1 / 1                 | -1.00 [-3.66, +1.67]; 1 / 1                |
| Q4 conv-linear sqrt2 | -0.29 [-3.99, +3.41]; 1 / 1                 | -0.83 [-3.31, +1.65]; 1 / 1                |
| Q4 conv-linear a2    | +0.37 [-1.78, +2.51]; 1 / 1                 | -0.42 [-2.83, +1.98]; 1 / 1                |
| conv sqrt2-EMD       | +10.66 [+7.21, +14.10]; 0.000223 / 0.00159  | +10.06 [+6.54, +13.57]; 0.00046 / 0.00159  |

Holm covers all 26 cells above, including both feature sets. The all-classifier table corrects 78 cells per metric. CIs are marginal, not multiplicity-adjusted. A nonsignificant result does not establish equivalence.

[animal macro-F1, accuracy and all classifiers](paired_all_metrics_animal.csv). [Fixed-classifier tests for all three metrics](paired_fixed_classifier_animal.csv). [Means and fold SDs](metric_summary_animal.csv).

Runtime 47.1 min; 450 new fits and 360 reused phase-2 fits, four single-thread workers.

116 new fits recorded warnings; [messages](fit_warnings_animal.csv) are retained rather than suppressed. These counts include inner-grid candidates when present.

## bird, fixed logreg

Convergence: -0.61 to -0.21 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 2.
H sensitivity: -0.07 to +0.40 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.
Matched-count contrasts against EMD: +3.89 to +5.71 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 4.
IRMF contrasts against linear IMF: -2.43 to -0.86 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.

| method             | without log_scale BA/F1/Acc   | with log_scale BA/F1/Acc   |
|:-------------------|:------------------------------|:---------------------------|
| raw                | 65.69/66.80/75.02             | 65.03/66.04/74.33          |
| EMD                | 67.22/68.02/74.80             | 66.35/67.04/73.93          |
| linear sqrt2       | 73.07/73.44/78.75             | 71.98/72.18/77.94          |
| IRMF GD60 sqrt2    | 70.85/71.11/76.92             | 70.37/70.60/76.33          |
| IRMF H2 conv sqrt2 | 70.64/70.89/76.61             | 69.76/69.95/75.96          |
| IRMF H1 conv sqrt2 | 70.97/71.28/76.67             | 70.00/70.09/75.86          |
| IRMF H4 conv sqrt2 | 71.04/71.39/77.04             | 69.97/70.15/76.21          |
| IRMF H2 conv a2    | 71.22/71.46/77.09             | 70.24/70.43/76.30          |
| linear a2          | 72.08/72.41/78.23             | 72.06/72.33/77.95          |

Scores are percentages. Paired differences below are percentage points, 95% Nadeau-Bengio CI; corrected-t Holm p / secondary Wilcoxon Holm p.

| contrast             | without log_scale                     | with log_scale                        |
|:---------------------|:--------------------------------------|:--------------------------------------|
| phase2 IRMF-EMD      | +3.63 [-0.32, +7.57]; 1 / 0.022       | +4.02 [+1.32, +6.73]; 0.164 / 0.00159 |
| phase2 linear-EMD    | +5.85 [+1.08, +10.62]; 0.453 / 0.0188 | +5.64 [+1.52, +9.75]; 0.26 / 0.00732  |
| phase2 IRMF-raw      | +5.16 [-1.44, +11.75]; 1 / 0.0752     | +5.34 [-1.14, +11.82]; 1 / 0.0641     |
| Q1 conv-GD60         | -0.21 [-2.08, +1.66]; 1 / 1           | -0.61 [-2.21, +0.98]; 1 / 0.851       |
| Q2 H1-H2             | +0.33 [-2.38, +3.04]; 1 / 1           | +0.25 [-1.41, +1.90]; 1 / 1           |
| Q2 H4-H2             | +0.40 [-1.20, +1.99]; 1 / 1           | +0.21 [-1.43, +1.86]; 1 / 1           |
| Q2 H1-H4             | -0.07 [-2.02, +1.88]; 1 / 1           | +0.03 [-1.81, +1.87]; 1 / 1           |
| Q3 IRMF a2-EMD       | +4.00 [+0.06, +7.94]; 0.991 / 0.0188  | +3.89 [+0.51, +7.27]; 0.593 / 0.00732 |
| Q3 linear a2-EMD     | +4.86 [-0.51, +10.22]; 1 / 0.042      | +5.71 [+1.95, +9.47]; 0.149 / 0.00159 |
| Q4 GD60-linear sqrt2 | -2.23 [-4.63, +0.18]; 1 / 0.0259      | -1.61 [-5.00, +1.78]; 1 / 0.413       |
| Q4 conv-linear sqrt2 | -2.43 [-6.10, +1.23]; 1 / 0.109       | -2.23 [-5.96, +1.51]; 1 / 0.181       |
| Q4 conv-linear a2    | -0.86 [-5.56, +3.85]; 1 / 1           | -1.81 [-4.66, +1.04]; 1 / 0.281       |
| conv sqrt2-EMD       | +3.42 [-0.43, +7.26]; 1 / 0.022       | +3.41 [+0.00, +6.82]; 1 / 0.0188      |

Holm covers all 26 cells above, including both feature sets. The all-classifier table corrects 78 cells per metric. CIs are marginal, not multiplicity-adjusted. A nonsignificant result does not establish equivalence.

[bird macro-F1, accuracy and all classifiers](paired_all_metrics_bird.csv). [Fixed-classifier tests for all three metrics](paired_fixed_classifier_bird.csv). [Means and fold SDs](metric_summary_bird.csv).

Runtime 9.9 min; 450 new fits and 360 reused phase-2 fits, four single-thread workers.

9 new fits recorded warnings; [messages](fit_warnings_bird.csv) are retained rather than suppressed. These counts include inner-grid candidates when present.

## Reproducibility and limits

LDA, logistic regression and linear SVM retain their phase-2 estimators, grids, grouped tuning where used and all 15 saved splits. No baseline was refitted. The reporting classifier maximizes phase-2 mean BA across the four main methods and both feature sets, with alphabetical tie breaking, and stays fixed across all contrasts. This outer-score selection makes the inference exploratory. [The original six-classifier phase-2 report](../classical/REPORT.md) retains its original comparison families and verdicts.

| tag                |   slots |   cpu_seconds_per_frame |
|:-------------------|--------:|------------------------:|
| irmf_h2_a1.41_conv |      18 |                 0.06440 |
| irmf_h1_a1.41_conv |      18 |                 0.07128 |
| irmf_h4_a1.41_conv |      18 |                 0.05835 |
| irmf_h2_a2_conv    |      10 |                 0.03726 |
| imf_linear_a2      |      10 |                 0.00410 |

Slot counts include residuals and EMD's zero padding. EMD has 9 slots; a2 has 9 slots on UAS and 10 on the L=2048 datasets. Thus a2 is a near-matched, not identical, component count. No dimension-matched feature selection was performed. A2 also changes the bandwidth schedule, so this is not a pure feature-count intervention.

All converged-tag channel-stages passed their stopping rule; all selected feature tables are finite, aligned and complete. Shared log_scale columns match raw exactly. The Newton tags solve pimf's lookup score to root tolerance 1e-12, with a 100-iteration cap; they preserve normalization, kernel, boundary and schedule. Reuse checks recompute BA/F1/accuracy from saved OOF predictions, match every held-out frame and label to the saved splits, and check fold sizes, seeds and IDs. Input hashes, estimator signatures, grids and readiness checks are in audit_*.json.

The optional GD60 a2 tag requires completed decomposition and feature caches and explicit inclusion within the budget; its readiness/skip reason is recorded in the audit. Known phase-2 driver limitations are missing native OOF probabilities and prediction filenames shared across batches. The isolated wrapper records probabilities when supported and writes only phase2b checkpoints. Linear SVM has no native probability output. The new wrapper's early object-dtype y_true serialization was fixed; a one-time repair reconstructs Unicode labels from verified frame IDs and prepared metadata, requiring every stored metric to reproduce. [Repair audit](../../uas_birds_60ghz/classical_ablations/checkpoint_label_repair.json). No new shared numerical bug was found.

```bash
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/baselines/ablations/run_ablations.py --dataset esc50_animals --task animal
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/baselines/ablations/run_ablations.py --dataset esc50_animals --task bird --allow-secondary
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/baselines/ablations/analyse_ablations.py
```

Code changes are confined to baselines/ablations/. Large matrices, predictions and per-fit caches remain under DATA_ROOT/<dataset>/results/phase2b/. No commit was made.
