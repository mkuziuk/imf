# uas_birds_60ghz: classical ablations

## bird_drone, fixed lda

Convergence: -0.17 to -0.16 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 2.
H sensitivity: -0.51 to +0.55 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.
Matched-count contrasts against EMD: -0.22 to +0.80 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 4.
IRMF contrasts against linear IMF: +0.31 to +0.91 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.

| method             | without log_scale BA/F1/Acc   | with log_scale BA/F1/Acc   |
|:-------------------|:------------------------------|:---------------------------|
| raw                | 95.28/95.32/95.50             | 97.41/97.42/97.50          |
| EMD                | 97.57/97.64/97.73             | 98.39/98.45/98.50          |
| linear sqrt2       | 98.48/98.55/98.60             | 98.71/98.76/98.81          |
| IRMF GD60 sqrt2    | 99.32/99.34/99.37             | 99.62/99.61/99.62          |
| IRMF H2 conv sqrt2 | 99.14/99.16/99.19             | 99.46/99.47/99.49          |
| IRMF H1 conv sqrt2 | 99.17/99.26/99.29             | 99.55/99.56/99.58          |
| IRMF H4 conv sqrt2 | 98.63/98.64/98.69             | 99.00/99.04/99.07          |
| IRMF H2 conv a2    | 98.36/98.50/98.56             | 98.82/98.94/98.99          |
| linear a2          | 98.05/98.15/98.23             | 98.17/98.28/98.35          |

Scores are percentages. Paired differences below are percentage points, 95% Nadeau-Bengio CI; corrected-t Holm p / secondary Wilcoxon Holm p.

| contrast             | without log_scale                | with log_scale                   |
|:---------------------|:---------------------------------|:---------------------------------|
| phase2 IRMF-EMD      | +1.75 [-0.38, +3.88]; 1 / 0.0846 | +1.23 [-0.37, +2.82]; 1 / 0.135  |
| phase2 linear-EMD    | +0.92 [-0.73, +2.56]; 1 / 0.283  | +0.32 [-1.00, +1.63]; 1 / 1      |
| phase2 IRMF-raw      | +4.03 [-0.30, +8.37]; 1 / 0.0462 | +2.20 [-0.62, +5.03]; 1 / 0.0732 |
| Q1 conv-GD60         | -0.17 [-0.60, +0.25]; 1 / 1      | -0.16 [-0.61, +0.30]; 1 / 1      |
| Q2 H1-H2             | +0.02 [-0.88, +0.93]; 1 / 1      | +0.09 [-0.54, +0.71]; 1 / 1      |
| Q2 H4-H2             | -0.51 [-1.65, +0.63]; 1 / 1      | -0.46 [-1.65, +0.73]; 1 / 1      |
| Q2 H1-H4             | +0.53 [-0.47, +1.54]; 1 / 0.531  | +0.55 [-0.41, +1.50]; 1 / 0.398  |
| Q3 IRMF a2-EMD       | +0.80 [-0.90, +2.49]; 1 / 0.656  | +0.43 [-1.02, +1.88]; 1 / 1      |
| Q3 linear a2-EMD     | +0.48 [-1.67, +2.63]; 1 / 1      | -0.22 [-1.76, +1.32]; 1 / 1      |
| Q4 GD60-linear sqrt2 | +0.83 [-0.63, +2.30]; 1 / 0.352  | +0.91 [-0.52, +2.34]; 1 / 0.205  |
| Q4 conv-linear sqrt2 | +0.66 [-0.85, +2.17]; 1 / 0.656  | +0.75 [-0.79, +2.30]; 1 / 0.656  |
| Q4 conv-linear a2    | +0.31 [-1.23, +1.86]; 1 / 1      | +0.65 [-0.55, +1.85]; 1 / 0.557  |
| conv sqrt2-EMD       | +1.58 [-0.65, +3.80]; 1 / 0.191  | +1.07 [-0.70, +2.84]; 1 / 0.254  |

Holm covers all 26 cells above, including both feature sets. The all-classifier table corrects 78 cells per metric. CIs are marginal, not multiplicity-adjusted. A nonsignificant result does not establish equivalence.

[bird_drone macro-F1, accuracy and all classifiers](paired_all_metrics_bird_drone.csv). [Fixed-classifier tests for all three metrics](paired_fixed_classifier_bird_drone.csv). [Means and fold SDs](metric_summary_bird_drone.csv).

Runtime 0.4 min; 450 new fits and 360 reused phase-2 fits, four single-thread workers.

0 new fits recorded warnings; [messages](fit_warnings_bird_drone.csv) are retained rather than suppressed.

## target3, fixed lda

Convergence: -0.24 to -0.17 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 2.
H sensitivity: -1.77 to +1.74 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.
Matched-count contrasts against EMD: +0.49 to +1.63 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 4.
IRMF contrasts against linear IMF: +1.08 to +1.70 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.

| method             | without log_scale BA/F1/Acc   | with log_scale BA/F1/Acc   |
|:-------------------|:------------------------------|:---------------------------|
| raw                | 91.01/88.65/90.18             | 92.88/90.65/91.89          |
| EMD                | 94.93/94.17/95.33             | 95.14/94.52/95.64          |
| linear sqrt2       | 95.87/95.21/96.15             | 96.35/95.87/96.75          |
| IRMF GD60 sqrt2    | 97.57/97.19/97.73             | 97.85/97.48/97.95          |
| IRMF H2 conv sqrt2 | 97.40/96.95/97.56             | 97.60/97.07/97.69          |
| IRMF H1 conv sqrt2 | 97.36/96.88/97.37             | 97.47/97.03/97.50          |
| IRMF H4 conv sqrt2 | 95.62/95.27/96.28             | 95.87/95.47/96.46          |
| IRMF H2 conv a2    | 96.56/95.98/96.70             | 96.74/96.15/96.87          |
| linear a2          | 95.49/94.86/95.75             | 95.62/94.93/95.87          |

Scores are percentages. Paired differences below are percentage points, 95% Nadeau-Bengio CI; corrected-t Holm p / secondary Wilcoxon Holm p.

| contrast             | without log_scale                     | with log_scale                        |
|:---------------------|:--------------------------------------|:--------------------------------------|
| phase2 IRMF-EMD      | +2.64 [-0.37, +5.65]; 1 / 0.0509      | +2.71 [-0.37, +5.79]; 1 / 0.041       |
| phase2 linear-EMD    | +0.94 [-3.37, +5.25]; 1 / 1           | +1.22 [-3.36, +5.79]; 1 / 1           |
| phase2 IRMF-raw      | +6.56 [+2.41, +10.72]; 0.115 / 0.0163 | +4.97 [+0.30, +9.63]; 0.967 / 0.00635 |
| Q1 conv-GD60         | -0.17 [-0.69, +0.34]; 1 / 1           | -0.24 [-0.82, +0.33]; 1 / 0.713       |
| Q2 H1-H2             | -0.03 [-1.53, +1.46]; 1 / 1           | -0.14 [-1.29, +1.01]; 1 / 1           |
| Q2 H4-H2             | -1.77 [-5.06, +1.52]; 1 / 0.0234      | -1.74 [-4.87, +1.40]; 1 / 0.0234      |
| Q2 H1-H4             | +1.74 [-2.30, +5.77]; 1 / 0.125       | +1.60 [-2.27, +5.46]; 1 / 0.105       |
| Q3 IRMF a2-EMD       | +1.63 [-2.25, +5.51]; 1 / 0.713       | +1.60 [-2.03, +5.22]; 1 / 0.713       |
| Q3 linear a2-EMD     | +0.56 [-3.46, +4.57]; 1 / 1           | +0.49 [-3.39, +4.37]; 1 / 1           |
| Q4 GD60-linear sqrt2 | +1.70 [-1.15, +4.55]; 1 / 0.105       | +1.49 [-1.23, +4.21]; 1 / 0.125       |
| Q4 conv-linear sqrt2 | +1.53 [-1.45, +4.51]; 1 / 0.152       | +1.25 [-1.48, +3.98]; 1 / 0.713       |
| Q4 conv-linear a2    | +1.08 [-0.86, +3.01]; 1 / 0.125       | +1.11 [-0.89, +3.11]; 1 / 0.223       |
| conv sqrt2-EMD       | +2.47 [-0.62, +5.55]; 1 / 0.0659      | +2.47 [-0.53, +5.46]; 1 / 0.0322      |

Holm covers all 26 cells above, including both feature sets. The all-classifier table corrects 78 cells per metric. CIs are marginal, not multiplicity-adjusted. A nonsignificant result does not establish equivalence.

[target3 macro-F1, accuracy and all classifiers](paired_all_metrics_target3.csv). [Fixed-classifier tests for all three metrics](paired_fixed_classifier_target3.csv). [Means and fold SDs](metric_summary_target3.csv).

Runtime 0.8 min; 450 new fits and 360 reused phase-2 fits, four single-thread workers.

0 new fits recorded warnings; [messages](fit_warnings_target3.csv) are retained rather than suppressed.

## Reproducibility and limits

LDA, logistic regression and linear SVM retain their phase-2 estimators, grids, nested grouped tuning and all 15 saved splits. No baseline was refitted. The reporting classifier maximizes phase-2 mean BA across the four main methods and both feature sets, with alphabetical tie breaking, and stays fixed across all contrasts. This outer-score selection makes the inference exploratory. [The original six-classifier phase-2 report](../classical/REPORT.md) retains its original comparison families and verdicts.

| tag                |   slots |   cpu_seconds_per_frame |
|:-------------------|--------:|------------------------:|
| irmf_h2_a1.41_conv |      16 |                 0.03141 |
| irmf_h1_a1.41_conv |      16 |                 0.03393 |
| irmf_h4_a1.41_conv |      16 |                 0.02833 |
| irmf_h2_a2_conv    |       9 |                 0.01811 |
| imf_linear_a2      |       9 |                 0.00198 |

Slot counts include residuals and EMD zero padding. EMD has 9 slots; a2 has 9 slots on UAS and 10 on the L=2048 datasets. Thus a2 is a near-matched, not identical, component count. No dimension-matched feature selection was performed. A2 also changes the bandwidth schedule, so this is not a pure feature-count intervention.

All converged-tag channel-stages passed their stopping rule; all selected feature tables are finite, aligned and complete. Shared log_scale columns match raw exactly. The Newton tags solve pimf's lookup score to root tolerance 1e-12, with a 100-iteration cap; they preserve normalization, kernel, boundary and schedule. Reuse checks recompute BA/F1/accuracy from saved OOF predictions, match every held-out frame and label to the saved splits, and check fold sizes, seeds and IDs. Input hashes, estimator signatures, grids and readiness checks are in audit_*.json.

The optional GD60 a2 tag requires completed decomposition and feature caches and explicit inclusion within the budget; its readiness/skip reason is recorded in the audit. Known phase-2 driver limitations are missing native OOF probabilities and prediction filenames shared across batches. The isolated wrapper records probabilities when supported and writes only phase2b checkpoints. Linear SVM has no native probability output. The new wrapper's early object-dtype y_true serialization was fixed; a one-time repair reconstructs Unicode labels from verified frame IDs and prepared metadata, requiring every stored metric to reproduce. [Repair audit](../../uas_birds_60ghz/classical_ablations/checkpoint_label_repair.json). No new shared numerical bug was found.

```bash
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/baselines/ablations/run_ablations.py --dataset uas_birds_60ghz --task bird_drone
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/baselines/ablations/run_ablations.py --dataset uas_birds_60ghz --task target3 --allow-secondary
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/baselines/ablations/analyse_ablations.py
```

Code changes are confined to baselines/ablations/. Large matrices, predictions and per-fit caches remain under DATA_ROOT/<dataset>/results/phase2b/. No commit was made.
