# dronerf: classical ablations

## 10class, fixed linsvm

Convergence: +0.73 to +0.83 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 2.
H sensitivity: -3.84 to +1.90 pp; corrected-t/Holm rejects 1 positive and 4 negative contrasts of 6.
Matched-count contrasts against EMD: +9.66 to +16.42 pp; corrected-t/Holm rejects 4 positive and 0 negative contrasts of 4.
IRMF contrasts against linear IMF: -7.94 to -3.92 pp; corrected-t/Holm rejects 0 positive and 6 negative contrasts of 6.

| method             | without log_scale BA/F1/Acc   | with log_scale BA/F1/Acc   |
|:-------------------|:------------------------------|:---------------------------|
| raw                | 34.95/30.84/34.85             | 39.05/35.77/38.94          |
| EMD                | 38.73/36.94/38.67             | 42.92/41.47/42.83          |
| linear sqrt2       | 57.47/56.76/57.48             | 59.49/58.77/59.48          |
| IRMF GD60 sqrt2    | 49.53/48.39/49.50             | 53.58/52.62/53.56          |
| IRMF H2 conv sqrt2 | 50.36/49.29/50.36             | 54.31/53.41/54.30          |
| IRMF H1 conv sqrt2 | 48.41/47.36/48.41             | 52.44/51.59/52.41          |
| IRMF H4 conv sqrt2 | 52.26/51.30/52.28             | 55.37/54.49/55.35          |
| IRMF H2 conv a2    | 48.39/47.00/48.38             | 52.82/51.73/52.80          |
| linear a2          | 55.15/54.14/55.16             | 56.74/55.80/56.75          |

Scores are percentages. Paired differences below are percentage points, 95% Nadeau-Bengio CI; corrected-t Holm p / secondary Wilcoxon Holm p.

| contrast             | without log_scale                           | with log_scale                              |
|:---------------------|:--------------------------------------------|:--------------------------------------------|
| phase2 IRMF-EMD      | +10.80 [+9.91, +11.69]; 6.91e-12 / 0.00159  | +10.67 [+9.80, +11.53]; 5.99e-12 / 0.00159  |
| phase2 linear-EMD    | +18.74 [+17.05, +20.43]; 2.16e-11 / 0.00159 | +16.57 [+15.30, +17.84]; 2.77e-12 / 0.00159 |
| phase2 IRMF-raw      | +14.58 [+12.98, +16.18]; 2.97e-10 / 0.00159 | +14.54 [+13.16, +15.91]; 3.99e-11 / 0.00159 |
| Q1 conv-GD60         | +0.83 [+0.02, +1.63]; 0.133 / 0.00159       | +0.73 [+0.02, +1.43]; 0.133 / 0.00159       |
| Q2 H1-H2             | -1.95 [-2.94, -0.95]; 0.00444 / 0.00159     | -1.87 [-2.79, -0.95]; 0.00444 / 0.00159     |
| Q2 H4-H2             | +1.90 [+0.96, +2.84]; 0.00444 / 0.00159     | +1.06 [-0.12, +2.24]; 0.133 / 0.00159       |
| Q2 H1-H4             | -3.84 [-5.56, -2.13]; 0.00224 / 0.00159     | -2.93 [-4.41, -1.45]; 0.00444 / 0.00159     |
| Q3 IRMF a2-EMD       | +9.66 [+8.31, +11.00]; 5e-09 / 0.00159      | +9.91 [+8.78, +11.03]; 4.05e-10 / 0.00159   |
| Q3 linear a2-EMD     | +16.42 [+14.38, +18.46]; 1.17e-09 / 0.00159 | +13.83 [+12.24, +15.41]; 4.29e-10 / 0.00159 |
| Q4 GD60-linear sqrt2 | -7.94 [-9.17, -6.72]; 1.81e-08 / 0.00159    | -5.90 [-7.01, -4.80]; 2.05e-07 / 0.00159    |
| Q4 conv-linear sqrt2 | -7.11 [-7.91, -6.31]; 3.68e-10 / 0.00159    | -5.18 [-6.32, -4.03]; 1.47e-06 / 0.00159    |
| Q4 conv-linear a2    | -6.76 [-8.52, -5.00]; 9.74e-06 / 0.00159    | -3.92 [-5.52, -2.32]; 0.00113 / 0.00159     |
| conv sqrt2-EMD       | +11.63 [+10.33, +12.93]; 3.52e-10 / 0.00159 | +11.40 [+10.46, +12.33]; 6.82e-12 / 0.00159 |

Holm covers all 26 cells above, including both feature sets. The all-classifier table corrects 78 cells per metric. CIs are marginal, not multiplicity-adjusted. A nonsignificant result does not establish equivalence.

[10class macro-F1, accuracy and all classifiers](paired_all_metrics_10class.csv). [Fixed-classifier tests for all three metrics](paired_fixed_classifier_10class.csv). [Means and fold SDs](metric_summary_10class.csv).

Runtime 46.6 min; 450 new fits and 360 reused phase-2 fits, four single-thread workers.

0 new fits recorded warnings; [messages](fit_warnings_10class.csv) are retained rather than suppressed. These counts include inner-grid candidates when present.

The run log records 1 worker-replacement warning(s). Final coverage and unique checkpoint counts are checked separately; no missing fit is silently dropped.

## Reproducibility and limits

LDA, logistic regression and linear SVM retain their phase-2 estimators, grids, grouped tuning where used and all 15 saved splits. No baseline was refitted. The reporting classifier maximizes phase-2 mean BA across the four main methods and both feature sets, with alphabetical tie breaking, and stays fixed across all contrasts. This outer-score selection makes the inference exploratory. [The original six-classifier phase-2 report](../classical/REPORT.md) retains its original comparison families and verdicts.

| tag                |   slots |   cpu_seconds_per_frame |
|:-------------------|--------:|------------------------:|
| irmf_h2_a1.41_conv |      18 |                 0.10761 |
| irmf_h1_a1.41_conv |      18 |                 0.11112 |
| irmf_h4_a1.41_conv |      18 |                 0.10395 |
| irmf_h2_a2_conv    |      10 |                 0.06225 |
| imf_linear_a2      |      10 |                 0.00854 |

Slot counts include residuals and EMD's zero padding. EMD has 9 slots; a2 has 9 slots on UAS and 10 on the L=2048 datasets. Thus a2 is a near-matched, not identical, component count. No dimension-matched feature selection was performed. A2 also changes the bandwidth schedule, so this is not a pure feature-count intervention.

All converged-tag channel-stages passed their stopping rule; all selected feature tables are finite, aligned and complete. Shared log_scale columns match raw exactly. The Newton tags solve pimf's lookup score to root tolerance 1e-12, with a 100-iteration cap; they preserve normalization, kernel, boundary and schedule. Reuse checks recompute BA/F1/accuracy from saved OOF predictions, match every held-out frame and label to the saved splits, and check fold sizes, seeds and IDs. Input hashes, estimator signatures, grids and readiness checks are in audit_*.json.

The optional GD60 a2 tag requires completed decomposition and feature caches and explicit inclusion within the budget; its readiness/skip reason is recorded in the audit. Known phase-2 driver limitations are missing native OOF probabilities and prediction filenames shared across batches. The isolated wrapper records probabilities when supported and writes only phase2b checkpoints. Linear SVM has no native probability output. The new wrapper's early object-dtype y_true serialization was fixed; a one-time repair reconstructs Unicode labels from verified frame IDs and prepared metadata, requiring every stored metric to reproduce. [Repair audit](../../uas_birds_60ghz/classical_ablations/checkpoint_label_repair.json). No new shared numerical bug was found.

```bash
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/baselines/ablations/run_ablations.py --dataset dronerf --task 10class
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/baselines/ablations/analyse_ablations.py
```

Code changes are confined to baselines/ablations/. Large matrices, predictions and per-fit caches remain under DATA_ROOT/<dataset>/results/phase2b/. No commit was made.

Secondary ablations deferred to preserve the four-dataset budget: 2class, 4class. Their phase-2 main results remain in the source report.

The primary SVM fixes C=0.1 without redundant inner fits; logistic regression uses C in {0.01, 0.1, 1}, exactly as in phase 2. Each BUI is one session, so segment-held-out CV cannot establish unseen-session recognition. L/H views are nonsimultaneous.
