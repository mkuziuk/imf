# ESC-50 animals: classical baselines

Percentages, balanced accuracy / macro-F1 / accuracy. Best-classifier frame scores are exploratory outer-CV means; SVM is linear. Without/with excludes/includes `log_scale`. Clip BA fixes logistic regression, averages probabilities within clip, pools 400 clips/seed, then averages three seeds.

| Task | Method/classifier | Without: BA/F1/accuracy | With: BA/F1/accuracy | Clip BA without/with |
|---|---|---|---|---|
| animal | Raw/kNN | 49.98/48.90/50.03 | 50.46/49.32/50.47 | 55.33/55.50 |
| animal | EMD/SVM | 54.32/52.69/54.11 | 56.55/55.15/56.37 | 68.58/70.58 |
| animal | Linear/SVM | 65.27/64.19/65.18 | 67.44/66.58/67.67 | 76.75/76.92 |
| animal | IRMF/SVM | 65.29/64.36/65.32 | 66.44/65.70/66.63 | 76.50/77.67 |
| bird | Raw/kNN | 71.82/72.48/77.77 | 72.16/72.62/77.56 | 66.19/65.77 |
| bird | EMD/logistic | 67.22/68.02/74.80 | 66.35/67.04/73.93 | 70.08/69.50 |
| bird | Linear/kNN | 74.36/74.48/78.78 | 74.20/74.33/78.66 | 76.01/75.00 |
| bird | IRMF/kNN | 74.00/73.86/77.97 | 73.84/73.62/77.71 | 74.94/73.77 |

Animal IRMF−EMD, SVM: without, +10.97 pp, 95% CI [7.75,14.19], corrected-t p=3.90e-6, Holm p=.000218; with, +9.89 [6.44,13.35], p=.0000255, Holm p=.00115. The gain survives removing level under all six classifiers. Bird evidence depends on classifier: QDA passes Holm without level; QDA/kNN pass with level. Bird SVM Holm p=1 in both settings.

Significant corrected-t/Holm cells/12, six classifiers × two feature sets:

| Paired contrast | Animal | Bird |
|---|---:|---:|
| IRMF > EMD | 12 | 3 |
| IRMF vs linear | 0 | 0 |
| Linear > EMD | 10 | 2 |
| IRMF > raw | 11 | 2 |
| Linear > raw | 10 | 1 |
| EMD > raw | 2 | 0 |

[Paired differences, 95% CIs, t/Wilcoxon p-values and Holm adjustments](paired_all_metrics.csv) use 72 tests/task/metric. [SVM table](paired_svm.csv) covers all six pairs.

IRMF−EMD SVM recall differences, without/with level: hen +27.2/+25.4 pp, crow +21.6/+22.0, insects −5.0/−1.3. [All classes](per_class_recall_differences.csv).

Scales w65–w129 dominate IMF features, observed centroids 420–680 Hz. IRMF w93, 557 Hz, has mean Kruskal-Wallis ε²/MI .435/.363 animal and .097/.076 bird. EMD peaks at imf1, 2,060 Hz, and imf4, 249 Hz. [Aggregates](discriminability_by_scale.csv) use `scale_order`/slot labels; MI is nats, statistics descriptive.

At max_iter=60, 2,294/5,100 frames fully converge; 21.6% of stages remain unconverged; overall mean 32.9 iterations. SVM converged/unconverged accuracy without/with level is animal 62.7/67.6 and 64.2/68.8%; bird 70.1/81.8 and 68.8/81.4%. All four Holm p=1; class composition confounds these associations.

Classes span 12 dB; one uploader supplies 36.7% of crow frames. CV holds out uploaders, 157 groups, 5 folds × 3 seeds. Uploader-balanced SVM IRMF−EMD differences retain positive signs. IRMF has 18 slots versus EMD's 9; no robustness-specific advantage over linear IMF is established.

CPU decomposition cost/frame: raw 0, EMD .0140 s, linear .00624 s, IRMF 1.759 s. All four H/schedule ablations were skipped: feature parquets were absent at 19:29 UTC.

Runtime: 42.0 minutes CV, approximately two minutes checks, ≤3 workers. Wrote tables, three [figures](figures/), two isolated scripts. [Reproductions/workarounds](RUN_NOTES.md) cover missing OOF probabilities, prediction overwrites, and QDA failures. All 42 SVM refits reproduce labels; two capped fits converge with unchanged BA. No shared edits by this task; no commit.
