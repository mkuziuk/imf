# Classical classification summary

Updated 2026-10-20T22:54:42+00:00. All differences are BA percentage points, A minus B, with 95% corrected intervals. Each dataset fixes the best phase-2 classifier across raw, EMD, linear IMF and GD60 IRMF, averaging both feature sets. The choice is exploratory and never changes by contrast.

Phase-2 reports establish IRMF versus EMD gains on ESC-50 animals and BirdCLEF species, with and without log_scale. UAS is inconclusive after corrected-t/Holm testing and its recording votes reach a ceiling. DroneRF's primary SVM IRMF-EMD gain is +10.80/+10.67 pp without/with log_scale, while IRMF trails linear IMF by 7.94/5.90 pp. The original reports establish no consistent robustness-specific IRMF gain over linear IMF; DroneRF gives clear evidence favoring linear IMF. BirdCLEF's original GD60 kNN is a classifier-specific positive exception, +2.53/+2.62 pp over linear IMF with original Holm significance; kNN convergence/schedule ablations are outside this budget. Source reports: [UAS](uas_birds_60ghz/classical/REPORT.md), [ESC-50](esc50_animals/classical/REPORT.md), [BirdCLEF](birdclef2023/classical/REPORT.md), [DroneRF](dronerf/classical/REPORT.md).

uas_birds_60ghz, lda.

Convergence: -0.17 to -0.16 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 2.
H sensitivity: -0.51 to +0.55 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.
Matched-count contrasts against EMD: -0.22 to +0.80 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 4.
IRMF contrasts against linear IMF: +0.31 to +0.91 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.

esc50_animals, linsvm.

Convergence: -0.31 to +0.16 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 2.
H sensitivity: -1.22 to +1.31 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.
Matched-count contrasts against EMD: +9.75 to +11.00 pp; corrected-t/Holm rejects 4 positive and 0 negative contrasts of 4.
IRMF contrasts against linear IMF: -1.00 to +0.37 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.

birdclef2023, logreg.

Convergence: -0.12 to +0.07 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 2.
H sensitivity: -0.19 to +0.36 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.
Matched-count contrasts against EMD: +3.81 to +4.43 pp; corrected-t/Holm rejects 4 positive and 0 negative contrasts of 4.
IRMF contrasts against linear IMF: +0.38 to +0.82 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.

dronerf, linsvm.

Convergence: +0.73 to +0.83 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 2.
H sensitivity: -3.84 to +1.90 pp; corrected-t/Holm rejects 0 positive and 2 negative contrasts of 6.
Matched-count contrasts against EMD: +9.66 to +16.42 pp; corrected-t/Holm rejects 4 positive and 0 negative contrasts of 4.
IRMF contrasts against linear IMF: -7.94 to -3.92 pp; corrected-t/Holm rejects 0 positive and 6 negative contrasts of 6.

pooled, dataset-fixed.

Convergence: +0.10 to +0.16 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 2.
H sensitivity: -0.91 to +0.26 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.
Matched-count contrasts against EMD: +6.07 to +7.86 pp; corrected-t/Holm rejects 4 positive and 0 negative contrasts of 4.
IRMF contrasts against linear IMF: -1.61 to -0.83 pp; corrected-t/Holm rejects 0 positive and 0 negative contrasts of 6.

| dataset / fixed classifier   | contrast             | without log_scale: delta [CI]; Holm p   | with log_scale: delta [CI]; Holm p   |
|:-----------------------------|:---------------------|:----------------------------------------|:-------------------------------------|
| uas_birds_60ghz / lda        | phase2 IRMF-EMD      | +1.75 [-0.38, +3.88]; 1                 | +1.23 [-0.37, +2.82]; 1              |
| uas_birds_60ghz / lda        | phase2 linear-EMD    | +0.92 [-0.73, +2.56]; 1                 | +0.32 [-1.00, +1.63]; 1              |
| uas_birds_60ghz / lda        | phase2 IRMF-raw      | +4.03 [-0.30, +8.37]; 1                 | +2.20 [-0.62, +5.03]; 1              |
| uas_birds_60ghz / lda        | Q1 conv-GD60         | -0.17 [-0.60, +0.25]; 1                 | -0.16 [-0.61, +0.30]; 1              |
| uas_birds_60ghz / lda        | Q2 H1-H2             | +0.02 [-0.88, +0.93]; 1                 | +0.09 [-0.54, +0.71]; 1              |
| uas_birds_60ghz / lda        | Q2 H4-H2             | -0.51 [-1.65, +0.63]; 1                 | -0.46 [-1.65, +0.73]; 1              |
| uas_birds_60ghz / lda        | Q2 H1-H4             | +0.53 [-0.47, +1.54]; 1                 | +0.55 [-0.41, +1.50]; 1              |
| uas_birds_60ghz / lda        | Q3 IRMF a2-EMD       | +0.80 [-0.90, +2.49]; 1                 | +0.43 [-1.02, +1.88]; 1              |
| uas_birds_60ghz / lda        | Q3 linear a2-EMD     | +0.48 [-1.67, +2.63]; 1                 | -0.22 [-1.76, +1.32]; 1              |
| uas_birds_60ghz / lda        | Q4 GD60-linear sqrt2 | +0.83 [-0.63, +2.30]; 1                 | +0.91 [-0.52, +2.34]; 1              |
| uas_birds_60ghz / lda        | Q4 conv-linear sqrt2 | +0.66 [-0.85, +2.17]; 1                 | +0.75 [-0.79, +2.30]; 1              |
| uas_birds_60ghz / lda        | Q4 conv-linear a2    | +0.31 [-1.23, +1.86]; 1                 | +0.65 [-0.55, +1.85]; 1              |
| uas_birds_60ghz / lda        | conv sqrt2-EMD       | +1.58 [-0.65, +3.80]; 1                 | +1.07 [-0.70, +2.84]; 1              |
| esc50_animals / linsvm       | phase2 IRMF-EMD      | +10.97 [+7.75, +14.19]; 0.000347        | +9.89 [+6.44, +13.35]; 0.00209       |
| esc50_animals / linsvm       | phase2 linear-EMD    | +10.95 [+7.71, +14.19]; 0.000369        | +10.89 [+8.22, +13.55]; 4.39e-05     |
| esc50_animals / linsvm       | phase2 IRMF-raw      | +20.95 [+14.36, +27.54]; 0.000727       | +20.53 [+12.43, +28.62]; 0.00691     |
| esc50_animals / linsvm       | Q1 conv-GD60         | -0.31 [-2.55, +1.93]; 1                 | +0.16 [-1.09, +1.41]; 1              |
| esc50_animals / linsvm       | Q2 H1-H2             | -0.55 [-4.21, +3.12]; 1                 | +0.08 [-1.28, +1.44]; 1              |
| esc50_animals / linsvm       | Q2 H4-H2             | -0.28 [-3.01, +2.44]; 1                 | -1.22 [-3.87, +1.43]; 1              |
| esc50_animals / linsvm       | Q2 H1-H4             | -0.26 [-5.40, +4.87]; 1                 | +1.31 [-1.72, +4.33]; 1              |
| esc50_animals / linsvm       | Q3 IRMF a2-EMD       | +11.00 [+7.40, +14.60]; 0.0011          | +9.75 [+5.94, +13.56]; 0.00632       |
| esc50_animals / linsvm       | Q3 linear a2-EMD     | +10.63 [+7.54, +13.72]; 0.000312        | +10.17 [+7.37, +12.98]; 0.000174     |
| esc50_animals / linsvm       | Q4 GD60-linear sqrt2 | +0.02 [-3.21, +3.24]; 1                 | -1.00 [-3.66, +1.67]; 1              |
| esc50_animals / linsvm       | Q4 conv-linear sqrt2 | -0.29 [-3.99, +3.41]; 1                 | -0.83 [-3.31, +1.65]; 1              |
| esc50_animals / linsvm       | Q4 conv-linear a2    | +0.37 [-1.78, +2.51]; 1                 | -0.42 [-2.83, +1.98]; 1              |
| esc50_animals / linsvm       | conv sqrt2-EMD       | +10.66 [+7.21, +14.10]; 0.00096         | +10.06 [+6.54, +13.57]; 0.00209      |
| birdclef2023 / logreg        | phase2 IRMF-EMD      | +6.05 [+4.72, +7.39]; 1.28e-05          | +6.05 [+4.49, +7.60]; 7.84e-05       |
| birdclef2023 / logreg        | phase2 linear-EMD    | +5.41 [+4.24, +6.57]; 1.01e-05          | +5.23 [+4.02, +6.44]; 2.22e-05       |
| birdclef2023 / logreg        | phase2 IRMF-raw      | +11.89 [+9.84, +13.94]; 5.92e-07        | +11.82 [+9.98, +13.66]; 1.61e-07     |
| birdclef2023 / logreg        | Q1 conv-GD60         | +0.07 [-0.72, +0.85]; 1                 | -0.12 [-0.95, +0.72]; 1              |
| birdclef2023 / logreg        | Q2 H1-H2             | -0.12 [-1.29, +1.05]; 1                 | +0.17 [-1.16, +1.50]; 1              |
| birdclef2023 / logreg        | Q2 H4-H2             | -0.04 [-1.10, +1.02]; 1                 | -0.19 [-1.11, +0.73]; 1              |
| birdclef2023 / logreg        | Q2 H1-H4             | -0.07 [-1.75, +1.60]; 1                 | +0.36 [-1.07, +1.80]; 1              |
| birdclef2023 / logreg        | Q3 IRMF a2-EMD       | +4.43 [+2.96, +5.91]; 0.0013            | +4.20 [+2.43, +5.96]; 0.0122         |
| birdclef2023 / logreg        | Q3 linear a2-EMD     | +3.89 [+3.32, +4.46]; 7.14e-08          | +3.81 [+2.53, +5.09]; 0.00139        |
| birdclef2023 / logreg        | Q4 GD60-linear sqrt2 | +0.65 [-0.60, +1.89]; 1                 | +0.82 [-0.41, +2.05]; 1              |
| birdclef2023 / logreg        | Q4 conv-linear sqrt2 | +0.71 [-0.46, +1.88]; 1                 | +0.70 [-0.41, +1.81]; 1              |
| birdclef2023 / logreg        | Q4 conv-linear a2    | +0.54 [-0.68, +1.76]; 1                 | +0.38 [-0.83, +1.60]; 1              |
| birdclef2023 / logreg        | conv sqrt2-EMD       | +6.12 [+4.84, +7.39]; 6.57e-06          | +5.93 [+4.32, +7.55]; 0.000153       |
| dronerf / linsvm             | phase2 IRMF-EMD      | +10.80 [+9.91, +11.69]; 3.51e-11        | +10.67 [+9.80, +11.53]; 2.85e-11     |
| dronerf / linsvm             | phase2 linear-EMD    | +18.74 [+17.05, +20.43]; 1.14e-10       | +16.57 [+15.30, +17.84]; 1.29e-11    |
| dronerf / linsvm             | phase2 IRMF-raw      | +14.58 [+12.98, +16.18]; 1.68e-09       | +14.54 [+13.16, +15.91]; 2.19e-10    |
| dronerf / linsvm             | Q1 conv-GD60         | +0.83 [+0.02, +1.63]; 1                 | +0.73 [+0.02, +1.43]; 1              |
| dronerf / linsvm             | Q2 H1-H2             | -1.95 [-2.94, -0.95]; 0.0625            | -1.87 [-2.79, -0.95]; 0.0475         |
| dronerf / linsvm             | Q2 H4-H2             | +1.90 [+0.96, +2.84]; 0.0521            | +1.06 [-0.12, +2.24]; 1              |
| dronerf / linsvm             | Q2 H1-H4             | -3.84 [-5.56, -2.13]; 0.0213            | -2.93 [-4.41, -1.45]; 0.059          |
| dronerf / linsvm             | Q3 IRMF a2-EMD       | +9.66 [+8.31, +11.00]; 3.79e-08         | +9.91 [+8.78, +11.03]; 2.62e-09      |
| dronerf / linsvm             | Q3 linear a2-EMD     | +16.42 [+14.38, +18.46]; 8.41e-09       | +13.83 [+12.24, +15.41]; 2.92e-09    |
| dronerf / linsvm             | Q4 GD60-linear sqrt2 | -7.94 [-9.17, -6.72]; 1.45e-07          | -5.90 [-7.01, -4.80]; 1.73e-06       |
| dronerf / linsvm             | Q4 conv-linear sqrt2 | -7.11 [-7.91, -6.31]; 2.27e-09          | -5.18 [-6.32, -4.03]; 1.29e-05       |
| dronerf / linsvm             | Q4 conv-linear a2    | -6.76 [-8.52, -5.00]; 9.06e-05          | -3.92 [-5.52, -2.32]; 0.00976        |
| dronerf / linsvm             | conv sqrt2-EMD       | +11.63 [+10.33, +12.93]; 2.07e-09       | +11.40 [+10.46, +12.33]; 3.35e-11    |
| pooled / dataset-fixed       | phase2 IRMF-EMD      | +7.39 [+6.40, +8.38]; 3.53e-14          | +6.96 [+5.96, +7.96]; 2.96e-12       |
| pooled / dataset-fixed       | phase2 linear-EMD    | +9.00 [+8.01, +9.99]; 5.28e-17          | +8.25 [+7.43, +9.07]; 1.14e-18       |
| pooled / dataset-fixed       | phase2 IRMF-raw      | +12.86 [+10.89, +14.84]; 7.04e-12       | +12.27 [+10.11, +14.43]; 1.95e-08    |
| pooled / dataset-fixed       | Q1 conv-GD60         | +0.10 [-0.51, +0.72]; 1                 | +0.16 [-0.25, +0.56]; 1              |
| pooled / dataset-fixed       | Q2 H1-H2             | -0.65 [-1.63, +0.34]; 1                 | -0.38 [-0.90, +0.14]; 1              |
| pooled / dataset-fixed       | Q2 H4-H2             | +0.26 [-0.52, +1.05]; 1                 | -0.20 [-0.98, +0.57]; 1              |
| pooled / dataset-fixed       | Q2 H1-H4             | -0.91 [-2.30, +0.48]; 1                 | -0.18 [-1.08, +0.72]; 1              |
| pooled / dataset-fixed       | Q3 IRMF a2-EMD       | +6.47 [+5.41, +7.53]; 2.79e-11          | +6.07 [+4.98, +7.16]; 8.36e-10       |
| pooled / dataset-fixed       | Q3 linear a2-EMD     | +7.86 [+6.84, +8.88]; 7.11e-16          | +6.90 [+6.00, +7.79]; 5.88e-16       |
| pooled / dataset-fixed       | Q4 GD60-linear sqrt2 | -1.61 [-2.55, -0.67]; 0.106             | -1.29 [-2.11, -0.48]; 0.195          |
| pooled / dataset-fixed       | Q4 conv-linear sqrt2 | -1.51 [-2.53, -0.49]; 0.382             | -1.14 [-1.92, -0.35]; 0.385          |
| pooled / dataset-fixed       | Q4 conv-linear a2    | -1.38 [-2.18, -0.59]; 0.0734            | -0.83 [-1.62, -0.04]; 1              |
| pooled / dataset-fixed       | conv sqrt2-EMD       | +7.50 [+6.43, +8.56]; 1.09e-13          | +7.11 [+6.08, +8.15]; 1.6e-12        |

Holm correction spans all 130 numeric cells in this table, separately for corrected-t and Wilcoxon. The original phase-2 reports used their own comparison families; their published verdicts above retain that context. The CSV contains raw p values, secondary Wilcoxon and separate F1/accuracy families. Intervals are marginal 95% intervals.

Pooled rows use the four primary tasks; secondary tasks remain dataset-specific. Pooled estimates weight each available dataset equally. Their SE propagates the independent dataset-specific Nadeau-Bengio corrected variances, with Welch-Satterthwaite degrees of freedom: SE_pool = sqrt(sum(SE_NB_i^2)) / D. This estimates the average across these observed studies, conditional on their chosen classifiers. It does not treat 60 overlapping folds as independent or establish an effect on unseen datasets. Pooled Wilcoxon uses one mean difference per dataset and has very low power. Between-dataset effect ranges are saved in the CSV. A separate [common-classifier pooled table](uas_birds_60ghz/classical_ablations/pooled_all_classifiers.csv) holds LDA, logistic regression or linear SVM fixed across every dataset, with Holm correction across all three classifier tables per metric.

Budget: at most four single-thread fit workers, three classifiers, no baseline refits. All 15 folds, both log_scale settings and five required tags are retained. Secondary tasks run when time permits. No PROTOCOL, shared package, baseline driver, ML source or pre-existing result was edited. No commit was made.

Files: each dataset's classical_ablations/REPORT.md, reused and new fold CSVs, all-classifier/fixed-classifier paired CSVs, metric summaries, warning tables and audit JSONs. [Cross-dataset BA/F1/accuracy and secondary tests](uas_birds_60ghz/classical_ablations/cross_dataset_all_metrics.csv). Code: experiments/classification-emd-irmf/baselines/ablations/.

Bugs: the inherited driver omits OOF probabilities and shares prediction filenames between batches. Its default comparisons collapse tags to one representative per method family, so explicit pairs are required for H/convergence/schedule ablations. The new runner avoids these limitations without editing shared code. Its initial object-label serialization bug was fixed locally, with predictions, probabilities and timings preserved and metrics checked in the [repair audit](uas_birds_60ghz/classical_ablations/checkpoint_label_repair.json).
