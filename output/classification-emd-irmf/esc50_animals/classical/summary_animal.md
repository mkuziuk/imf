# Classical baselines: esc50_animals / animal

5100 frames, 157 groups, 10 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 2116 s with 3 workers.

## Decomposition cost

| method           |   slots |   cpu_s_per_frame |   failed | emd_imf_counts                                                | irmf_all_stages_converged   |
|:-----------------|--------:|------------------:|---------:|:--------------------------------------------------------------|:----------------------------|
| emd              |       9 |            0.0140 |        0 | {'4': 13, '5': 143, '6': 1121, '7': 2598, '8': 1171, '9': 54} | nan                         |
| imf_linear_a1.41 |      18 |            0.0062 |        0 | nan                                                           | nan                         |
| irmf_h2_a1.41    |      18 |            1.7586 |        0 | nan                                                           | 2294/5100                   |

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.360 ± 0.034 | 0.500 ± 0.028 | 0.419 ± 0.042 | 0.443 ± 0.044 | 0.459 ± 0.041 | 0.357 ± 0.025 |
| emd              | 0.352 ± 0.034 | 0.443 ± 0.033 | 0.523 ± 0.043 | 0.543 ± 0.045 | 0.537 ± 0.054 | 0.390 ± 0.029 |
| imf_linear_a1.41 | 0.486 ± 0.041 | 0.587 ± 0.044 | 0.637 ± 0.033 | 0.653 ± 0.036 | 0.651 ± 0.034 | 0.498 ± 0.055 |
| irmf_h2_a1.41    | 0.490 ± 0.045 | 0.588 ± 0.039 | 0.627 ± 0.032 | 0.653 ± 0.037 | 0.651 ± 0.043 | 0.501 ± 0.039 |

Feature set `comp+log_scale`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.370 ± 0.037 | 0.505 ± 0.031 | 0.446 ± 0.052 | 0.459 ± 0.048 | 0.474 ± 0.045 | 0.372 ± 0.035 |
| emd              | 0.356 ± 0.034 | 0.444 ± 0.035 | 0.539 ± 0.041 | 0.565 ± 0.041 | 0.557 ± 0.051 | 0.399 ± 0.030 |
| imf_linear_a1.41 | 0.487 ± 0.040 | 0.588 ± 0.043 | 0.649 ± 0.033 | 0.674 ± 0.032 | 0.665 ± 0.035 | 0.506 ± 0.050 |
| irmf_h2_a1.41    | 0.490 ± 0.046 | 0.591 ± 0.039 | 0.639 ± 0.030 | 0.664 ± 0.041 | 0.661 ± 0.043 | 0.509 ± 0.032 |

## Best classifier per method

| feature_set    | method           | classifier   |   balanced_accuracy |
|:---------------|:-----------------|:-------------|--------------------:|
| comp           | emd              | linsvm       |               0.543 |
| comp           | imf_linear_a1.41 | linsvm       |               0.653 |
| comp           | irmf_h2_a1.41    | linsvm       |               0.653 |
| comp           | raw              | knn          |               0.500 |
| comp+log_scale | emd              | linsvm       |               0.565 |
| comp+log_scale | imf_linear_a1.41 | linsvm       |               0.674 |
| comp+log_scale | irmf_h2_a1.41    | linsvm       |               0.664 |
| comp+log_scale | raw              | knn          |               0.505 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a         | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-----------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41    | emd              |  0.1030 | [+0.071, +0.136] |      0.0004 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0100 | [-0.033, +0.013] |      1.0000 |            0.9558 | 4/0/11 |
| comp           | lda          | imf_linear_a1.41 | emd              |  0.1130 | [+0.091, +0.135] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | raw              |  0.2080 | [+0.164, +0.251] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41 | raw              |  0.2170 | [+0.169, +0.266] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | emd              | raw              |  0.1040 | [+0.048, +0.161] |      0.0453 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | emd              |  0.1120 | [+0.061, +0.162] |      0.0110 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0030 | [-0.021, +0.027] |      1.0000 |            1.0000 | 7/0/8  |
| comp           | qda          | imf_linear_a1.41 | emd              |  0.1090 | [+0.041, +0.176] |      0.0910 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | raw              |  0.1450 | [+0.088, +0.201] |      0.0034 |            0.0044 | 15/0/0 |
| comp           | qda          | imf_linear_a1.41 | raw              |  0.1410 | [+0.064, +0.218] |      0.0453 |            0.0044 | 15/0/0 |
| comp           | qda          | emd              | raw              |  0.0330 | [-0.019, +0.085] |      1.0000 |            0.2412 | 13/0/2 |
| comp           | gnb          | irmf_h2_a1.41    | emd              |  0.1380 | [+0.100, +0.176] |      0.0001 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0040 | [-0.017, +0.025] |      1.0000 |            1.0000 | 8/0/7  |
| comp           | gnb          | imf_linear_a1.41 | emd              |  0.1340 | [+0.091, +0.176] |      0.0005 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | raw              |  0.1300 | [+0.061, +0.199] |      0.0386 |            0.0044 | 15/0/0 |
| comp           | gnb          | imf_linear_a1.41 | raw              |  0.1260 | [+0.069, +0.183] |      0.0110 |            0.0044 | 15/0/0 |
| comp           | gnb          | emd              | raw              | -0.0070 | [-0.070, +0.056] |      1.0000 |            1.0000 | 7/0/8  |
| comp           | logreg       | irmf_h2_a1.41    | emd              |  0.1140 | [+0.081, +0.147] |      0.0002 |            0.0044 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0010 | [-0.025, +0.024] |      1.0000 |            1.0000 | 8/0/7  |
| comp           | logreg       | imf_linear_a1.41 | emd              |  0.1150 | [+0.081, +0.149] |      0.0002 |            0.0044 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | raw              |  0.1920 | [+0.140, +0.243] |      0.0001 |            0.0044 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41 | raw              |  0.1930 | [+0.147, +0.238] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | emd              | raw              |  0.0780 | [+0.030, +0.126] |      0.0910 |            0.0044 | 14/0/1 |
| comp           | knn          | irmf_h2_a1.41    | emd              |  0.1450 | [+0.093, +0.198] |      0.0017 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0010 | [-0.032, +0.034] |      1.0000 |            1.0000 | 6/0/9  |
| comp           | knn          | imf_linear_a1.41 | emd              |  0.1450 | [+0.103, +0.186] |      0.0002 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | raw              |  0.0890 | [+0.054, +0.124] |      0.0036 |            0.0044 | 15/0/0 |
| comp           | knn          | imf_linear_a1.41 | raw              |  0.0880 | [+0.045, +0.130] |      0.0196 |            0.0044 | 15/0/0 |
| comp           | knn          | emd              | raw              | -0.0570 | [-0.106, -0.008] |      0.4620 |            0.0073 | 2/0/13 |
| comp           | linsvm       | irmf_h2_a1.41    | emd              |  0.1100 | [+0.077, +0.142] |      0.0002 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0000 | [-0.032, +0.032] |      1.0000 |            1.0000 | 9/0/6  |
| comp           | linsvm       | imf_linear_a1.41 | emd              |  0.1090 | [+0.077, +0.142] |      0.0002 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | raw              |  0.2100 | [+0.144, +0.275] |      0.0004 |            0.0044 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41 | raw              |  0.2090 | [+0.156, +0.263] |      0.0001 |            0.0044 | 15/0/0 |
| comp           | linsvm       | emd              | raw              |  0.1000 | [+0.041, +0.158] |      0.0652 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | emd              |  0.1000 | [+0.074, +0.127] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0110 | [-0.037, +0.016] |      1.0000 |            1.0000 | 5/0/10 |
| comp+log_scale | lda          | imf_linear_a1.41 | emd              |  0.1110 | [+0.089, +0.133] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | raw              |  0.1930 | [+0.136, +0.249] |      0.0002 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41 | raw              |  0.2030 | [+0.139, +0.268] |      0.0005 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | emd              | raw              |  0.0930 | [+0.027, +0.158] |      0.1807 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | emd              |  0.1100 | [+0.066, +0.155] |      0.0040 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0030 | [-0.023, +0.029] |      1.0000 |            1.0000 | 8/0/7  |
| comp+log_scale | qda          | imf_linear_a1.41 | emd              |  0.1080 | [+0.045, +0.170] |      0.0643 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | raw              |  0.1370 | [+0.082, +0.193] |      0.0040 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | imf_linear_a1.41 | raw              |  0.1350 | [+0.062, +0.208] |      0.0453 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | emd              | raw              |  0.0270 | [-0.038, +0.092] |      1.0000 |            1.0000 | 11/0/4 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | emd              |  0.1340 | [+0.097, +0.172] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0030 | [-0.017, +0.024] |      1.0000 |            1.0000 | 8/0/7  |
| comp+log_scale | gnb          | imf_linear_a1.41 | emd              |  0.1310 | [+0.089, +0.173] |      0.0005 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | raw              |  0.1200 | [+0.042, +0.198] |      0.1152 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | imf_linear_a1.41 | raw              |  0.1170 | [+0.050, +0.183] |      0.0578 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | emd              | raw              | -0.0140 | [-0.086, +0.058] |      1.0000 |            1.0000 | 5/0/10 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | emd              |  0.1030 | [+0.068, +0.139] |      0.0010 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0050 | [-0.026, +0.016] |      1.0000 |            1.0000 | 6/0/9  |
| comp+log_scale | logreg       | imf_linear_a1.41 | emd              |  0.1080 | [+0.077, +0.139] |      0.0002 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | raw              |  0.1860 | [+0.112, +0.261] |      0.0037 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41 | raw              |  0.1910 | [+0.128, +0.255] |      0.0007 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | emd              | raw              |  0.0830 | [+0.020, +0.146] |      0.2542 |            0.0055 | 13/0/2 |
| comp+log_scale | knn          | irmf_h2_a1.41    | emd              |  0.1470 | [+0.095, +0.198] |      0.0012 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0020 | [-0.028, +0.033] |      1.0000 |            1.0000 | 7/0/8  |
| comp+log_scale | knn          | imf_linear_a1.41 | emd              |  0.1440 | [+0.107, +0.182] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | raw              |  0.0860 | [+0.039, +0.133] |      0.0453 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | imf_linear_a1.41 | raw              |  0.0840 | [+0.029, +0.139] |      0.1190 |            0.0044 | 14/0/1 |
| comp+log_scale | knn          | emd              | raw              | -0.0610 | [-0.114, -0.007] |      0.4805 |            0.0044 | 1/0/14 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | emd              |  0.0990 | [+0.064, +0.133] |      0.0011 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0100 | [-0.037, +0.017] |      1.0000 |            1.0000 | 5/0/10 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | emd              |  0.1090 | [+0.082, +0.135] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | raw              |  0.2050 | [+0.124, +0.286] |      0.0036 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | raw              |  0.2150 | [+0.148, +0.283] |      0.0004 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | emd              | raw              |  0.1060 | [+0.049, +0.164] |      0.0453 |            0.0044 | 15/0/0 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method           |   max_eps2 |   max_mi |   n_features |
|:-----------------|-----------:|---------:|-------------:|
| raw              |      0.419 |    0.551 |       21.000 |
| emd              |      0.492 |    0.394 |      181.000 |
| imf_linear_a1.41 |      0.596 |    0.530 |      361.000 |
| irmf_h2_a1.41    |      0.612 |    0.547 |      361.000 |

| method           | feature              | slot_label   |   epsilon2 |    mi |
|:-----------------|:---------------------|:-------------|-----------:|------:|
| irmf_h2_a1.41    | ch0_c8_hjorth_mob    | w65          |      0.612 | 0.547 |
| imf_linear_a1.41 | ch0_c8_hjorth_mob    | w65          |      0.596 | 0.530 |
| irmf_h2_a1.41    | ch0_c8_hilb_if_mean  | w65          |      0.594 | 0.516 |
| irmf_h2_a1.41    | ch0_c10_energy_frac  | w129         |      0.593 | 0.495 |
| imf_linear_a1.41 | ch0_c10_energy_frac  | w129         |      0.592 | 0.507 |
| irmf_h2_a1.41    | ch0_c8_spec_centroid | w65          |      0.588 | 0.506 |
| irmf_h2_a1.41    | ch0_c9_hjorth_mob    | w93          |      0.585 | 0.506 |
| imf_linear_a1.41 | ch0_c8_hilb_if_mean  | w65          |      0.584 | 0.508 |
| imf_linear_a1.41 | ch0_c9_energy_frac   | w93          |      0.582 | 0.519 |
| imf_linear_a1.41 | ch0_c8_spec_centroid | w65          |      0.580 | 0.499 |
| emd              | ch0_c3_energy_frac   | imf4         |      0.492 | 0.370 |
| emd              | ch0_c3_mean_abs      | imf4         |      0.447 | 0.331 |
| emd              | ch0_c3_hilb_amp_mean | imf4         |      0.446 | 0.330 |
| emd              | ch0_c3_log_energy    | imf4         |      0.426 | 0.314 |
| emd              | ch0_c3_std           | imf4         |      0.426 | 0.315 |
| raw              | ch0_c0_hjorth_cpx    | frame        |      0.419 | 0.366 |
| raw              | ch0_c0_hilb_if_mean  | frame        |      0.416 | 0.428 |
| raw              | ch0_c0_spec_centroid | frame        |      0.382 | 0.398 |
| raw              | ch0_c0_hjorth_mob    | frame        |      0.375 | 0.355 |
| raw              | ch0_c0_zcr           | frame        |      0.367 | 0.345 |
