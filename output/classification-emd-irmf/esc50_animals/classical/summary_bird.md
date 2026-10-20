# Classical baselines: esc50_animals / bird

5100 frames, 157 groups, 2 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 405 s with 3 workers.

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
| raw              | 0.551 ± 0.037 | 0.718 ± 0.039 | 0.493 ± 0.028 | 0.630 ± 0.030 | 0.657 ± 0.029 | 0.584 ± 0.036 |
| emd              | 0.538 ± 0.040 | 0.646 ± 0.027 | 0.631 ± 0.048 | 0.667 ± 0.039 | 0.672 ± 0.041 | 0.584 ± 0.035 |
| imf_linear_a1.41 | 0.609 ± 0.072 | 0.744 ± 0.029 | 0.728 ± 0.084 | 0.729 ± 0.070 | 0.731 ± 0.063 | 0.653 ± 0.053 |
| irmf_h2_a1.41    | 0.589 ± 0.054 | 0.740 ± 0.035 | 0.696 ± 0.068 | 0.706 ± 0.060 | 0.708 ± 0.059 | 0.643 ± 0.038 |

Feature set `comp+log_scale`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.551 ± 0.037 | 0.722 ± 0.040 | 0.489 ± 0.028 | 0.622 ± 0.034 | 0.650 ± 0.032 | 0.582 ± 0.036 |
| emd              | 0.538 ± 0.040 | 0.642 ± 0.028 | 0.624 ± 0.058 | 0.657 ± 0.052 | 0.663 ± 0.054 | 0.587 ± 0.035 |
| imf_linear_a1.41 | 0.609 ± 0.072 | 0.742 ± 0.027 | 0.723 ± 0.092 | 0.721 ± 0.083 | 0.720 ± 0.079 | 0.655 ± 0.054 |
| irmf_h2_a1.41    | 0.589 ± 0.054 | 0.738 ± 0.034 | 0.690 ± 0.075 | 0.695 ± 0.073 | 0.704 ± 0.062 | 0.644 ± 0.038 |

## Best classifier per method

| feature_set    | method           | classifier   |   balanced_accuracy |
|:---------------|:-----------------|:-------------|--------------------:|
| comp           | emd              | logreg       |               0.672 |
| comp           | imf_linear_a1.41 | knn          |               0.744 |
| comp           | irmf_h2_a1.41    | knn          |               0.740 |
| comp           | raw              | knn          |               0.718 |
| comp+log_scale | emd              | logreg       |               0.663 |
| comp+log_scale | imf_linear_a1.41 | knn          |               0.742 |
| comp+log_scale | irmf_h2_a1.41    | knn          |               0.738 |
| comp+log_scale | raw              | knn          |               0.722 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a         | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-----------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41    | emd              |  0.0650 | [+0.022, +0.109] |      0.3252 |            0.0062 | 14/0/1 |
| comp           | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0310 | [-0.053, -0.010] |      0.3619 |            0.0062 | 1/0/14 |
| comp           | lda          | imf_linear_a1.41 | emd              |  0.0970 | [+0.037, +0.157] |      0.2031 |            0.0062 | 14/0/1 |
| comp           | lda          | irmf_h2_a1.41    | raw              |  0.2030 | [+0.110, +0.296] |      0.0249 |            0.0044 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41 | raw              |  0.2340 | [+0.122, +0.347] |      0.0358 |            0.0044 | 15/0/0 |
| comp           | lda          | emd              | raw              |  0.1380 | [+0.065, +0.210] |      0.0676 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | emd              |  0.0590 | [+0.035, +0.084] |      0.0100 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0100 | [-0.034, +0.014] |      1.0000 |            0.8123 | 6/0/9  |
| comp           | qda          | imf_linear_a1.41 | emd              |  0.0690 | [+0.034, +0.104] |      0.0502 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | raw              |  0.0590 | [+0.015, +0.103] |      0.4546 |            0.0134 | 13/0/2 |
| comp           | qda          | imf_linear_a1.41 | raw              |  0.0690 | [+0.011, +0.127] |      0.7864 |            0.0134 | 13/0/2 |
| comp           | qda          | emd              | raw              | -0.0000 | [-0.041, +0.041] |      1.0000 |            1.0000 | 8/0/7  |
| comp           | gnb          | irmf_h2_a1.41    | emd              |  0.0510 | [+0.021, +0.081] |      0.1555 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0200 | [-0.047, +0.007] |      1.0000 |            0.1141 | 2/0/13 |
| comp           | gnb          | imf_linear_a1.41 | emd              |  0.0710 | [+0.022, +0.119] |      0.3601 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | raw              |  0.0380 | [+0.000, +0.075] |      1.0000 |            0.0195 | 14/0/1 |
| comp           | gnb          | imf_linear_a1.41 | raw              |  0.0580 | [+0.000, +0.115] |      1.0000 |            0.0154 | 13/0/2 |
| comp           | gnb          | emd              | raw              | -0.0130 | [-0.038, +0.012] |      1.0000 |            0.3920 | 4/0/11 |
| comp           | logreg       | irmf_h2_a1.41    | emd              |  0.0360 | [-0.003, +0.076] |      1.0000 |            0.0313 | 13/0/2 |
| comp           | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0220 | [-0.046, +0.002] |      1.0000 |            0.0366 | 2/0/13 |
| comp           | logreg       | imf_linear_a1.41 | emd              |  0.0590 | [+0.011, +0.106] |      0.7153 |            0.0248 | 14/0/1 |
| comp           | logreg       | irmf_h2_a1.41    | raw              |  0.0520 | [-0.014, +0.118] |      1.0000 |            0.1021 | 12/0/3 |
| comp           | logreg       | imf_linear_a1.41 | raw              |  0.0740 | [+0.002, +0.145] |      1.0000 |            0.0313 | 13/0/2 |
| comp           | logreg       | emd              | raw              |  0.0150 | [-0.024, +0.054] |      1.0000 |            0.6372 | 11/0/4 |
| comp           | knn          | irmf_h2_a1.41    | emd              |  0.0940 | [+0.039, +0.148] |      0.1387 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0040 | [-0.037, +0.030] |      1.0000 |            1.0000 | 7/0/8  |
| comp           | knn          | imf_linear_a1.41 | emd              |  0.0970 | [+0.060, +0.135] |      0.0049 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | raw              |  0.0220 | [-0.022, +0.066] |      1.0000 |            0.6372 | 9/0/6  |
| comp           | knn          | imf_linear_a1.41 | raw              |  0.0250 | [-0.017, +0.068] |      1.0000 |            0.2261 | 11/0/4 |
| comp           | knn          | emd              | raw              | -0.0720 | [-0.122, -0.021] |      0.3727 |            0.0062 | 1/0/14 |
| comp           | linsvm       | irmf_h2_a1.41    | emd              |  0.0380 | [-0.001, +0.077] |      1.0000 |            0.0366 | 13/0/2 |
| comp           | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0230 | [-0.056, +0.009] |      1.0000 |            0.1021 | 3/0/12 |
| comp           | linsvm       | imf_linear_a1.41 | emd              |  0.0620 | [+0.014, +0.109] |      0.5908 |            0.0134 | 13/0/2 |
| comp           | linsvm       | irmf_h2_a1.41    | raw              |  0.0750 | [+0.020, +0.131] |      0.4546 |            0.0134 | 14/0/1 |
| comp           | linsvm       | imf_linear_a1.41 | raw              |  0.0990 | [+0.035, +0.162] |      0.2540 |            0.0134 | 13/0/2 |
| comp           | linsvm       | emd              | raw              |  0.0370 | [+0.002, +0.072] |      1.0000 |            0.0154 | 14/0/1 |
| comp+log_scale | lda          | irmf_h2_a1.41    | emd              |  0.0660 | [+0.025, +0.107] |      0.2031 |            0.0062 | 14/0/1 |
| comp+log_scale | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0330 | [-0.055, -0.010] |      0.3619 |            0.0084 | 1/0/14 |
| comp+log_scale | lda          | imf_linear_a1.41 | emd              |  0.0990 | [+0.042, +0.156] |      0.1358 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | raw              |  0.2010 | [+0.102, +0.300] |      0.0426 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41 | raw              |  0.2330 | [+0.116, +0.351] |      0.0505 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | emd              | raw              |  0.1350 | [+0.055, +0.214] |      0.1523 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | emd              |  0.0570 | [+0.033, +0.082] |      0.0123 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0110 | [-0.037, +0.015] |      1.0000 |            0.6089 | 4/0/11 |
| comp+log_scale | qda          | imf_linear_a1.41 | emd              |  0.0680 | [+0.033, +0.103] |      0.0546 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | raw              |  0.0630 | [+0.020, +0.105] |      0.3601 |            0.0084 | 14/0/1 |
| comp+log_scale | qda          | imf_linear_a1.41 | raw              |  0.0730 | [+0.014, +0.133] |      0.7153 |            0.0154 | 13/0/2 |
| comp+log_scale | qda          | emd              | raw              |  0.0050 | [-0.036, +0.046] |      1.0000 |            1.0000 | 9/0/6  |
| comp+log_scale | gnb          | irmf_h2_a1.41    | emd              |  0.0510 | [+0.021, +0.081] |      0.1576 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0200 | [-0.047, +0.007] |      1.0000 |            0.1141 | 2/0/13 |
| comp+log_scale | gnb          | imf_linear_a1.41 | emd              |  0.0710 | [+0.022, +0.119] |      0.3601 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | raw              |  0.0380 | [+0.000, +0.076] |      1.0000 |            0.0195 | 14/0/1 |
| comp+log_scale | gnb          | imf_linear_a1.41 | raw              |  0.0580 | [+0.000, +0.116] |      1.0000 |            0.0154 | 13/0/2 |
| comp+log_scale | gnb          | emd              | raw              | -0.0120 | [-0.037, +0.012] |      1.0000 |            0.3580 | 4/0/11 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | emd              |  0.0400 | [+0.013, +0.067] |      0.3278 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0160 | [-0.050, +0.018] |      1.0000 |            0.4951 | 3/0/12 |
| comp+log_scale | logreg       | imf_linear_a1.41 | emd              |  0.0560 | [+0.015, +0.098] |      0.4546 |            0.0134 | 13/0/2 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | raw              |  0.0530 | [-0.011, +0.118] |      1.0000 |            0.0854 | 12/0/3 |
| comp+log_scale | logreg       | imf_linear_a1.41 | raw              |  0.0700 | [-0.012, +0.151] |      1.0000 |            0.0705 | 13/0/2 |
| comp+log_scale | logreg       | emd              | raw              |  0.0130 | [-0.037, +0.063] |      1.0000 |            1.0000 | 10/0/5 |
| comp+log_scale | knn          | irmf_h2_a1.41    | emd              |  0.0960 | [+0.050, +0.143] |      0.0359 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0040 | [-0.036, +0.029] |      1.0000 |            1.0000 | 8/0/7  |
| comp+log_scale | knn          | imf_linear_a1.41 | emd              |  0.1000 | [+0.072, +0.129] |      0.0002 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | raw              |  0.0170 | [-0.035, +0.069] |      1.0000 |            0.7490 | 10/0/5 |
| comp+log_scale | knn          | imf_linear_a1.41 | raw              |  0.0200 | [-0.023, +0.064] |      1.0000 |            0.6372 | 11/0/4 |
| comp+log_scale | knn          | emd              | raw              | -0.0800 | [-0.126, -0.034] |      0.1387 |            0.0044 | 0/0/15 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | emd              |  0.0380 | [+0.002, +0.074] |      1.0000 |            0.0248 | 13/0/2 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0260 | [-0.056, +0.004] |      1.0000 |            0.0443 | 2/0/13 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | emd              |  0.0640 | [+0.020, +0.108] |      0.3619 |            0.0134 | 13/0/2 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | raw              |  0.0740 | [+0.010, +0.137] |      0.8874 |            0.0195 | 13/0/2 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | raw              |  0.1000 | [+0.027, +0.172] |      0.4546 |            0.0134 | 13/0/2 |
| comp+log_scale | linsvm       | emd              | raw              |  0.0350 | [-0.005, +0.076] |      1.0000 |            0.0313 | 12/0/3 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method           |   max_eps2 |   max_mi |   n_features |
|:-----------------|-----------:|---------:|-------------:|
| raw              |      0.051 |    0.113 |       21.000 |
| emd              |      0.101 |    0.060 |      181.000 |
| imf_linear_a1.41 |      0.184 |    0.133 |      361.000 |
| irmf_h2_a1.41    |      0.188 |    0.143 |      361.000 |

| method           | feature              | slot_label   |   epsilon2 |    mi |
|:-----------------|:---------------------|:-------------|-----------:|------:|
| irmf_h2_a1.41    | ch0_c8_hilb_if_mean  | w65          |      0.188 | 0.143 |
| imf_linear_a1.41 | ch0_c5_energy_frac   | w25          |      0.184 | 0.101 |
| irmf_h2_a1.41    | ch0_c8_spec_centroid | w65          |      0.178 | 0.142 |
| imf_linear_a1.41 | ch0_c8_hilb_if_mean  | w65          |      0.178 | 0.127 |
| irmf_h2_a1.41    | ch0_c8_zcr           | w65          |      0.176 | 0.121 |
| irmf_h2_a1.41    | ch0_c5_energy_frac   | w25          |      0.176 | 0.096 |
| irmf_h2_a1.41    | ch0_c9_zcr           | w93          |      0.173 | 0.109 |
| imf_linear_a1.41 | ch0_c8_spec_centroid | w65          |      0.169 | 0.129 |
| imf_linear_a1.41 | ch0_c8_zcr           | w65          |      0.168 | 0.118 |
| imf_linear_a1.41 | ch0_c4_energy_frac   | w17          |      0.160 | 0.087 |
| emd              | ch0_c3_hilb_amp_mean | imf4         |      0.101 | 0.058 |
| emd              | ch0_c3_mean_abs      | imf4         |      0.101 | 0.058 |
| emd              | ch0_c3_log_energy    | imf4         |      0.098 | 0.056 |
| emd              | ch0_c3_std           | imf4         |      0.098 | 0.056 |
| emd              | ch0_c3_hilb_amp_std  | imf4         |      0.083 | 0.050 |
| raw              | ch0_c0_hilb_if_mean  | frame        |      0.051 | 0.085 |
| raw              | ch0_c0_spec_peak     | frame        |      0.050 | 0.113 |
| raw              | ch0_c0_zcr           | frame        |      0.045 | 0.070 |
| raw              | ch0_c0_spec_centroid | frame        |      0.042 | 0.083 |
| raw              | ch0_c0_log_energy    | frame        |      0.028 | 0.019 |
