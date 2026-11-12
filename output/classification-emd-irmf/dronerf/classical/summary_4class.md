# Classical baselines: dronerf / 4class

12000 frames, 227 groups, 4 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 1902 s with 3 workers.

## Decomposition cost

| method           |   slots |   cpu_s_per_frame |   failed | emd_imf_counts                                | irmf_all_stages_converged   |
|:-----------------|--------:|------------------:|---------:|:----------------------------------------------|:----------------------------|
| emd              |       9 |            0.0294 |        0 | {'7': 518, '8': 13095, '9': 10226, '10': 161} | nan                         |
| imf_linear_a1.41 |      18 |            0.0126 |        0 | nan                                           | nan                         |
| irmf_h2_a1.41    |      18 |            1.2787 |        0 | nan                                           | 21985/24000                 |

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.435 ± 0.011 | 0.769 ± 0.015 | 0.656 ± 0.021 | 0.685 ± 0.020 | 0.680 ± 0.017 | 0.498 ± 0.018 |
| emd              | 0.426 ± 0.009 | 0.576 ± 0.016 | 0.704 ± 0.024 | 0.737 ± 0.019 | 0.736 ± 0.017 | 0.461 ± 0.015 |
| imf_linear_a1.41 | 0.630 ± 0.023 | 0.822 ± 0.013 | 0.836 ± 0.015 | 0.850 ± 0.014 | 0.846 ± 0.015 | 0.424 ± 0.015 |
| irmf_h2_a1.41    | 0.586 ± 0.014 | 0.762 ± 0.011 | 0.786 ± 0.014 | 0.809 ± 0.014 | 0.803 ± 0.014 | 0.428 ± 0.018 |

Feature set `comp+log_scale`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.557 ± 0.014 | 0.827 ± 0.019 | 0.724 ± 0.026 | 0.731 ± 0.025 | 0.743 ± 0.021 | 0.523 ± 0.018 |
| emd              | 0.428 ± 0.010 | 0.601 ± 0.014 | 0.764 ± 0.024 | 0.799 ± 0.021 | 0.795 ± 0.018 | 0.464 ± 0.015 |
| imf_linear_a1.41 | 0.630 ± 0.023 | 0.825 ± 0.014 | 0.864 ± 0.019 | 0.874 ± 0.015 | 0.868 ± 0.016 | 0.431 ± 0.015 |
| irmf_h2_a1.41    | 0.586 ± 0.014 | 0.771 ± 0.012 | 0.849 ± 0.017 | 0.862 ± 0.016 | 0.855 ± 0.014 | 0.434 ± 0.019 |

## Best classifier per method

| feature_set    | method           | classifier   |   balanced_accuracy |
|:---------------|:-----------------|:-------------|--------------------:|
| comp           | emd              | linsvm       |               0.737 |
| comp           | imf_linear_a1.41 | linsvm       |               0.850 |
| comp           | irmf_h2_a1.41    | linsvm       |               0.809 |
| comp           | raw              | knn          |               0.769 |
| comp+log_scale | emd              | linsvm       |               0.799 |
| comp+log_scale | imf_linear_a1.41 | linsvm       |               0.874 |
| comp+log_scale | irmf_h2_a1.41    | linsvm       |               0.862 |
| comp+log_scale | raw              | knn          |               0.827 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a         | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-----------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41    | emd              |  0.0820 | [+0.066, +0.098] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0500 | [-0.061, -0.038] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | lda          | imf_linear_a1.41 | emd              |  0.1320 | [+0.113, +0.150] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | raw              |  0.1300 | [+0.115, +0.146] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41 | raw              |  0.1800 | [+0.163, +0.196] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | emd              | raw              |  0.0480 | [+0.032, +0.064] |      0.0003 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | emd              | -0.0330 | [-0.047, -0.019] |      0.0022 |            0.0044 | 0/0/15 |
| comp           | qda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0040 | [-0.010, +0.019] |      1.0000 |            0.8306 | 9/0/6  |
| comp           | qda          | imf_linear_a1.41 | emd              | -0.0370 | [-0.056, -0.018] |      0.0114 |            0.0044 | 0/0/15 |
| comp           | qda          | irmf_h2_a1.41    | raw              | -0.0700 | [-0.087, -0.053] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | qda          | imf_linear_a1.41 | raw              | -0.0740 | [-0.098, -0.050] |      0.0002 |            0.0044 | 0/0/15 |
| comp           | qda          | emd              | raw              | -0.0370 | [-0.047, -0.027] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | gnb          | irmf_h2_a1.41    | emd              |  0.1600 | [+0.145, +0.176] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0440 | [-0.060, -0.027] |      0.0007 |            0.0044 | 0/0/15 |
| comp           | gnb          | imf_linear_a1.41 | emd              |  0.2040 | [+0.176, +0.231] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | raw              |  0.1510 | [+0.141, +0.161] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | gnb          | imf_linear_a1.41 | raw              |  0.1940 | [+0.173, +0.216] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | gnb          | emd              | raw              | -0.0100 | [-0.019, +0.000] |      0.2841 |            0.0044 | 1/0/14 |
| comp           | logreg       | irmf_h2_a1.41    | emd              |  0.0670 | [+0.053, +0.082] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0420 | [-0.054, -0.030] |      0.0001 |            0.0044 | 0/0/15 |
| comp           | logreg       | imf_linear_a1.41 | emd              |  0.1100 | [+0.096, +0.123] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | raw              |  0.1230 | [+0.109, +0.137] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41 | raw              |  0.1650 | [+0.148, +0.183] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | emd              | raw              |  0.0560 | [+0.041, +0.070] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | emd              |  0.1860 | [+0.165, +0.207] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0600 | [-0.072, -0.049] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | knn          | imf_linear_a1.41 | emd              |  0.2460 | [+0.227, +0.266] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | raw              | -0.0070 | [-0.021, +0.007] |      1.0000 |            0.4280 | 6/0/9  |
| comp           | knn          | imf_linear_a1.41 | raw              |  0.0540 | [+0.041, +0.066] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | knn          | emd              | raw              | -0.1930 | [-0.212, -0.173] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | linsvm       | irmf_h2_a1.41    | emd              |  0.0720 | [+0.057, +0.087] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0410 | [-0.051, -0.031] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | linsvm       | imf_linear_a1.41 | emd              |  0.1130 | [+0.098, +0.128] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | raw              |  0.1240 | [+0.109, +0.139] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41 | raw              |  0.1650 | [+0.148, +0.182] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | emd              | raw              |  0.0520 | [+0.039, +0.065] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | emd              |  0.0850 | [+0.071, +0.099] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0160 | [-0.024, -0.007] |      0.0118 |            0.0044 | 0/0/15 |
| comp+log_scale | lda          | imf_linear_a1.41 | emd              |  0.1000 | [+0.084, +0.116] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | raw              |  0.1250 | [+0.106, +0.144] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41 | raw              |  0.1400 | [+0.120, +0.160] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | emd              | raw              |  0.0400 | [+0.029, +0.052] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | emd              | -0.0310 | [-0.044, -0.018] |      0.0022 |            0.0044 | 0/0/15 |
| comp+log_scale | qda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0030 | [-0.011, +0.016] |      1.0000 |            1.0000 | 9/0/6  |
| comp+log_scale | qda          | imf_linear_a1.41 | emd              | -0.0330 | [-0.051, -0.016] |      0.0114 |            0.0044 | 0/0/15 |
| comp+log_scale | qda          | irmf_h2_a1.41    | raw              | -0.0890 | [-0.107, -0.071] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | qda          | imf_linear_a1.41 | raw              | -0.0920 | [-0.115, -0.069] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | qda          | emd              | raw              | -0.0580 | [-0.072, -0.045] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | emd              |  0.1580 | [+0.143, +0.173] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0430 | [-0.060, -0.027] |      0.0008 |            0.0044 | 0/0/15 |
| comp+log_scale | gnb          | imf_linear_a1.41 | emd              |  0.2020 | [+0.175, +0.229] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | raw              |  0.0290 | [+0.010, +0.047] |      0.0282 |            0.0044 | 14/0/1 |
| comp+log_scale | gnb          | imf_linear_a1.41 | raw              |  0.0720 | [+0.041, +0.104] |      0.0026 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | emd              | raw              | -0.1300 | [-0.144, -0.115] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | emd              |  0.0590 | [+0.047, +0.072] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0130 | [-0.020, -0.006] |      0.0130 |            0.0044 | 0/0/15 |
| comp+log_scale | logreg       | imf_linear_a1.41 | emd              |  0.0730 | [+0.060, +0.085] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | raw              |  0.1110 | [+0.098, +0.125] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41 | raw              |  0.1250 | [+0.110, +0.139] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | emd              | raw              |  0.0520 | [+0.039, +0.065] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | emd              |  0.1700 | [+0.149, +0.191] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0540 | [-0.066, -0.042] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | knn          | imf_linear_a1.41 | emd              |  0.2240 | [+0.204, +0.245] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | raw              | -0.0560 | [-0.075, -0.038] |      0.0002 |            0.0044 | 0/0/15 |
| comp+log_scale | knn          | imf_linear_a1.41 | raw              | -0.0020 | [-0.017, +0.013] |      1.0000 |            1.0000 | 8/0/7  |
| comp+log_scale | knn          | emd              | raw              | -0.2260 | [-0.249, -0.203] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | emd              |  0.0630 | [+0.050, +0.077] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0120 | [-0.019, -0.005] |      0.0130 |            0.0044 | 1/0/14 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | emd              |  0.0760 | [+0.062, +0.089] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | raw              |  0.1310 | [+0.112, +0.150] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | raw              |  0.1430 | [+0.123, +0.163] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | emd              | raw              |  0.0680 | [+0.055, +0.080] |      0.0000 |            0.0044 | 15/0/0 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method           |   max_eps2 |   max_mi |   n_features |
|:-----------------|-----------:|---------:|-------------:|
| raw              |      0.241 |    0.339 |       42.000 |
| emd              |      0.267 |    0.339 |      362.000 |
| imf_linear_a1.41 |      0.369 |    0.339 |      722.000 |
| irmf_h2_a1.41    |      0.310 |    0.339 |      722.000 |

| method           | feature              | slot_label   |   epsilon2 |    mi |
|:-----------------|:---------------------|:-------------|-----------:|------:|
| imf_linear_a1.41 | ch0_c10_spec_bw      | w129         |      0.369 | 0.337 |
| imf_linear_a1.41 | ch0_c10_hilb_if_std  | w129         |      0.369 | 0.318 |
| imf_linear_a1.41 | ch0_c9_hilb_if_std   | w93          |      0.368 | 0.325 |
| imf_linear_a1.41 | ch0_c7_hilb_if_mean  | w47          |      0.361 | 0.335 |
| imf_linear_a1.41 | ch0_c8_hjorth_mob    | w65          |      0.355 | 0.319 |
| irmf_h2_a1.41    | ch0_c9_hilb_if_std   | w93          |      0.310 | 0.274 |
| irmf_h2_a1.41    | ch0_c10_hilb_if_std  | w129         |      0.309 | 0.255 |
| irmf_h2_a1.41    | ch0_c8_hjorth_mob    | w65          |      0.308 | 0.264 |
| irmf_h2_a1.41    | ch0_c9_hjorth_mob    | w93          |      0.307 | 0.253 |
| irmf_h2_a1.41    | ch0_c7_hjorth_mob    | w47          |      0.305 | 0.257 |
| emd              | ch0_c3_energy_frac   | imf4         |      0.267 | 0.233 |
| emd              | ch0_c4_energy_frac   | imf5         |      0.258 | 0.217 |
| emd              | ch0_c0_spec_centroid | imf1         |      0.246 | 0.252 |
| emd              | ch0_c5_energy_frac   | imf6         |      0.246 | 0.194 |
| emd              | ch0_c0_hjorth_mob    | imf1         |      0.244 | 0.249 |
| raw              | ch0_c0_spec_bw       | frame        |      0.241 | 0.243 |
| raw              | ch0_c0_hjorth_cpx    | frame        |      0.217 | 0.264 |
| raw              | ch0_c0_hilb_if_std   | frame        |      0.211 | 0.287 |
| raw              | ch0_c0_extrema_rate  | frame        |      0.182 | 0.215 |
| raw              | ch0_c0_hilb_if_mean  | frame        |      0.181 | 0.245 |
