# Classical baselines: birdclef2023 / species

12000 frames, 451 groups, 16 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 1584 s with 3 workers.

## Decomposition cost

| method           |   slots |   cpu_s_per_frame |   failed | emd_imf_counts                                                          | irmf_all_stages_converged   |
|:-----------------|--------:|------------------:|---------:|:------------------------------------------------------------------------|:----------------------------|
| emd              |       9 |            0.0116 |        0 | {'4': 11, '5': 207, '6': 1620, '7': 5594, '8': 4113, '9': 451, '10': 4} | nan                         |
| imf_linear_a1.41 |      18 |            0.0063 |        0 | nan                                                                     | nan                         |
| irmf_h2_a1.41    |      18 |            1.3205 |        0 | nan                                                                     | 7888/12000                  |

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.249 ± 0.034 | 0.423 ± 0.012 | 0.340 ± 0.009 | 0.337 ± 0.010 | 0.376 ± 0.011 | 0.218 ± 0.021 |
| emd              | 0.201 ± 0.018 | 0.322 ± 0.019 | 0.403 ± 0.018 | 0.420 ± 0.015 | 0.435 ± 0.016 | 0.195 ± 0.017 |
| imf_linear_a1.41 | 0.298 ± 0.033 | 0.392 ± 0.019 | 0.477 ± 0.021 | 0.493 ± 0.019 | 0.489 ± 0.019 | 0.262 ± 0.038 |
| irmf_h2_a1.41    | 0.307 ± 0.038 | 0.417 ± 0.024 | 0.474 ± 0.022 | 0.496 ± 0.019 | 0.495 ± 0.018 | 0.260 ± 0.028 |

Feature set `comp+log_scale`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.249 ± 0.036 | 0.404 ± 0.011 | 0.343 ± 0.011 | 0.343 ± 0.011 | 0.376 ± 0.017 | 0.224 ± 0.025 |
| emd              | 0.201 ± 0.019 | 0.324 ± 0.018 | 0.405 ± 0.019 | 0.420 ± 0.011 | 0.433 ± 0.016 | 0.196 ± 0.017 |
| imf_linear_a1.41 | 0.298 ± 0.033 | 0.391 ± 0.019 | 0.476 ± 0.020 | 0.493 ± 0.017 | 0.486 ± 0.017 | 0.261 ± 0.038 |
| irmf_h2_a1.41    | 0.307 ± 0.038 | 0.418 ± 0.024 | 0.475 ± 0.021 | 0.497 ± 0.017 | 0.494 ± 0.017 | 0.260 ± 0.029 |

## Best classifier per method

| feature_set    | method           | classifier   |   balanced_accuracy |
|:---------------|:-----------------|:-------------|--------------------:|
| comp           | emd              | logreg       |               0.435 |
| comp           | imf_linear_a1.41 | linsvm       |               0.493 |
| comp           | irmf_h2_a1.41    | linsvm       |               0.496 |
| comp           | raw              | knn          |               0.423 |
| comp+log_scale | emd              | logreg       |               0.433 |
| comp+log_scale | imf_linear_a1.41 | linsvm       |               0.493 |
| comp+log_scale | irmf_h2_a1.41    | linsvm       |               0.497 |
| comp+log_scale | raw              | knn          |               0.404 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a         | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-----------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41    | emd              |  0.0710 | [+0.047, +0.095] |      0.0007 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0030 | [-0.015, +0.009] |      1.0000 |            1.0000 | 6/0/9  |
| comp           | lda          | imf_linear_a1.41 | emd              |  0.0740 | [+0.054, +0.094] |      0.0001 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | raw              |  0.1330 | [+0.105, +0.162] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41 | raw              |  0.1360 | [+0.110, +0.163] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | emd              | raw              |  0.0630 | [+0.043, +0.083] |      0.0004 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | emd              |  0.0650 | [+0.031, +0.099] |      0.0266 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0010 | [-0.034, +0.031] |      1.0000 |            1.0000 | 9/0/6  |
| comp           | qda          | imf_linear_a1.41 | emd              |  0.0660 | [+0.021, +0.112] |      0.1418 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | raw              |  0.0430 | [+0.023, +0.062] |      0.0099 |            0.0044 | 15/0/0 |
| comp           | qda          | imf_linear_a1.41 | raw              |  0.0440 | [+0.010, +0.078] |      0.2498 |            0.0044 | 15/0/0 |
| comp           | qda          | emd              | raw              | -0.0230 | [-0.052, +0.007] |      1.0000 |            0.0470 | 3/0/12 |
| comp           | gnb          | irmf_h2_a1.41    | emd              |  0.1070 | [+0.069, +0.144] |      0.0009 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0090 | [-0.013, +0.032] |      1.0000 |            0.6089 | 12/0/3 |
| comp           | gnb          | imf_linear_a1.41 | emd              |  0.0970 | [+0.062, +0.132] |      0.0012 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | raw              |  0.0590 | [+0.042, +0.075] |      0.0001 |            0.0044 | 15/0/0 |
| comp           | gnb          | imf_linear_a1.41 | raw              |  0.0490 | [+0.022, +0.077] |      0.0424 |            0.0044 | 15/0/0 |
| comp           | gnb          | emd              | raw              | -0.0480 | [-0.081, -0.015] |      0.1496 |            0.0044 | 0/0/15 |
| comp           | logreg       | irmf_h2_a1.41    | emd              |  0.0610 | [+0.047, +0.074] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0060 | [-0.006, +0.019] |      1.0000 |            0.3069 | 10/0/5 |
| comp           | logreg       | imf_linear_a1.41 | emd              |  0.0540 | [+0.042, +0.066] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | raw              |  0.1190 | [+0.098, +0.139] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41 | raw              |  0.1120 | [+0.092, +0.133] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | emd              | raw              |  0.0580 | [+0.043, +0.073] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | emd              |  0.0950 | [+0.076, +0.114] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0250 | [+0.013, +0.038] |      0.0191 |            0.0044 | 15/0/0 |
| comp           | knn          | imf_linear_a1.41 | emd              |  0.0700 | [+0.055, +0.084] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | raw              | -0.0060 | [-0.027, +0.015] |      1.0000 |            1.0000 | 4/0/11 |
| comp           | knn          | imf_linear_a1.41 | raw              | -0.0320 | [-0.049, -0.014] |      0.0424 |            0.0044 | 0/0/15 |
| comp           | knn          | emd              | raw              | -0.1010 | [-0.122, -0.080] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | linsvm       | irmf_h2_a1.41    | emd              |  0.0760 | [+0.054, +0.097] |      0.0001 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0030 | [-0.010, +0.015] |      1.0000 |            1.0000 | 10/0/5 |
| comp           | linsvm       | imf_linear_a1.41 | emd              |  0.0730 | [+0.055, +0.090] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | raw              |  0.1580 | [+0.128, +0.189] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41 | raw              |  0.1550 | [+0.128, +0.182] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | emd              | raw              |  0.0830 | [+0.063, +0.103] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | emd              |  0.0700 | [+0.046, +0.093] |      0.0006 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0020 | [-0.014, +0.010] |      1.0000 |            1.0000 | 7/0/8  |
| comp+log_scale | lda          | imf_linear_a1.41 | emd              |  0.0720 | [+0.052, +0.091] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | raw              |  0.1320 | [+0.113, +0.152] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41 | raw              |  0.1340 | [+0.117, +0.151] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | emd              | raw              |  0.0620 | [+0.048, +0.077] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | emd              |  0.0650 | [+0.033, +0.097] |      0.0191 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0010 | [-0.032, +0.030] |      1.0000 |            1.0000 | 9/0/6  |
| comp+log_scale | qda          | imf_linear_a1.41 | emd              |  0.0660 | [+0.022, +0.110] |      0.1371 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | raw              |  0.0370 | [+0.016, +0.058] |      0.0438 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | imf_linear_a1.41 | raw              |  0.0380 | [+0.009, +0.067] |      0.2380 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | emd              | raw              | -0.0280 | [-0.060, +0.004] |      1.0000 |            0.0174 | 3/0/12 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | emd              |  0.1050 | [+0.068, +0.143] |      0.0009 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0090 | [-0.014, +0.032] |      1.0000 |            0.6570 | 12/0/3 |
| comp+log_scale | gnb          | imf_linear_a1.41 | emd              |  0.0970 | [+0.062, +0.132] |      0.0012 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | raw              |  0.0580 | [+0.043, +0.074] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | imf_linear_a1.41 | raw              |  0.0490 | [+0.022, +0.076] |      0.0423 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | emd              | raw              | -0.0470 | [-0.083, -0.012] |      0.2231 |            0.0044 | 0/0/15 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | emd              |  0.0600 | [+0.045, +0.076] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0080 | [-0.004, +0.020] |      1.0000 |            0.1087 | 13/0/2 |
| comp+log_scale | logreg       | imf_linear_a1.41 | emd              |  0.0520 | [+0.040, +0.064] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | raw              |  0.1180 | [+0.100, +0.137] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41 | raw              |  0.1100 | [+0.093, +0.127] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | emd              | raw              |  0.0580 | [+0.043, +0.073] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | emd              |  0.0940 | [+0.077, +0.110] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0260 | [+0.014, +0.039] |      0.0133 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | imf_linear_a1.41 | emd              |  0.0670 | [+0.056, +0.079] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | raw              |  0.0130 | [-0.017, +0.043] |      1.0000 |            1.0000 | 9/0/6  |
| comp+log_scale | knn          | imf_linear_a1.41 | raw              | -0.0130 | [-0.038, +0.012] |      1.0000 |            0.6089 | 4/0/11 |
| comp+log_scale | knn          | emd              | raw              | -0.0800 | [-0.105, -0.055] |      0.0003 |            0.0044 | 0/0/15 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | emd              |  0.0770 | [+0.061, +0.093] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0040 | [-0.007, +0.015] |      1.0000 |            1.0000 | 11/0/4 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | emd              |  0.0730 | [+0.058, +0.088] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | raw              |  0.1530 | [+0.133, +0.174] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | raw              |  0.1500 | [+0.132, +0.167] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | emd              | raw              |  0.0760 | [+0.061, +0.092] |      0.0000 |            0.0044 | 15/0/0 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method           |   max_eps2 |   max_mi |   n_features |
|:-----------------|-----------:|---------:|-------------:|
| raw              |      0.362 |    0.657 |       21.000 |
| emd              |      0.415 |    0.514 |      181.000 |
| imf_linear_a1.41 |      0.483 |    0.633 |      361.000 |
| irmf_h2_a1.41    |      0.472 |    0.647 |      361.000 |

| method           | feature              | slot_label   |   epsilon2 |    mi |
|:-----------------|:---------------------|:-------------|-----------:|------:|
| imf_linear_a1.41 | ch0_c2_hilb_if_mean  | w9           |      0.483 | 0.456 |
| imf_linear_a1.41 | ch0_c2_zcr           | w9           |      0.477 | 0.439 |
| imf_linear_a1.41 | ch0_c2_spec_centroid | w9           |      0.473 | 0.440 |
| irmf_h2_a1.41    | ch0_c2_hilb_if_mean  | w9           |      0.472 | 0.451 |
| irmf_h2_a1.41    | ch0_c4_extrema_rate  | w17          |      0.470 | 0.479 |
| imf_linear_a1.41 | ch0_c4_extrema_rate  | w17          |      0.464 | 0.496 |
| imf_linear_a1.41 | ch0_c5_energy_frac   | w25          |      0.464 | 0.486 |
| irmf_h2_a1.41    | ch0_c2_spec_centroid | w9           |      0.464 | 0.439 |
| irmf_h2_a1.41    | ch0_c2_zcr           | w9           |      0.464 | 0.430 |
| irmf_h2_a1.41    | ch0_c2_hjorth_mob    | w9           |      0.460 | 0.444 |
| emd              | ch0_c1_energy_frac   | imf2         |      0.415 | 0.301 |
| emd              | ch0_c1_mean_abs      | imf2         |      0.392 | 0.274 |
| emd              | ch0_c1_hilb_amp_mean | imf2         |      0.392 | 0.274 |
| emd              | ch0_c1_log_energy    | imf2         |      0.384 | 0.269 |
| emd              | ch0_c1_std           | imf2         |      0.384 | 0.269 |
| raw              | ch0_c0_extrema_rate  | frame        |      0.362 | 0.405 |
| raw              | ch0_c0_hjorth_mob    | frame        |      0.333 | 0.555 |
| raw              | ch0_c0_zcr           | frame        |      0.324 | 0.522 |
| raw              | ch0_c0_spec_centroid | frame        |      0.314 | 0.564 |
| raw              | ch0_c0_hilb_if_mean  | frame        |      0.313 | 0.571 |
