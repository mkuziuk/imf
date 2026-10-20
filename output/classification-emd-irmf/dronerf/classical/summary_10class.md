# Classical baselines: dronerf / 10class

12000 frames, 227 groups, 10 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 2879 s with 3 workers.

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
| raw              | 0.197 ± 0.009 | 0.419 ± 0.015 | 0.332 ± 0.011 | 0.349 ± 0.014 | 0.354 ± 0.013 | 0.212 ± 0.006 |
| emd              | 0.188 ± 0.007 | 0.276 ± 0.009 | 0.355 ± 0.013 | 0.387 ± 0.012 | 0.388 ± 0.013 | 0.200 ± 0.010 |
| imf_linear_a1.41 | 0.262 ± 0.007 | 0.471 ± 0.011 | 0.518 ± 0.017 | 0.575 ± 0.012 | 0.567 ± 0.012 | 0.190 ± 0.012 |
| irmf_h2_a1.41    | 0.237 ± 0.008 | 0.401 ± 0.010 | 0.450 ± 0.012 | 0.495 ± 0.010 | 0.493 ± 0.010 | 0.179 ± 0.012 |

Feature set `comp+log_scale`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.211 ± 0.012 | 0.477 ± 0.012 | 0.364 ± 0.014 | 0.390 ± 0.014 | 0.398 ± 0.016 | 0.224 ± 0.006 |
| emd              | 0.199 ± 0.007 | 0.284 ± 0.009 | 0.396 ± 0.013 | 0.429 ± 0.009 | 0.429 ± 0.012 | 0.202 ± 0.010 |
| imf_linear_a1.41 | 0.262 ± 0.007 | 0.474 ± 0.012 | 0.537 ± 0.017 | 0.595 ± 0.012 | 0.589 ± 0.011 | 0.193 ± 0.011 |
| irmf_h2_a1.41    | 0.237 ± 0.008 | 0.406 ± 0.009 | 0.502 ± 0.012 | 0.536 ± 0.011 | 0.536 ± 0.011 | 0.181 ± 0.012 |

## Best classifier per method

| feature_set    | method           | classifier   |   balanced_accuracy |
|:---------------|:-----------------|:-------------|--------------------:|
| comp           | emd              | logreg       |               0.388 |
| comp           | imf_linear_a1.41 | linsvm       |               0.575 |
| comp           | irmf_h2_a1.41    | linsvm       |               0.495 |
| comp           | raw              | knn          |               0.419 |
| comp+log_scale | emd              | linsvm       |               0.429 |
| comp+log_scale | imf_linear_a1.41 | linsvm       |               0.595 |
| comp+log_scale | irmf_h2_a1.41    | logreg       |               0.536 |
| comp+log_scale | raw              | knn          |               0.477 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a         | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-----------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41    | emd              |  0.0940 | [+0.084, +0.104] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0680 | [-0.085, -0.051] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | lda          | imf_linear_a1.41 | emd              |  0.1630 | [+0.144, +0.182] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | raw              |  0.1180 | [+0.106, +0.130] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41 | raw              |  0.1870 | [+0.160, +0.213] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | emd              | raw              |  0.0240 | [+0.010, +0.038] |      0.0282 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | emd              | -0.0210 | [-0.027, -0.015] |      0.0001 |            0.0044 | 0/0/15 |
| comp           | qda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0110 | [-0.018, -0.003] |      0.0590 |            0.0044 | 0/0/15 |
| comp           | qda          | imf_linear_a1.41 | emd              | -0.0110 | [-0.020, -0.002] |      0.1271 |            0.0044 | 1/0/14 |
| comp           | qda          | irmf_h2_a1.41    | raw              | -0.0330 | [-0.046, -0.021] |      0.0011 |            0.0044 | 0/0/15 |
| comp           | qda          | imf_linear_a1.41 | raw              | -0.0230 | [-0.036, -0.009] |      0.0282 |            0.0044 | 0/0/15 |
| comp           | qda          | emd              | raw              | -0.0120 | [-0.023, -0.001] |      0.1468 |            0.0044 | 0/0/15 |
| comp           | gnb          | irmf_h2_a1.41    | emd              |  0.0490 | [+0.037, +0.061] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0250 | [-0.031, -0.020] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | gnb          | imf_linear_a1.41 | emd              |  0.0740 | [+0.062, +0.086] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | raw              |  0.0400 | [+0.030, +0.050] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | gnb          | imf_linear_a1.41 | raw              |  0.0650 | [+0.055, +0.076] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | gnb          | emd              | raw              | -0.0090 | [-0.018, +0.001] |      0.1951 |            0.0044 | 2/0/13 |
| comp           | logreg       | irmf_h2_a1.41    | emd              |  0.1060 | [+0.091, +0.120] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0740 | [-0.083, -0.065] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | logreg       | imf_linear_a1.41 | emd              |  0.1790 | [+0.160, +0.199] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | raw              |  0.1400 | [+0.121, +0.158] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41 | raw              |  0.2140 | [+0.190, +0.237] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | emd              | raw              |  0.0340 | [+0.022, +0.047] |      0.0009 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | emd              |  0.1260 | [+0.112, +0.139] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0700 | [-0.086, -0.053] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | knn          | imf_linear_a1.41 | emd              |  0.1950 | [+0.183, +0.208] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | raw              | -0.0180 | [-0.033, -0.004] |      0.1188 |            0.0044 | 0/0/15 |
| comp           | knn          | imf_linear_a1.41 | raw              |  0.0510 | [+0.030, +0.072] |      0.0023 |            0.0044 | 15/0/0 |
| comp           | knn          | emd              | raw              | -0.1440 | [-0.160, -0.127] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | linsvm       | irmf_h2_a1.41    | emd              |  0.1080 | [+0.099, +0.117] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0790 | [-0.092, -0.067] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | linsvm       | imf_linear_a1.41 | emd              |  0.1870 | [+0.171, +0.204] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | raw              |  0.1460 | [+0.130, +0.162] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41 | raw              |  0.2250 | [+0.203, +0.247] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | emd              | raw              |  0.0380 | [+0.022, +0.053] |      0.0022 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | emd              |  0.1060 | [+0.092, +0.119] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0350 | [-0.050, -0.020] |      0.0026 |            0.0044 | 0/0/15 |
| comp+log_scale | lda          | imf_linear_a1.41 | emd              |  0.1410 | [+0.121, +0.160] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | raw              |  0.1370 | [+0.122, +0.152] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41 | raw              |  0.1720 | [+0.148, +0.197] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | emd              | raw              |  0.0320 | [+0.016, +0.048] |      0.0104 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | emd              | -0.0220 | [-0.028, -0.015] |      0.0001 |            0.0044 | 0/0/15 |
| comp+log_scale | qda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0120 | [-0.020, -0.005] |      0.0361 |            0.0044 | 0/0/15 |
| comp+log_scale | qda          | imf_linear_a1.41 | emd              | -0.0090 | [-0.018, -0.000] |      0.1632 |            0.0044 | 2/0/13 |
| comp+log_scale | qda          | irmf_h2_a1.41    | raw              | -0.0430 | [-0.058, -0.028] |      0.0005 |            0.0044 | 0/0/15 |
| comp+log_scale | qda          | imf_linear_a1.41 | raw              | -0.0310 | [-0.045, -0.017] |      0.0051 |            0.0044 | 0/0/15 |
| comp+log_scale | qda          | emd              | raw              | -0.0220 | [-0.034, -0.009] |      0.0224 |            0.0044 | 0/0/15 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | emd              |  0.0380 | [+0.026, +0.050] |      0.0002 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0250 | [-0.031, -0.020] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | gnb          | imf_linear_a1.41 | emd              |  0.0630 | [+0.051, +0.076] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | raw              |  0.0260 | [+0.013, +0.039] |      0.0084 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | imf_linear_a1.41 | raw              |  0.0510 | [+0.036, +0.066] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | emd              | raw              | -0.0120 | [-0.026, +0.002] |      0.1951 |            0.0044 | 0/0/15 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | emd              |  0.1070 | [+0.090, +0.124] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0530 | [-0.062, -0.044] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | logreg       | imf_linear_a1.41 | emd              |  0.1600 | [+0.143, +0.178] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | raw              |  0.1380 | [+0.120, +0.157] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41 | raw              |  0.1910 | [+0.174, +0.209] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | emd              | raw              |  0.0310 | [+0.016, +0.047] |      0.0104 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | emd              |  0.1220 | [+0.108, +0.136] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0670 | [-0.084, -0.051] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | knn          | imf_linear_a1.41 | emd              |  0.1890 | [+0.175, +0.204] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | raw              | -0.0710 | [-0.085, -0.056] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | knn          | imf_linear_a1.41 | raw              | -0.0030 | [-0.018, +0.011] |      0.6489 |            0.3303 | 6/0/9  |
| comp+log_scale | knn          | emd              | raw              | -0.1920 | [-0.206, -0.178] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | emd              |  0.1070 | [+0.098, +0.115] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0590 | [-0.070, -0.048] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | emd              |  0.1660 | [+0.153, +0.178] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | raw              |  0.1450 | [+0.132, +0.159] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | raw              |  0.2040 | [+0.187, +0.221] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | emd              | raw              |  0.0390 | [+0.025, +0.053] |      0.0008 |            0.0044 | 15/0/0 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method           |   max_eps2 |   max_mi |   n_features |
|:-----------------|-----------:|---------:|-------------:|
| raw              |      0.261 |    0.408 |       42.000 |
| emd              |      0.284 |    0.408 |      362.000 |
| imf_linear_a1.41 |      0.381 |    0.423 |      722.000 |
| irmf_h2_a1.41    |      0.318 |    0.408 |      722.000 |

| method           | feature              | slot_label   |   epsilon2 |    mi |
|:-----------------|:---------------------|:-------------|-----------:|------:|
| imf_linear_a1.41 | ch0_c10_spec_bw      | w129         |      0.381 | 0.388 |
| imf_linear_a1.41 | ch0_c7_hilb_if_mean  | w47          |      0.381 | 0.422 |
| imf_linear_a1.41 | ch0_c10_hilb_if_std  | w129         |      0.379 | 0.356 |
| imf_linear_a1.41 | ch0_c9_hilb_if_std   | w93          |      0.379 | 0.356 |
| imf_linear_a1.41 | ch0_c8_hilb_if_mean  | w65          |      0.374 | 0.372 |
| irmf_h2_a1.41    | ch0_c10_hilb_if_std  | w129         |      0.318 | 0.296 |
| irmf_h2_a1.41    | ch0_c9_hilb_if_std   | w93          |      0.318 | 0.317 |
| irmf_h2_a1.41    | ch0_c8_hjorth_mob    | w65          |      0.317 | 0.299 |
| irmf_h2_a1.41    | ch0_c9_hjorth_mob    | w93          |      0.316 | 0.285 |
| irmf_h2_a1.41    | ch0_c7_hjorth_mob    | w47          |      0.314 | 0.304 |
| emd              | ch0_c3_energy_frac   | imf4         |      0.284 | 0.266 |
| emd              | ch0_c4_energy_frac   | imf5         |      0.273 | 0.239 |
| emd              | ch0_c0_hjorth_mob    | imf1         |      0.263 | 0.330 |
| emd              | ch0_c0_spec_centroid | imf1         |      0.263 | 0.321 |
| raw              | ch0_c0_spec_bw       | frame        |      0.261 | 0.320 |
| emd              | ch0_c5_energy_frac   | imf6         |      0.260 | 0.211 |
| raw              | ch0_c0_hjorth_cpx    | frame        |      0.237 | 0.353 |
| raw              | ch0_c0_hilb_if_std   | frame        |      0.229 | 0.344 |
| raw              | ch0_c0_extrema_rate  | frame        |      0.204 | 0.247 |
| raw              | ch0_c0_hilb_if_mean  | frame        |      0.195 | 0.317 |
