# Classical baselines: synthetic / signal

1500 frames, 150 groups, 5 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 68 s with 6 workers.

## Decomposition cost

| method           |   slots |   cpu_s_per_frame |   failed | emd_imf_counts                                 | irmf_all_stages_converged   |
|:-----------------|--------:|------------------:|---------:|:-----------------------------------------------|:----------------------------|
| emd              |       9 |            0.0129 |        0 | {'5': 3, '6': 265, '7': 914, '8': 313, '9': 5} | nan                         |
| imf_linear_a1.41 |      16 |            0.0016 |        0 | nan                                            | nan                         |
| irmf_h2_a1.41    |      16 |            0.1007 |        0 | nan                                            | 516/1500                    |

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.286 ± 0.043 | 0.296 ± 0.038 | 0.460 ± 0.041 | 0.460 ± 0.041 | 0.474 ± 0.044 | 0.423 ± 0.051 |
| emd              | 0.245 ± 0.054 | 0.244 ± 0.024 | 0.475 ± 0.041 | 0.505 ± 0.043 | 0.504 ± 0.047 | 0.373 ± 0.052 |
| imf_linear_a1.41 | 0.309 ± 0.080 | 0.352 ± 0.042 | 0.749 ± 0.040 | 0.697 ± 0.037 | 0.730 ± 0.042 | 0.632 ± 0.051 |
| irmf_h2_a1.41    | 0.483 ± 0.074 | 0.440 ± 0.035 | 0.869 ± 0.033 | 0.826 ± 0.028 | 0.848 ± 0.035 | 0.768 ± 0.046 |

Feature set `comp+log_scale`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.290 ± 0.042 | 0.286 ± 0.029 | 0.452 ± 0.051 | 0.450 ± 0.041 | 0.463 ± 0.043 | 0.424 ± 0.052 |
| emd              | 0.244 ± 0.055 | 0.253 ± 0.021 | 0.470 ± 0.050 | 0.499 ± 0.050 | 0.502 ± 0.051 | 0.366 ± 0.053 |
| imf_linear_a1.41 | 0.308 ± 0.080 | 0.358 ± 0.039 | 0.754 ± 0.041 | 0.689 ± 0.032 | 0.732 ± 0.047 | 0.632 ± 0.056 |
| irmf_h2_a1.41    | 0.482 ± 0.074 | 0.445 ± 0.026 | 0.869 ± 0.033 | 0.826 ± 0.033 | 0.849 ± 0.034 | 0.762 ± 0.048 |

## Best classifier per method

| feature_set    | method           | classifier   |   balanced_accuracy |
|:---------------|:-----------------|:-------------|--------------------:|
| comp           | emd              | linsvm       |               0.505 |
| comp           | imf_linear_a1.41 | lda          |               0.749 |
| comp           | irmf_h2_a1.41    | lda          |               0.869 |
| comp           | raw              | logreg       |               0.474 |
| comp+log_scale | emd              | logreg       |               0.502 |
| comp+log_scale | imf_linear_a1.41 | lda          |               0.754 |
| comp+log_scale | irmf_h2_a1.41    | lda          |               0.869 |
| comp+log_scale | raw              | logreg       |               0.463 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a         | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-----------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41    | emd              |  0.3940 | [+0.346, +0.442] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.1190 | [+0.082, +0.156] |      0.0003 |            0.0044 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41 | emd              |  0.2750 | [+0.233, +0.317] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | raw              |  0.4090 | [+0.346, +0.472] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41 | raw              |  0.2900 | [+0.241, +0.338] |      0.0000 |            0.0253 | 15/0/0 |
| comp           | lda          | emd              | raw              |  0.0150 | [-0.046, +0.076] |      1.0000 |            0.9084 | 10/1/4 |
| comp           | qda          | irmf_h2_a1.41    | emd              |  0.3950 | [+0.324, +0.466] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.1370 | [+0.109, +0.165] |      0.0000 |            0.0253 | 15/0/0 |
| comp           | qda          | imf_linear_a1.41 | emd              |  0.2580 | [+0.188, +0.329] |      0.0001 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | raw              |  0.3460 | [+0.272, +0.420] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | qda          | imf_linear_a1.41 | raw              |  0.2090 | [+0.129, +0.288] |      0.0015 |            0.0044 | 15/0/0 |
| comp           | qda          | emd              | raw              | -0.0500 | [-0.139, +0.040] |      1.0000 |            0.2011 | 5/0/10 |
| comp           | gnb          | irmf_h2_a1.41    | emd              |  0.2370 | [+0.149, +0.326] |      0.0013 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.1740 | [+0.120, +0.228] |      0.0003 |            0.0253 | 15/0/0 |
| comp           | gnb          | imf_linear_a1.41 | emd              |  0.0630 | [-0.033, +0.160] |      1.0000 |            0.1370 | 11/0/4 |
| comp           | gnb          | irmf_h2_a1.41    | raw              |  0.1970 | [+0.112, +0.281] |      0.0045 |            0.0253 | 15/0/0 |
| comp           | gnb          | imf_linear_a1.41 | raw              |  0.0230 | [-0.068, +0.113] |      1.0000 |            0.9084 | 10/0/5 |
| comp           | gnb          | emd              | raw              | -0.0410 | [-0.124, +0.043] |      1.0000 |            0.2768 | 4/0/11 |
| comp           | logreg       | irmf_h2_a1.41    | emd              |  0.3440 | [+0.297, +0.391] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.1180 | [+0.083, +0.153] |      0.0001 |            0.0044 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41 | emd              |  0.2260 | [+0.175, +0.277] |      0.0000 |            0.0253 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | raw              |  0.3740 | [+0.322, +0.426] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41 | raw              |  0.2560 | [+0.198, +0.313] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | emd              | raw              |  0.0300 | [-0.030, +0.090] |      1.0000 |            0.2624 | 10/0/5 |
| comp           | knn          | irmf_h2_a1.41    | emd              |  0.1960 | [+0.143, +0.249] |      0.0001 |            0.0253 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0870 | [+0.045, +0.129] |      0.0102 |            0.0253 | 15/0/0 |
| comp           | knn          | imf_linear_a1.41 | emd              |  0.1090 | [+0.058, +0.159] |      0.0087 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | raw              |  0.1440 | [+0.098, +0.190] |      0.0003 |            0.0253 | 15/0/0 |
| comp           | knn          | imf_linear_a1.41 | raw              |  0.0560 | [+0.008, +0.105] |      0.4447 |            0.0253 | 13/0/2 |
| comp           | knn          | emd              | raw              | -0.0520 | [-0.098, -0.006] |      0.4578 |            0.0044 | 0/0/15 |
| comp           | linsvm       | irmf_h2_a1.41    | emd              |  0.3210 | [+0.277, +0.365] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.1300 | [+0.087, +0.172] |      0.0004 |            0.0253 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41 | emd              |  0.1920 | [+0.142, +0.241] |      0.0000 |            0.0253 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | raw              |  0.3660 | [+0.311, +0.421] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41 | raw              |  0.2370 | [+0.179, +0.295] |      0.0000 |            0.0253 | 15/0/0 |
| comp           | linsvm       | emd              | raw              |  0.0450 | [-0.008, +0.098] |      1.0000 |            0.0253 | 13/0/2 |
| comp+log_scale | lda          | irmf_h2_a1.41    | emd              |  0.3980 | [+0.342, +0.455] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.1150 | [+0.088, +0.141] |      0.0000 |            0.0253 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41 | emd              |  0.2840 | [+0.232, +0.335] |      0.0000 |            0.0253 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | raw              |  0.4160 | [+0.355, +0.477] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41 | raw              |  0.3020 | [+0.247, +0.356] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | emd              | raw              |  0.0180 | [-0.043, +0.078] |      1.0000 |            0.4468 | 10/0/5 |
| comp+log_scale | qda          | irmf_h2_a1.41    | emd              |  0.3960 | [+0.326, +0.467] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.1300 | [+0.102, +0.159] |      0.0000 |            0.0253 | 15/0/0 |
| comp+log_scale | qda          | imf_linear_a1.41 | emd              |  0.2660 | [+0.195, +0.336] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | raw              |  0.3380 | [+0.273, +0.403] |      0.0000 |            0.0253 | 15/0/0 |
| comp+log_scale | qda          | imf_linear_a1.41 | raw              |  0.2080 | [+0.130, +0.285] |      0.0013 |            0.0253 | 15/0/0 |
| comp+log_scale | qda          | emd              | raw              | -0.0580 | [-0.142, +0.025] |      1.0000 |            0.1494 | 5/1/9  |
| comp+log_scale | gnb          | irmf_h2_a1.41    | emd              |  0.2380 | [+0.150, +0.325] |      0.0012 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.1740 | [+0.119, +0.229] |      0.0003 |            0.0253 | 15/0/0 |
| comp+log_scale | gnb          | imf_linear_a1.41 | emd              |  0.0640 | [-0.034, +0.162] |      1.0000 |            0.1370 | 11/0/4 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | raw              |  0.1920 | [+0.116, +0.269] |      0.0022 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | imf_linear_a1.41 | raw              |  0.0180 | [-0.067, +0.104] |      1.0000 |            0.9084 | 9/0/6  |
| comp+log_scale | gnb          | emd              | raw              | -0.0450 | [-0.127, +0.037] |      1.0000 |            0.1713 | 3/0/12 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | emd              |  0.3460 | [+0.304, +0.388] |      0.0000 |            0.0253 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.1170 | [+0.076, +0.158] |      0.0008 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41 | emd              |  0.2290 | [+0.177, +0.281] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | raw              |  0.3860 | [+0.343, +0.429] |      0.0000 |            0.0253 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41 | raw              |  0.2690 | [+0.216, +0.322] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | emd              | raw              |  0.0400 | [-0.022, +0.101] |      1.0000 |            0.0909 | 11/0/4 |
| comp+log_scale | knn          | irmf_h2_a1.41    | emd              |  0.1920 | [+0.150, +0.234] |      0.0000 |            0.0253 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0870 | [+0.048, +0.126] |      0.0066 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | imf_linear_a1.41 | emd              |  0.1050 | [+0.055, +0.156] |      0.0100 |            0.0253 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | raw              |  0.1590 | [+0.114, +0.204] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | imf_linear_a1.41 | raw              |  0.0720 | [+0.025, +0.119] |      0.0950 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | emd              | raw              | -0.0330 | [-0.073, +0.007] |      1.0000 |            0.0253 | 1/1/13 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | emd              |  0.3270 | [+0.281, +0.373] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.1370 | [+0.095, +0.179] |      0.0002 |            0.0253 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | emd              |  0.1900 | [+0.135, +0.245] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | raw              |  0.3760 | [+0.325, +0.428] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | raw              |  0.2390 | [+0.197, +0.282] |      0.0000 |            0.0253 | 15/0/0 |
| comp+log_scale | linsvm       | emd              | raw              |  0.0490 | [-0.001, +0.100] |      0.8211 |            0.0253 | 12/0/3 |

## Balanced accuracy by subset (noise_regime)

Feature set `comp`, best classifier per method:

| subset                     |   raw |   emd |   imf_linear_a1.41 |   irmf_h2_a1.41 |
|:---------------------------|------:|------:|-------------------:|----------------:|
| noise_regime=gaussian      | 0.551 | 0.633 |              0.817 |           0.878 |
| noise_regime=impulsive_one | 0.462 | 0.446 |              0.708 |           0.884 |
| noise_regime=impulsive_two | 0.442 | 0.453 |              0.748 |           0.882 |

Feature set `comp+log_scale`, best classifier per method:

| subset                     |   raw |   emd |   imf_linear_a1.41 |   irmf_h2_a1.41 |
|:---------------------------|------:|------:|-------------------:|----------------:|
| noise_regime=gaussian      | 0.518 | 0.617 |              0.817 |           0.873 |
| noise_regime=impulsive_one | 0.438 | 0.439 |              0.729 |           0.881 |
| noise_regime=impulsive_two | 0.441 | 0.453 |              0.745 |           0.885 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method           |   max_eps2 |   max_mi |   n_features |
|:-----------------|-----------:|---------:|-------------:|
| raw              |      0.218 |    0.173 |       21.000 |
| emd              |      0.110 |    0.126 |      181.000 |
| imf_linear_a1.41 |      0.242 |    0.280 |      321.000 |
| irmf_h2_a1.41    |      0.568 |    0.517 |      321.000 |

| method           | feature              | slot_label   |   epsilon2 |    mi |
|:-----------------|:---------------------|:-------------|-----------:|------:|
| irmf_h2_a1.41    | ch0_c5_kurt          | w25          |      0.568 | 0.517 |
| irmf_h2_a1.41    | ch0_c6_kurt          | w33          |      0.559 | 0.505 |
| irmf_h2_a1.41    | ch0_c7_kurt          | w47          |      0.490 | 0.380 |
| irmf_h2_a1.41    | ch0_c5_hilb_amp_std  | w25          |      0.462 | 0.385 |
| irmf_h2_a1.41    | ch0_c6_crest         | w33          |      0.398 | 0.292 |
| imf_linear_a1.41 | ch0_c5_spec_entropy  | w25          |      0.242 | 0.214 |
| imf_linear_a1.41 | ch0_c5_hilb_amp_std  | w25          |      0.242 | 0.174 |
| imf_linear_a1.41 | ch0_c5_perm_entropy  | w25          |      0.225 | 0.180 |
| imf_linear_a1.41 | ch0_c5_extrema_rate  | w25          |      0.218 | 0.177 |
| raw              | ch0_c0_spec_peak     | frame        |      0.218 | 0.173 |
| imf_linear_a1.41 | ch0_c8_spec_entropy  | w65          |      0.211 | 0.144 |
| emd              | ch0_c3_hilb_if_std   | imf4         |      0.110 | 0.057 |
| emd              | ch0_c2_kurt          | imf3         |      0.105 | 0.084 |
| raw              | ch0_c0_spec_entropy  | frame        |      0.089 | 0.052 |
| emd              | ch0_c3_perm_entropy  | imf4         |      0.077 | 0.044 |
| emd              | ch0_c3_extrema_rate  | imf4         |      0.077 | 0.047 |
| emd              | ch0_c3_zcr           | imf4         |      0.074 | 0.045 |
| raw              | ch0_c0_hilb_if_mean  | frame        |      0.050 | 0.050 |
| raw              | ch0_c0_mean_abs      | frame        |      0.041 | 0.035 |
| raw              | ch0_c0_hilb_amp_mean | frame        |      0.027 | 0.030 |
