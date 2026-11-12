# Classical baselines: uas_birds_60ghz / target3

784 frames, 45 groups, 3 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 123 s with 3 workers.

## Decomposition cost

| method           |   slots |   cpu_s_per_frame |   failed | emd_imf_counts                                  | irmf_all_stages_converged   |
|:-----------------|--------:|------------------:|---------:|:------------------------------------------------|:----------------------------|
| emd              |       9 |            0.0305 |        0 | {'5': 13, '6': 260, '7': 972, '8': 314, '9': 9} | nan                         |
| imf_linear_a1.41 |      16 |            0.0044 |        0 | nan                                             | nan                         |
| irmf_h2_a1.41    |      16 |            0.3547 |        0 | nan                                             | 1374/1568                   |

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.839 ± 0.049 | 0.867 ± 0.049 | 0.910 ± 0.045 | 0.899 ± 0.029 | 0.911 ± 0.035 | 0.891 ± 0.047 |
| emd              | 0.875 ± 0.040 | 0.882 ± 0.032 | 0.949 ± 0.025 | 0.922 ± 0.028 | 0.944 ± 0.032 | 0.888 ± 0.046 |
| imf_linear_a1.41 | 0.922 ± 0.037 | 0.951 ± 0.022 | 0.959 ± 0.031 | 0.949 ± 0.033 | 0.954 ± 0.029 | 0.931 ± 0.040 |
| irmf_h2_a1.41    | 0.929 ± 0.034 | 0.950 ± 0.034 | 0.976 ± 0.016 | 0.963 ± 0.032 | 0.959 ± 0.026 | 0.925 ± 0.042 |

Feature set `comp+log_scale`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.849 ± 0.040 | 0.919 ± 0.030 | 0.929 ± 0.048 | 0.928 ± 0.020 | 0.934 ± 0.025 | 0.923 ± 0.036 |
| emd              | 0.879 ± 0.040 | 0.888 ± 0.021 | 0.951 ± 0.026 | 0.934 ± 0.030 | 0.947 ± 0.026 | 0.900 ± 0.042 |
| imf_linear_a1.41 | 0.923 ± 0.039 | 0.952 ± 0.022 | 0.964 ± 0.029 | 0.947 ± 0.037 | 0.952 ± 0.028 | 0.934 ± 0.040 |
| irmf_h2_a1.41    | 0.930 ± 0.034 | 0.951 ± 0.033 | 0.978 ± 0.016 | 0.966 ± 0.029 | 0.959 ± 0.027 | 0.930 ± 0.042 |

## Best classifier per method

| feature_set    | method           | classifier   |   balanced_accuracy |
|:---------------|:-----------------|:-------------|--------------------:|
| comp           | emd              | lda          |               0.949 |
| comp           | imf_linear_a1.41 | lda          |               0.959 |
| comp           | irmf_h2_a1.41    | lda          |               0.976 |
| comp           | raw              | logreg       |               0.911 |
| comp+log_scale | emd              | lda          |               0.951 |
| comp+log_scale | imf_linear_a1.41 | lda          |               0.964 |
| comp+log_scale | irmf_h2_a1.41    | lda          |               0.978 |
| comp+log_scale | raw              | logreg       |               0.934 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a         | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-----------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41    | emd              |  0.0260 | [-0.004, +0.056] |      1.0000 |            0.1350 | 12/1/2 |
| comp           | lda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0170 | [-0.011, +0.045] |      1.0000 |            0.2285 | 9/5/1  |
| comp           | lda          | imf_linear_a1.41 | emd              |  0.0090 | [-0.034, +0.052] |      1.0000 |            1.0000 | 9/1/5  |
| comp           | lda          | irmf_h2_a1.41    | raw              |  0.0660 | [+0.024, +0.107] |      0.2921 |            0.0459 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41 | raw              |  0.0490 | [-0.000, +0.098] |      1.0000 |            0.0767 | 13/0/2 |
| comp           | lda          | emd              | raw              |  0.0390 | [-0.016, +0.095] |      1.0000 |            0.2807 | 11/1/3 |
| comp           | qda          | irmf_h2_a1.41    | emd              |  0.0370 | [-0.004, +0.078] |      1.0000 |            0.1350 | 13/0/2 |
| comp           | qda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0060 | [-0.044, +0.032] |      1.0000 |            1.0000 | 4/1/10 |
| comp           | qda          | imf_linear_a1.41 | emd              |  0.0430 | [-0.006, +0.093] |      1.0000 |            0.1145 | 12/1/2 |
| comp           | qda          | irmf_h2_a1.41    | raw              |  0.0340 | [-0.020, +0.089] |      1.0000 |            0.3307 | 11/1/3 |
| comp           | qda          | imf_linear_a1.41 | raw              |  0.0410 | [-0.014, +0.095] |      1.0000 |            0.1523 | 12/1/2 |
| comp           | qda          | emd              | raw              | -0.0030 | [-0.061, +0.055] |      1.0000 |            1.0000 | 7/0/8  |
| comp           | gnb          | irmf_h2_a1.41    | emd              |  0.0540 | [-0.006, +0.114] |      1.0000 |            0.1350 | 13/0/2 |
| comp           | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0070 | [-0.013, +0.026] |      1.0000 |            1.0000 | 10/2/3 |
| comp           | gnb          | imf_linear_a1.41 | emd              |  0.0470 | [-0.010, +0.104] |      1.0000 |            0.1350 | 13/0/2 |
| comp           | gnb          | irmf_h2_a1.41    | raw              |  0.0910 | [+0.038, +0.143] |      0.1600 |            0.0459 | 15/0/0 |
| comp           | gnb          | imf_linear_a1.41 | raw              |  0.0840 | [+0.035, +0.132] |      0.1600 |            0.0459 | 15/0/0 |
| comp           | gnb          | emd              | raw              |  0.0360 | [-0.024, +0.097] |      1.0000 |            0.3380 | 13/0/2 |
| comp           | logreg       | irmf_h2_a1.41    | emd              |  0.0150 | [-0.010, +0.041] |      1.0000 |            0.4077 | 12/0/3 |
| comp           | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0060 | [-0.015, +0.026] |      1.0000 |            1.0000 | 8/3/4  |
| comp           | logreg       | imf_linear_a1.41 | emd              |  0.0100 | [-0.026, +0.045] |      1.0000 |            1.0000 | 10/1/4 |
| comp           | logreg       | irmf_h2_a1.41    | raw              |  0.0480 | [-0.002, +0.099] |      1.0000 |            0.1350 | 14/0/1 |
| comp           | logreg       | imf_linear_a1.41 | raw              |  0.0430 | [-0.007, +0.092] |      1.0000 |            0.1350 | 13/0/2 |
| comp           | logreg       | emd              | raw              |  0.0330 | [-0.033, +0.099] |      1.0000 |            0.7864 | 12/1/2 |
| comp           | knn          | irmf_h2_a1.41    | emd              |  0.0690 | [+0.009, +0.128] |      1.0000 |            0.0731 | 14/0/1 |
| comp           | knn          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0010 | [-0.032, +0.031] |      1.0000 |            1.0000 | 7/2/6  |
| comp           | knn          | imf_linear_a1.41 | emd              |  0.0690 | [+0.030, +0.109] |      0.1434 |            0.0459 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | raw              |  0.0830 | [+0.002, +0.164] |      1.0000 |            0.1070 | 13/1/1 |
| comp           | knn          | imf_linear_a1.41 | raw              |  0.0840 | [+0.023, +0.145] |      0.6704 |            0.0459 | 15/0/0 |
| comp           | knn          | emd              | raw              |  0.0140 | [-0.038, +0.066] |      1.0000 |            1.0000 | 9/0/6  |
| comp           | linsvm       | irmf_h2_a1.41    | emd              |  0.0400 | [-0.005, +0.085] |      1.0000 |            0.1424 | 12/1/2 |
| comp           | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0140 | [-0.012, +0.039] |      1.0000 |            0.4284 | 12/0/3 |
| comp           | linsvm       | imf_linear_a1.41 | emd              |  0.0270 | [-0.025, +0.078] |      1.0000 |            0.9551 | 10/1/4 |
| comp           | linsvm       | irmf_h2_a1.41    | raw              |  0.0630 | [+0.022, +0.105] |      0.3656 |            0.0502 | 14/0/1 |
| comp           | linsvm       | imf_linear_a1.41 | raw              |  0.0500 | [+0.014, +0.085] |      0.5930 |            0.0502 | 14/0/1 |
| comp           | linsvm       | emd              | raw              |  0.0230 | [-0.035, +0.081] |      1.0000 |            1.0000 | 12/0/3 |
| comp+log_scale | lda          | irmf_h2_a1.41    | emd              |  0.0270 | [-0.004, +0.058] |      1.0000 |            0.1094 | 11/3/1 |
| comp+log_scale | lda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0150 | [-0.012, +0.042] |      1.0000 |            0.3076 | 10/2/3 |
| comp+log_scale | lda          | imf_linear_a1.41 | emd              |  0.0120 | [-0.034, +0.058] |      1.0000 |            1.0000 | 10/0/5 |
| comp+log_scale | lda          | irmf_h2_a1.41    | raw              |  0.0500 | [+0.003, +0.096] |      1.0000 |            0.0176 | 13/2/0 |
| comp+log_scale | lda          | imf_linear_a1.41 | raw              |  0.0350 | [-0.016, +0.086] |      1.0000 |            0.3098 | 11/1/3 |
| comp+log_scale | lda          | emd              | raw              |  0.0230 | [-0.028, +0.073] |      1.0000 |            1.0000 | 9/2/4  |
| comp+log_scale | qda          | irmf_h2_a1.41    | emd              |  0.0300 | [-0.011, +0.072] |      1.0000 |            0.1734 | 11/1/3 |
| comp+log_scale | qda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0040 | [-0.041, +0.033] |      1.0000 |            1.0000 | 6/0/9  |
| comp+log_scale | qda          | imf_linear_a1.41 | emd              |  0.0340 | [-0.007, +0.076] |      1.0000 |            0.1350 | 13/1/1 |
| comp+log_scale | qda          | irmf_h2_a1.41    | raw              |  0.0070 | [-0.046, +0.061] |      1.0000 |            1.0000 | 8/1/6  |
| comp+log_scale | qda          | imf_linear_a1.41 | raw              |  0.0110 | [-0.036, +0.059] |      1.0000 |            1.0000 | 10/1/4 |
| comp+log_scale | qda          | emd              | raw              | -0.0230 | [-0.077, +0.031] |      1.0000 |            1.0000 | 3/0/12 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | emd              |  0.0510 | [-0.010, +0.111] |      1.0000 |            0.1350 | 12/1/2 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0070 | [-0.014, +0.027] |      1.0000 |            1.0000 | 10/1/4 |
| comp+log_scale | gnb          | imf_linear_a1.41 | emd              |  0.0440 | [-0.015, +0.104] |      1.0000 |            0.1523 | 12/1/2 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | raw              |  0.0810 | [+0.039, +0.123] |      0.0776 |            0.0459 | 15/0/0 |
| comp+log_scale | gnb          | imf_linear_a1.41 | raw              |  0.0740 | [+0.033, +0.116] |      0.1269 |            0.0459 | 15/0/0 |
| comp+log_scale | gnb          | emd              | raw              |  0.0300 | [-0.027, +0.087] |      1.0000 |            0.5340 | 13/0/2 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | emd              |  0.0120 | [-0.008, +0.032] |      1.0000 |            0.4385 | 11/1/3 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0070 | [-0.014, +0.028] |      1.0000 |            1.0000 | 7/5/3  |
| comp+log_scale | logreg       | imf_linear_a1.41 | emd              |  0.0050 | [-0.021, +0.032] |      1.0000 |            1.0000 | 10/0/5 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | raw              |  0.0250 | [-0.013, +0.063] |      1.0000 |            0.2066 | 13/0/2 |
| comp+log_scale | logreg       | imf_linear_a1.41 | raw              |  0.0180 | [-0.024, +0.061] |      1.0000 |            1.0000 | 11/0/4 |
| comp+log_scale | logreg       | emd              | raw              |  0.0130 | [-0.028, +0.055] |      1.0000 |            1.0000 | 9/2/4  |
| comp+log_scale | knn          | irmf_h2_a1.41    | emd              |  0.0630 | [+0.012, +0.115] |      1.0000 |            0.0731 | 14/0/1 |
| comp+log_scale | knn          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0010 | [-0.031, +0.029] |      1.0000 |            1.0000 | 7/2/6  |
| comp+log_scale | knn          | imf_linear_a1.41 | emd              |  0.0640 | [+0.029, +0.100] |      0.1171 |            0.0459 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | raw              |  0.0320 | [-0.019, +0.083] |      1.0000 |            0.4390 | 11/0/4 |
| comp+log_scale | knn          | imf_linear_a1.41 | raw              |  0.0330 | [-0.001, +0.068] |      1.0000 |            0.1151 | 13/0/2 |
| comp+log_scale | knn          | emd              | raw              | -0.0310 | [-0.065, +0.003] |      1.0000 |            0.1523 | 2/1/12 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | emd              |  0.0320 | [-0.007, +0.071] |      1.0000 |            0.2422 | 13/0/2 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0190 | [-0.022, +0.059] |      1.0000 |            0.3076 | 11/2/2 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | emd              |  0.0130 | [-0.040, +0.066] |      1.0000 |            1.0000 | 10/1/4 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | raw              |  0.0380 | [+0.009, +0.066] |      0.7680 |            0.0731 | 13/1/1 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | raw              |  0.0190 | [-0.026, +0.063] |      1.0000 |            1.0000 | 11/2/2 |
| comp+log_scale | linsvm       | emd              | raw              |  0.0060 | [-0.027, +0.039] |      1.0000 |            1.0000 | 10/1/4 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method           |   max_eps2 |   max_mi |   n_features |
|:-----------------|-----------:|---------:|-------------:|
| raw              |      0.684 |    0.563 |       42.000 |
| emd              |      0.588 |    0.498 |      362.000 |
| imf_linear_a1.41 |      0.787 |    0.701 |      642.000 |
| irmf_h2_a1.41    |      0.796 |    0.722 |      642.000 |

| method           | feature               | slot_label   |   epsilon2 |    mi |
|:-----------------|:----------------------|:-------------|-----------:|------:|
| irmf_h2_a1.41    | ch0_c10_log_energy    | w127         |      0.796 | 0.714 |
| irmf_h2_a1.41    | ch0_c10_hilb_amp_mean | w127         |      0.796 | 0.722 |
| irmf_h2_a1.41    | ch0_c10_std           | w127         |      0.796 | 0.714 |
| irmf_h2_a1.41    | ch0_c10_mean_abs      | w127         |      0.796 | 0.719 |
| irmf_h2_a1.41    | ch1_c10_log_energy    | w127         |      0.795 | 0.715 |
| imf_linear_a1.41 | ch1_c10_std           | w127         |      0.787 | 0.700 |
| imf_linear_a1.41 | ch1_c10_log_energy    | w127         |      0.787 | 0.700 |
| imf_linear_a1.41 | ch1_c10_hilb_amp_mean | w127         |      0.786 | 0.701 |
| imf_linear_a1.41 | ch1_c10_mean_abs      | w127         |      0.786 | 0.692 |
| imf_linear_a1.41 | ch0_c10_mean_abs      | w127         |      0.785 | 0.695 |
| raw              | ch1_c0_zcr            | frame        |      0.684 | 0.548 |
| raw              | ch1_c0_spec_centroid  | frame        |      0.672 | 0.563 |
| raw              | ch1_c0_extrema_rate   | frame        |      0.670 | 0.514 |
| raw              | ch0_c0_zcr            | frame        |      0.669 | 0.518 |
| raw              | ch1_c0_hjorth_mob     | frame        |      0.665 | 0.546 |
| emd              | ch0_c3_hjorth_mob     | imf4         |      0.588 | 0.433 |
| emd              | ch1_c3_hjorth_mob     | imf4         |      0.577 | 0.408 |
| emd              | ch0_c3_hilb_if_mean   | imf4         |      0.569 | 0.404 |
| emd              | ch1_c0_mean_abs       | imf1         |      0.567 | 0.418 |
| emd              | ch1_c1_hilb_if_mean   | imf2         |      0.562 | 0.405 |
