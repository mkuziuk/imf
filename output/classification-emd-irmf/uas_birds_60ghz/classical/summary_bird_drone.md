# Classical baselines: uas_birds_60ghz / bird_drone

784 frames, 45 groups, 2 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 96 s with 3 workers.

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
| raw              | 0.875 ± 0.064 | 0.907 ± 0.043 | 0.953 ± 0.036 | 0.965 ± 0.023 | 0.963 ± 0.025 | 0.930 ± 0.043 |
| emd              | 0.920 ± 0.040 | 0.949 ± 0.032 | 0.976 ± 0.019 | 0.976 ± 0.021 | 0.976 ± 0.022 | 0.937 ± 0.023 |
| imf_linear_a1.41 | 0.959 ± 0.021 | 0.966 ± 0.029 | 0.985 ± 0.016 | 0.977 ± 0.018 | 0.974 ± 0.020 | 0.969 ± 0.024 |
| irmf_h2_a1.41    | 0.956 ± 0.024 | 0.973 ± 0.029 | 0.993 ± 0.007 | 0.976 ± 0.019 | 0.975 ± 0.021 | 0.982 ± 0.017 |

Feature set `comp+log_scale`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.888 ± 0.060 | 0.950 ± 0.046 | 0.974 ± 0.026 | 0.977 ± 0.021 | 0.967 ± 0.033 | 0.953 ± 0.044 |
| emd              | 0.921 ± 0.040 | 0.950 ± 0.036 | 0.984 ± 0.015 | 0.981 ± 0.021 | 0.980 ± 0.020 | 0.943 ± 0.020 |
| imf_linear_a1.41 | 0.961 ± 0.021 | 0.964 ± 0.031 | 0.987 ± 0.014 | 0.976 ± 0.019 | 0.977 ± 0.020 | 0.971 ± 0.024 |
| irmf_h2_a1.41    | 0.956 ± 0.024 | 0.974 ± 0.028 | 0.996 ± 0.004 | 0.979 ± 0.018 | 0.978 ± 0.020 | 0.981 ± 0.017 |

## Best classifier per method

| feature_set    | method           | classifier   |   balanced_accuracy |
|:---------------|:-----------------|:-------------|--------------------:|
| comp           | emd              | logreg       |               0.976 |
| comp           | imf_linear_a1.41 | lda          |               0.985 |
| comp           | irmf_h2_a1.41    | lda          |               0.993 |
| comp           | raw              | linsvm       |               0.965 |
| comp+log_scale | emd              | lda          |               0.984 |
| comp+log_scale | imf_linear_a1.41 | lda          |               0.987 |
| comp+log_scale | irmf_h2_a1.41    | lda          |               0.996 |
| comp+log_scale | raw              | linsvm       |               0.977 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a         | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-----------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41    | emd              |  0.0170 | [-0.004, +0.039] |      1.0000 |            0.1938 | 13/0/2 |
| comp           | lda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0080 | [-0.006, +0.023] |      1.0000 |            0.7227 | 8/6/1  |
| comp           | lda          | imf_linear_a1.41 | emd              |  0.0090 | [-0.007, +0.026] |      1.0000 |            0.5957 | 10/2/3 |
| comp           | lda          | irmf_h2_a1.41    | raw              |  0.0400 | [-0.003, +0.084] |      1.0000 |            0.1084 | 13/0/2 |
| comp           | lda          | imf_linear_a1.41 | raw              |  0.0320 | [-0.010, +0.074] |      1.0000 |            0.2871 | 12/0/3 |
| comp           | lda          | emd              | raw              |  0.0230 | [-0.010, +0.055] |      1.0000 |            0.5649 | 11/0/4 |
| comp           | qda          | irmf_h2_a1.41    | emd              |  0.0450 | [+0.018, +0.071] |      0.1982 |            0.0457 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0130 | [-0.011, +0.038] |      1.0000 |            0.5649 | 11/3/1 |
| comp           | qda          | imf_linear_a1.41 | emd              |  0.0310 | [+0.001, +0.062] |      1.0000 |            0.1164 | 14/0/1 |
| comp           | qda          | irmf_h2_a1.41    | raw              |  0.0510 | [+0.002, +0.100] |      1.0000 |            0.0586 | 14/0/1 |
| comp           | qda          | imf_linear_a1.41 | raw              |  0.0380 | [-0.005, +0.081] |      1.0000 |            0.2029 | 12/0/3 |
| comp           | qda          | emd              | raw              |  0.0070 | [-0.038, +0.051] |      1.0000 |            1.0000 | 9/1/5  |
| comp           | gnb          | irmf_h2_a1.41    | emd              |  0.0360 | [-0.013, +0.086] |      1.0000 |            0.1913 | 13/0/2 |
| comp           | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0030 | [-0.025, +0.019] |      1.0000 |            1.0000 | 3/4/8  |
| comp           | gnb          | imf_linear_a1.41 | emd              |  0.0390 | [-0.006, +0.085] |      1.0000 |            0.1368 | 12/1/2 |
| comp           | gnb          | irmf_h2_a1.41    | raw              |  0.0810 | [+0.018, +0.145] |      1.0000 |            0.0538 | 14/0/1 |
| comp           | gnb          | imf_linear_a1.41 | raw              |  0.0840 | [+0.018, +0.150] |      1.0000 |            0.0088 | 14/0/1 |
| comp           | gnb          | emd              | raw              |  0.0450 | [-0.009, +0.099] |      1.0000 |            0.1922 | 13/0/2 |
| comp           | logreg       | irmf_h2_a1.41    | emd              | -0.0010 | [-0.031, +0.029] |      1.0000 |            1.0000 | 8/0/7  |
| comp           | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0010 | [-0.017, +0.019] |      1.0000 |            1.0000 | 7/3/5  |
| comp           | logreg       | imf_linear_a1.41 | emd              | -0.0020 | [-0.026, +0.022] |      1.0000 |            1.0000 | 6/2/7  |
| comp           | logreg       | irmf_h2_a1.41    | raw              |  0.0120 | [-0.014, +0.038] |      1.0000 |            0.8285 | 12/0/3 |
| comp           | logreg       | imf_linear_a1.41 | raw              |  0.0110 | [-0.010, +0.032] |      1.0000 |            1.0000 | 10/1/4 |
| comp           | logreg       | emd              | raw              |  0.0130 | [-0.002, +0.028] |      1.0000 |            0.2047 | 12/1/2 |
| comp           | knn          | irmf_h2_a1.41    | emd              |  0.0230 | [-0.017, +0.064] |      1.0000 |            0.6982 | 11/0/4 |
| comp           | knn          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0070 | [-0.007, +0.021] |      1.0000 |            1.0000 | 10/3/2 |
| comp           | knn          | imf_linear_a1.41 | emd              |  0.0160 | [-0.025, +0.058] |      1.0000 |            1.0000 | 12/0/3 |
| comp           | knn          | irmf_h2_a1.41    | raw              |  0.0660 | [+0.016, +0.115] |      0.9240 |            0.0457 | 15/0/0 |
| comp           | knn          | imf_linear_a1.41 | raw              |  0.0590 | [+0.009, +0.108] |      1.0000 |            0.0782 | 13/1/1 |
| comp           | knn          | emd              | raw              |  0.0420 | [+0.007, +0.077] |      1.0000 |            0.0782 | 13/1/1 |
| comp           | linsvm       | irmf_h2_a1.41    | emd              |  0.0000 | [-0.031, +0.032] |      1.0000 |            1.0000 | 6/1/8  |
| comp           | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0000 | [-0.014, +0.013] |      1.0000 |            1.0000 | 4/4/7  |
| comp           | linsvm       | imf_linear_a1.41 | emd              |  0.0010 | [-0.024, +0.025] |      1.0000 |            1.0000 | 7/1/7  |
| comp           | linsvm       | irmf_h2_a1.41    | raw              |  0.0110 | [-0.016, +0.038] |      1.0000 |            1.0000 | 9/2/4  |
| comp           | linsvm       | imf_linear_a1.41 | raw              |  0.0110 | [-0.005, +0.028] |      1.0000 |            0.3594 | 10/3/2 |
| comp           | linsvm       | emd              | raw              |  0.0110 | [-0.009, +0.031] |      1.0000 |            1.0000 | 11/1/3 |
| comp+log_scale | lda          | irmf_h2_a1.41    | emd              |  0.0120 | [-0.004, +0.028] |      1.0000 |            0.2871 | 9/5/1  |
| comp+log_scale | lda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0090 | [-0.005, +0.023] |      1.0000 |            0.4297 | 9/5/1  |
| comp+log_scale | lda          | imf_linear_a1.41 | emd              |  0.0030 | [-0.010, +0.016] |      1.0000 |            1.0000 | 7/5/3  |
| comp+log_scale | lda          | irmf_h2_a1.41    | raw              |  0.0220 | [-0.006, +0.050] |      1.0000 |            0.1699 | 10/4/1 |
| comp+log_scale | lda          | imf_linear_a1.41 | raw              |  0.0130 | [-0.009, +0.035] |      1.0000 |            0.6855 | 10/3/2 |
| comp+log_scale | lda          | emd              | raw              |  0.0100 | [-0.011, +0.030] |      1.0000 |            1.0000 | 8/4/3  |
| comp+log_scale | qda          | irmf_h2_a1.41    | emd              |  0.0390 | [+0.014, +0.064] |      0.3722 |            0.0457 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0110 | [-0.012, +0.033] |      1.0000 |            1.0000 | 10/2/3 |
| comp+log_scale | qda          | imf_linear_a1.41 | emd              |  0.0280 | [+0.002, +0.054] |      1.0000 |            0.0907 | 14/0/1 |
| comp+log_scale | qda          | irmf_h2_a1.41    | raw              |  0.0280 | [-0.017, +0.073] |      1.0000 |            0.4919 | 12/0/3 |
| comp+log_scale | qda          | imf_linear_a1.41 | raw              |  0.0170 | [-0.020, +0.055] |      1.0000 |            1.0000 | 8/1/6  |
| comp+log_scale | qda          | emd              | raw              | -0.0110 | [-0.053, +0.031] |      1.0000 |            1.0000 | 6/0/9  |
| comp+log_scale | gnb          | irmf_h2_a1.41    | emd              |  0.0350 | [-0.015, +0.085] |      1.0000 |            0.2095 | 12/1/2 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0050 | [-0.027, +0.018] |      1.0000 |            1.0000 | 3/4/8  |
| comp+log_scale | gnb          | imf_linear_a1.41 | emd              |  0.0390 | [-0.006, +0.085] |      1.0000 |            0.2176 | 12/1/2 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | raw              |  0.0680 | [+0.009, +0.127] |      1.0000 |            0.0217 | 13/0/2 |
| comp+log_scale | gnb          | imf_linear_a1.41 | raw              |  0.0730 | [+0.012, +0.133] |      1.0000 |            0.0782 | 13/0/2 |
| comp+log_scale | gnb          | emd              | raw              |  0.0330 | [-0.015, +0.081] |      1.0000 |            0.3156 | 11/1/3 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | emd              | -0.0010 | [-0.030, +0.027] |      1.0000 |            1.0000 | 5/2/8  |
| comp+log_scale | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0020 | [-0.010, +0.013] |      1.0000 |            1.0000 | 8/3/4  |
| comp+log_scale | logreg       | imf_linear_a1.41 | emd              | -0.0030 | [-0.028, +0.022] |      1.0000 |            1.0000 | 4/4/7  |
| comp+log_scale | logreg       | irmf_h2_a1.41    | raw              |  0.0110 | [-0.017, +0.039] |      1.0000 |            1.0000 | 7/3/5  |
| comp+log_scale | logreg       | imf_linear_a1.41 | raw              |  0.0090 | [-0.018, +0.037] |      1.0000 |            1.0000 | 7/4/4  |
| comp+log_scale | logreg       | emd              | raw              |  0.0120 | [-0.014, +0.038] |      1.0000 |            1.0000 | 8/3/4  |
| comp+log_scale | knn          | irmf_h2_a1.41    | emd              |  0.0240 | [-0.012, +0.061] |      1.0000 |            0.3761 | 12/1/2 |
| comp+log_scale | knn          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0100 | [-0.011, +0.031] |      1.0000 |            1.0000 | 9/4/2  |
| comp+log_scale | knn          | imf_linear_a1.41 | emd              |  0.0150 | [-0.029, +0.058] |      1.0000 |            1.0000 | 10/1/4 |
| comp+log_scale | knn          | irmf_h2_a1.41    | raw              |  0.0240 | [-0.011, +0.058] |      1.0000 |            0.2441 | 9/4/2  |
| comp+log_scale | knn          | imf_linear_a1.41 | raw              |  0.0140 | [-0.023, +0.051] |      1.0000 |            1.0000 | 9/1/5  |
| comp+log_scale | knn          | emd              | raw              | -0.0000 | [-0.032, +0.031] |      1.0000 |            1.0000 | 5/3/7  |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | emd              | -0.0010 | [-0.035, +0.033] |      1.0000 |            1.0000 | 6/2/7  |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0040 | [-0.013, +0.020] |      1.0000 |            1.0000 | 6/4/5  |
| comp+log_scale | linsvm       | imf_linear_a1.41 | emd              | -0.0050 | [-0.031, +0.022] |      1.0000 |            1.0000 | 5/3/7  |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | raw              |  0.0030 | [-0.023, +0.028] |      1.0000 |            1.0000 | 6/1/8  |
| comp+log_scale | linsvm       | imf_linear_a1.41 | raw              | -0.0010 | [-0.014, +0.012] |      1.0000 |            1.0000 | 5/2/8  |
| comp+log_scale | linsvm       | emd              | raw              |  0.0040 | [-0.017, +0.025] |      1.0000 |            1.0000 | 7/2/6  |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method           |   max_eps2 |   max_mi |   n_features |
|:-----------------|-----------:|---------:|-------------:|
| raw              |      0.582 |    0.416 |       42.000 |
| emd              |      0.571 |    0.407 |      362.000 |
| imf_linear_a1.41 |      0.684 |    0.541 |      642.000 |
| irmf_h2_a1.41    |      0.682 |    0.525 |      642.000 |

| method           | feature               | slot_label   |   epsilon2 |    mi |
|:-----------------|:----------------------|:-------------|-----------:|------:|
| imf_linear_a1.41 | ch0_c10_energy_frac   | w127         |      0.684 | 0.529 |
| irmf_h2_a1.41    | ch0_c10_log_energy    | w127         |      0.682 | 0.520 |
| irmf_h2_a1.41    | ch0_c10_hilb_amp_mean | w127         |      0.682 | 0.525 |
| irmf_h2_a1.41    | ch0_c10_std           | w127         |      0.682 | 0.520 |
| irmf_h2_a1.41    | ch0_c10_mean_abs      | w127         |      0.682 | 0.524 |
| irmf_h2_a1.41    | ch1_c10_hilb_amp_mean | w127         |      0.682 | 0.523 |
| imf_linear_a1.41 | ch1_c10_energy_frac   | w127         |      0.681 | 0.541 |
| imf_linear_a1.41 | ch1_c10_hilb_amp_mean | w127         |      0.675 | 0.511 |
| imf_linear_a1.41 | ch1_c10_mean_abs      | w127         |      0.674 | 0.508 |
| imf_linear_a1.41 | ch1_c10_log_energy    | w127         |      0.674 | 0.510 |
| raw              | ch1_c0_extrema_rate   | frame        |      0.582 | 0.395 |
| raw              | ch0_c0_extrema_rate   | frame        |      0.574 | 0.388 |
| emd              | ch0_c3_hjorth_mob     | imf4         |      0.571 | 0.407 |
| raw              | ch1_c0_zcr            | frame        |      0.571 | 0.389 |
| raw              | ch0_c0_zcr            | frame        |      0.562 | 0.369 |
| emd              | ch0_c3_hilb_if_mean   | imf4         |      0.556 | 0.384 |
| emd              | ch1_c3_hjorth_mob     | imf4         |      0.556 | 0.383 |
| raw              | ch1_c0_spec_centroid  | frame        |      0.556 | 0.386 |
| emd              | ch1_c3_hilb_if_mean   | imf4         |      0.540 | 0.365 |
| emd              | ch0_c3_spec_centroid  | imf4         |      0.539 | 0.351 |
