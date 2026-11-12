# Classical baselines: birdclef2023 / passerine

12000 frames, 451 groups, 2 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 496 s with 3 workers.

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
| raw              | 0.651 ± 0.033 | 0.761 ± 0.017 | 0.676 ± 0.023 | 0.692 ± 0.024 | 0.706 ± 0.024 | 0.635 ± 0.049 |
| emd              | 0.624 ± 0.028 | 0.715 ± 0.018 | 0.745 ± 0.019 | 0.749 ± 0.017 | 0.752 ± 0.015 | 0.629 ± 0.044 |
| imf_linear_a1.41 | 0.710 ± 0.020 | 0.754 ± 0.020 | 0.770 ± 0.019 | 0.771 ± 0.019 | 0.780 ± 0.017 | 0.687 ± 0.053 |
| irmf_h2_a1.41    | 0.695 ± 0.022 | 0.760 ± 0.015 | 0.767 ± 0.014 | 0.775 ± 0.016 | 0.776 ± 0.016 | 0.638 ± 0.059 |

Feature set `comp+log_scale`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.651 ± 0.033 | 0.752 ± 0.023 | 0.676 ± 0.023 | 0.691 ± 0.024 | 0.705 ± 0.025 | 0.637 ± 0.048 |
| emd              | 0.624 ± 0.029 | 0.717 ± 0.018 | 0.744 ± 0.019 | 0.748 ± 0.016 | 0.752 ± 0.016 | 0.631 ± 0.045 |
| imf_linear_a1.41 | 0.710 ± 0.020 | 0.754 ± 0.020 | 0.769 ± 0.018 | 0.770 ± 0.019 | 0.778 ± 0.017 | 0.687 ± 0.053 |
| irmf_h2_a1.41    | 0.695 ± 0.022 | 0.759 ± 0.018 | 0.765 ± 0.014 | 0.773 ± 0.016 | 0.776 ± 0.017 | 0.637 ± 0.059 |

## Best classifier per method

| feature_set    | method           | classifier   |   balanced_accuracy |
|:---------------|:-----------------|:-------------|--------------------:|
| comp           | emd              | logreg       |               0.752 |
| comp           | imf_linear_a1.41 | logreg       |               0.780 |
| comp           | irmf_h2_a1.41    | logreg       |               0.776 |
| comp           | raw              | knn          |               0.761 |
| comp+log_scale | emd              | logreg       |               0.752 |
| comp+log_scale | imf_linear_a1.41 | logreg       |               0.778 |
| comp+log_scale | irmf_h2_a1.41    | logreg       |               0.776 |
| comp+log_scale | raw              | knn          |               0.752 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a         | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-----------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41    | emd              |  0.0210 | [+0.010, +0.032] |      0.0379 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0030 | [-0.017, +0.010] |      1.0000 |            1.0000 | 5/0/10 |
| comp           | lda          | imf_linear_a1.41 | emd              |  0.0250 | [+0.010, +0.039] |      0.0967 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | raw              |  0.0910 | [+0.070, +0.112] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41 | raw              |  0.0940 | [+0.068, +0.120] |      0.0002 |            0.0044 | 15/0/0 |
| comp           | lda          | emd              | raw              |  0.0690 | [+0.044, +0.095] |      0.0024 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | emd              |  0.0080 | [-0.102, +0.119] |      1.0000 |            1.0000 | 7/0/8  |
| comp           | qda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0490 | [-0.079, -0.020] |      0.1060 |            0.0044 | 0/0/15 |
| comp           | qda          | imf_linear_a1.41 | emd              |  0.0570 | [-0.050, +0.165] |      1.0000 |            0.4740 | 11/0/4 |
| comp           | qda          | irmf_h2_a1.41    | raw              |  0.0020 | [-0.054, +0.058] |      1.0000 |            1.0000 | 9/0/6  |
| comp           | qda          | imf_linear_a1.41 | raw              |  0.0510 | [-0.001, +0.103] |      1.0000 |            0.0197 | 14/0/1 |
| comp           | qda          | emd              | raw              | -0.0060 | [-0.090, +0.078] |      1.0000 |            1.0000 | 6/0/9  |
| comp           | gnb          | irmf_h2_a1.41    | emd              |  0.0710 | [+0.024, +0.118] |      0.1723 |            0.0048 | 14/0/1 |
| comp           | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0150 | [-0.027, -0.004] |      0.3752 |            0.0044 | 0/0/15 |
| comp           | gnb          | imf_linear_a1.41 | emd              |  0.0860 | [+0.046, +0.126] |      0.0206 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | raw              |  0.0440 | [+0.012, +0.075] |      0.2714 |            0.0044 | 15/0/0 |
| comp           | gnb          | imf_linear_a1.41 | raw              |  0.0590 | [+0.033, +0.085] |      0.0134 |            0.0044 | 15/0/0 |
| comp           | gnb          | emd              | raw              | -0.0270 | [-0.082, +0.027] |      1.0000 |            0.6332 | 5/0/10 |
| comp           | logreg       | irmf_h2_a1.41    | emd              |  0.0230 | [+0.009, +0.037] |      0.1220 |            0.0044 | 14/0/1 |
| comp           | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0040 | [-0.019, +0.010] |      1.0000 |            1.0000 | 6/0/9  |
| comp           | logreg       | imf_linear_a1.41 | emd              |  0.0270 | [+0.014, +0.041] |      0.0332 |            0.0044 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | raw              |  0.0690 | [+0.045, +0.093] |      0.0014 |            0.0044 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41 | raw              |  0.0740 | [+0.047, +0.100] |      0.0021 |            0.0044 | 15/0/0 |
| comp           | logreg       | emd              | raw              |  0.0460 | [+0.021, +0.072] |      0.0609 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | emd              |  0.0450 | [+0.030, +0.061] |      0.0016 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0060 | [-0.008, +0.021] |      1.0000 |            0.7427 | 12/0/3 |
| comp           | knn          | imf_linear_a1.41 | emd              |  0.0390 | [+0.018, +0.059] |      0.0478 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | raw              | -0.0010 | [-0.015, +0.014] |      1.0000 |            1.0000 | 7/0/8  |
| comp           | knn          | imf_linear_a1.41 | raw              | -0.0070 | [-0.031, +0.016] |      1.0000 |            1.0000 | 5/0/10 |
| comp           | knn          | emd              | raw              | -0.0460 | [-0.057, -0.035] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | linsvm       | irmf_h2_a1.41    | emd              |  0.0250 | [+0.016, +0.035] |      0.0031 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0030 | [-0.008, +0.015] |      1.0000 |            1.0000 | 9/0/6  |
| comp           | linsvm       | imf_linear_a1.41 | emd              |  0.0220 | [+0.008, +0.036] |      0.1324 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | raw              |  0.0830 | [+0.060, +0.105] |      0.0001 |            0.0044 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41 | raw              |  0.0790 | [+0.055, +0.104] |      0.0004 |            0.0044 | 15/0/0 |
| comp           | linsvm       | emd              | raw              |  0.0570 | [+0.035, +0.079] |      0.0037 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | emd              |  0.0210 | [+0.009, +0.032] |      0.0609 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0040 | [-0.017, +0.008] |      1.0000 |            1.0000 | 6/0/9  |
| comp+log_scale | lda          | imf_linear_a1.41 | emd              |  0.0250 | [+0.011, +0.038] |      0.0609 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | raw              |  0.0890 | [+0.068, +0.110] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41 | raw              |  0.0930 | [+0.067, +0.120] |      0.0002 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | emd              | raw              |  0.0680 | [+0.043, +0.094] |      0.0028 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | emd              |  0.0060 | [-0.104, +0.117] |      1.0000 |            1.0000 | 7/0/8  |
| comp+log_scale | qda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0490 | [-0.078, -0.020] |      0.0967 |            0.0044 | 0/0/15 |
| comp+log_scale | qda          | imf_linear_a1.41 | emd              |  0.0560 | [-0.053, +0.164] |      1.0000 |            0.6332 | 11/0/4 |
| comp+log_scale | qda          | irmf_h2_a1.41    | raw              |  0.0000 | [-0.054, +0.054] |      1.0000 |            1.0000 | 9/0/6  |
| comp+log_scale | qda          | imf_linear_a1.41 | raw              |  0.0500 | [-0.001, +0.100] |      1.0000 |            0.0146 | 14/0/1 |
| comp+log_scale | qda          | emd              | raw              | -0.0060 | [-0.091, +0.079] |      1.0000 |            1.0000 | 6/0/9  |
| comp+log_scale | gnb          | irmf_h2_a1.41    | emd              |  0.0710 | [+0.024, +0.118] |      0.1723 |            0.0048 | 14/0/1 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0150 | [-0.027, -0.003] |      0.3752 |            0.0044 | 0/0/15 |
| comp+log_scale | gnb          | imf_linear_a1.41 | emd              |  0.0860 | [+0.046, +0.127] |      0.0207 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | raw              |  0.0440 | [+0.013, +0.075] |      0.2468 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | imf_linear_a1.41 | raw              |  0.0590 | [+0.034, +0.085] |      0.0109 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | emd              | raw              | -0.0270 | [-0.081, +0.027] |      1.0000 |            0.6332 | 5/0/10 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | emd              |  0.0240 | [+0.013, +0.035] |      0.0135 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0030 | [-0.016, +0.011] |      1.0000 |            1.0000 | 7/0/8  |
| comp+log_scale | logreg       | imf_linear_a1.41 | emd              |  0.0270 | [+0.013, +0.040] |      0.0314 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | raw              |  0.0700 | [+0.047, +0.094] |      0.0010 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41 | raw              |  0.0730 | [+0.045, +0.101] |      0.0037 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | emd              | raw              |  0.0460 | [+0.022, +0.071] |      0.0478 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | emd              |  0.0420 | [+0.021, +0.062] |      0.0329 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0050 | [-0.009, +0.019] |      1.0000 |            1.0000 | 11/0/4 |
| comp+log_scale | knn          | imf_linear_a1.41 | emd              |  0.0370 | [+0.017, +0.056] |      0.0524 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | raw              |  0.0070 | [-0.012, +0.026] |      1.0000 |            1.0000 | 10/0/5 |
| comp+log_scale | knn          | imf_linear_a1.41 | raw              |  0.0020 | [-0.020, +0.024] |      1.0000 |            1.0000 | 9/0/6  |
| comp+log_scale | knn          | emd              | raw              | -0.0350 | [-0.049, -0.021] |      0.0050 |            0.0044 | 0/0/15 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | emd              |  0.0250 | [+0.015, +0.035] |      0.0050 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 |  0.0040 | [-0.008, +0.015] |      1.0000 |            1.0000 | 9/0/6  |
| comp+log_scale | linsvm       | imf_linear_a1.41 | emd              |  0.0220 | [+0.008, +0.035] |      0.1220 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | raw              |  0.0820 | [+0.061, +0.104] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | raw              |  0.0790 | [+0.055, +0.102] |      0.0004 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | emd              | raw              |  0.0570 | [+0.034, +0.080] |      0.0065 |            0.0044 | 15/0/0 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method           |   max_eps2 |   max_mi |   n_features |
|:-----------------|-----------:|---------:|-------------:|
| raw              |      0.151 |    0.106 |       21.000 |
| emd              |      0.149 |    0.093 |      181.000 |
| imf_linear_a1.41 |      0.188 |    0.120 |      361.000 |
| irmf_h2_a1.41    |      0.190 |    0.113 |      361.000 |

| method           | feature              | slot_label   |   epsilon2 |    mi |
|:-----------------|:---------------------|:-------------|-----------:|------:|
| irmf_h2_a1.41    | ch0_c4_hilb_if_mean  | w17          |      0.190 | 0.107 |
| irmf_h2_a1.41    | ch0_c4_hjorth_mob    | w17          |      0.190 | 0.106 |
| imf_linear_a1.41 | ch0_c4_hilb_if_mean  | w17          |      0.188 | 0.106 |
| imf_linear_a1.41 | ch0_c4_zcr           | w17          |      0.188 | 0.106 |
| imf_linear_a1.41 | ch0_c4_hjorth_mob    | w17          |      0.187 | 0.105 |
| irmf_h2_a1.41    | ch0_c5_hjorth_mob    | w25          |      0.186 | 0.113 |
| imf_linear_a1.41 | ch0_c6_energy_frac   | w33          |      0.185 | 0.109 |
| irmf_h2_a1.41    | ch0_c4_zcr           | w17          |      0.184 | 0.103 |
| irmf_h2_a1.41    | ch0_c4_spec_centroid | w17          |      0.184 | 0.102 |
| imf_linear_a1.41 | ch0_c5_extrema_rate  | w25          |      0.184 | 0.103 |
| raw              | ch0_c0_hilb_if_mean  | frame        |      0.151 | 0.106 |
| raw              | ch0_c0_hjorth_mob    | frame        |      0.150 | 0.101 |
| emd              | ch0_c0_hjorth_cpx    | imf1         |      0.149 | 0.093 |
| raw              | ch0_c0_spec_centroid | frame        |      0.145 | 0.100 |
| emd              | ch0_c1_energy_frac   | imf2         |      0.144 | 0.086 |
| raw              | ch0_c0_zcr           | frame        |      0.143 | 0.096 |
| emd              | ch0_c0_hilb_amp_mean | imf1         |      0.140 | 0.090 |
| emd              | ch0_c0_mean_abs      | imf1         |      0.139 | 0.090 |
| raw              | ch0_c0_hjorth_cpx    | frame        |      0.134 | 0.096 |
| emd              | ch0_c0_energy_frac   | imf1         |      0.129 | 0.080 |
