# Classical baselines: dronerf / 2class

12000 frames, 227 groups, 2 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 3279 s with 3 workers.

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
| raw              | 0.579 ± 0.020 | 0.866 ± 0.024 | 0.684 ± 0.039 | 0.758 ± 0.022 | 0.768 ± 0.019 | 0.805 ± 0.060 |
| emd              | 0.585 ± 0.011 | 0.659 ± 0.011 | 0.800 ± 0.027 | 0.825 ± 0.029 | 0.822 ± 0.030 | 0.770 ± 0.034 |
| imf_linear_a1.41 | 0.856 ± 0.008 | 0.903 ± 0.021 | 0.939 ± 0.018 | 0.941 ± 0.015 | 0.938 ± 0.018 | 0.889 ± 0.023 |
| irmf_h2_a1.41    | 0.842 ± 0.011 | 0.876 ± 0.017 | 0.928 ± 0.019 | 0.929 ± 0.018 | 0.927 ± 0.017 | 0.848 ± 0.022 |

Feature set `comp+log_scale`:

| method           | gnb           | knn           | lda           | linsvm        | logreg        | qda           |
|:-----------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| raw              | 0.669 ± 0.034 | 0.902 ± 0.026 | 0.791 ± 0.038 | 0.776 ± 0.031 | 0.779 ± 0.030 | 0.872 ± 0.020 |
| emd              | 0.587 ± 0.011 | 0.677 ± 0.014 | 0.835 ± 0.032 | 0.863 ± 0.026 | 0.865 ± 0.026 | 0.814 ± 0.043 |
| imf_linear_a1.41 | 0.856 ± 0.008 | 0.903 ± 0.020 | 0.944 ± 0.018 | 0.954 ± 0.016 | 0.948 ± 0.018 | 0.891 ± 0.021 |
| irmf_h2_a1.41    | 0.843 ± 0.010 | 0.878 ± 0.016 | 0.935 ± 0.017 | 0.942 ± 0.018 | 0.938 ± 0.017 | 0.856 ± 0.021 |

## Best classifier per method

| feature_set    | method           | classifier   |   balanced_accuracy |
|:---------------|:-----------------|:-------------|--------------------:|
| comp           | emd              | linsvm       |               0.825 |
| comp           | imf_linear_a1.41 | linsvm       |               0.941 |
| comp           | irmf_h2_a1.41    | linsvm       |               0.929 |
| comp           | raw              | knn          |               0.866 |
| comp+log_scale | emd              | logreg       |               0.865 |
| comp+log_scale | imf_linear_a1.41 | linsvm       |               0.954 |
| comp+log_scale | irmf_h2_a1.41    | linsvm       |               0.942 |
| comp+log_scale | raw              | knn          |               0.902 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a         | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-----------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41    | emd              |  0.1280 | [+0.102, +0.155] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0100 | [-0.020, -0.000] |      0.4505 |            0.0044 | 2/0/13 |
| comp           | lda          | imf_linear_a1.41 | emd              |  0.1390 | [+0.115, +0.162] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41    | raw              |  0.2440 | [+0.207, +0.281] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41 | raw              |  0.2550 | [+0.220, +0.289] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | lda          | emd              | raw              |  0.1160 | [+0.077, +0.155] |      0.0005 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | emd              |  0.0780 | [+0.029, +0.128] |      0.0902 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0410 | [-0.054, -0.027] |      0.0005 |            0.0044 | 0/0/15 |
| comp           | qda          | imf_linear_a1.41 | emd              |  0.1190 | [+0.070, +0.168] |      0.0037 |            0.0044 | 15/0/0 |
| comp           | qda          | irmf_h2_a1.41    | raw              |  0.0430 | [-0.024, +0.110] |      0.9573 |            0.0168 | 12/0/3 |
| comp           | qda          | imf_linear_a1.41 | raw              |  0.0840 | [+0.013, +0.155] |      0.3138 |            0.0044 | 15/0/0 |
| comp           | qda          | emd              | raw              | -0.0350 | [-0.123, +0.053] |      1.0000 |            0.3330 | 3/0/12 |
| comp           | gnb          | irmf_h2_a1.41    | emd              |  0.2570 | [+0.237, +0.276] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0140 | [-0.022, -0.005] |      0.0823 |            0.0044 | 1/0/14 |
| comp           | gnb          | imf_linear_a1.41 | emd              |  0.2700 | [+0.251, +0.290] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | gnb          | irmf_h2_a1.41    | raw              |  0.2630 | [+0.232, +0.293] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | gnb          | imf_linear_a1.41 | raw              |  0.2760 | [+0.247, +0.306] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | gnb          | emd              | raw              |  0.0060 | [-0.010, +0.023] |      1.0000 |            0.4543 | 10/0/5 |
| comp           | logreg       | irmf_h2_a1.41    | emd              |  0.1050 | [+0.074, +0.136] |      0.0001 |            0.0044 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0110 | [-0.019, -0.004] |      0.1121 |            0.0044 | 0/0/15 |
| comp           | logreg       | imf_linear_a1.41 | emd              |  0.1160 | [+0.088, +0.145] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41    | raw              |  0.1590 | [+0.128, +0.190] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41 | raw              |  0.1710 | [+0.140, +0.201] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | logreg       | emd              | raw              |  0.0540 | [+0.028, +0.081] |      0.0172 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | emd              |  0.2170 | [+0.193, +0.241] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0270 | [-0.042, -0.012] |      0.0487 |            0.0044 | 0/0/15 |
| comp           | knn          | imf_linear_a1.41 | emd              |  0.2440 | [+0.220, +0.268] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | knn          | irmf_h2_a1.41    | raw              |  0.0090 | [-0.016, +0.034] |      1.0000 |            0.4543 | 9/0/6  |
| comp           | knn          | imf_linear_a1.41 | raw              |  0.0360 | [+0.018, +0.054] |      0.0194 |            0.0044 | 15/0/0 |
| comp           | knn          | emd              | raw              | -0.2080 | [-0.232, -0.183] |      0.0000 |            0.0044 | 0/0/15 |
| comp           | linsvm       | irmf_h2_a1.41    | emd              |  0.1040 | [+0.076, +0.131] |      0.0001 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0120 | [-0.023, -0.002] |      0.3138 |            0.0044 | 1/0/14 |
| comp           | linsvm       | imf_linear_a1.41 | emd              |  0.1160 | [+0.089, +0.143] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41    | raw              |  0.1710 | [+0.135, +0.206] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41 | raw              |  0.1830 | [+0.152, +0.214] |      0.0000 |            0.0044 | 15/0/0 |
| comp           | linsvm       | emd              | raw              |  0.0670 | [+0.036, +0.098] |      0.0116 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | emd              |  0.0990 | [+0.068, +0.131] |      0.0003 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0090 | [-0.019, -0.000] |      0.4512 |            0.0044 | 0/0/15 |
| comp+log_scale | lda          | imf_linear_a1.41 | emd              |  0.1090 | [+0.081, +0.137] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41    | raw              |  0.1440 | [+0.102, +0.186] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41 | raw              |  0.1540 | [+0.115, +0.192] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | lda          | emd              | raw              |  0.0450 | [+0.016, +0.074] |      0.0929 |            0.0044 | 14/0/1 |
| comp+log_scale | qda          | irmf_h2_a1.41    | emd              |  0.0410 | [-0.003, +0.086] |      0.5205 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0360 | [-0.050, -0.022] |      0.0020 |            0.0044 | 0/0/15 |
| comp+log_scale | qda          | imf_linear_a1.41 | emd              |  0.0770 | [+0.037, +0.118] |      0.0266 |            0.0044 | 15/0/0 |
| comp+log_scale | qda          | irmf_h2_a1.41    | raw              | -0.0160 | [-0.031, -0.002] |      0.3810 |            0.0044 | 2/0/13 |
| comp+log_scale | qda          | imf_linear_a1.41 | raw              |  0.0190 | [+0.000, +0.038] |      0.4512 |            0.0044 | 13/0/2 |
| comp+log_scale | qda          | emd              | raw              | -0.0580 | [-0.100, -0.016] |      0.1640 |            0.0044 | 0/0/15 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | emd              |  0.2560 | [+0.237, +0.276] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0130 | [-0.021, -0.005] |      0.0902 |            0.0044 | 1/0/14 |
| comp+log_scale | gnb          | imf_linear_a1.41 | emd              |  0.2690 | [+0.250, +0.289] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | irmf_h2_a1.41    | raw              |  0.1740 | [+0.130, +0.218] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | imf_linear_a1.41 | raw              |  0.1870 | [+0.142, +0.232] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | gnb          | emd              | raw              | -0.0820 | [-0.112, -0.053] |      0.0011 |            0.0044 | 0/0/15 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | emd              |  0.0730 | [+0.047, +0.099] |      0.0011 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0090 | [-0.021, +0.003] |      0.7238 |            0.0044 | 2/0/13 |
| comp+log_scale | logreg       | imf_linear_a1.41 | emd              |  0.0830 | [+0.057, +0.108] |      0.0002 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41    | raw              |  0.1590 | [+0.124, +0.193] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41 | raw              |  0.1680 | [+0.137, +0.200] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | logreg       | emd              | raw              |  0.0860 | [+0.062, +0.110] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | emd              |  0.2010 | [+0.176, +0.225] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0250 | [-0.041, -0.010] |      0.0850 |            0.0044 | 0/0/15 |
| comp+log_scale | knn          | imf_linear_a1.41 | emd              |  0.2260 | [+0.203, +0.249] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | knn          | irmf_h2_a1.41    | raw              | -0.0250 | [-0.052, +0.002] |      0.5205 |            0.0157 | 4/0/11 |
| comp+log_scale | knn          | imf_linear_a1.41 | raw              |  0.0010 | [-0.021, +0.023] |      1.0000 |            0.9341 | 7/0/8  |
| comp+log_scale | knn          | emd              | raw              | -0.2260 | [-0.253, -0.198] |      0.0000 |            0.0044 | 0/0/15 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | emd              |  0.0780 | [+0.058, +0.099] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | imf_linear_a1.41 | -0.0120 | [-0.021, -0.003] |      0.2327 |            0.0044 | 0/0/15 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | emd              |  0.0900 | [+0.065, +0.115] |      0.0001 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41    | raw              |  0.1660 | [+0.134, +0.197] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41 | raw              |  0.1770 | [+0.147, +0.208] |      0.0000 |            0.0044 | 15/0/0 |
| comp+log_scale | linsvm       | emd              | raw              |  0.0870 | [+0.065, +0.109] |      0.0000 |            0.0044 | 15/0/0 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method           |   max_eps2 |   max_mi |   n_features |
|:-----------------|-----------:|---------:|-------------:|
| raw              |      0.090 |    0.116 |       42.000 |
| emd              |      0.099 |    0.116 |      362.000 |
| imf_linear_a1.41 |      0.136 |    0.126 |      722.000 |
| irmf_h2_a1.41    |      0.123 |    0.126 |      722.000 |

| method           | feature              | slot_label   |   epsilon2 |    mi |
|:-----------------|:---------------------|:-------------|-----------:|------:|
| imf_linear_a1.41 | ch0_c10_hjorth_cpx   | w129         |      0.136 | 0.115 |
| imf_linear_a1.41 | ch0_c11_hjorth_cpx   | w183         |      0.131 | 0.098 |
| imf_linear_a1.41 | ch0_c9_hjorth_cpx    | w93          |      0.126 | 0.111 |
| imf_linear_a1.41 | ch1_c1_hjorth_cpx    | w7           |      0.125 | 0.114 |
| irmf_h2_a1.41    | ch1_c1_hjorth_cpx    | w7           |      0.123 | 0.106 |
| irmf_h2_a1.41    | ch1_c2_hjorth_cpx    | w9           |      0.120 | 0.113 |
| imf_linear_a1.41 | ch0_c11_hilb_if_mean | w183         |      0.116 | 0.091 |
| irmf_h2_a1.41    | ch1_c2_hjorth_mob    | w9           |      0.115 | 0.098 |
| irmf_h2_a1.41    | ch0_c11_crest        | w183         |      0.107 | 0.072 |
| irmf_h2_a1.41    | ch0_c10_crest        | w129         |      0.107 | 0.075 |
| emd              | ch0_c0_hilb_amp_std  | imf1         |      0.099 | 0.085 |
| raw              | ch0_c0_std           | frame        |      0.090 | 0.075 |
| raw              | ch0_c0_log_energy    | frame        |      0.089 | 0.075 |
| raw              | ch0_c0_mean_abs      | frame        |      0.088 | 0.075 |
| raw              | ch0_c0_hilb_amp_mean | frame        |      0.086 | 0.080 |
| raw              | ch0_c0_hilb_amp_std  | frame        |      0.085 | 0.074 |
| emd              | ch0_c0_log_energy    | imf1         |      0.082 | 0.082 |
| emd              | ch0_c0_std           | imf1         |      0.082 | 0.082 |
| emd              | ch0_log_scale        | frame        |      0.081 | 0.116 |
| emd              | ch0_c0_hilb_amp_mean | imf1         |      0.076 | 0.068 |
