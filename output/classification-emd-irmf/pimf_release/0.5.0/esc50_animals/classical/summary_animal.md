# Classical baselines: esc50_animals / animal

5100 frames, 157 groups, 10 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 1625 s with 4 workers.

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| raw                | 0.419 ± 0.042 | 0.443 ± 0.044 | 0.459 ± 0.041 |
| emd                | 0.523 ± 0.043 | 0.543 ± 0.045 | 0.537 ± 0.054 |
| imf_linear_a1.41   | 0.637 ± 0.033 | 0.653 ± 0.036 | 0.651 ± 0.034 |
| irmf_h2_a1.41_conv | 0.625 ± 0.029 | 0.650 ± 0.036 | 0.651 ± 0.039 |

Feature set `comp+log_scale`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| raw                | 0.446 ± 0.052 | 0.459 ± 0.048 | 0.474 ± 0.045 |
| emd                | 0.539 ± 0.041 | 0.565 ± 0.041 | 0.557 ± 0.051 |
| imf_linear_a1.41   | 0.649 ± 0.033 | 0.674 ± 0.032 | 0.665 ± 0.035 |
| irmf_h2_a1.41_conv | 0.639 ± 0.028 | 0.666 ± 0.034 | 0.663 ± 0.039 |

## Best classifier per method

| feature_set    | method             | classifier   |   balanced_accuracy |
|:---------------|:-------------------|:-------------|--------------------:|
| comp           | emd                | linsvm       |               0.543 |
| comp           | imf_linear_a1.41   | linsvm       |               0.653 |
| comp           | irmf_h2_a1.41_conv | logreg       |               0.651 |
| comp           | raw                | logreg       |               0.459 |
| comp+log_scale | emd                | linsvm       |               0.565 |
| comp+log_scale | imf_linear_a1.41   | linsvm       |               0.674 |
| comp+log_scale | irmf_h2_a1.41_conv | linsvm       |               0.666 |
| comp+log_scale | raw                | logreg       |               0.474 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a           | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-------------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41_conv | emd              |  0.1020 | [+0.068, +0.136] |      0.0003 |            0.0022 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0110 | [-0.035, +0.012] |      1.0000 |            0.1809 | 3/0/12 |
| comp           | lda          | imf_linear_a1.41   | emd              |  0.1130 | [+0.091, +0.135] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41_conv | raw              |  0.2060 | [+0.160, +0.252] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41   | raw              |  0.2170 | [+0.169, +0.266] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | lda          | emd                | raw              |  0.1040 | [+0.048, +0.161] |      0.0170 |            0.0022 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41_conv | emd              |  0.1140 | [+0.077, +0.151] |      0.0002 |            0.0022 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0010 | [-0.024, +0.022] |      1.0000 |            1.0000 | 7/0/8  |
| comp           | logreg       | imf_linear_a1.41   | emd              |  0.1150 | [+0.081, +0.149] |      0.0001 |            0.0022 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41_conv | raw              |  0.1920 | [+0.142, +0.242] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41   | raw              |  0.1930 | [+0.147, +0.238] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | logreg       | emd                | raw              |  0.0780 | [+0.030, +0.126] |      0.0342 |            0.0022 | 14/0/1 |
| comp           | linsvm       | irmf_h2_a1.41_conv | emd              |  0.1070 | [+0.072, +0.141] |      0.0002 |            0.0022 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0030 | [-0.040, +0.034] |      1.0000 |            1.0000 | 8/0/7  |
| comp           | linsvm       | imf_linear_a1.41   | emd              |  0.1090 | [+0.077, +0.142] |      0.0001 |            0.0022 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41_conv | raw              |  0.2060 | [+0.146, +0.267] |      0.0001 |            0.0022 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41   | raw              |  0.2090 | [+0.156, +0.263] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | linsvm       | emd                | raw              |  0.1000 | [+0.041, +0.158] |      0.0261 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41_conv | emd              |  0.1000 | [+0.070, +0.130] |      0.0001 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0110 | [-0.037, +0.016] |      1.0000 |            0.5350 | 6/0/9  |
| comp+log_scale | lda          | imf_linear_a1.41   | emd              |  0.1110 | [+0.089, +0.133] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41_conv | raw              |  0.1930 | [+0.134, +0.252] |      0.0001 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41   | raw              |  0.2030 | [+0.139, +0.268] |      0.0002 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | emd                | raw              |  0.0930 | [+0.027, +0.158] |      0.0723 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41_conv | emd              |  0.1060 | [+0.070, +0.141] |      0.0003 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0030 | [-0.022, +0.017] |      1.0000 |            1.0000 | 6/0/9  |
| comp+log_scale | logreg       | imf_linear_a1.41   | emd              |  0.1080 | [+0.077, +0.139] |      0.0001 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41_conv | raw              |  0.1890 | [+0.119, +0.258] |      0.0006 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41   | raw              |  0.1910 | [+0.128, +0.255] |      0.0003 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | emd                | raw              |  0.0830 | [+0.020, +0.146] |      0.0936 |            0.0022 | 13/0/2 |
| comp+log_scale | linsvm       | irmf_h2_a1.41_conv | emd              |  0.1010 | [+0.066, +0.136] |      0.0004 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0080 | [-0.033, +0.016] |      1.0000 |            0.5350 | 4/0/11 |
| comp+log_scale | linsvm       | imf_linear_a1.41   | emd              |  0.1090 | [+0.082, +0.135] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41_conv | raw              |  0.2070 | [+0.130, +0.284] |      0.0006 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41   | raw              |  0.2150 | [+0.148, +0.283] |      0.0002 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | emd                | raw              |  0.1060 | [+0.049, +0.164] |      0.0170 |            0.0022 | 15/0/0 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method             |   max_eps2 |   max_mi |   n_features |
|:-------------------|-----------:|---------:|-------------:|
| raw                |      0.419 |    0.551 |       21.000 |
| emd                |      0.492 |    0.394 |      181.000 |
| imf_linear_a1.41   |      0.596 |    0.530 |      361.000 |
| irmf_h2_a1.41_conv |      0.610 |    0.547 |      361.000 |

| method             | feature              | slot_label   |   epsilon2 |    mi |
|:-------------------|:---------------------|:-------------|-----------:|------:|
| irmf_h2_a1.41_conv | ch0_c8_hjorth_mob    | w65          |      0.610 | 0.547 |
| imf_linear_a1.41   | ch0_c8_hjorth_mob    | w65          |      0.596 | 0.530 |
| irmf_h2_a1.41_conv | ch0_c8_hilb_if_mean  | w65          |      0.593 | 0.514 |
| imf_linear_a1.41   | ch0_c10_energy_frac  | w129         |      0.592 | 0.507 |
| irmf_h2_a1.41_conv | ch0_c10_energy_frac  | w129         |      0.592 | 0.492 |
| irmf_h2_a1.41_conv | ch0_c8_spec_centroid | w65          |      0.588 | 0.505 |
| irmf_h2_a1.41_conv | ch0_c9_hjorth_mob    | w93          |      0.585 | 0.504 |
| imf_linear_a1.41   | ch0_c8_hilb_if_mean  | w65          |      0.584 | 0.508 |
| imf_linear_a1.41   | ch0_c9_energy_frac   | w93          |      0.582 | 0.519 |
| imf_linear_a1.41   | ch0_c8_spec_centroid | w65          |      0.580 | 0.499 |
| emd                | ch0_c3_energy_frac   | imf4         |      0.492 | 0.370 |
| emd                | ch0_c3_mean_abs      | imf4         |      0.447 | 0.331 |
| emd                | ch0_c3_hilb_amp_mean | imf4         |      0.446 | 0.330 |
| emd                | ch0_c3_log_energy    | imf4         |      0.426 | 0.314 |
| emd                | ch0_c3_std           | imf4         |      0.426 | 0.315 |
| raw                | ch0_c0_hjorth_cpx    | frame        |      0.419 | 0.366 |
| raw                | ch0_c0_hilb_if_mean  | frame        |      0.416 | 0.428 |
| raw                | ch0_c0_spec_centroid | frame        |      0.382 | 0.398 |
| raw                | ch0_c0_hjorth_mob    | frame        |      0.375 | 0.355 |
| raw                | ch0_c0_zcr           | frame        |      0.367 | 0.345 |
