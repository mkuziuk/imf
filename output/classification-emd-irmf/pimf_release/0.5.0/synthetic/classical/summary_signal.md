# Classical baselines: synthetic / signal

1500 frames, 150 groups, 5 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 97 s with 4 workers.

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| raw                | 0.460 ± 0.041 | 0.460 ± 0.041 | 0.474 ± 0.044 |
| emd                | 0.475 ± 0.041 | 0.505 ± 0.043 | 0.504 ± 0.047 |
| imf_linear_a1.41   | 0.749 ± 0.040 | 0.697 ± 0.037 | 0.730 ± 0.042 |
| irmf_h2_a1.41_conv | 0.870 ± 0.035 | 0.823 ± 0.030 | 0.845 ± 0.037 |

Feature set `comp+log_scale`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| raw                | 0.452 ± 0.051 | 0.450 ± 0.041 | 0.463 ± 0.043 |
| emd                | 0.470 ± 0.050 | 0.499 ± 0.050 | 0.502 ± 0.051 |
| imf_linear_a1.41   | 0.754 ± 0.041 | 0.689 ± 0.032 | 0.732 ± 0.047 |
| irmf_h2_a1.41_conv | 0.869 ± 0.034 | 0.822 ± 0.033 | 0.848 ± 0.033 |

## Best classifier per method

| feature_set    | method             | classifier   |   balanced_accuracy |
|:---------------|:-------------------|:-------------|--------------------:|
| comp           | emd                | linsvm       |               0.505 |
| comp           | imf_linear_a1.41   | lda          |               0.749 |
| comp           | irmf_h2_a1.41_conv | lda          |               0.870 |
| comp           | raw                | logreg       |               0.474 |
| comp+log_scale | emd                | logreg       |               0.502 |
| comp+log_scale | imf_linear_a1.41   | lda          |               0.754 |
| comp+log_scale | irmf_h2_a1.41_conv | lda          |               0.869 |
| comp+log_scale | raw                | logreg       |               0.463 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a           | method_b         |   diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-------------------|:-----------------|-------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41_conv | emd              | 0.3950 | [+0.349, +0.442] |      0.0000 |            0.0136 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41_conv | imf_linear_a1.41 | 0.1210 | [+0.083, +0.158] |      0.0001 |            0.0136 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41   | emd              | 0.2750 | [+0.233, +0.317] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41_conv | raw              | 0.4100 | [+0.345, +0.475] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41   | raw              | 0.2900 | [+0.241, +0.338] |      0.0000 |            0.0136 | 15/0/0 |
| comp           | lda          | emd                | raw              | 0.0150 | [-0.046, +0.076] |      1.0000 |            0.3626 | 10/1/4 |
| comp           | logreg       | irmf_h2_a1.41_conv | emd              | 0.3410 | [+0.294, +0.388] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41_conv | imf_linear_a1.41 | 0.1150 | [+0.077, +0.153] |      0.0001 |            0.0136 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41   | emd              | 0.2260 | [+0.175, +0.277] |      0.0000 |            0.0136 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41_conv | raw              | 0.3710 | [+0.323, +0.418] |      0.0000 |            0.0136 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41   | raw              | 0.2560 | [+0.198, +0.313] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | logreg       | emd                | raw              | 0.0300 | [-0.030, +0.090] |      0.9183 |            0.1312 | 10/0/5 |
| comp           | linsvm       | irmf_h2_a1.41_conv | emd              | 0.3180 | [+0.274, +0.362] |      0.0000 |            0.0136 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41_conv | imf_linear_a1.41 | 0.1260 | [+0.079, +0.174] |      0.0004 |            0.0136 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41   | emd              | 0.1920 | [+0.142, +0.241] |      0.0000 |            0.0136 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41_conv | raw              | 0.3630 | [+0.306, +0.421] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41   | raw              | 0.2370 | [+0.179, +0.295] |      0.0000 |            0.0136 | 15/0/0 |
| comp           | linsvm       | emd                | raw              | 0.0450 | [-0.008, +0.098] |      0.4425 |            0.0136 | 13/0/2 |
| comp+log_scale | lda          | irmf_h2_a1.41_conv | emd              | 0.3990 | [+0.343, +0.455] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41_conv | imf_linear_a1.41 | 0.1150 | [+0.090, +0.141] |      0.0000 |            0.0136 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41   | emd              | 0.2840 | [+0.232, +0.335] |      0.0000 |            0.0136 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41_conv | raw              | 0.4170 | [+0.356, +0.478] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41   | raw              | 0.3020 | [+0.247, +0.356] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | emd                | raw              | 0.0180 | [-0.043, +0.078] |      1.0000 |            0.2234 | 10/0/5 |
| comp+log_scale | logreg       | irmf_h2_a1.41_conv | emd              | 0.3450 | [+0.300, +0.390] |      0.0000 |            0.0136 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41_conv | imf_linear_a1.41 | 0.1160 | [+0.072, +0.160] |      0.0004 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41   | emd              | 0.2290 | [+0.177, +0.281] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41_conv | raw              | 0.3850 | [+0.344, +0.425] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41   | raw              | 0.2690 | [+0.216, +0.322] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | emd                | raw              | 0.0400 | [-0.022, +0.101] |      0.7471 |            0.0303 | 11/0/4 |
| comp+log_scale | linsvm       | irmf_h2_a1.41_conv | emd              | 0.3220 | [+0.272, +0.373] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41_conv | imf_linear_a1.41 | 0.1320 | [+0.089, +0.176] |      0.0001 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41   | emd              | 0.1900 | [+0.135, +0.245] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41_conv | raw              | 0.3720 | [+0.316, +0.427] |      0.0000 |            0.0136 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41   | raw              | 0.2390 | [+0.197, +0.282] |      0.0000 |            0.0136 | 15/0/0 |
| comp+log_scale | linsvm       | emd                | raw              | 0.0490 | [-0.001, +0.100] |      0.3285 |            0.0136 | 12/0/3 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method             |   max_eps2 |   max_mi |   n_features |
|:-------------------|-----------:|---------:|-------------:|
| raw                |      0.218 |    0.173 |       21.000 |
| emd                |      0.110 |    0.126 |      181.000 |
| imf_linear_a1.41   |      0.242 |    0.280 |      321.000 |
| irmf_h2_a1.41_conv |      0.568 |    0.517 |      321.000 |

| method             | feature              | slot_label   |   epsilon2 |    mi |
|:-------------------|:---------------------|:-------------|-----------:|------:|
| irmf_h2_a1.41_conv | ch0_c5_kurt          | w25          |      0.568 | 0.517 |
| irmf_h2_a1.41_conv | ch0_c6_kurt          | w33          |      0.559 | 0.505 |
| irmf_h2_a1.41_conv | ch0_c7_kurt          | w47          |      0.490 | 0.380 |
| irmf_h2_a1.41_conv | ch0_c5_hilb_amp_std  | w25          |      0.462 | 0.385 |
| irmf_h2_a1.41_conv | ch0_c6_crest         | w33          |      0.398 | 0.292 |
| imf_linear_a1.41   | ch0_c5_spec_entropy  | w25          |      0.242 | 0.214 |
| imf_linear_a1.41   | ch0_c5_hilb_amp_std  | w25          |      0.242 | 0.174 |
| imf_linear_a1.41   | ch0_c5_perm_entropy  | w25          |      0.225 | 0.180 |
| imf_linear_a1.41   | ch0_c5_extrema_rate  | w25          |      0.218 | 0.177 |
| raw                | ch0_c0_spec_peak     | frame        |      0.218 | 0.173 |
| imf_linear_a1.41   | ch0_c8_spec_entropy  | w65          |      0.211 | 0.144 |
| emd                | ch0_c3_hilb_if_std   | imf4         |      0.110 | 0.057 |
| emd                | ch0_c2_kurt          | imf3         |      0.105 | 0.084 |
| raw                | ch0_c0_spec_entropy  | frame        |      0.089 | 0.052 |
| emd                | ch0_c3_perm_entropy  | imf4         |      0.077 | 0.044 |
| emd                | ch0_c3_extrema_rate  | imf4         |      0.077 | 0.047 |
| emd                | ch0_c3_zcr           | imf4         |      0.074 | 0.045 |
| raw                | ch0_c0_hilb_if_mean  | frame        |      0.050 | 0.050 |
| raw                | ch0_c0_mean_abs      | frame        |      0.041 | 0.035 |
| raw                | ch0_c0_hilb_amp_mean | frame        |      0.027 | 0.030 |
