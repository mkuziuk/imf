# Classical baselines: dronerf / 10class

12000 frames, 227 groups, 10 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 41416 s with 4 workers.

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| raw                | 0.332 ± 0.011 | 0.350 ± 0.014 | 0.355 ± 0.014 |
| emd                | 0.355 ± 0.013 | 0.389 ± 0.012 | 0.388 ± 0.012 |
| imf_linear_a1.41   | 0.518 ± 0.017 | 0.579 ± 0.012 | 0.567 ± 0.012 |
| irmf_h2_a1.41_conv | 0.465 ± 0.014 | 0.503 ± 0.010 | 0.500 ± 0.013 |

Feature set `comp+log_scale`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| raw                | 0.364 ± 0.014 | 0.391 ± 0.014 | 0.400 ± 0.014 |
| emd                | 0.396 ± 0.013 | 0.429 ± 0.010 | 0.429 ± 0.012 |
| imf_linear_a1.41   | 0.537 ± 0.017 | 0.600 ± 0.011 | 0.590 ± 0.012 |
| irmf_h2_a1.41_conv | 0.514 ± 0.016 | 0.544 ± 0.011 | 0.544 ± 0.008 |

## Best classifier per method

| feature_set    | method             | classifier   |   balanced_accuracy |
|:---------------|:-------------------|:-------------|--------------------:|
| comp           | emd                | linsvm       |               0.389 |
| comp           | imf_linear_a1.41   | linsvm       |               0.579 |
| comp           | irmf_h2_a1.41_conv | linsvm       |               0.503 |
| comp           | raw                | logreg       |               0.355 |
| comp+log_scale | emd                | linsvm       |               0.429 |
| comp+log_scale | imf_linear_a1.41   | linsvm       |               0.600 |
| comp+log_scale | irmf_h2_a1.41_conv | logreg       |               0.544 |
| comp+log_scale | raw                | logreg       |               0.400 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a           | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-------------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41_conv | emd              |  0.1100 | [+0.099, +0.120] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0530 | [-0.071, -0.035] |      0.0001 |            0.0022 | 0/0/15 |
| comp           | lda          | imf_linear_a1.41   | emd              |  0.1630 | [+0.144, +0.182] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41_conv | raw              |  0.1330 | [+0.119, +0.147] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41   | raw              |  0.1870 | [+0.160, +0.213] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | lda          | emd                | raw              |  0.0240 | [+0.010, +0.038] |      0.0051 |            0.0022 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41_conv | emd              |  0.1120 | [+0.097, +0.128] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0670 | [-0.078, -0.056] |      0.0000 |            0.0022 | 0/0/15 |
| comp           | logreg       | imf_linear_a1.41   | emd              |  0.1790 | [+0.161, +0.198] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41_conv | raw              |  0.1460 | [+0.125, +0.166] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41   | raw              |  0.2120 | [+0.188, +0.237] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | logreg       | emd                | raw              |  0.0330 | [+0.020, +0.047] |      0.0005 |            0.0022 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41_conv | emd              |  0.1150 | [+0.103, +0.126] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0760 | [-0.088, -0.063] |      0.0000 |            0.0022 | 0/0/15 |
| comp           | linsvm       | imf_linear_a1.41   | emd              |  0.1900 | [+0.172, +0.209] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41_conv | raw              |  0.1540 | [+0.136, +0.171] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41   | raw              |  0.2290 | [+0.207, +0.251] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | linsvm       | emd                | raw              |  0.0390 | [+0.025, +0.053] |      0.0002 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41_conv | emd              |  0.1180 | [+0.105, +0.132] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0220 | [-0.037, -0.007] |      0.0063 |            0.0022 | 1/0/14 |
| comp+log_scale | lda          | imf_linear_a1.41   | emd              |  0.1410 | [+0.121, +0.160] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41_conv | raw              |  0.1500 | [+0.134, +0.166] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41   | raw              |  0.1720 | [+0.148, +0.197] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | emd                | raw              |  0.0320 | [+0.016, +0.048] |      0.0022 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41_conv | emd              |  0.1150 | [+0.102, +0.128] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0450 | [-0.056, -0.035] |      0.0000 |            0.0022 | 0/0/15 |
| comp+log_scale | logreg       | imf_linear_a1.41   | emd              |  0.1600 | [+0.143, +0.178] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41_conv | raw              |  0.1450 | [+0.132, +0.158] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41   | raw              |  0.1900 | [+0.173, +0.207] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | emd                | raw              |  0.0300 | [+0.017, +0.042] |      0.0007 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41_conv | emd              |  0.1140 | [+0.104, +0.124] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0560 | [-0.068, -0.045] |      0.0000 |            0.0022 | 0/0/15 |
| comp+log_scale | linsvm       | imf_linear_a1.41   | emd              |  0.1700 | [+0.157, +0.184] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41_conv | raw              |  0.1530 | [+0.142, +0.164] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41   | raw              |  0.2090 | [+0.194, +0.224] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | emd                | raw              |  0.0390 | [+0.025, +0.052] |      0.0002 |            0.0022 | 15/0/0 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method             |   max_eps2 |   max_mi |   n_features |
|:-------------------|-----------:|---------:|-------------:|
| raw                |      0.261 |    0.408 |       42.000 |
| emd                |      0.284 |    0.408 |      362.000 |
| imf_linear_a1.41   |      0.381 |    0.423 |      722.000 |
| irmf_h2_a1.41_conv |      0.311 |    0.408 |      722.000 |

| method             | feature              | slot_label   |   epsilon2 |    mi |
|:-------------------|:---------------------|:-------------|-----------:|------:|
| imf_linear_a1.41   | ch0_c10_spec_bw      | w129         |      0.381 | 0.388 |
| imf_linear_a1.41   | ch0_c7_hilb_if_mean  | w47          |      0.381 | 0.422 |
| imf_linear_a1.41   | ch0_c10_hilb_if_std  | w129         |      0.379 | 0.356 |
| imf_linear_a1.41   | ch0_c9_hilb_if_std   | w93          |      0.379 | 0.356 |
| imf_linear_a1.41   | ch0_c8_hilb_if_mean  | w65          |      0.374 | 0.372 |
| irmf_h2_a1.41_conv | ch0_c8_hjorth_mob    | w65          |      0.311 | 0.300 |
| irmf_h2_a1.41_conv | ch0_c7_hjorth_mob    | w47          |      0.309 | 0.310 |
| irmf_h2_a1.41_conv | ch0_c11_energy_frac  | w183         |      0.308 | 0.279 |
| irmf_h2_a1.41_conv | ch0_c10_energy_frac  | w129         |      0.307 | 0.289 |
| irmf_h2_a1.41_conv | ch0_c12_energy_frac  | w257         |      0.305 | 0.270 |
| emd                | ch0_c3_energy_frac   | imf4         |      0.284 | 0.266 |
| emd                | ch0_c4_energy_frac   | imf5         |      0.273 | 0.239 |
| emd                | ch0_c0_hjorth_mob    | imf1         |      0.263 | 0.330 |
| emd                | ch0_c0_spec_centroid | imf1         |      0.263 | 0.321 |
| raw                | ch0_c0_spec_bw       | frame        |      0.261 | 0.320 |
| emd                | ch0_c5_energy_frac   | imf6         |      0.260 | 0.211 |
| raw                | ch0_c0_hjorth_cpx    | frame        |      0.237 | 0.353 |
| raw                | ch0_c0_hilb_if_std   | frame        |      0.229 | 0.344 |
| raw                | ch0_c0_extrema_rate  | frame        |      0.204 | 0.247 |
| raw                | ch0_c0_hilb_if_mean  | frame        |      0.195 | 0.317 |
