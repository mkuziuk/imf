# Classical baselines: birdclef2023 / species

12000 frames, 451 groups, 16 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 7687 s with 4 workers.

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| raw                | 0.340 ± 0.009 | 0.353 ± 0.008 | 0.376 ± 0.011 |
| emd                | 0.403 ± 0.018 | 0.433 ± 0.016 | 0.435 ± 0.016 |
| imf_linear_a1.41   | 0.477 ± 0.021 | 0.493 ± 0.019 | 0.489 ± 0.019 |
| irmf_h2_a1.41_conv | 0.475 ± 0.022 | 0.495 ± 0.019 | 0.496 ± 0.016 |

Feature set `comp+log_scale`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| raw                | 0.343 ± 0.011 | 0.355 ± 0.012 | 0.376 ± 0.017 |
| emd                | 0.405 ± 0.019 | 0.436 ± 0.016 | 0.433 ± 0.016 |
| imf_linear_a1.41   | 0.476 ± 0.020 | 0.492 ± 0.017 | 0.486 ± 0.017 |
| irmf_h2_a1.41_conv | 0.474 ± 0.021 | 0.496 ± 0.019 | 0.493 ± 0.018 |

## Best classifier per method

| feature_set    | method             | classifier   |   balanced_accuracy |
|:---------------|:-------------------|:-------------|--------------------:|
| comp           | emd                | logreg       |               0.435 |
| comp           | imf_linear_a1.41   | linsvm       |               0.493 |
| comp           | irmf_h2_a1.41_conv | logreg       |               0.496 |
| comp           | raw                | logreg       |               0.376 |
| comp+log_scale | emd                | linsvm       |               0.436 |
| comp+log_scale | imf_linear_a1.41   | linsvm       |               0.492 |
| comp+log_scale | irmf_h2_a1.41_conv | linsvm       |               0.496 |
| comp+log_scale | raw                | logreg       |               0.376 |

## Paired comparisons, balanced accuracy (A - B; Holm within the table)

CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.

| feature_set    | classifier   | method_a           | method_b         |    diff | ci               |   p_nb_holm |   p_wilcoxon_holm | wins   |
|:---------------|:-------------|:-------------------|:-----------------|--------:|:-----------------|------------:|------------------:|:-------|
| comp           | lda          | irmf_h2_a1.41_conv | emd              |  0.0720 | [+0.048, +0.096] |      0.0001 |            0.0022 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0020 | [-0.013, +0.009] |      1.0000 |            1.0000 | 6/0/9  |
| comp           | lda          | imf_linear_a1.41   | emd              |  0.0740 | [+0.054, +0.094] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | lda          | irmf_h2_a1.41_conv | raw              |  0.1340 | [+0.106, +0.163] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | lda          | imf_linear_a1.41   | raw              |  0.1360 | [+0.110, +0.163] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | lda          | emd                | raw              |  0.0630 | [+0.043, +0.083] |      0.0001 |            0.0022 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41_conv | emd              |  0.0610 | [+0.048, +0.074] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41_conv | imf_linear_a1.41 |  0.0070 | [-0.005, +0.019] |      1.0000 |            0.0903 | 11/0/4 |
| comp           | logreg       | imf_linear_a1.41   | emd              |  0.0540 | [+0.042, +0.066] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | logreg       | irmf_h2_a1.41_conv | raw              |  0.1200 | [+0.099, +0.140] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | logreg       | imf_linear_a1.41   | raw              |  0.1120 | [+0.092, +0.133] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | logreg       | emd                | raw              |  0.0580 | [+0.043, +0.073] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41_conv | emd              |  0.0620 | [+0.049, +0.075] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41_conv | imf_linear_a1.41 |  0.0020 | [-0.011, +0.015] |      1.0000 |            1.0000 | 8/0/7  |
| comp           | linsvm       | imf_linear_a1.41   | emd              |  0.0600 | [+0.048, +0.071] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | linsvm       | irmf_h2_a1.41_conv | raw              |  0.1420 | [+0.118, +0.166] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | linsvm       | imf_linear_a1.41   | raw              |  0.1400 | [+0.118, +0.161] |      0.0000 |            0.0022 | 15/0/0 |
| comp           | linsvm       | emd                | raw              |  0.0800 | [+0.060, +0.100] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41_conv | emd              |  0.0690 | [+0.047, +0.092] |      0.0001 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41_conv | imf_linear_a1.41 | -0.0020 | [-0.012, +0.007] |      1.0000 |            1.0000 | 7/0/8  |
| comp+log_scale | lda          | imf_linear_a1.41   | emd              |  0.0720 | [+0.052, +0.091] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | irmf_h2_a1.41_conv | raw              |  0.1320 | [+0.113, +0.150] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | imf_linear_a1.41   | raw              |  0.1340 | [+0.117, +0.151] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | lda          | emd                | raw              |  0.0620 | [+0.048, +0.077] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41_conv | emd              |  0.0590 | [+0.043, +0.075] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41_conv | imf_linear_a1.41 |  0.0070 | [-0.004, +0.018] |      1.0000 |            0.0615 | 12/0/3 |
| comp+log_scale | logreg       | imf_linear_a1.41   | emd              |  0.0520 | [+0.040, +0.064] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | irmf_h2_a1.41_conv | raw              |  0.1170 | [+0.100, +0.134] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | imf_linear_a1.41   | raw              |  0.1100 | [+0.093, +0.127] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | logreg       | emd                | raw              |  0.0580 | [+0.043, +0.073] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41_conv | emd              |  0.0590 | [+0.047, +0.072] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41_conv | imf_linear_a1.41 |  0.0030 | [-0.009, +0.015] |      1.0000 |            1.0000 | 11/0/4 |
| comp+log_scale | linsvm       | imf_linear_a1.41   | emd              |  0.0560 | [+0.046, +0.067] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | irmf_h2_a1.41_conv | raw              |  0.1400 | [+0.125, +0.155] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | imf_linear_a1.41   | raw              |  0.1370 | [+0.124, +0.151] |      0.0000 |            0.0022 | 15/0/0 |
| comp+log_scale | linsvm       | emd                | raw              |  0.0810 | [+0.068, +0.093] |      0.0000 |            0.0022 | 15/0/0 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method             |   max_eps2 |   max_mi |   n_features |
|:-------------------|-----------:|---------:|-------------:|
| raw                |      0.362 |    0.657 |       21.000 |
| emd                |      0.415 |    0.514 |      181.000 |
| imf_linear_a1.41   |      0.483 |    0.633 |      361.000 |
| irmf_h2_a1.41_conv |      0.472 |    0.646 |      361.000 |

| method             | feature              | slot_label   |   epsilon2 |    mi |
|:-------------------|:---------------------|:-------------|-----------:|------:|
| imf_linear_a1.41   | ch0_c2_hilb_if_mean  | w9           |      0.483 | 0.456 |
| imf_linear_a1.41   | ch0_c2_zcr           | w9           |      0.477 | 0.439 |
| imf_linear_a1.41   | ch0_c2_spec_centroid | w9           |      0.473 | 0.440 |
| irmf_h2_a1.41_conv | ch0_c2_hilb_if_mean  | w9           |      0.472 | 0.450 |
| irmf_h2_a1.41_conv | ch0_c4_extrema_rate  | w17          |      0.470 | 0.479 |
| irmf_h2_a1.41_conv | ch0_c2_spec_centroid | w9           |      0.465 | 0.439 |
| imf_linear_a1.41   | ch0_c4_extrema_rate  | w17          |      0.464 | 0.496 |
| imf_linear_a1.41   | ch0_c5_energy_frac   | w25          |      0.464 | 0.486 |
| irmf_h2_a1.41_conv | ch0_c2_zcr           | w9           |      0.462 | 0.430 |
| irmf_h2_a1.41_conv | ch0_c2_hjorth_mob    | w9           |      0.461 | 0.443 |
| emd                | ch0_c1_energy_frac   | imf2         |      0.415 | 0.301 |
| emd                | ch0_c1_mean_abs      | imf2         |      0.392 | 0.274 |
| emd                | ch0_c1_hilb_amp_mean | imf2         |      0.392 | 0.274 |
| emd                | ch0_c1_log_energy    | imf2         |      0.384 | 0.269 |
| emd                | ch0_c1_std           | imf2         |      0.384 | 0.269 |
| raw                | ch0_c0_extrema_rate  | frame        |      0.362 | 0.405 |
| raw                | ch0_c0_hjorth_mob    | frame        |      0.333 | 0.555 |
| raw                | ch0_c0_zcr           | frame        |      0.324 | 0.522 |
| raw                | ch0_c0_spec_centroid | frame        |      0.314 | 0.564 |
| raw                | ch0_c0_hilb_if_mean  | frame        |      0.313 | 0.571 |
