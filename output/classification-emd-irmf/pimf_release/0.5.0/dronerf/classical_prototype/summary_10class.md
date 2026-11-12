# Classical baselines: dronerf / 10class

12000 frames, 227 groups, 10 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 26850 s with 3 workers.

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| irmf_h2_a1.41_conv | 0.464 ± 0.014 | 0.503 ± 0.010 | 0.501 ± 0.012 |

Feature set `comp+log_scale`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| irmf_h2_a1.41_conv | 0.514 ± 0.016 | 0.543 ± 0.011 | 0.544 ± 0.009 |

## Best classifier per method

| feature_set    | method             | classifier   |   balanced_accuracy |
|:---------------|:-------------------|:-------------|--------------------:|
| comp           | irmf_h2_a1.41_conv | linsvm       |               0.503 |
| comp+log_scale | irmf_h2_a1.41_conv | logreg       |               0.544 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method             |   max_eps2 |   max_mi |   n_features |
|:-------------------|-----------:|---------:|-------------:|
| irmf_h2_a1.41_conv |      0.311 |    0.408 |      722.000 |

| method             | feature             | slot_label   |   epsilon2 |    mi |
|:-------------------|:--------------------|:-------------|-----------:|------:|
| irmf_h2_a1.41_conv | ch0_c8_hjorth_mob   | w65          |      0.311 | 0.300 |
| irmf_h2_a1.41_conv | ch0_c7_hjorth_mob   | w47          |      0.309 | 0.310 |
| irmf_h2_a1.41_conv | ch0_c11_energy_frac | w183         |      0.308 | 0.279 |
| irmf_h2_a1.41_conv | ch0_c10_energy_frac | w129         |      0.307 | 0.289 |
| irmf_h2_a1.41_conv | ch0_c12_energy_frac | w257         |      0.305 | 0.270 |
