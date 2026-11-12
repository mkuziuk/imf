# Classical baselines: birdclef2023 / species

12000 frames, 451 groups, 16 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 4432 s with 4 workers.

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| irmf_h2_a1.41_conv | 0.475 ± 0.022 | 0.496 ± 0.019 | 0.496 ± 0.016 |

Feature set `comp+log_scale`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| irmf_h2_a1.41_conv | 0.474 ± 0.021 | 0.496 ± 0.019 | 0.493 ± 0.018 |

## Best classifier per method

| feature_set    | method             | classifier   |   balanced_accuracy |
|:---------------|:-------------------|:-------------|--------------------:|
| comp           | irmf_h2_a1.41_conv | logreg       |               0.496 |
| comp+log_scale | irmf_h2_a1.41_conv | linsvm       |               0.496 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method             |   max_eps2 |   max_mi |   n_features |
|:-------------------|-----------:|---------:|-------------:|
| irmf_h2_a1.41_conv |      0.472 |    0.646 |      361.000 |

| method             | feature              | slot_label   |   epsilon2 |    mi |
|:-------------------|:---------------------|:-------------|-----------:|------:|
| irmf_h2_a1.41_conv | ch0_c2_hilb_if_mean  | w9           |      0.472 | 0.450 |
| irmf_h2_a1.41_conv | ch0_c4_extrema_rate  | w17          |      0.470 | 0.479 |
| irmf_h2_a1.41_conv | ch0_c2_spec_centroid | w9           |      0.465 | 0.439 |
| irmf_h2_a1.41_conv | ch0_c2_zcr           | w9           |      0.462 | 0.430 |
| irmf_h2_a1.41_conv | ch0_c2_hjorth_mob    | w9           |      0.461 | 0.443 |
