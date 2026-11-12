# Classical baselines: esc50_animals / animal

5100 frames, 157 groups, 10 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 1039 s with 4 workers.

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| irmf_h2_a1.41_conv | 0.625 ± 0.029 | 0.650 ± 0.036 | 0.653 ± 0.040 |

Feature set `comp+log_scale`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| irmf_h2_a1.41_conv | 0.639 ± 0.028 | 0.666 ± 0.034 | 0.663 ± 0.039 |

## Best classifier per method

| feature_set    | method             | classifier   |   balanced_accuracy |
|:---------------|:-------------------|:-------------|--------------------:|
| comp           | irmf_h2_a1.41_conv | logreg       |               0.653 |
| comp+log_scale | irmf_h2_a1.41_conv | linsvm       |               0.666 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method             |   max_eps2 |   max_mi |   n_features |
|:-------------------|-----------:|---------:|-------------:|
| irmf_h2_a1.41_conv |      0.610 |    0.547 |      361.000 |

| method             | feature              | slot_label   |   epsilon2 |    mi |
|:-------------------|:---------------------|:-------------|-----------:|------:|
| irmf_h2_a1.41_conv | ch0_c8_hjorth_mob    | w65          |      0.610 | 0.547 |
| irmf_h2_a1.41_conv | ch0_c8_hilb_if_mean  | w65          |      0.593 | 0.514 |
| irmf_h2_a1.41_conv | ch0_c10_energy_frac  | w129         |      0.592 | 0.492 |
| irmf_h2_a1.41_conv | ch0_c8_spec_centroid | w65          |      0.588 | 0.505 |
| irmf_h2_a1.41_conv | ch0_c9_hjorth_mob    | w93          |      0.585 | 0.504 |
