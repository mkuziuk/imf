# Classical baselines: synthetic / signal

1500 frames, 150 groups, 5 classes; 15 grouped splits (5 folds x 3 seeds). Runtime 38 s with 4 workers.

## Balanced accuracy (mean ± sd over splits)

Feature set `comp`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| irmf_h2_a1.41_conv | 0.870 ± 0.035 | 0.824 ± 0.031 | 0.847 ± 0.036 |

Feature set `comp+log_scale`:

| method             | lda           | linsvm        | logreg        |
|:-------------------|:--------------|:--------------|:--------------|
| irmf_h2_a1.41_conv | 0.869 ± 0.034 | 0.822 ± 0.033 | 0.847 ± 0.033 |

## Best classifier per method

| feature_set    | method             | classifier   |   balanced_accuracy |
|:---------------|:-------------------|:-------------|--------------------:|
| comp           | irmf_h2_a1.41_conv | lda          |               0.870 |
| comp+log_scale | irmf_h2_a1.41_conv | lda          |               0.869 |

## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)

| method             |   max_eps2 |   max_mi |   n_features |
|:-------------------|-----------:|---------:|-------------:|
| irmf_h2_a1.41_conv |      0.568 |    0.517 |      321.000 |

| method             | feature             | slot_label   |   epsilon2 |    mi |
|:-------------------|:--------------------|:-------------|-----------:|------:|
| irmf_h2_a1.41_conv | ch0_c5_kurt         | w25          |      0.568 | 0.517 |
| irmf_h2_a1.41_conv | ch0_c6_kurt         | w33          |      0.559 | 0.505 |
| irmf_h2_a1.41_conv | ch0_c7_kurt         | w47          |      0.490 | 0.380 |
| irmf_h2_a1.41_conv | ch0_c5_hilb_amp_std | w25          |      0.462 | 0.385 |
| irmf_h2_a1.41_conv | ch0_c6_crest        | w33          |      0.398 | 0.292 |
