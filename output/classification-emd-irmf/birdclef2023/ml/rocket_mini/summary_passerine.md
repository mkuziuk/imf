# rocket_mini: birdclef2023 / passerine

50 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 3207.7 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'minirocket')                          | 0.7612 | 0.0109 |
| ('emd', 'comp+log_scale', 'minirocket')                | 0.7612 | 0.0124 |
| ('imf_linear_a1.41', 'comp', 'minirocket')             | 0.7701 | 0.0211 |
| ('imf_linear_a1.41', 'comp+log_scale', 'minirocket')   | 0.7697 | 0.0209 |
| ('irmf_h2_a1.41', 'comp', 'minirocket')                | 0.7730 | 0.0232 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'minirocket')      | 0.7747 | 0.0216 |
| ('irmf_h2_a1.41_conv', 'comp', 'minirocket')           | 0.7744 | 0.0222 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'minirocket') | 0.7734 | 0.0212 |
| ('raw', 'comp', 'minirocket')                          | 0.7613 | 0.0113 |
| ('raw', 'comp+log_scale', 'minirocket')                | 0.7632 | 0.0130 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | minirocket   | raw              | emd                |      0.0001 |  -0.0201 |    0.0203 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw              | imf_linear_a1.41   |     -0.0087 |  -0.0407 |    0.0232 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw              | irmf_h2_a1.41      |     -0.0116 |  -0.0503 |    0.0271 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw              | irmf_h2_a1.41_conv |     -0.0130 |  -0.0482 |    0.0221 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd              | imf_linear_a1.41   |     -0.0088 |  -0.0301 |    0.0124 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd              | irmf_h2_a1.41      |     -0.0117 |  -0.0371 |    0.0137 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd              | irmf_h2_a1.41_conv |     -0.0131 |  -0.0353 |    0.0090 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0029 |  -0.0107 |    0.0049 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0043 |  -0.0181 |    0.0094 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0014 |  -0.0163 |    0.0134 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | emd                |      0.0020 |  -0.0224 |    0.0265 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | imf_linear_a1.41   |     -0.0065 |  -0.0388 |    0.0259 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | irmf_h2_a1.41      |     -0.0114 |  -0.0396 |    0.0167 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | irmf_h2_a1.41_conv |     -0.0101 |  -0.0425 |    0.0222 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd              | imf_linear_a1.41   |     -0.0085 |  -0.0322 |    0.0152 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd              | irmf_h2_a1.41      |     -0.0135 |  -0.0362 |    0.0093 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd              | irmf_h2_a1.41_conv |     -0.0122 |  -0.0335 |    0.0092 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0050 |  -0.0195 |    0.0095 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0037 |  -0.0144 |    0.0070 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0013 |  -0.0070 |    0.0096 |      1.0000 |            1.0000 |
