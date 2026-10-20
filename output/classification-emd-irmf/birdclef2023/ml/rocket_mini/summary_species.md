# rocket_mini: birdclef2023 / species

50 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 3977.8 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'minirocket')                          | 0.5075 | 0.0142 |
| ('emd', 'comp+log_scale', 'minirocket')                | 0.5061 | 0.0122 |
| ('imf_linear_a1.41', 'comp', 'minirocket')             | 0.5195 | 0.0210 |
| ('imf_linear_a1.41', 'comp+log_scale', 'minirocket')   | 0.5179 | 0.0172 |
| ('irmf_h2_a1.41', 'comp', 'minirocket')                | 0.5205 | 0.0153 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'minirocket')      | 0.5214 | 0.0158 |
| ('irmf_h2_a1.41_conv', 'comp', 'minirocket')           | 0.5215 | 0.0135 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'minirocket') | 0.5211 | 0.0198 |
| ('raw', 'comp', 'minirocket')                          | 0.5051 | 0.0205 |
| ('raw', 'comp+log_scale', 'minirocket')                | 0.5049 | 0.0225 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | minirocket   | raw              | emd                |     -0.0024 |  -0.0184 |    0.0136 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw              | imf_linear_a1.41   |     -0.0145 |  -0.0255 |   -0.0034 |      0.4234 |            1.0000 |
| comp           | minirocket   | raw              | irmf_h2_a1.41      |     -0.0155 |  -0.0285 |   -0.0024 |      0.4998 |            1.0000 |
| comp           | minirocket   | raw              | irmf_h2_a1.41_conv |     -0.0164 |  -0.0337 |    0.0010 |      0.8200 |            1.0000 |
| comp           | minirocket   | emd              | imf_linear_a1.41   |     -0.0121 |  -0.0277 |    0.0036 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd              | irmf_h2_a1.41      |     -0.0131 |  -0.0281 |    0.0020 |      0.9539 |            1.0000 |
| comp           | minirocket   | emd              | irmf_h2_a1.41_conv |     -0.0140 |  -0.0305 |    0.0026 |      0.9539 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0010 |  -0.0168 |    0.0148 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0019 |  -0.0245 |    0.0207 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0009 |  -0.0094 |    0.0076 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | emd                |     -0.0012 |  -0.0255 |    0.0231 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | imf_linear_a1.41   |     -0.0130 |  -0.0356 |    0.0095 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | irmf_h2_a1.41      |     -0.0164 |  -0.0296 |   -0.0033 |      0.4621 |            1.0000 |
| comp+log_scale | minirocket   | raw              | irmf_h2_a1.41_conv |     -0.0162 |  -0.0217 |   -0.0107 |      0.0246 |            1.0000 |
| comp+log_scale | minirocket   | emd              | imf_linear_a1.41   |     -0.0118 |  -0.0217 |   -0.0019 |      0.4998 |            1.0000 |
| comp+log_scale | minirocket   | emd              | irmf_h2_a1.41      |     -0.0153 |  -0.0293 |   -0.0013 |      0.5849 |            1.0000 |
| comp+log_scale | minirocket   | emd              | irmf_h2_a1.41_conv |     -0.0150 |  -0.0345 |    0.0044 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0034 |  -0.0198 |    0.0129 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0032 |  -0.0221 |    0.0157 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0002 |  -0.0078 |    0.0083 |      1.0000 |            1.0000 |
