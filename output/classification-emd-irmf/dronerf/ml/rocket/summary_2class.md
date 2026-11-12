# dronerf_phase3_rocket: dronerf / 2class

70 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 1315.9 s with 6 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'minirocket')                          | 0.8521 | 0.0276 |
| ('emd', 'comp+log_scale', 'minirocket')                | 0.8629 | 0.0332 |
| ('imf_linear_a1.41', 'comp', 'minirocket')             | 0.8743 | 0.0216 |
| ('imf_linear_a1.41', 'comp+log_scale', 'minirocket')   | 0.8760 | 0.0226 |
| ('imf_linear_a2', 'comp', 'minirocket')                | 0.8730 | 0.0187 |
| ('imf_linear_a2', 'comp+log_scale', 'minirocket')      | 0.8731 | 0.0208 |
| ('irmf_h2_a1.41', 'comp', 'minirocket')                | 0.8724 | 0.0213 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'minirocket')      | 0.8736 | 0.0256 |
| ('irmf_h2_a1.41_conv', 'comp', 'minirocket')           | 0.8692 | 0.0228 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'minirocket') | 0.8707 | 0.0235 |
| ('irmf_h2_a2_conv', 'comp', 'minirocket')              | 0.8679 | 0.0213 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'minirocket')    | 0.8728 | 0.0195 |
| ('raw', 'comp', 'minirocket')                          | 0.8705 | 0.0251 |
| ('raw', 'comp+log_scale', 'minirocket')                | 0.8721 | 0.0237 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | minirocket   | raw                | emd                |      0.0184 |   0.0119 |    0.0249 |      0.0598 |            1.0000 |
| comp           | minirocket   | raw                | imf_linear_a1.41   |     -0.0038 |  -0.0222 |    0.0145 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a1.41      |     -0.0019 |  -0.0157 |    0.0119 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a1.41_conv |      0.0013 |  -0.0081 |    0.0106 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a2_conv    |      0.0026 |  -0.0199 |    0.0250 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | imf_linear_a2      |     -0.0025 |  -0.0165 |    0.0116 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | imf_linear_a1.41   |     -0.0222 |  -0.0449 |    0.0006 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a1.41      |     -0.0203 |  -0.0404 |   -0.0002 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a1.41_conv |     -0.0171 |  -0.0323 |   -0.0019 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a2_conv    |     -0.0158 |  -0.0431 |    0.0115 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | imf_linear_a2      |     -0.0208 |  -0.0407 |   -0.0010 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0019 |  -0.0099 |    0.0137 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0051 |  -0.0055 |    0.0157 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0064 |  -0.0157 |    0.0284 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | imf_linear_a2      |      0.0013 |  -0.0097 |    0.0124 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0032 |  -0.0025 |    0.0088 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0045 |  -0.0125 |    0.0214 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0006 |  -0.0072 |    0.0061 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0013 |  -0.0169 |    0.0195 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0037 |  -0.0116 |    0.0042 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0050 |  -0.0237 |    0.0136 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | emd                |      0.0092 |  -0.0095 |    0.0279 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | imf_linear_a1.41   |     -0.0040 |  -0.0134 |    0.0055 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a1.41      |     -0.0016 |  -0.0071 |    0.0040 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a1.41_conv |      0.0013 |  -0.0057 |    0.0084 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a2_conv    |     -0.0008 |  -0.0110 |    0.0095 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | imf_linear_a2      |     -0.0010 |  -0.0095 |    0.0075 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | imf_linear_a1.41   |     -0.0132 |  -0.0361 |    0.0098 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a1.41      |     -0.0108 |  -0.0257 |    0.0042 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a1.41_conv |     -0.0079 |  -0.0262 |    0.0105 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a2_conv    |     -0.0100 |  -0.0358 |    0.0158 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | imf_linear_a2      |     -0.0102 |  -0.0337 |    0.0133 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0024 |  -0.0102 |    0.0150 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0053 |  -0.0052 |    0.0158 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0032 |  -0.0089 |    0.0153 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | imf_linear_a2      |      0.0030 |  -0.0067 |    0.0126 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0029 |  -0.0046 |    0.0104 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0008 |  -0.0116 |    0.0132 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | imf_linear_a2      |      0.0006 |  -0.0110 |    0.0122 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0021 |  -0.0099 |    0.0057 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0023 |  -0.0079 |    0.0032 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0002 |  -0.0051 |    0.0047 |      1.0000 |            1.0000 |
