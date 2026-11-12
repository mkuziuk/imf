# dronerf_phase3_rocket: dronerf / 4class

70 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 1695.1 s with 4 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'minirocket')                          | 0.7487 | 0.0159 |
| ('emd', 'comp+log_scale', 'minirocket')                | 0.7895 | 0.0176 |
| ('imf_linear_a1.41', 'comp', 'minirocket')             | 0.7979 | 0.0249 |
| ('imf_linear_a1.41', 'comp+log_scale', 'minirocket')   | 0.8330 | 0.0273 |
| ('imf_linear_a2', 'comp', 'minirocket')                | 0.8015 | 0.0216 |
| ('imf_linear_a2', 'comp+log_scale', 'minirocket')      | 0.8315 | 0.0199 |
| ('irmf_h2_a1.41', 'comp', 'minirocket')                | 0.7698 | 0.0205 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'minirocket')      | 0.8129 | 0.0250 |
| ('irmf_h2_a1.41_conv', 'comp', 'minirocket')           | 0.7711 | 0.0202 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'minirocket') | 0.8162 | 0.0228 |
| ('irmf_h2_a2_conv', 'comp', 'minirocket')              | 0.7723 | 0.0200 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'minirocket')    | 0.8139 | 0.0198 |
| ('raw', 'comp', 'minirocket')                          | 0.7623 | 0.0263 |
| ('raw', 'comp+log_scale', 'minirocket')                | 0.7938 | 0.0274 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | minirocket   | raw                | emd                |      0.0135 |  -0.0113 |    0.0384 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | imf_linear_a1.41   |     -0.0356 |  -0.0504 |   -0.0209 |      0.0981 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a1.41      |     -0.0075 |  -0.0227 |    0.0076 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a1.41_conv |     -0.0088 |  -0.0252 |    0.0075 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a2_conv    |     -0.0101 |  -0.0329 |    0.0128 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | imf_linear_a2      |     -0.0392 |  -0.0660 |   -0.0124 |      0.3846 |            1.0000 |
| comp           | minirocket   | emd                | imf_linear_a1.41   |     -0.0492 |  -0.0713 |   -0.0271 |      0.1258 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a1.41      |     -0.0211 |  -0.0377 |   -0.0045 |      0.5808 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a1.41_conv |     -0.0223 |  -0.0361 |   -0.0086 |      0.3086 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a2_conv    |     -0.0236 |  -0.0497 |    0.0025 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | imf_linear_a2      |     -0.0527 |  -0.0697 |   -0.0358 |      0.0403 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0281 |   0.0181 |    0.0380 |      0.0558 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0268 |   0.0176 |    0.0360 |      0.0503 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0256 |  -0.0021 |    0.0533 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | imf_linear_a2      |     -0.0036 |  -0.0254 |    0.0183 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0013 |  -0.0075 |    0.0050 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0025 |  -0.0215 |    0.0165 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0317 |  -0.0496 |   -0.0137 |      0.2483 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0012 |  -0.0253 |    0.0228 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0304 |  -0.0493 |   -0.0115 |      0.3113 |            1.0000 |
| comp           | minirocket   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0291 |  -0.0557 |   -0.0025 |      0.8440 |            1.0000 |
| comp+log_scale | minirocket   | raw                | emd                |      0.0043 |  -0.0209 |    0.0295 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | imf_linear_a1.41   |     -0.0392 |  -0.0463 |   -0.0322 |      0.0042 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a1.41      |     -0.0191 |  -0.0293 |   -0.0090 |      0.2059 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a1.41_conv |     -0.0224 |  -0.0334 |   -0.0115 |      0.1653 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a2_conv    |     -0.0201 |  -0.0423 |    0.0020 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | imf_linear_a2      |     -0.0377 |  -0.0623 |   -0.0131 |      0.3407 |            1.0000 |
| comp+log_scale | minirocket   | emd                | imf_linear_a1.41   |     -0.0435 |  -0.0659 |   -0.0211 |      0.1939 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a1.41      |     -0.0234 |  -0.0453 |   -0.0015 |      0.8665 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a1.41_conv |     -0.0267 |  -0.0425 |   -0.0109 |      0.2832 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a2_conv    |     -0.0244 |  -0.0516 |    0.0028 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | imf_linear_a2      |     -0.0420 |  -0.0642 |   -0.0197 |      0.2059 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0201 |   0.0117 |    0.0285 |      0.0981 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0168 |   0.0061 |    0.0276 |      0.3305 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0191 |  -0.0064 |    0.0447 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | imf_linear_a2      |      0.0015 |  -0.0211 |    0.0242 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0033 |  -0.0137 |    0.0072 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0010 |  -0.0224 |    0.0204 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0185 |  -0.0338 |   -0.0033 |      0.6463 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0023 |  -0.0194 |    0.0240 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0153 |  -0.0357 |    0.0051 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0176 |  -0.0395 |    0.0043 |      1.0000 |            1.0000 |
