# rocket_multi: birdclef2023 / species

50 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 1697.6 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                         |   mean |    std |
|:--------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'multirocket')                          | 0.4752 | 0.0152 |
| ('emd', 'comp+log_scale', 'multirocket')                | 0.4746 | 0.0123 |
| ('imf_linear_a1.41', 'comp', 'multirocket')             | 0.4985 | 0.0173 |
| ('imf_linear_a1.41', 'comp+log_scale', 'multirocket')   | 0.4976 | 0.0182 |
| ('irmf_h2_a1.41', 'comp', 'multirocket')                | 0.5040 | 0.0120 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'multirocket')      | 0.5055 | 0.0146 |
| ('irmf_h2_a1.41_conv', 'comp', 'multirocket')           | 0.4993 | 0.0151 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'multirocket') | 0.5011 | 0.0167 |
| ('raw', 'comp', 'multirocket')                          | 0.4805 | 0.0150 |
| ('raw', 'comp+log_scale', 'multirocket')                | 0.4799 | 0.0188 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | multirocket  | raw              | emd                |      0.0053 |  -0.0149 |    0.0255 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw              | imf_linear_a1.41   |     -0.0179 |  -0.0347 |   -0.0012 |      0.4922 |            1.0000 |
| comp           | multirocket  | raw              | irmf_h2_a1.41      |     -0.0235 |  -0.0321 |   -0.0148 |      0.0318 |            1.0000 |
| comp           | multirocket  | raw              | irmf_h2_a1.41_conv |     -0.0188 |  -0.0235 |   -0.0141 |      0.0075 |            1.0000 |
| comp           | multirocket  | emd              | imf_linear_a1.41   |     -0.0232 |  -0.0551 |    0.0086 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd              | irmf_h2_a1.41      |     -0.0288 |  -0.0479 |   -0.0096 |      0.1823 |            1.0000 |
| comp           | multirocket  | emd              | irmf_h2_a1.41_conv |     -0.0241 |  -0.0476 |   -0.0006 |      0.5121 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0055 |  -0.0208 |    0.0098 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0009 |  -0.0193 |    0.0175 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0047 |  -0.0067 |    0.0160 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw              | emd                |      0.0052 |  -0.0158 |    0.0263 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw              | imf_linear_a1.41   |     -0.0178 |  -0.0273 |   -0.0082 |      0.1205 |            1.0000 |
| comp+log_scale | multirocket  | raw              | irmf_h2_a1.41      |     -0.0257 |  -0.0412 |   -0.0101 |      0.1528 |            1.0000 |
| comp+log_scale | multirocket  | raw              | irmf_h2_a1.41_conv |     -0.0213 |  -0.0343 |   -0.0082 |      0.1528 |            1.0000 |
| comp+log_scale | multirocket  | emd              | imf_linear_a1.41   |     -0.0230 |  -0.0462 |    0.0002 |      0.5121 |            1.0000 |
| comp+log_scale | multirocket  | emd              | irmf_h2_a1.41      |     -0.0309 |  -0.0488 |   -0.0130 |      0.1389 |            1.0000 |
| comp+log_scale | multirocket  | emd              | irmf_h2_a1.41_conv |     -0.0265 |  -0.0413 |   -0.0117 |      0.1295 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0079 |  -0.0299 |    0.0141 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0035 |  -0.0239 |    0.0169 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0044 |  -0.0047 |    0.0136 |      1.0000 |            1.0000 |
