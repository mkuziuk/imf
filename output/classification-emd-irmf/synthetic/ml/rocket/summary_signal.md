# rocket: synthetic / signal

80 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 154.8 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                       |   mean |    std |
|:------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'minirocket')                         | 0.4160 | 0.0166 |
| ('emd', 'comp', 'multirocket')                        | 0.3500 | 0.0391 |
| ('emd', 'comp+log_scale', 'minirocket')               | 0.4080 | 0.0398 |
| ('emd', 'comp+log_scale', 'multirocket')              | 0.3540 | 0.0368 |
| ('imf_linear_a1.41', 'comp', 'minirocket')            | 0.5427 | 0.0161 |
| ('imf_linear_a1.41', 'comp', 'multirocket')           | 0.4620 | 0.0258 |
| ('imf_linear_a1.41', 'comp+log_scale', 'minirocket')  | 0.5407 | 0.0206 |
| ('imf_linear_a1.41', 'comp+log_scale', 'multirocket') | 0.4333 | 0.0337 |
| ('irmf_h2_a1.41', 'comp', 'minirocket')               | 0.5800 | 0.0279 |
| ('irmf_h2_a1.41', 'comp', 'multirocket')              | 0.5520 | 0.0233 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'minirocket')     | 0.5773 | 0.0224 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'multirocket')    | 0.5433 | 0.0267 |
| ('raw', 'comp', 'minirocket')                         | 0.5000 | 0.0286 |
| ('raw', 'comp', 'multirocket')                        | 0.4347 | 0.0277 |
| ('raw', 'comp+log_scale', 'minirocket')               | 0.5180 | 0.0322 |
| ('raw', 'comp+log_scale', 'multirocket')              | 0.4247 | 0.0326 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b         |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-----------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | minirocket   | raw              | emd              |      0.0840 |   0.0454 |    0.1226 |      0.0758 |            1.0000 |
| comp           | minirocket   | raw              | imf_linear_a1.41 |     -0.0427 |  -0.1223 |    0.0370 |      0.8451 |            1.0000 |
| comp           | minirocket   | raw              | irmf_h2_a1.41    |     -0.0800 |  -0.1683 |    0.0083 |      0.4997 |            1.0000 |
| comp           | minirocket   | emd              | imf_linear_a1.41 |     -0.1267 |  -0.1733 |   -0.0800 |      0.0382 |            1.0000 |
| comp           | minirocket   | emd              | irmf_h2_a1.41    |     -0.1640 |  -0.2162 |   -0.1118 |      0.0228 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41    |     -0.0373 |  -0.0688 |   -0.0059 |      0.3915 |            1.0000 |
| comp+log_scale | minirocket   | raw              | emd              |      0.1100 |   0.0148 |    0.2052 |      0.3915 |            1.0000 |
| comp+log_scale | minirocket   | raw              | imf_linear_a1.41 |     -0.0227 |  -0.1190 |    0.0737 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | irmf_h2_a1.41    |     -0.0593 |  -0.1287 |    0.0100 |      0.4997 |            1.0000 |
| comp+log_scale | minirocket   | emd              | imf_linear_a1.41 |     -0.1327 |  -0.2268 |   -0.0386 |      0.2599 |            1.0000 |
| comp+log_scale | minirocket   | emd              | irmf_h2_a1.41    |     -0.1693 |  -0.2666 |   -0.0720 |      0.1353 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41    |     -0.0367 |  -0.0861 |    0.0128 |      0.5436 |            1.0000 |
| comp           | multirocket  | raw              | emd              |      0.0847 |  -0.0071 |    0.1764 |      0.4997 |            1.0000 |
| comp           | multirocket  | raw              | imf_linear_a1.41 |     -0.0273 |  -0.1087 |    0.0540 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw              | irmf_h2_a1.41    |     -0.1173 |  -0.1745 |   -0.0602 |      0.0889 |            1.0000 |
| comp           | multirocket  | emd              | imf_linear_a1.41 |     -0.1120 |  -0.1573 |   -0.0667 |      0.0520 |            1.0000 |
| comp           | multirocket  | emd              | irmf_h2_a1.41    |     -0.2020 |  -0.3028 |   -0.1012 |      0.0921 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41    |     -0.0900 |  -0.1662 |   -0.0138 |      0.3915 |            1.0000 |
| comp+log_scale | multirocket  | raw              | emd              |      0.0707 |   0.0059 |    0.1354 |      0.3915 |            1.0000 |
| comp+log_scale | multirocket  | raw              | imf_linear_a1.41 |     -0.0087 |  -0.0924 |    0.0750 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw              | irmf_h2_a1.41    |     -0.1187 |  -0.1705 |   -0.0668 |      0.0661 |            1.0000 |
| comp+log_scale | multirocket  | emd              | imf_linear_a1.41 |     -0.0793 |  -0.1590 |    0.0003 |      0.4555 |            1.0000 |
| comp+log_scale | multirocket  | emd              | irmf_h2_a1.41    |     -0.1893 |  -0.2855 |   -0.0932 |      0.0924 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41    |     -0.1100 |  -0.1972 |   -0.0228 |      0.3484 |            1.0000 |
