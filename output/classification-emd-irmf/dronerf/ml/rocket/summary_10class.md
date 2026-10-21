# dronerf_phase3_rocket: dronerf / 10class

70 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 5173.1 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'minirocket')                          | 0.4183 | 0.0096 |
| ('emd', 'comp+log_scale', 'minirocket')                | 0.4415 | 0.0150 |
| ('imf_linear_a1.41', 'comp', 'minirocket')             | 0.4958 | 0.0112 |
| ('imf_linear_a1.41', 'comp+log_scale', 'minirocket')   | 0.5102 | 0.0126 |
| ('imf_linear_a2', 'comp', 'minirocket')                | 0.4910 | 0.0172 |
| ('imf_linear_a2', 'comp+log_scale', 'minirocket')      | 0.5108 | 0.0110 |
| ('irmf_h2_a1.41', 'comp', 'minirocket')                | 0.4689 | 0.0094 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'minirocket')      | 0.4967 | 0.0124 |
| ('irmf_h2_a1.41_conv', 'comp', 'minirocket')           | 0.4726 | 0.0086 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'minirocket') | 0.4970 | 0.0128 |
| ('irmf_h2_a2_conv', 'comp', 'minirocket')              | 0.4661 | 0.0122 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'minirocket')    | 0.4971 | 0.0155 |
| ('raw', 'comp', 'minirocket')                          | 0.4655 | 0.0167 |
| ('raw', 'comp+log_scale', 'minirocket')                | 0.4860 | 0.0156 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | minirocket   | raw                | emd                |      0.0472 |   0.0181 |    0.0764 |      0.3249 |            1.0000 |
| comp           | minirocket   | raw                | imf_linear_a1.41   |     -0.0302 |  -0.0539 |   -0.0066 |      0.5738 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a1.41      |     -0.0034 |  -0.0207 |    0.0139 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a1.41_conv |     -0.0070 |  -0.0288 |    0.0147 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a2_conv    |     -0.0006 |  -0.0282 |    0.0271 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | imf_linear_a2      |     -0.0254 |  -0.0438 |   -0.0071 |      0.4936 |            1.0000 |
| comp           | minirocket   | emd                | imf_linear_a1.41   |     -0.0775 |  -0.0999 |   -0.0551 |      0.0263 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a1.41      |     -0.0506 |  -0.0759 |   -0.0254 |      0.1739 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a1.41_conv |     -0.0543 |  -0.0816 |   -0.0270 |      0.1739 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a2_conv    |     -0.0478 |  -0.0642 |   -0.0314 |      0.0492 |            1.0000 |
| comp           | minirocket   | emd                | imf_linear_a2      |     -0.0727 |  -0.1009 |   -0.0444 |      0.0712 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0268 |   0.0097 |    0.0440 |      0.3421 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0232 |   0.0042 |    0.0422 |      0.6076 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0297 |   0.0110 |    0.0483 |      0.3350 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | imf_linear_a2      |      0.0048 |  -0.0089 |    0.0185 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0036 |  -0.0104 |    0.0032 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0028 |  -0.0203 |    0.0260 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0220 |  -0.0447 |    0.0006 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0065 |  -0.0214 |    0.0344 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0184 |  -0.0451 |    0.0083 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0249 |  -0.0472 |   -0.0026 |      0.7626 |            1.0000 |
| comp+log_scale | minirocket   | raw                | emd                |      0.0445 |   0.0273 |    0.0617 |      0.0712 |            1.0000 |
| comp+log_scale | minirocket   | raw                | imf_linear_a1.41   |     -0.0242 |  -0.0421 |   -0.0063 |      0.5093 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a1.41      |     -0.0107 |  -0.0243 |    0.0029 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a1.41_conv |     -0.0110 |  -0.0352 |    0.0133 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a2_conv    |     -0.0111 |  -0.0262 |    0.0040 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | imf_linear_a2      |     -0.0248 |  -0.0395 |   -0.0101 |      0.2931 |            1.0000 |
| comp+log_scale | minirocket   | emd                | imf_linear_a1.41   |     -0.0687 |  -0.0858 |   -0.0516 |      0.0155 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a1.41      |     -0.0552 |  -0.0756 |   -0.0348 |      0.0625 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a1.41_conv |     -0.0555 |  -0.0839 |   -0.0270 |      0.1809 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a2_conv    |     -0.0556 |  -0.0757 |   -0.0355 |      0.0589 |            1.0000 |
| comp+log_scale | minirocket   | emd                | imf_linear_a2      |     -0.0693 |  -0.0889 |   -0.0498 |      0.0244 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0135 |   0.0036 |    0.0234 |      0.5093 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0132 |  -0.0056 |    0.0320 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0131 |  -0.0110 |    0.0372 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | imf_linear_a2      |     -0.0006 |  -0.0168 |    0.0156 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0003 |  -0.0128 |    0.0122 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0004 |  -0.0246 |    0.0238 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0141 |  -0.0309 |    0.0026 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0001 |  -0.0354 |    0.0351 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0138 |  -0.0410 |    0.0133 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0137 |  -0.0246 |   -0.0028 |      0.5755 |            1.0000 |
