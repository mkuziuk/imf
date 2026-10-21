# dronerf_phase3_rocket_extra: dronerf / 10class

70 fits on shared split IDs [5, 6, 7, 8, 9]; wall time 1532.3 s with 8 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'minirocket')                          | 0.4213 | 0.0195 |
| ('emd', 'comp+log_scale', 'minirocket')                | 0.4447 | 0.0166 |
| ('imf_linear_a1.41', 'comp', 'minirocket')             | 0.4950 | 0.0174 |
| ('imf_linear_a1.41', 'comp+log_scale', 'minirocket')   | 0.5075 | 0.0155 |
| ('imf_linear_a2', 'comp', 'minirocket')                | 0.4933 | 0.0183 |
| ('imf_linear_a2', 'comp+log_scale', 'minirocket')      | 0.5080 | 0.0182 |
| ('irmf_h2_a1.41', 'comp', 'minirocket')                | 0.4724 | 0.0116 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'minirocket')      | 0.4974 | 0.0185 |
| ('irmf_h2_a1.41_conv', 'comp', 'minirocket')           | 0.4743 | 0.0107 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'minirocket') | 0.4998 | 0.0147 |
| ('irmf_h2_a2_conv', 'comp', 'minirocket')              | 0.4725 | 0.0120 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'minirocket')    | 0.5001 | 0.0181 |
| ('raw', 'comp', 'minirocket')                          | 0.4599 | 0.0198 |
| ('raw', 'comp+log_scale', 'minirocket')                | 0.4782 | 0.0173 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | minirocket   | raw                | emd                |      0.0386 |   0.0186 |    0.0585 |      0.1743 |            1.0000 |
| comp           | minirocket   | raw                | imf_linear_a1.41   |     -0.0351 |  -0.0510 |   -0.0193 |      0.1129 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a1.41      |     -0.0125 |  -0.0361 |    0.0112 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a1.41_conv |     -0.0144 |  -0.0398 |    0.0110 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a2_conv    |     -0.0126 |  -0.0335 |    0.0084 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | imf_linear_a2      |     -0.0334 |  -0.0401 |   -0.0267 |      0.0066 |            1.0000 |
| comp           | minirocket   | emd                | imf_linear_a1.41   |     -0.0737 |  -0.0929 |   -0.0545 |      0.0175 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a1.41      |     -0.0510 |  -0.0697 |   -0.0324 |      0.0566 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a1.41_conv |     -0.0529 |  -0.0716 |   -0.0342 |      0.0510 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a2_conv    |     -0.0511 |  -0.0739 |   -0.0283 |      0.1121 |            1.0000 |
| comp           | minirocket   | emd                | imf_linear_a2      |     -0.0720 |  -0.0925 |   -0.0515 |      0.0242 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0226 |  -0.0016 |    0.0468 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0207 |  -0.0034 |    0.0449 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0226 |   0.0005 |    0.0447 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | imf_linear_a2      |      0.0017 |  -0.0168 |    0.0202 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0019 |  -0.0054 |    0.0016 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0001 |  -0.0094 |    0.0092 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0209 |  -0.0429 |    0.0010 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0018 |  -0.0098 |    0.0135 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0190 |  -0.0425 |    0.0044 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0209 |  -0.0417 |   -0.0001 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | emd                |      0.0335 |   0.0044 |    0.0627 |      0.8905 |            1.0000 |
| comp+log_scale | minirocket   | raw                | imf_linear_a1.41   |     -0.0293 |  -0.0466 |   -0.0120 |      0.2612 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a1.41      |     -0.0192 |  -0.0372 |   -0.0012 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a1.41_conv |     -0.0216 |  -0.0404 |   -0.0027 |      0.8905 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a2_conv    |     -0.0218 |  -0.0433 |   -0.0004 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | imf_linear_a2      |     -0.0297 |  -0.0399 |   -0.0196 |      0.0463 |            1.0000 |
| comp+log_scale | minirocket   | emd                | imf_linear_a1.41   |     -0.0628 |  -0.0971 |   -0.0286 |      0.2030 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a1.41      |     -0.0527 |  -0.0679 |   -0.0375 |      0.0245 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a1.41_conv |     -0.0551 |  -0.0691 |   -0.0411 |      0.0162 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a2_conv    |     -0.0554 |  -0.0832 |   -0.0275 |      0.1635 |            1.0000 |
| comp+log_scale | minirocket   | emd                | imf_linear_a2      |     -0.0633 |  -0.0907 |   -0.0359 |      0.1037 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0101 |  -0.0194 |    0.0397 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0077 |  -0.0207 |    0.0362 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0075 |  -0.0286 |    0.0436 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | imf_linear_a2      |     -0.0004 |  -0.0209 |    0.0200 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0024 |  -0.0131 |    0.0083 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0026 |  -0.0206 |    0.0153 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0106 |  -0.0263 |    0.0052 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0003 |  -0.0176 |    0.0171 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0082 |  -0.0290 |    0.0127 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0079 |  -0.0296 |    0.0137 |      1.0000 |            1.0000 |
