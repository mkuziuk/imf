# esc50_phase3_multirocket: esc50_animals / animal

50 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 406.9 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                         |   mean |    std |
|:--------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'multirocket')                          | 0.5427 | 0.0391 |
| ('emd', 'comp+log_scale', 'multirocket')                | 0.5493 | 0.0396 |
| ('imf_linear_a1.41', 'comp', 'multirocket')             | 0.5963 | 0.0470 |
| ('imf_linear_a1.41', 'comp+log_scale', 'multirocket')   | 0.6042 | 0.0454 |
| ('irmf_h2_a1.41', 'comp', 'multirocket')                | 0.5892 | 0.0418 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'multirocket')      | 0.6001 | 0.0395 |
| ('irmf_h2_a1.41_conv', 'comp', 'multirocket')           | 0.5822 | 0.0455 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'multirocket') | 0.5924 | 0.0369 |
| ('raw', 'comp', 'multirocket')                          | 0.5587 | 0.0259 |
| ('raw', 'comp+log_scale', 'multirocket')                | 0.5792 | 0.0346 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | multirocket  | raw              | emd                |      0.0160 |  -0.0803 |    0.1124 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw              | imf_linear_a1.41   |     -0.0376 |  -0.0933 |    0.0181 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw              | irmf_h2_a1.41      |     -0.0305 |  -0.0813 |    0.0203 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw              | irmf_h2_a1.41_conv |     -0.0235 |  -0.0861 |    0.0390 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd              | imf_linear_a1.41   |     -0.0536 |  -0.1485 |    0.0412 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd              | irmf_h2_a1.41      |     -0.0465 |  -0.1313 |    0.0382 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd              | irmf_h2_a1.41_conv |     -0.0396 |  -0.1197 |    0.0406 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0071 |  -0.0057 |    0.0199 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0141 |  -0.0071 |    0.0353 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0070 |  -0.0071 |    0.0210 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw              | emd                |      0.0299 |  -0.0648 |    0.1247 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw              | imf_linear_a1.41   |     -0.0250 |  -0.0826 |    0.0325 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw              | irmf_h2_a1.41      |     -0.0208 |  -0.0746 |    0.0329 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw              | irmf_h2_a1.41_conv |     -0.0132 |  -0.0683 |    0.0420 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd              | imf_linear_a1.41   |     -0.0550 |  -0.1202 |    0.0102 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd              | irmf_h2_a1.41      |     -0.0508 |  -0.1315 |    0.0299 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd              | irmf_h2_a1.41_conv |     -0.0431 |  -0.1107 |    0.0245 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0042 |  -0.0340 |    0.0424 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0118 |  -0.0214 |    0.0451 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0077 |  -0.0075 |    0.0229 |      1.0000 |            1.0000 |
