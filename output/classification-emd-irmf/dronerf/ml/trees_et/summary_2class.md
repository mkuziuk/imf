# dronerf_phase3_trees_et: dronerf / 2class

210 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 393.8 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'extratrees')                          | 0.8550 | 0.0259 |
| ('emd', 'comp+log_scale', 'extratrees')                | 0.8893 | 0.0242 |
| ('imf_linear_a1.41', 'comp', 'extratrees')             | 0.9701 | 0.0148 |
| ('imf_linear_a1.41', 'comp+log_scale', 'extratrees')   | 0.9695 | 0.0147 |
| ('imf_linear_a2', 'comp', 'extratrees')                | 0.9697 | 0.0140 |
| ('imf_linear_a2', 'comp+log_scale', 'extratrees')      | 0.9719 | 0.0142 |
| ('irmf_h2_a1.41', 'comp', 'extratrees')                | 0.9622 | 0.0154 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'extratrees')      | 0.9629 | 0.0135 |
| ('irmf_h2_a1.41_conv', 'comp', 'extratrees')           | 0.9634 | 0.0133 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.9656 | 0.0151 |
| ('irmf_h2_a2_conv', 'comp', 'extratrees')              | 0.9594 | 0.0128 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'extratrees')    | 0.9597 | 0.0146 |
| ('raw', 'comp', 'extratrees')                          | 0.9411 | 0.0209 |
| ('raw', 'comp+log_scale', 'extratrees')                | 0.9607 | 0.0190 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | extratrees   | raw                | emd                |      0.0861 |   0.0652 |    0.1070 |      0.0000 |            0.0026 |
| comp           | extratrees   | raw                | imf_linear_a1.41   |     -0.0290 |  -0.0429 |   -0.0151 |      0.0156 |            0.0026 |
| comp           | extratrees   | raw                | irmf_h2_a1.41      |     -0.0211 |  -0.0376 |   -0.0046 |      0.3866 |            0.0026 |
| comp           | extratrees   | raw                | irmf_h2_a1.41_conv |     -0.0223 |  -0.0392 |   -0.0054 |      0.3640 |            0.0026 |
| comp           | extratrees   | raw                | irmf_h2_a2_conv    |     -0.0183 |  -0.0343 |   -0.0024 |      0.6043 |            0.0026 |
| comp           | extratrees   | raw                | imf_linear_a2      |     -0.0286 |  -0.0440 |   -0.0131 |      0.0404 |            0.0026 |
| comp           | extratrees   | emd                | imf_linear_a1.41   |     -0.1151 |  -0.1374 |   -0.0929 |      0.0000 |            0.0026 |
| comp           | extratrees   | emd                | irmf_h2_a1.41      |     -0.1072 |  -0.1308 |   -0.0835 |      0.0000 |            0.0026 |
| comp           | extratrees   | emd                | irmf_h2_a1.41_conv |     -0.1084 |  -0.1343 |   -0.0825 |      0.0000 |            0.0026 |
| comp           | extratrees   | emd                | irmf_h2_a2_conv    |     -0.1044 |  -0.1288 |   -0.0800 |      0.0000 |            0.0026 |
| comp           | extratrees   | emd                | imf_linear_a2      |     -0.1147 |  -0.1385 |   -0.0908 |      0.0000 |            0.0026 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0079 |   0.0006 |    0.0153 |      0.7667 |            0.0070 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0067 |  -0.0018 |    0.0153 |      1.0000 |            0.0220 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0107 |   0.0022 |    0.0192 |      0.4024 |            0.0094 |
| comp           | extratrees   | imf_linear_a1.41   | imf_linear_a2      |      0.0005 |  -0.0065 |    0.0074 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0012 |  -0.0073 |    0.0049 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0028 |  -0.0056 |    0.0111 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0075 |  -0.0146 |   -0.0004 |      0.7912 |            0.0122 |
| comp           | extratrees   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0040 |  -0.0033 |    0.0113 |      1.0000 |            0.4085 |
| comp           | extratrees   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0063 |  -0.0146 |    0.0021 |      1.0000 |            0.0470 |
| comp           | extratrees   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0103 |  -0.0176 |   -0.0029 |      0.2703 |            0.0059 |
| comp+log_scale | extratrees   | raw                | emd                |      0.0714 |   0.0509 |    0.0919 |      0.0001 |            0.0026 |
| comp+log_scale | extratrees   | raw                | imf_linear_a1.41   |     -0.0088 |  -0.0223 |    0.0047 |      1.0000 |            0.0555 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a1.41      |     -0.0021 |  -0.0139 |    0.0096 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a1.41_conv |     -0.0049 |  -0.0176 |    0.0078 |      1.0000 |            0.6660 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a2_conv    |      0.0010 |  -0.0145 |    0.0164 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | imf_linear_a2      |     -0.0112 |  -0.0245 |    0.0021 |      1.0000 |            0.0322 |
| comp+log_scale | extratrees   | emd                | imf_linear_a1.41   |     -0.0802 |  -0.1014 |   -0.0591 |      0.0000 |            0.0026 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a1.41      |     -0.0736 |  -0.0932 |   -0.0539 |      0.0000 |            0.0124 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a1.41_conv |     -0.0763 |  -0.0953 |   -0.0574 |      0.0000 |            0.0026 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a2_conv    |     -0.0704 |  -0.0903 |   -0.0506 |      0.0001 |            0.0026 |
| comp+log_scale | extratrees   | emd                | imf_linear_a2      |     -0.0826 |  -0.1011 |   -0.0642 |      0.0000 |            0.0026 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0067 |  -0.0006 |    0.0139 |      1.0000 |            0.0220 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0039 |  -0.0021 |    0.0099 |      1.0000 |            0.1128 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0098 |   0.0023 |    0.0173 |      0.3802 |            0.0026 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | imf_linear_a2      |     -0.0024 |  -0.0097 |    0.0049 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0028 |  -0.0094 |    0.0038 |      1.0000 |            0.8438 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0031 |  -0.0033 |    0.0096 |      1.0000 |            0.4085 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0091 |  -0.0176 |   -0.0005 |      0.7912 |            0.0094 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0059 |  -0.0026 |    0.0144 |      1.0000 |            0.0645 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0063 |  -0.0141 |    0.0016 |      1.0000 |            0.0348 |
| comp+log_scale | extratrees   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0122 |  -0.0217 |   -0.0027 |      0.3854 |            0.0026 |
