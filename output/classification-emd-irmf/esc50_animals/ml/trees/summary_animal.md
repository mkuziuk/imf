# esc50_phase3_trees: esc50_animals / animal

450 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 2463.2 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'extratrees')                          | 0.5622 | 0.0425 |
| ('emd', 'comp', 'lightgbm')                            | 0.5946 | 0.0401 |
| ('emd', 'comp', 'xgboost')                             | 0.5724 | 0.0373 |
| ('emd', 'comp+log_scale', 'extratrees')                | 0.5709 | 0.0375 |
| ('emd', 'comp+log_scale', 'lightgbm')                  | 0.5970 | 0.0286 |
| ('emd', 'comp+log_scale', 'xgboost')                   | 0.5733 | 0.0283 |
| ('imf_linear_a1.41', 'comp', 'extratrees')             | 0.6579 | 0.0367 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')               | 0.6718 | 0.0323 |
| ('imf_linear_a1.41', 'comp', 'xgboost')                | 0.6577 | 0.0332 |
| ('imf_linear_a1.41', 'comp+log_scale', 'extratrees')   | 0.6581 | 0.0352 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')     | 0.6748 | 0.0270 |
| ('imf_linear_a1.41', 'comp+log_scale', 'xgboost')      | 0.6607 | 0.0295 |
| ('irmf_h2_a1.41', 'comp', 'extratrees')                | 0.6494 | 0.0364 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                  | 0.6632 | 0.0305 |
| ('irmf_h2_a1.41', 'comp', 'xgboost')                   | 0.6408 | 0.0321 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'extratrees')      | 0.6493 | 0.0342 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')        | 0.6708 | 0.0314 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')         | 0.6488 | 0.0300 |
| ('irmf_h2_a1.41_conv', 'comp', 'extratrees')           | 0.6503 | 0.0397 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')             | 0.6589 | 0.0312 |
| ('irmf_h2_a1.41_conv', 'comp', 'xgboost')              | 0.6405 | 0.0255 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.6512 | 0.0381 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm')   | 0.6640 | 0.0309 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'xgboost')    | 0.6445 | 0.0245 |
| ('raw', 'comp', 'extratrees')                          | 0.5214 | 0.0189 |
| ('raw', 'comp', 'lightgbm')                            | 0.5237 | 0.0312 |
| ('raw', 'comp', 'xgboost')                             | 0.5028 | 0.0222 |
| ('raw', 'comp+log_scale', 'extratrees')                | 0.5316 | 0.0265 |
| ('raw', 'comp+log_scale', 'lightgbm')                  | 0.5358 | 0.0379 |
| ('raw', 'comp+log_scale', 'xgboost')                   | 0.5151 | 0.0313 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | raw              | emd                |     -0.0709 |  -0.1283 |   -0.0135 |      0.4058 |            0.0058 |
| comp           | lightgbm     | raw              | imf_linear_a1.41   |     -0.1481 |  -0.1989 |   -0.0973 |      0.0010 |            0.0037 |
| comp           | lightgbm     | raw              | irmf_h2_a1.41      |     -0.1395 |  -0.1935 |   -0.0855 |      0.0029 |            0.0037 |
| comp           | lightgbm     | raw              | irmf_h2_a1.41_conv |     -0.1352 |  -0.1889 |   -0.0815 |      0.0036 |            0.0037 |
| comp           | lightgbm     | emd              | imf_linear_a1.41   |     -0.0772 |  -0.1087 |   -0.0457 |      0.0043 |            0.0037 |
| comp           | lightgbm     | emd              | irmf_h2_a1.41      |     -0.0686 |  -0.1042 |   -0.0330 |      0.0262 |            0.0037 |
| comp           | lightgbm     | emd              | irmf_h2_a1.41_conv |     -0.0643 |  -0.0954 |   -0.0331 |      0.0166 |            0.0037 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0086 |  -0.0034 |    0.0206 |      1.0000 |            0.0806 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0129 |   0.0002 |    0.0256 |      0.7996 |            0.0037 |
| comp           | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0043 |  -0.0075 |    0.0162 |      1.0000 |            0.9644 |
| comp           | xgboost      | raw              | emd                |     -0.0696 |  -0.1161 |   -0.0231 |      0.1514 |            0.0037 |
| comp           | xgboost      | raw              | imf_linear_a1.41   |     -0.1549 |  -0.1897 |   -0.1201 |      0.0000 |            0.0037 |
| comp           | xgboost      | raw              | irmf_h2_a1.41      |     -0.1380 |  -0.1776 |   -0.0984 |      0.0002 |            0.0037 |
| comp           | xgboost      | raw              | irmf_h2_a1.41_conv |     -0.1377 |  -0.1684 |   -0.1071 |      0.0000 |            0.0037 |
| comp           | xgboost      | emd              | imf_linear_a1.41   |     -0.0853 |  -0.1171 |   -0.0535 |      0.0022 |            0.0037 |
| comp           | xgboost      | emd              | irmf_h2_a1.41      |     -0.0684 |  -0.1039 |   -0.0329 |      0.0262 |            0.0037 |
| comp           | xgboost      | emd              | irmf_h2_a1.41_conv |     -0.0681 |  -0.1000 |   -0.0362 |      0.0136 |            0.0037 |
| comp           | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0169 |   0.0036 |    0.0302 |      0.3837 |            0.0037 |
| comp           | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0172 |   0.0034 |    0.0310 |      0.4058 |            0.0058 |
| comp           | xgboost      | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0003 |  -0.0107 |    0.0113 |      1.0000 |            1.0000 |
| comp           | extratrees   | raw              | emd                |     -0.0408 |  -0.0880 |    0.0065 |      1.0000 |            0.0470 |
| comp           | extratrees   | raw              | imf_linear_a1.41   |     -0.1365 |  -0.1794 |   -0.0936 |      0.0004 |            0.0037 |
| comp           | extratrees   | raw              | irmf_h2_a1.41      |     -0.1280 |  -0.1706 |   -0.0853 |      0.0008 |            0.0037 |
| comp           | extratrees   | raw              | irmf_h2_a1.41_conv |     -0.1289 |  -0.1787 |   -0.0791 |      0.0029 |            0.0037 |
| comp           | extratrees   | emd              | imf_linear_a1.41   |     -0.0957 |  -0.1269 |   -0.0646 |      0.0006 |            0.0037 |
| comp           | extratrees   | emd              | irmf_h2_a1.41      |     -0.0872 |  -0.1310 |   -0.0434 |      0.0211 |            0.0037 |
| comp           | extratrees   | emd              | irmf_h2_a1.41_conv |     -0.0881 |  -0.1323 |   -0.0440 |      0.0211 |            0.0037 |
| comp           | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0085 |  -0.0135 |    0.0305 |      1.0000 |            0.6372 |
| comp           | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0076 |  -0.0129 |    0.0280 |      1.0000 |            0.7493 |
| comp           | extratrees   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0009 |  -0.0154 |    0.0135 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw              | emd                |     -0.0612 |  -0.1107 |   -0.0116 |      0.4058 |            0.0037 |
| comp+log_scale | lightgbm     | raw              | imf_linear_a1.41   |     -0.1390 |  -0.1887 |   -0.0894 |      0.0014 |            0.0037 |
| comp+log_scale | lightgbm     | raw              | irmf_h2_a1.41      |     -0.1350 |  -0.1883 |   -0.0817 |      0.0034 |            0.0037 |
| comp+log_scale | lightgbm     | raw              | irmf_h2_a1.41_conv |     -0.1282 |  -0.1818 |   -0.0746 |      0.0054 |            0.0037 |
| comp+log_scale | lightgbm     | emd              | imf_linear_a1.41   |     -0.0779 |  -0.1027 |   -0.0531 |      0.0005 |            0.0037 |
| comp+log_scale | lightgbm     | emd              | irmf_h2_a1.41      |     -0.0739 |  -0.1015 |   -0.0463 |      0.0022 |            0.0037 |
| comp+log_scale | lightgbm     | emd              | irmf_h2_a1.41_conv |     -0.0670 |  -0.0942 |   -0.0398 |      0.0043 |            0.0037 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0040 |  -0.0094 |    0.0174 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0109 |  -0.0023 |    0.0241 |      1.0000 |            0.0470 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0069 |  -0.0013 |    0.0150 |      1.0000 |            0.0186 |
| comp+log_scale | xgboost      | raw              | emd                |     -0.0582 |  -0.1058 |   -0.0105 |      0.4058 |            0.0037 |
| comp+log_scale | xgboost      | raw              | imf_linear_a1.41   |     -0.1456 |  -0.1817 |   -0.1095 |      0.0000 |            0.0037 |
| comp+log_scale | xgboost      | raw              | irmf_h2_a1.41      |     -0.1336 |  -0.1809 |   -0.0863 |      0.0014 |            0.0037 |
| comp+log_scale | xgboost      | raw              | irmf_h2_a1.41_conv |     -0.1294 |  -0.1705 |   -0.0883 |      0.0005 |            0.0037 |
| comp+log_scale | xgboost      | emd              | imf_linear_a1.41   |     -0.0874 |  -0.1181 |   -0.0568 |      0.0012 |            0.0037 |
| comp+log_scale | xgboost      | emd              | irmf_h2_a1.41      |     -0.0754 |  -0.1111 |   -0.0398 |      0.0138 |            0.0037 |
| comp+log_scale | xgboost      | emd              | irmf_h2_a1.41_conv |     -0.0712 |  -0.1025 |   -0.0399 |      0.0083 |            0.0037 |
| comp+log_scale | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0120 |  -0.0050 |    0.0290 |      1.0000 |            0.1128 |
| comp+log_scale | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0162 |   0.0008 |    0.0317 |      0.7304 |            0.0073 |
| comp+log_scale | xgboost      | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0042 |  -0.0082 |    0.0166 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw              | emd                |     -0.0392 |  -0.0830 |    0.0045 |      1.0000 |            0.0186 |
| comp+log_scale | extratrees   | raw              | imf_linear_a1.41   |     -0.1264 |  -0.1634 |   -0.0895 |      0.0002 |            0.0037 |
| comp+log_scale | extratrees   | raw              | irmf_h2_a1.41      |     -0.1177 |  -0.1561 |   -0.0793 |      0.0006 |            0.0037 |
| comp+log_scale | extratrees   | raw              | irmf_h2_a1.41_conv |     -0.1196 |  -0.1636 |   -0.0757 |      0.0019 |            0.0037 |
| comp+log_scale | extratrees   | emd              | imf_linear_a1.41   |     -0.0872 |  -0.1114 |   -0.0630 |      0.0001 |            0.0037 |
| comp+log_scale | extratrees   | emd              | irmf_h2_a1.41      |     -0.0785 |  -0.1149 |   -0.0421 |      0.0130 |            0.0037 |
| comp+log_scale | extratrees   | emd              | irmf_h2_a1.41_conv |     -0.0804 |  -0.1180 |   -0.0428 |      0.0136 |            0.0037 |
| comp+log_scale | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0087 |  -0.0157 |    0.0332 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0068 |  -0.0159 |    0.0296 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0019 |  -0.0170 |    0.0132 |      1.0000 |            1.0000 |
