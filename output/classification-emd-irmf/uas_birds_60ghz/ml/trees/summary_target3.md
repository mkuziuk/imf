# uas_phase3_trees: uas_birds_60ghz / target3

810 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 2126.5 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'extratrees')                          | 0.9292 | 0.0234 |
| ('emd', 'comp', 'lightgbm')                            | 0.9434 | 0.0239 |
| ('emd', 'comp', 'xgboost')                             | 0.9424 | 0.0248 |
| ('emd', 'comp+log_scale', 'extratrees')                | 0.9278 | 0.0269 |
| ('emd', 'comp+log_scale', 'lightgbm')                  | 0.9479 | 0.0287 |
| ('emd', 'comp+log_scale', 'xgboost')                   | 0.9434 | 0.0289 |
| ('imf_linear_a1.41', 'comp', 'extratrees')             | 0.9545 | 0.0245 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')               | 0.9451 | 0.0320 |
| ('imf_linear_a1.41', 'comp', 'xgboost')                | 0.9378 | 0.0330 |
| ('imf_linear_a1.41', 'comp+log_scale', 'extratrees')   | 0.9517 | 0.0241 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')     | 0.9434 | 0.0337 |
| ('imf_linear_a1.41', 'comp+log_scale', 'xgboost')      | 0.9365 | 0.0351 |
| ('imf_linear_a2', 'comp', 'extratrees')                | 0.9531 | 0.0238 |
| ('imf_linear_a2', 'comp', 'lightgbm')                  | 0.9455 | 0.0315 |
| ('imf_linear_a2', 'comp', 'xgboost')                   | 0.9417 | 0.0327 |
| ('imf_linear_a2', 'comp+log_scale', 'extratrees')      | 0.9528 | 0.0207 |
| ('imf_linear_a2', 'comp+log_scale', 'lightgbm')        | 0.9465 | 0.0315 |
| ('imf_linear_a2', 'comp+log_scale', 'xgboost')         | 0.9424 | 0.0317 |
| ('irmf_h1_a1.41_conv', 'comp', 'extratrees')           | 0.9691 | 0.0186 |
| ('irmf_h1_a1.41_conv', 'comp', 'lightgbm')             | 0.9726 | 0.0267 |
| ('irmf_h1_a1.41_conv', 'comp', 'xgboost')              | 0.9719 | 0.0288 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.9649 | 0.0205 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'lightgbm')   | 0.9736 | 0.0272 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'xgboost')    | 0.9708 | 0.0302 |
| ('irmf_h2_a1.41', 'comp', 'extratrees')                | 0.9646 | 0.0216 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                  | 0.9646 | 0.0269 |
| ('irmf_h2_a1.41', 'comp', 'xgboost')                   | 0.9635 | 0.0309 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'extratrees')      | 0.9625 | 0.0231 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')        | 0.9653 | 0.0266 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')         | 0.9622 | 0.0315 |
| ('irmf_h2_a1.41_conv', 'comp', 'extratrees')           | 0.9594 | 0.0219 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')             | 0.9660 | 0.0273 |
| ('irmf_h2_a1.41_conv', 'comp', 'xgboost')              | 0.9615 | 0.0319 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.9604 | 0.0230 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm')   | 0.9649 | 0.0280 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'xgboost')    | 0.9601 | 0.0320 |
| ('irmf_h2_a2_conv', 'comp', 'extratrees')              | 0.9608 | 0.0224 |
| ('irmf_h2_a2_conv', 'comp', 'lightgbm')                | 0.9594 | 0.0288 |
| ('irmf_h2_a2_conv', 'comp', 'xgboost')                 | 0.9580 | 0.0326 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'extratrees')    | 0.9639 | 0.0224 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'lightgbm')      | 0.9580 | 0.0299 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'xgboost')       | 0.9583 | 0.0325 |
| ('irmf_h4_a1.41_conv', 'comp', 'extratrees')           | 0.9580 | 0.0265 |
| ('irmf_h4_a1.41_conv', 'comp', 'lightgbm')             | 0.9549 | 0.0341 |
| ('irmf_h4_a1.41_conv', 'comp', 'xgboost')              | 0.9455 | 0.0362 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.9566 | 0.0237 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'lightgbm')   | 0.9528 | 0.0361 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'xgboost')    | 0.9444 | 0.0363 |
| ('raw', 'comp', 'extratrees')                          | 0.8958 | 0.0427 |
| ('raw', 'comp', 'lightgbm')                            | 0.9149 | 0.0258 |
| ('raw', 'comp', 'xgboost')                             | 0.9139 | 0.0274 |
| ('raw', 'comp+log_scale', 'extratrees')                | 0.9160 | 0.0384 |
| ('raw', 'comp+log_scale', 'lightgbm')                  | 0.9340 | 0.0356 |
| ('raw', 'comp+log_scale', 'xgboost')                   | 0.9358 | 0.0336 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | raw                | emd                |     -0.0285 |  -0.0600 |    0.0030 |      1.0000 |            0.2734 |
| comp           | lightgbm     | raw                | imf_linear_a1.41   |     -0.0302 |  -0.0675 |    0.0071 |      1.0000 |            0.6420 |
| comp           | lightgbm     | raw                | irmf_h2_a1.41      |     -0.0497 |  -0.0822 |   -0.0171 |      1.0000 |            0.1563 |
| comp           | lightgbm     | raw                | irmf_h2_a1.41_conv |     -0.0510 |  -0.0852 |   -0.0168 |      1.0000 |            0.1563 |
| comp           | lightgbm     | raw                | irmf_h2_a2_conv    |     -0.0444 |  -0.0789 |   -0.0100 |      1.0000 |            0.1790 |
| comp           | lightgbm     | raw                | imf_linear_a2      |     -0.0306 |  -0.0665 |    0.0054 |      1.0000 |            0.6765 |
| comp           | lightgbm     | raw                | irmf_h1_a1.41_conv |     -0.0576 |  -0.0957 |   -0.0196 |      1.0000 |            0.1563 |
| comp           | lightgbm     | raw                | irmf_h4_a1.41_conv |     -0.0399 |  -0.0763 |   -0.0036 |      1.0000 |            0.2188 |
| comp           | lightgbm     | emd                | imf_linear_a1.41   |     -0.0017 |  -0.0220 |    0.0185 |      1.0000 |            1.0000 |
| comp           | lightgbm     | emd                | irmf_h2_a1.41      |     -0.0212 |  -0.0448 |    0.0024 |      1.0000 |            0.7712 |
| comp           | lightgbm     | emd                | irmf_h2_a1.41_conv |     -0.0226 |  -0.0400 |   -0.0052 |      1.0000 |            0.2030 |
| comp           | lightgbm     | emd                | irmf_h2_a2_conv    |     -0.0160 |  -0.0289 |   -0.0031 |      1.0000 |            0.2030 |
| comp           | lightgbm     | emd                | imf_linear_a2      |     -0.0021 |  -0.0250 |    0.0208 |      1.0000 |            1.0000 |
| comp           | lightgbm     | emd                | irmf_h1_a1.41_conv |     -0.0292 |  -0.0454 |   -0.0129 |      0.3816 |            0.1310 |
| comp           | lightgbm     | emd                | irmf_h4_a1.41_conv |     -0.0115 |  -0.0342 |    0.0112 |      1.0000 |            1.0000 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0194 |  -0.0404 |    0.0015 |      1.0000 |            0.6315 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0208 |  -0.0381 |   -0.0035 |      1.0000 |            0.1790 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0142 |  -0.0336 |    0.0051 |      1.0000 |            0.6934 |
| comp           | lightgbm     | imf_linear_a1.41   | imf_linear_a2      |     -0.0003 |  -0.0331 |    0.0324 |      1.0000 |            1.0000 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0274 |  -0.0459 |   -0.0090 |      1.0000 |            0.1306 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0097 |  -0.0314 |    0.0119 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0014 |  -0.0145 |    0.0117 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0052 |  -0.0160 |    0.0265 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | imf_linear_a2      |      0.0191 |  -0.0061 |    0.0443 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0080 |  -0.0243 |    0.0083 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0097 |  -0.0151 |    0.0345 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0066 |  -0.0067 |    0.0199 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0205 |  -0.0011 |    0.0421 |      1.0000 |            0.3210 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0066 |  -0.0168 |    0.0036 |      1.0000 |            0.7910 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0111 |  -0.0077 |    0.0300 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a2_conv    | imf_linear_a2      |      0.0139 |  -0.0101 |    0.0379 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0132 |  -0.0280 |    0.0016 |      1.0000 |            0.1790 |
| comp           | lightgbm     | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0045 |  -0.0146 |    0.0237 |      1.0000 |            1.0000 |
| comp           | lightgbm     | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0271 |  -0.0495 |   -0.0046 |      1.0000 |            0.0527 |
| comp           | lightgbm     | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0094 |  -0.0344 |    0.0157 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0177 |  -0.0064 |    0.0418 |      1.0000 |            1.0000 |
| comp           | xgboost      | raw                | emd                |     -0.0285 |  -0.0602 |    0.0032 |      1.0000 |            0.5750 |
| comp           | xgboost      | raw                | imf_linear_a1.41   |     -0.0240 |  -0.0617 |    0.0138 |      1.0000 |            1.0000 |
| comp           | xgboost      | raw                | irmf_h2_a1.41      |     -0.0497 |  -0.0913 |   -0.0080 |      1.0000 |            0.1790 |
| comp           | xgboost      | raw                | irmf_h2_a1.41_conv |     -0.0476 |  -0.0909 |   -0.0042 |      1.0000 |            0.1790 |
| comp           | xgboost      | raw                | irmf_h2_a2_conv    |     -0.0441 |  -0.0804 |   -0.0078 |      1.0000 |            0.1790 |
| comp           | xgboost      | raw                | imf_linear_a2      |     -0.0278 |  -0.0679 |    0.0123 |      1.0000 |            0.7358 |
| comp           | xgboost      | raw                | irmf_h1_a1.41_conv |     -0.0580 |  -0.0939 |   -0.0221 |      0.8042 |            0.1790 |
| comp           | xgboost      | raw                | irmf_h4_a1.41_conv |     -0.0316 |  -0.0712 |    0.0080 |      1.0000 |            0.2388 |
| comp           | xgboost      | emd                | imf_linear_a1.41   |      0.0045 |  -0.0201 |    0.0291 |      1.0000 |            1.0000 |
| comp           | xgboost      | emd                | irmf_h2_a1.41      |     -0.0212 |  -0.0558 |    0.0135 |      1.0000 |            1.0000 |
| comp           | xgboost      | emd                | irmf_h2_a1.41_conv |     -0.0191 |  -0.0564 |    0.0182 |      1.0000 |            1.0000 |
| comp           | xgboost      | emd                | irmf_h2_a2_conv    |     -0.0156 |  -0.0466 |    0.0154 |      1.0000 |            1.0000 |
| comp           | xgboost      | emd                | imf_linear_a2      |      0.0007 |  -0.0307 |    0.0321 |      1.0000 |            1.0000 |
| comp           | xgboost      | emd                | irmf_h1_a1.41_conv |     -0.0295 |  -0.0575 |   -0.0015 |      1.0000 |            0.0527 |
| comp           | xgboost      | emd                | irmf_h4_a1.41_conv |     -0.0031 |  -0.0390 |    0.0327 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0257 |  -0.0528 |    0.0014 |      1.0000 |            0.3047 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0236 |  -0.0517 |    0.0045 |      1.0000 |            0.6593 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0201 |  -0.0465 |    0.0062 |      1.0000 |            0.1790 |
| comp           | xgboost      | imf_linear_a1.41   | imf_linear_a2      |     -0.0038 |  -0.0356 |    0.0279 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0340 |  -0.0582 |   -0.0098 |      1.0000 |            0.1306 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0076 |  -0.0307 |    0.0154 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0021 |  -0.0036 |    0.0078 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0056 |  -0.0126 |    0.0237 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | imf_linear_a2      |      0.0219 |  -0.0077 |    0.0515 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0083 |  -0.0254 |    0.0088 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0181 |  -0.0037 |    0.0398 |      1.0000 |            0.1790 |
| comp           | xgboost      | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0035 |  -0.0168 |    0.0237 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0198 |  -0.0103 |    0.0499 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0104 |  -0.0304 |    0.0096 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0160 |  -0.0071 |    0.0391 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a2_conv    | imf_linear_a2      |      0.0163 |  -0.0072 |    0.0398 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0139 |  -0.0298 |    0.0020 |      1.0000 |            0.2388 |
| comp           | xgboost      | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0125 |  -0.0106 |    0.0356 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0302 |  -0.0538 |   -0.0067 |      1.0000 |            0.0527 |
| comp           | xgboost      | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0038 |  -0.0339 |    0.0263 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0264 |   0.0011 |    0.0517 |      1.0000 |            0.1021 |
| comp           | extratrees   | raw                | emd                |     -0.0333 |  -0.0666 |   -0.0001 |      1.0000 |            0.2384 |
| comp           | extratrees   | raw                | imf_linear_a1.41   |     -0.0587 |  -0.1094 |   -0.0080 |      1.0000 |            0.1563 |
| comp           | extratrees   | raw                | irmf_h2_a1.41      |     -0.0688 |  -0.1168 |   -0.0207 |      1.0000 |            0.1310 |
| comp           | extratrees   | raw                | irmf_h2_a1.41_conv |     -0.0635 |  -0.1107 |   -0.0164 |      1.0000 |            0.1310 |
| comp           | extratrees   | raw                | irmf_h2_a2_conv    |     -0.0649 |  -0.1113 |   -0.0186 |      1.0000 |            0.1310 |
| comp           | extratrees   | raw                | imf_linear_a2      |     -0.0573 |  -0.1089 |   -0.0057 |      1.0000 |            0.2030 |
| comp           | extratrees   | raw                | irmf_h1_a1.41_conv |     -0.0733 |  -0.1169 |   -0.0296 |      0.6210 |            0.1310 |
| comp           | extratrees   | raw                | irmf_h4_a1.41_conv |     -0.0622 |  -0.1121 |   -0.0122 |      1.0000 |            0.1563 |
| comp           | extratrees   | emd                | imf_linear_a1.41   |     -0.0253 |  -0.0589 |    0.0082 |      1.0000 |            0.8256 |
| comp           | extratrees   | emd                | irmf_h2_a1.41      |     -0.0354 |  -0.0630 |   -0.0078 |      1.0000 |            0.1563 |
| comp           | extratrees   | emd                | irmf_h2_a1.41_conv |     -0.0302 |  -0.0564 |   -0.0040 |      1.0000 |            0.1847 |
| comp           | extratrees   | emd                | irmf_h2_a2_conv    |     -0.0316 |  -0.0569 |   -0.0063 |      1.0000 |            0.1657 |
| comp           | extratrees   | emd                | imf_linear_a2      |     -0.0240 |  -0.0557 |    0.0078 |      1.0000 |            1.0000 |
| comp           | extratrees   | emd                | irmf_h1_a1.41_conv |     -0.0399 |  -0.0580 |   -0.0219 |      0.0676 |            0.1302 |
| comp           | extratrees   | emd                | irmf_h4_a1.41_conv |     -0.0288 |  -0.0645 |    0.0068 |      1.0000 |            0.8216 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0101 |  -0.0314 |    0.0112 |      1.0000 |            1.0000 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0049 |  -0.0256 |    0.0159 |      1.0000 |            1.0000 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0063 |  -0.0292 |    0.0167 |      1.0000 |            1.0000 |
| comp           | extratrees   | imf_linear_a1.41   | imf_linear_a2      |      0.0014 |  -0.0106 |    0.0134 |      1.0000 |            1.0000 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0146 |  -0.0417 |    0.0125 |      1.0000 |            1.0000 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0035 |  -0.0212 |    0.0143 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0052 |  -0.0054 |    0.0158 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0038 |  -0.0052 |    0.0129 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | imf_linear_a2      |      0.0115 |  -0.0109 |    0.0338 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0045 |  -0.0223 |    0.0132 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0066 |  -0.0094 |    0.0226 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0014 |  -0.0098 |    0.0070 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0063 |  -0.0137 |    0.0262 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0097 |  -0.0249 |    0.0055 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0014 |  -0.0141 |    0.0169 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a2_conv    | imf_linear_a2      |      0.0076 |  -0.0143 |    0.0295 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0083 |  -0.0227 |    0.0061 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0028 |  -0.0143 |    0.0199 |      1.0000 |            1.0000 |
| comp           | extratrees   | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0160 |  -0.0387 |    0.0068 |      1.0000 |            0.3086 |
| comp           | extratrees   | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0049 |  -0.0272 |    0.0175 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0111 |  -0.0148 |    0.0370 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | emd                |     -0.0139 |  -0.0603 |    0.0325 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | imf_linear_a1.41   |     -0.0094 |  -0.0635 |    0.0448 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a1.41      |     -0.0312 |  -0.0840 |    0.0215 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a1.41_conv |     -0.0309 |  -0.0850 |    0.0232 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a2_conv    |     -0.0240 |  -0.0747 |    0.0268 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | imf_linear_a2      |     -0.0125 |  -0.0708 |    0.0458 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | irmf_h1_a1.41_conv |     -0.0396 |  -0.0930 |    0.0139 |      1.0000 |            0.4048 |
| comp+log_scale | lightgbm     | raw                | irmf_h4_a1.41_conv |     -0.0187 |  -0.0774 |    0.0399 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | emd                | imf_linear_a1.41   |      0.0045 |  -0.0171 |    0.0261 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a1.41      |     -0.0174 |  -0.0429 |    0.0082 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a1.41_conv |     -0.0170 |  -0.0383 |    0.0043 |      1.0000 |            0.6726 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a2_conv    |     -0.0101 |  -0.0266 |    0.0065 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | emd                | imf_linear_a2      |      0.0014 |  -0.0243 |    0.0270 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | emd                | irmf_h1_a1.41_conv |     -0.0257 |  -0.0482 |   -0.0032 |      1.0000 |            0.1847 |
| comp+log_scale | lightgbm     | emd                | irmf_h4_a1.41_conv |     -0.0049 |  -0.0275 |    0.0177 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0219 |  -0.0406 |   -0.0031 |      1.0000 |            0.0527 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0215 |  -0.0376 |   -0.0054 |      1.0000 |            0.1790 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0146 |  -0.0333 |    0.0042 |      1.0000 |            0.6934 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | imf_linear_a2      |     -0.0031 |  -0.0323 |    0.0261 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0302 |  -0.0495 |   -0.0109 |      1.0000 |            0.1790 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0094 |  -0.0281 |    0.0094 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0003 |  -0.0096 |    0.0103 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0073 |  -0.0133 |    0.0278 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | imf_linear_a2      |      0.0187 |  -0.0062 |    0.0437 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0083 |  -0.0231 |    0.0065 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0125 |  -0.0129 |    0.0379 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0069 |  -0.0074 |    0.0213 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0184 |  -0.0041 |    0.0409 |      1.0000 |            0.7088 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0087 |  -0.0197 |    0.0024 |      1.0000 |            0.5938 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0122 |  -0.0084 |    0.0327 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a2_conv    | imf_linear_a2      |      0.0115 |  -0.0107 |    0.0336 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0156 |  -0.0305 |   -0.0008 |      1.0000 |            0.1021 |
| comp+log_scale | lightgbm     | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0052 |  -0.0142 |    0.0247 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0271 |  -0.0482 |   -0.0059 |      1.0000 |            0.0527 |
| comp+log_scale | lightgbm     | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0062 |  -0.0309 |    0.0184 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0208 |  -0.0044 |    0.0461 |      1.0000 |            0.6934 |
| comp+log_scale | xgboost      | raw                | emd                |     -0.0076 |  -0.0505 |    0.0352 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | imf_linear_a1.41   |     -0.0007 |  -0.0478 |    0.0464 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a1.41      |     -0.0264 |  -0.0833 |    0.0305 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a1.41_conv |     -0.0243 |  -0.0845 |    0.0359 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a2_conv    |     -0.0226 |  -0.0759 |    0.0308 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | imf_linear_a2      |     -0.0066 |  -0.0600 |    0.0468 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | irmf_h1_a1.41_conv |     -0.0351 |  -0.0860 |    0.0159 |      1.0000 |            0.6315 |
| comp+log_scale | xgboost      | raw                | irmf_h4_a1.41_conv |     -0.0087 |  -0.0617 |    0.0443 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | imf_linear_a1.41   |      0.0069 |  -0.0210 |    0.0349 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a1.41      |     -0.0187 |  -0.0541 |    0.0166 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a1.41_conv |     -0.0167 |  -0.0543 |    0.0209 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a2_conv    |     -0.0149 |  -0.0431 |    0.0133 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | imf_linear_a2      |      0.0010 |  -0.0278 |    0.0299 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | irmf_h1_a1.41_conv |     -0.0274 |  -0.0529 |   -0.0020 |      1.0000 |            0.0527 |
| comp+log_scale | xgboost      | emd                | irmf_h4_a1.41_conv |     -0.0010 |  -0.0377 |    0.0356 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0257 |  -0.0565 |    0.0052 |      1.0000 |            0.6576 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0236 |  -0.0566 |    0.0094 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0219 |  -0.0520 |    0.0083 |      1.0000 |            0.4919 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | imf_linear_a2      |     -0.0059 |  -0.0374 |    0.0256 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0344 |  -0.0581 |   -0.0106 |      1.0000 |            0.1310 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0080 |  -0.0311 |    0.0152 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0021 |  -0.0054 |    0.0095 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0038 |  -0.0151 |    0.0228 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | imf_linear_a2      |      0.0198 |  -0.0113 |    0.0509 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0087 |  -0.0278 |    0.0104 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0177 |  -0.0046 |    0.0400 |      1.0000 |            0.2388 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0017 |  -0.0205 |    0.0240 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0177 |  -0.0164 |    0.0518 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0108 |  -0.0325 |    0.0109 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0156 |  -0.0109 |    0.0422 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a2_conv    | imf_linear_a2      |      0.0160 |  -0.0098 |    0.0417 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0125 |  -0.0294 |    0.0044 |      1.0000 |            0.5938 |
| comp+log_scale | xgboost      | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0139 |  -0.0138 |    0.0416 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0285 |  -0.0537 |   -0.0032 |      1.0000 |            0.1021 |
| comp+log_scale | xgboost      | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0021 |  -0.0331 |    0.0290 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0264 |   0.0010 |    0.0518 |      1.0000 |            0.0527 |
| comp+log_scale | extratrees   | raw                | emd                |     -0.0118 |  -0.0466 |    0.0230 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | imf_linear_a1.41   |     -0.0358 |  -0.0794 |    0.0079 |      1.0000 |            0.6934 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a1.41      |     -0.0465 |  -0.0835 |   -0.0095 |      1.0000 |            0.1790 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a1.41_conv |     -0.0444 |  -0.0809 |   -0.0079 |      1.0000 |            0.1563 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a2_conv    |     -0.0479 |  -0.0816 |   -0.0142 |      1.0000 |            0.1310 |
| comp+log_scale | extratrees   | raw                | imf_linear_a2      |     -0.0368 |  -0.0797 |    0.0061 |      1.0000 |            0.5369 |
| comp+log_scale | extratrees   | raw                | irmf_h1_a1.41_conv |     -0.0490 |  -0.0853 |   -0.0126 |      1.0000 |            0.1790 |
| comp+log_scale | extratrees   | raw                | irmf_h4_a1.41_conv |     -0.0406 |  -0.0790 |   -0.0022 |      1.0000 |            0.1790 |
| comp+log_scale | extratrees   | emd                | imf_linear_a1.41   |     -0.0240 |  -0.0569 |    0.0090 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a1.41      |     -0.0347 |  -0.0647 |   -0.0048 |      1.0000 |            0.2023 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a1.41_conv |     -0.0326 |  -0.0588 |   -0.0064 |      1.0000 |            0.2142 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a2_conv    |     -0.0361 |  -0.0632 |   -0.0091 |      1.0000 |            0.1646 |
| comp+log_scale | extratrees   | emd                | imf_linear_a2      |     -0.0250 |  -0.0563 |    0.0063 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | emd                | irmf_h1_a1.41_conv |     -0.0372 |  -0.0672 |   -0.0071 |      1.0000 |            0.1790 |
| comp+log_scale | extratrees   | emd                | irmf_h4_a1.41_conv |     -0.0288 |  -0.0617 |    0.0040 |      1.0000 |            0.6930 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0108 |  -0.0273 |    0.0058 |      1.0000 |            0.7910 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0087 |  -0.0288 |    0.0114 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0122 |  -0.0367 |    0.0124 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | imf_linear_a2      |     -0.0010 |  -0.0141 |    0.0120 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0132 |  -0.0451 |    0.0187 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0049 |  -0.0236 |    0.0139 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0021 |  -0.0082 |    0.0124 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0014 |  -0.0155 |    0.0128 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | imf_linear_a2      |      0.0097 |  -0.0057 |    0.0251 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0024 |  -0.0266 |    0.0218 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0059 |  -0.0091 |    0.0209 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0035 |  -0.0153 |    0.0083 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0076 |  -0.0093 |    0.0246 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0045 |  -0.0270 |    0.0180 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0038 |  -0.0093 |    0.0169 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a2_conv    | imf_linear_a2      |      0.0111 |  -0.0095 |    0.0317 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0010 |  -0.0155 |    0.0135 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0073 |  -0.0067 |    0.0213 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0122 |  -0.0388 |    0.0145 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0038 |  -0.0204 |    0.0127 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0083 |  -0.0150 |    0.0317 |      1.0000 |            1.0000 |
