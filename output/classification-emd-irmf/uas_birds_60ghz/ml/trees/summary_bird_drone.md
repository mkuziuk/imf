# uas_phase3_trees: uas_birds_60ghz / bird_drone

810 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 872.1 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'extratrees')                          | 0.9543 | 0.0274 |
| ('emd', 'comp', 'lightgbm')                            | 0.9622 | 0.0208 |
| ('emd', 'comp', 'xgboost')                             | 0.9679 | 0.0183 |
| ('emd', 'comp+log_scale', 'extratrees')                | 0.9585 | 0.0251 |
| ('emd', 'comp+log_scale', 'lightgbm')                  | 0.9724 | 0.0174 |
| ('emd', 'comp+log_scale', 'xgboost')                   | 0.9740 | 0.0139 |
| ('imf_linear_a1.41', 'comp', 'extratrees')             | 0.9710 | 0.0235 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')               | 0.9789 | 0.0210 |
| ('imf_linear_a1.41', 'comp', 'xgboost')                | 0.9799 | 0.0177 |
| ('imf_linear_a1.41', 'comp+log_scale', 'extratrees')   | 0.9730 | 0.0228 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')     | 0.9802 | 0.0178 |
| ('imf_linear_a1.41', 'comp+log_scale', 'xgboost')      | 0.9790 | 0.0178 |
| ('imf_linear_a2', 'comp', 'extratrees')                | 0.9705 | 0.0241 |
| ('imf_linear_a2', 'comp', 'lightgbm')                  | 0.9835 | 0.0148 |
| ('imf_linear_a2', 'comp', 'xgboost')                   | 0.9812 | 0.0164 |
| ('imf_linear_a2', 'comp+log_scale', 'extratrees')      | 0.9701 | 0.0228 |
| ('imf_linear_a2', 'comp+log_scale', 'lightgbm')        | 0.9828 | 0.0153 |
| ('imf_linear_a2', 'comp+log_scale', 'xgboost')         | 0.9816 | 0.0158 |
| ('irmf_h1_a1.41_conv', 'comp', 'extratrees')           | 0.9860 | 0.0140 |
| ('irmf_h1_a1.41_conv', 'comp', 'lightgbm')             | 0.9904 | 0.0141 |
| ('irmf_h1_a1.41_conv', 'comp', 'xgboost')              | 0.9866 | 0.0205 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.9833 | 0.0145 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'lightgbm')   | 0.9911 | 0.0139 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'xgboost')    | 0.9866 | 0.0205 |
| ('irmf_h2_a1.41', 'comp', 'extratrees')                | 0.9825 | 0.0158 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                  | 0.9898 | 0.0124 |
| ('irmf_h2_a1.41', 'comp', 'xgboost')                   | 0.9886 | 0.0119 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'extratrees')      | 0.9819 | 0.0178 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')        | 0.9899 | 0.0119 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')         | 0.9880 | 0.0126 |
| ('irmf_h2_a1.41_conv', 'comp', 'extratrees')           | 0.9819 | 0.0162 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')             | 0.9911 | 0.0124 |
| ('irmf_h2_a1.41_conv', 'comp', 'xgboost')              | 0.9870 | 0.0136 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.9831 | 0.0166 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm')   | 0.9907 | 0.0122 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'xgboost')    | 0.9858 | 0.0146 |
| ('irmf_h2_a2_conv', 'comp', 'extratrees')              | 0.9781 | 0.0201 |
| ('irmf_h2_a2_conv', 'comp', 'lightgbm')                | 0.9892 | 0.0154 |
| ('irmf_h2_a2_conv', 'comp', 'xgboost')                 | 0.9834 | 0.0187 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'extratrees')    | 0.9789 | 0.0178 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'lightgbm')      | 0.9873 | 0.0159 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'xgboost')       | 0.9834 | 0.0192 |
| ('irmf_h4_a1.41_conv', 'comp', 'extratrees')           | 0.9812 | 0.0166 |
| ('irmf_h4_a1.41_conv', 'comp', 'lightgbm')             | 0.9878 | 0.0154 |
| ('irmf_h4_a1.41_conv', 'comp', 'xgboost')              | 0.9832 | 0.0171 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.9803 | 0.0165 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'lightgbm')   | 0.9867 | 0.0147 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'xgboost')    | 0.9833 | 0.0164 |
| ('raw', 'comp', 'extratrees')                          | 0.9368 | 0.0446 |
| ('raw', 'comp', 'lightgbm')                            | 0.9592 | 0.0260 |
| ('raw', 'comp', 'xgboost')                             | 0.9576 | 0.0285 |
| ('raw', 'comp+log_scale', 'extratrees')                | 0.9551 | 0.0408 |
| ('raw', 'comp+log_scale', 'lightgbm')                  | 0.9711 | 0.0268 |
| ('raw', 'comp+log_scale', 'xgboost')                   | 0.9734 | 0.0221 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | raw                | emd                |     -0.0030 |  -0.0271 |    0.0210 |      1.0000 |            1.0000 |
| comp           | lightgbm     | raw                | imf_linear_a1.41   |     -0.0197 |  -0.0486 |    0.0092 |      1.0000 |            1.0000 |
| comp           | lightgbm     | raw                | irmf_h2_a1.41      |     -0.0306 |  -0.0613 |    0.0001 |      1.0000 |            0.4252 |
| comp           | lightgbm     | raw                | irmf_h2_a1.41_conv |     -0.0319 |  -0.0644 |    0.0005 |      1.0000 |            0.5447 |
| comp           | lightgbm     | raw                | irmf_h2_a2_conv    |     -0.0300 |  -0.0632 |    0.0031 |      1.0000 |            0.6024 |
| comp           | lightgbm     | raw                | imf_linear_a2      |     -0.0243 |  -0.0521 |    0.0034 |      1.0000 |            0.6993 |
| comp           | lightgbm     | raw                | irmf_h1_a1.41_conv |     -0.0312 |  -0.0638 |    0.0015 |      1.0000 |            0.5081 |
| comp           | lightgbm     | raw                | irmf_h4_a1.41_conv |     -0.0286 |  -0.0610 |    0.0039 |      1.0000 |            0.6391 |
| comp           | lightgbm     | emd                | imf_linear_a1.41   |     -0.0166 |  -0.0428 |    0.0096 |      1.0000 |            1.0000 |
| comp           | lightgbm     | emd                | irmf_h2_a1.41      |     -0.0275 |  -0.0511 |   -0.0040 |      1.0000 |            0.1873 |
| comp           | lightgbm     | emd                | irmf_h2_a1.41_conv |     -0.0289 |  -0.0541 |   -0.0037 |      1.0000 |            0.2053 |
| comp           | lightgbm     | emd                | irmf_h2_a2_conv    |     -0.0270 |  -0.0557 |    0.0017 |      1.0000 |            0.3791 |
| comp           | lightgbm     | emd                | imf_linear_a2      |     -0.0213 |  -0.0428 |    0.0003 |      1.0000 |            0.3717 |
| comp           | lightgbm     | emd                | irmf_h1_a1.41_conv |     -0.0282 |  -0.0526 |   -0.0038 |      1.0000 |            0.2053 |
| comp           | lightgbm     | emd                | irmf_h4_a1.41_conv |     -0.0256 |  -0.0543 |    0.0032 |      1.0000 |            0.3951 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0109 |  -0.0260 |    0.0042 |      1.0000 |            0.3828 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0123 |  -0.0278 |    0.0033 |      1.0000 |            0.1055 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0104 |  -0.0241 |    0.0033 |      1.0000 |            0.5975 |
| comp           | lightgbm     | imf_linear_a1.41   | imf_linear_a2      |     -0.0047 |  -0.0188 |    0.0095 |      1.0000 |            1.0000 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0115 |  -0.0270 |    0.0040 |      1.0000 |            0.8594 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0089 |  -0.0205 |    0.0026 |      1.0000 |            0.3828 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0014 |  -0.0060 |    0.0032 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0005 |  -0.0081 |    0.0091 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | imf_linear_a2      |      0.0063 |  -0.0083 |    0.0208 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0006 |  -0.0125 |    0.0113 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0020 |  -0.0077 |    0.0116 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0019 |  -0.0043 |    0.0081 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0076 |  -0.0067 |    0.0220 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |      0.0008 |  -0.0099 |    0.0114 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0034 |  -0.0054 |    0.0121 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a2_conv    | imf_linear_a2      |      0.0057 |  -0.0095 |    0.0209 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0011 |  -0.0124 |    0.0101 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0015 |  -0.0056 |    0.0085 |      1.0000 |            1.0000 |
| comp           | lightgbm     | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0069 |  -0.0219 |    0.0081 |      1.0000 |            1.0000 |
| comp           | lightgbm     | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0043 |  -0.0183 |    0.0097 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0026 |  -0.0087 |    0.0139 |      1.0000 |            1.0000 |
| comp           | xgboost      | raw                | emd                |     -0.0103 |  -0.0317 |    0.0111 |      1.0000 |            1.0000 |
| comp           | xgboost      | raw                | imf_linear_a1.41   |     -0.0223 |  -0.0502 |    0.0057 |      1.0000 |            0.7013 |
| comp           | xgboost      | raw                | irmf_h2_a1.41      |     -0.0310 |  -0.0649 |    0.0028 |      1.0000 |            0.3365 |
| comp           | xgboost      | raw                | irmf_h2_a1.41_conv |     -0.0293 |  -0.0613 |    0.0026 |      1.0000 |            0.3717 |
| comp           | xgboost      | raw                | irmf_h2_a2_conv    |     -0.0258 |  -0.0628 |    0.0113 |      1.0000 |            1.0000 |
| comp           | xgboost      | raw                | imf_linear_a2      |     -0.0235 |  -0.0504 |    0.0033 |      1.0000 |            0.5081 |
| comp           | xgboost      | raw                | irmf_h1_a1.41_conv |     -0.0290 |  -0.0636 |    0.0057 |      1.0000 |            0.6391 |
| comp           | xgboost      | raw                | irmf_h4_a1.41_conv |     -0.0256 |  -0.0637 |    0.0126 |      1.0000 |            1.0000 |
| comp           | xgboost      | emd                | imf_linear_a1.41   |     -0.0120 |  -0.0326 |    0.0087 |      1.0000 |            1.0000 |
| comp           | xgboost      | emd                | irmf_h2_a1.41      |     -0.0207 |  -0.0454 |    0.0039 |      1.0000 |            0.5447 |
| comp           | xgboost      | emd                | irmf_h2_a1.41_conv |     -0.0191 |  -0.0432 |    0.0051 |      1.0000 |            0.6391 |
| comp           | xgboost      | emd                | irmf_h2_a2_conv    |     -0.0155 |  -0.0457 |    0.0148 |      1.0000 |            1.0000 |
| comp           | xgboost      | emd                | imf_linear_a2      |     -0.0133 |  -0.0334 |    0.0069 |      1.0000 |            1.0000 |
| comp           | xgboost      | emd                | irmf_h1_a1.41_conv |     -0.0187 |  -0.0485 |    0.0111 |      1.0000 |            1.0000 |
| comp           | xgboost      | emd                | irmf_h4_a1.41_conv |     -0.0153 |  -0.0464 |    0.0158 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0088 |  -0.0218 |    0.0043 |      1.0000 |            0.7910 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0071 |  -0.0187 |    0.0046 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0035 |  -0.0222 |    0.0151 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a1.41   | imf_linear_a2      |     -0.0013 |  -0.0127 |    0.0102 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0067 |  -0.0242 |    0.0108 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0033 |  -0.0203 |    0.0137 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0017 |  -0.0047 |    0.0080 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0052 |  -0.0102 |    0.0206 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | imf_linear_a2      |      0.0075 |  -0.0071 |    0.0220 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h1_a1.41_conv |      0.0020 |  -0.0156 |    0.0197 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0055 |  -0.0088 |    0.0197 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0036 |  -0.0089 |    0.0161 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0058 |  -0.0082 |    0.0198 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |      0.0004 |  -0.0179 |    0.0187 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0038 |  -0.0100 |    0.0175 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a2_conv    | imf_linear_a2      |      0.0022 |  -0.0178 |    0.0223 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0032 |  -0.0237 |    0.0173 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0002 |  -0.0163 |    0.0167 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0054 |  -0.0247 |    0.0139 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0020 |  -0.0185 |    0.0145 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0034 |  -0.0136 |    0.0204 |      1.0000 |            1.0000 |
| comp           | extratrees   | raw                | emd                |     -0.0175 |  -0.0516 |    0.0165 |      1.0000 |            1.0000 |
| comp           | extratrees   | raw                | imf_linear_a1.41   |     -0.0343 |  -0.0868 |    0.0182 |      1.0000 |            1.0000 |
| comp           | extratrees   | raw                | irmf_h2_a1.41      |     -0.0457 |  -0.0991 |    0.0077 |      1.0000 |            0.2527 |
| comp           | extratrees   | raw                | irmf_h2_a1.41_conv |     -0.0452 |  -0.1013 |    0.0110 |      1.0000 |            0.6504 |
| comp           | extratrees   | raw                | irmf_h2_a2_conv    |     -0.0413 |  -0.0946 |    0.0120 |      1.0000 |            0.7013 |
| comp           | extratrees   | raw                | imf_linear_a2      |     -0.0337 |  -0.0872 |    0.0197 |      1.0000 |            1.0000 |
| comp           | extratrees   | raw                | irmf_h1_a1.41_conv |     -0.0493 |  -0.1022 |    0.0036 |      1.0000 |            0.1560 |
| comp           | extratrees   | raw                | irmf_h4_a1.41_conv |     -0.0444 |  -0.1013 |    0.0125 |      1.0000 |            0.4443 |
| comp           | extratrees   | emd                | imf_linear_a1.41   |     -0.0167 |  -0.0456 |    0.0121 |      1.0000 |            1.0000 |
| comp           | extratrees   | emd                | irmf_h2_a1.41      |     -0.0282 |  -0.0582 |    0.0018 |      1.0000 |            0.1055 |
| comp           | extratrees   | emd                | irmf_h2_a1.41_conv |     -0.0276 |  -0.0597 |    0.0044 |      1.0000 |            0.2527 |
| comp           | extratrees   | emd                | irmf_h2_a2_conv    |     -0.0237 |  -0.0525 |    0.0050 |      1.0000 |            0.5081 |
| comp           | extratrees   | emd                | imf_linear_a2      |     -0.0162 |  -0.0460 |    0.0135 |      1.0000 |            1.0000 |
| comp           | extratrees   | emd                | irmf_h1_a1.41_conv |     -0.0317 |  -0.0611 |   -0.0024 |      1.0000 |            0.2053 |
| comp           | extratrees   | emd                | irmf_h4_a1.41_conv |     -0.0269 |  -0.0592 |    0.0055 |      1.0000 |            0.4655 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0114 |  -0.0259 |    0.0031 |      1.0000 |            0.2527 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0109 |  -0.0257 |    0.0039 |      1.0000 |            0.3828 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0070 |  -0.0202 |    0.0062 |      1.0000 |            1.0000 |
| comp           | extratrees   | imf_linear_a1.41   | imf_linear_a2      |      0.0005 |  -0.0040 |    0.0050 |      1.0000 |            1.0000 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0150 |  -0.0328 |    0.0028 |      1.0000 |            0.3828 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0101 |  -0.0223 |    0.0020 |      1.0000 |            0.3828 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0005 |  -0.0038 |    0.0048 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0044 |  -0.0070 |    0.0158 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | imf_linear_a2      |      0.0119 |  -0.0036 |    0.0275 |      1.0000 |            0.2527 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0036 |  -0.0123 |    0.0052 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0013 |  -0.0063 |    0.0089 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0039 |  -0.0075 |    0.0153 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0114 |  -0.0042 |    0.0270 |      1.0000 |            0.3828 |
| comp           | extratrees   | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0041 |  -0.0124 |    0.0042 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0008 |  -0.0055 |    0.0071 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a2_conv    | imf_linear_a2      |      0.0075 |  -0.0050 |    0.0201 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0080 |  -0.0224 |    0.0064 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |     -0.0031 |  -0.0164 |    0.0102 |      1.0000 |            1.0000 |
| comp           | extratrees   | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0155 |  -0.0342 |    0.0032 |      1.0000 |            0.3828 |
| comp           | extratrees   | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0107 |  -0.0241 |    0.0028 |      1.0000 |            0.3828 |
| comp           | extratrees   | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0049 |  -0.0058 |    0.0155 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | emd                |     -0.0013 |  -0.0243 |    0.0218 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | imf_linear_a1.41   |     -0.0091 |  -0.0381 |    0.0199 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a1.41      |     -0.0188 |  -0.0499 |    0.0123 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a1.41_conv |     -0.0196 |  -0.0512 |    0.0120 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a2_conv    |     -0.0162 |  -0.0477 |    0.0153 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | imf_linear_a2      |     -0.0117 |  -0.0358 |    0.0124 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | irmf_h1_a1.41_conv |     -0.0200 |  -0.0533 |    0.0133 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | irmf_h4_a1.41_conv |     -0.0156 |  -0.0489 |    0.0176 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | emd                | imf_linear_a1.41   |     -0.0078 |  -0.0306 |    0.0149 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a1.41      |     -0.0175 |  -0.0396 |    0.0045 |      1.0000 |            0.3951 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a1.41_conv |     -0.0183 |  -0.0403 |    0.0036 |      1.0000 |            0.3252 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a2_conv    |     -0.0149 |  -0.0380 |    0.0081 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | emd                | imf_linear_a2      |     -0.0104 |  -0.0305 |    0.0097 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | emd                | irmf_h1_a1.41_conv |     -0.0187 |  -0.0424 |    0.0050 |      1.0000 |            0.2475 |
| comp+log_scale | lightgbm     | emd                | irmf_h4_a1.41_conv |     -0.0143 |  -0.0412 |    0.0125 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0097 |  -0.0214 |    0.0020 |      1.0000 |            0.3828 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0105 |  -0.0225 |    0.0015 |      1.0000 |            0.1055 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0071 |  -0.0182 |    0.0041 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | imf_linear_a2      |     -0.0026 |  -0.0149 |    0.0098 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0108 |  -0.0259 |    0.0042 |      1.0000 |            0.8928 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0065 |  -0.0174 |    0.0044 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0008 |  -0.0062 |    0.0046 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0026 |  -0.0037 |    0.0089 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | imf_linear_a2      |      0.0071 |  -0.0081 |    0.0224 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0011 |  -0.0124 |    0.0101 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0032 |  -0.0057 |    0.0121 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0034 |  -0.0024 |    0.0092 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0079 |  -0.0077 |    0.0235 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0003 |  -0.0117 |    0.0110 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0040 |  -0.0052 |    0.0132 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a2_conv    | imf_linear_a2      |      0.0045 |  -0.0124 |    0.0215 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0038 |  -0.0154 |    0.0079 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0006 |  -0.0096 |    0.0108 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0083 |  -0.0278 |    0.0112 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0039 |  -0.0184 |    0.0106 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0043 |  -0.0100 |    0.0187 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | emd                |     -0.0006 |  -0.0221 |    0.0209 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | imf_linear_a1.41   |     -0.0056 |  -0.0373 |    0.0261 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a1.41      |     -0.0146 |  -0.0428 |    0.0137 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a1.41_conv |     -0.0123 |  -0.0408 |    0.0162 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a2_conv    |     -0.0100 |  -0.0429 |    0.0229 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | imf_linear_a2      |     -0.0082 |  -0.0323 |    0.0159 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | irmf_h1_a1.41_conv |     -0.0132 |  -0.0505 |    0.0241 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | irmf_h4_a1.41_conv |     -0.0098 |  -0.0405 |    0.0208 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | imf_linear_a1.41   |     -0.0050 |  -0.0250 |    0.0150 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a1.41      |     -0.0140 |  -0.0331 |    0.0051 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a1.41_conv |     -0.0117 |  -0.0309 |    0.0075 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a2_conv    |     -0.0094 |  -0.0316 |    0.0129 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | imf_linear_a2      |     -0.0076 |  -0.0245 |    0.0093 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | irmf_h1_a1.41_conv |     -0.0126 |  -0.0348 |    0.0096 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | emd                | irmf_h4_a1.41_conv |     -0.0092 |  -0.0307 |    0.0123 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0090 |  -0.0220 |    0.0041 |      1.0000 |            0.6523 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0067 |  -0.0184 |    0.0049 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0044 |  -0.0213 |    0.0125 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | imf_linear_a2      |     -0.0026 |  -0.0139 |    0.0087 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0076 |  -0.0242 |    0.0090 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0042 |  -0.0203 |    0.0118 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0023 |  -0.0018 |    0.0064 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0046 |  -0.0095 |    0.0187 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | imf_linear_a2      |      0.0064 |  -0.0078 |    0.0206 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h1_a1.41_conv |      0.0014 |  -0.0175 |    0.0204 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0048 |  -0.0091 |    0.0186 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0024 |  -0.0102 |    0.0149 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0041 |  -0.0100 |    0.0182 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0008 |  -0.0193 |    0.0176 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0025 |  -0.0113 |    0.0163 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a2_conv    | imf_linear_a2      |      0.0018 |  -0.0175 |    0.0211 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0032 |  -0.0229 |    0.0165 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0001 |  -0.0167 |    0.0169 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0050 |  -0.0247 |    0.0148 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0016 |  -0.0190 |    0.0157 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0033 |  -0.0139 |    0.0206 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | emd                |     -0.0034 |  -0.0285 |    0.0217 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | imf_linear_a1.41   |     -0.0179 |  -0.0610 |    0.0253 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a1.41      |     -0.0268 |  -0.0744 |    0.0208 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a1.41_conv |     -0.0280 |  -0.0757 |    0.0197 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a2_conv    |     -0.0237 |  -0.0670 |    0.0195 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | imf_linear_a2      |     -0.0149 |  -0.0548 |    0.0249 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | irmf_h1_a1.41_conv |     -0.0282 |  -0.0756 |    0.0193 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | irmf_h4_a1.41_conv |     -0.0252 |  -0.0695 |    0.0191 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | emd                | imf_linear_a1.41   |     -0.0145 |  -0.0419 |    0.0129 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a1.41      |     -0.0234 |  -0.0525 |    0.0057 |      1.0000 |            0.4443 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a1.41_conv |     -0.0245 |  -0.0538 |    0.0047 |      1.0000 |            0.3418 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a2_conv    |     -0.0203 |  -0.0460 |    0.0054 |      1.0000 |            0.3828 |
| comp+log_scale | extratrees   | emd                | imf_linear_a2      |     -0.0115 |  -0.0368 |    0.0137 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | emd                | irmf_h1_a1.41_conv |     -0.0248 |  -0.0531 |    0.0036 |      1.0000 |            0.2668 |
| comp+log_scale | extratrees   | emd                | irmf_h4_a1.41_conv |     -0.0218 |  -0.0489 |    0.0053 |      1.0000 |            0.4443 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0089 |  -0.0207 |    0.0028 |      1.0000 |            0.9258 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0101 |  -0.0244 |    0.0043 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0058 |  -0.0197 |    0.0081 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | imf_linear_a2      |      0.0030 |  -0.0047 |    0.0106 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0103 |  -0.0298 |    0.0092 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0073 |  -0.0204 |    0.0058 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0011 |  -0.0067 |    0.0044 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0031 |  -0.0054 |    0.0116 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | imf_linear_a2      |      0.0119 |  -0.0030 |    0.0267 |      1.0000 |            0.3828 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0014 |  -0.0130 |    0.0103 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0016 |  -0.0076 |    0.0109 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0042 |  -0.0027 |    0.0112 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0130 |  -0.0040 |    0.0301 |      1.0000 |            0.7910 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0002 |  -0.0091 |    0.0086 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0028 |  -0.0058 |    0.0114 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a2_conv    | imf_linear_a2      |      0.0088 |  -0.0063 |    0.0239 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0044 |  -0.0128 |    0.0039 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |     -0.0015 |  -0.0115 |    0.0085 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0132 |  -0.0345 |    0.0081 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0102 |  -0.0238 |    0.0033 |      1.0000 |            0.6523 |
| comp+log_scale | extratrees   | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0030 |  -0.0100 |    0.0159 |      1.0000 |            1.0000 |
