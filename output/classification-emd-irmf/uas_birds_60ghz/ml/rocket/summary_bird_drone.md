# uas_phase3_rocket: uas_birds_60ghz / bird_drone

540 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 956.4 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                         |   mean |    std |
|:--------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'minirocket')                           | 0.9680 | 0.0296 |
| ('emd', 'comp', 'multirocket')                          | 0.9621 | 0.0242 |
| ('emd', 'comp+log_scale', 'minirocket')                 | 0.9689 | 0.0274 |
| ('emd', 'comp+log_scale', 'multirocket')                | 0.9727 | 0.0244 |
| ('imf_linear_a1.41', 'comp', 'minirocket')              | 0.9745 | 0.0195 |
| ('imf_linear_a1.41', 'comp', 'multirocket')             | 0.9624 | 0.0216 |
| ('imf_linear_a1.41', 'comp+log_scale', 'minirocket')    | 0.9780 | 0.0186 |
| ('imf_linear_a1.41', 'comp+log_scale', 'multirocket')   | 0.9709 | 0.0223 |
| ('imf_linear_a2', 'comp', 'minirocket')                 | 0.9727 | 0.0172 |
| ('imf_linear_a2', 'comp', 'multirocket')                | 0.9605 | 0.0283 |
| ('imf_linear_a2', 'comp+log_scale', 'minirocket')       | 0.9712 | 0.0190 |
| ('imf_linear_a2', 'comp+log_scale', 'multirocket')      | 0.9695 | 0.0211 |
| ('irmf_h1_a1.41_conv', 'comp', 'minirocket')            | 0.9725 | 0.0237 |
| ('irmf_h1_a1.41_conv', 'comp', 'multirocket')           | 0.9544 | 0.0262 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'minirocket')  | 0.9732 | 0.0211 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'multirocket') | 0.9699 | 0.0182 |
| ('irmf_h2_a1.41', 'comp', 'minirocket')                 | 0.9733 | 0.0200 |
| ('irmf_h2_a1.41', 'comp', 'multirocket')                | 0.9604 | 0.0295 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'minirocket')       | 0.9759 | 0.0193 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'multirocket')      | 0.9673 | 0.0212 |
| ('irmf_h2_a1.41_conv', 'comp', 'minirocket')            | 0.9736 | 0.0211 |
| ('irmf_h2_a1.41_conv', 'comp', 'multirocket')           | 0.9545 | 0.0282 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'minirocket')  | 0.9763 | 0.0195 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'multirocket') | 0.9671 | 0.0187 |
| ('irmf_h2_a2_conv', 'comp', 'minirocket')               | 0.9698 | 0.0224 |
| ('irmf_h2_a2_conv', 'comp', 'multirocket')              | 0.9563 | 0.0305 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'minirocket')     | 0.9708 | 0.0241 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'multirocket')    | 0.9698 | 0.0259 |
| ('irmf_h4_a1.41_conv', 'comp', 'minirocket')            | 0.9758 | 0.0218 |
| ('irmf_h4_a1.41_conv', 'comp', 'multirocket')           | 0.9613 | 0.0244 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'minirocket')  | 0.9756 | 0.0201 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'multirocket') | 0.9726 | 0.0208 |
| ('raw', 'comp', 'minirocket')                           | 0.9620 | 0.0289 |
| ('raw', 'comp', 'multirocket')                          | 0.9379 | 0.0395 |
| ('raw', 'comp+log_scale', 'minirocket')                 | 0.9664 | 0.0238 |
| ('raw', 'comp+log_scale', 'multirocket')                | 0.9566 | 0.0271 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | minirocket   | raw                | emd                |     -0.0060 |  -0.0488 |    0.0368 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | imf_linear_a1.41   |     -0.0125 |  -0.0437 |    0.0187 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a1.41      |     -0.0114 |  -0.0447 |    0.0220 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a1.41_conv |     -0.0116 |  -0.0436 |    0.0204 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h2_a2_conv    |     -0.0078 |  -0.0397 |    0.0240 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | imf_linear_a2      |     -0.0107 |  -0.0383 |    0.0169 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h1_a1.41_conv |     -0.0105 |  -0.0463 |    0.0253 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h4_a1.41_conv |     -0.0138 |  -0.0491 |    0.0215 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | imf_linear_a1.41   |     -0.0065 |  -0.0266 |    0.0136 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a1.41      |     -0.0053 |  -0.0294 |    0.0188 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a1.41_conv |     -0.0056 |  -0.0291 |    0.0179 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h2_a2_conv    |     -0.0018 |  -0.0249 |    0.0212 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | imf_linear_a2      |     -0.0047 |  -0.0291 |    0.0197 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h1_a1.41_conv |     -0.0045 |  -0.0328 |    0.0238 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | irmf_h4_a1.41_conv |     -0.0078 |  -0.0273 |    0.0117 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0011 |  -0.0173 |    0.0196 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0009 |  -0.0154 |    0.0172 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0047 |  -0.0102 |    0.0196 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | imf_linear_a2      |      0.0018 |  -0.0129 |    0.0165 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h1_a1.41_conv |      0.0020 |  -0.0210 |    0.0250 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0013 |  -0.0170 |    0.0144 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0002 |  -0.0111 |    0.0106 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0035 |  -0.0111 |    0.0182 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | imf_linear_a2      |      0.0007 |  -0.0158 |    0.0171 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h1_a1.41_conv |      0.0009 |  -0.0124 |    0.0141 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h4_a1.41_conv |     -0.0025 |  -0.0144 |    0.0095 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0038 |  -0.0084 |    0.0160 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0009 |  -0.0187 |    0.0205 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |      0.0011 |  -0.0124 |    0.0147 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |     -0.0022 |  -0.0152 |    0.0107 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0029 |  -0.0211 |    0.0154 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0027 |  -0.0208 |    0.0155 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |     -0.0060 |  -0.0162 |    0.0042 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a2      | irmf_h1_a1.41_conv |      0.0002 |  -0.0223 |    0.0227 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0031 |  -0.0186 |    0.0123 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |     -0.0033 |  -0.0179 |    0.0113 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | emd                |     -0.0024 |  -0.0351 |    0.0302 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | imf_linear_a1.41   |     -0.0115 |  -0.0374 |    0.0144 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a1.41      |     -0.0095 |  -0.0376 |    0.0187 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a1.41_conv |     -0.0099 |  -0.0409 |    0.0211 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a2_conv    |     -0.0044 |  -0.0319 |    0.0230 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | imf_linear_a2      |     -0.0048 |  -0.0203 |    0.0107 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h1_a1.41_conv |     -0.0067 |  -0.0345 |    0.0211 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h4_a1.41_conv |     -0.0091 |  -0.0388 |    0.0205 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | imf_linear_a1.41   |     -0.0091 |  -0.0264 |    0.0082 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a1.41      |     -0.0070 |  -0.0286 |    0.0145 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a1.41_conv |     -0.0075 |  -0.0354 |    0.0205 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a2_conv    |     -0.0020 |  -0.0280 |    0.0240 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | imf_linear_a2      |     -0.0024 |  -0.0329 |    0.0281 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h1_a1.41_conv |     -0.0043 |  -0.0266 |    0.0180 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h4_a1.41_conv |     -0.0067 |  -0.0287 |    0.0153 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0020 |  -0.0103 |    0.0144 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0016 |  -0.0147 |    0.0180 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0071 |  -0.0099 |    0.0241 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | imf_linear_a2      |      0.0067 |  -0.0162 |    0.0296 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h1_a1.41_conv |      0.0048 |  -0.0061 |    0.0157 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h4_a1.41_conv |      0.0024 |  -0.0102 |    0.0150 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0004 |  -0.0108 |    0.0100 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0051 |  -0.0080 |    0.0181 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | imf_linear_a2      |      0.0047 |  -0.0217 |    0.0310 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h1_a1.41_conv |      0.0027 |  -0.0061 |    0.0116 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0003 |  -0.0076 |    0.0083 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0055 |  -0.0084 |    0.0194 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0051 |  -0.0235 |    0.0337 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |      0.0032 |  -0.0083 |    0.0146 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0008 |  -0.0111 |    0.0127 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0004 |  -0.0288 |    0.0280 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0023 |  -0.0171 |    0.0124 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |     -0.0047 |  -0.0187 |    0.0092 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0019 |  -0.0294 |    0.0256 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0043 |  -0.0323 |    0.0237 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |     -0.0024 |  -0.0115 |    0.0067 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | emd                |     -0.0242 |  -0.0643 |    0.0159 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | imf_linear_a1.41   |     -0.0245 |  -0.0674 |    0.0183 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | irmf_h2_a1.41      |     -0.0225 |  -0.0503 |    0.0053 |      1.0000 |            0.4074 |
| comp           | multirocket  | raw                | irmf_h2_a1.41_conv |     -0.0166 |  -0.0461 |    0.0128 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | irmf_h2_a2_conv    |     -0.0184 |  -0.0533 |    0.0164 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | imf_linear_a2      |     -0.0226 |  -0.0656 |    0.0204 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | irmf_h1_a1.41_conv |     -0.0166 |  -0.0424 |    0.0093 |      1.0000 |            0.8574 |
| comp           | multirocket  | raw                | irmf_h4_a1.41_conv |     -0.0234 |  -0.0546 |    0.0078 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | imf_linear_a1.41   |     -0.0003 |  -0.0285 |    0.0278 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | irmf_h2_a1.41      |      0.0017 |  -0.0261 |    0.0295 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | irmf_h2_a1.41_conv |      0.0076 |  -0.0145 |    0.0296 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | irmf_h2_a2_conv    |      0.0058 |  -0.0177 |    0.0292 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | imf_linear_a2      |      0.0016 |  -0.0257 |    0.0290 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | irmf_h1_a1.41_conv |      0.0076 |  -0.0228 |    0.0380 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | irmf_h4_a1.41_conv |      0.0008 |  -0.0219 |    0.0235 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0020 |  -0.0233 |    0.0274 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0079 |  -0.0185 |    0.0344 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0061 |  -0.0174 |    0.0296 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41   | imf_linear_a2      |      0.0020 |  -0.0120 |    0.0160 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41   | irmf_h1_a1.41_conv |      0.0080 |  -0.0216 |    0.0376 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41   | irmf_h4_a1.41_conv |      0.0011 |  -0.0233 |    0.0255 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0059 |  -0.0056 |    0.0174 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0041 |  -0.0108 |    0.0190 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41      | imf_linear_a2      |     -0.0001 |  -0.0276 |    0.0275 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41      | irmf_h1_a1.41_conv |      0.0059 |  -0.0200 |    0.0319 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41      | irmf_h4_a1.41_conv |     -0.0009 |  -0.0269 |    0.0251 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0018 |  -0.0126 |    0.0090 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0059 |  -0.0315 |    0.0196 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |      0.0001 |  -0.0238 |    0.0239 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |     -0.0068 |  -0.0294 |    0.0159 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0041 |  -0.0244 |    0.0161 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |      0.0019 |  -0.0278 |    0.0316 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |     -0.0050 |  -0.0290 |    0.0191 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a2      | irmf_h1_a1.41_conv |      0.0060 |  -0.0256 |    0.0376 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0008 |  -0.0233 |    0.0217 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |     -0.0068 |  -0.0248 |    0.0111 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | emd                |     -0.0161 |  -0.0440 |    0.0118 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | imf_linear_a1.41   |     -0.0142 |  -0.0397 |    0.0112 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | irmf_h2_a1.41      |     -0.0107 |  -0.0314 |    0.0100 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | irmf_h2_a1.41_conv |     -0.0105 |  -0.0312 |    0.0103 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | irmf_h2_a2_conv    |     -0.0131 |  -0.0400 |    0.0137 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | imf_linear_a2      |     -0.0129 |  -0.0419 |    0.0161 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | irmf_h1_a1.41_conv |     -0.0132 |  -0.0410 |    0.0146 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | irmf_h4_a1.41_conv |     -0.0159 |  -0.0390 |    0.0071 |      1.0000 |            0.3516 |
| comp+log_scale | multirocket  | emd                | imf_linear_a1.41   |      0.0018 |  -0.0169 |    0.0205 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | irmf_h2_a1.41      |      0.0054 |  -0.0148 |    0.0256 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | irmf_h2_a1.41_conv |      0.0056 |  -0.0113 |    0.0226 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | irmf_h2_a2_conv    |      0.0030 |  -0.0196 |    0.0255 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | imf_linear_a2      |      0.0032 |  -0.0204 |    0.0267 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | irmf_h1_a1.41_conv |      0.0028 |  -0.0123 |    0.0180 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | irmf_h4_a1.41_conv |      0.0001 |  -0.0142 |    0.0144 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0035 |  -0.0178 |    0.0249 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0038 |  -0.0111 |    0.0186 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0011 |  -0.0161 |    0.0183 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41   | imf_linear_a2      |      0.0013 |  -0.0116 |    0.0143 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41   | irmf_h1_a1.41_conv |      0.0010 |  -0.0165 |    0.0185 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0017 |  -0.0141 |    0.0107 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0002 |  -0.0114 |    0.0119 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0024 |  -0.0236 |    0.0187 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41      | imf_linear_a2      |     -0.0022 |  -0.0298 |    0.0254 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0025 |  -0.0245 |    0.0194 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41      | irmf_h4_a1.41_conv |     -0.0052 |  -0.0224 |    0.0119 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0027 |  -0.0197 |    0.0144 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0025 |  -0.0221 |    0.0172 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0028 |  -0.0207 |    0.0152 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |     -0.0055 |  -0.0198 |    0.0088 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a2_conv    | imf_linear_a2      |      0.0002 |  -0.0242 |    0.0247 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0001 |  -0.0251 |    0.0249 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |     -0.0028 |  -0.0244 |    0.0187 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0003 |  -0.0169 |    0.0163 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0030 |  -0.0210 |    0.0149 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |     -0.0027 |  -0.0174 |    0.0119 |      1.0000 |            1.0000 |
