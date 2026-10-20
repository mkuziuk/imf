# uas_phase3_rocket: uas_birds_60ghz / target3

540 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 996.2 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                         |   mean |    std |
|:--------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'minirocket')                           | 0.9319 | 0.0370 |
| ('emd', 'comp', 'multirocket')                          | 0.8816 | 0.0501 |
| ('emd', 'comp+log_scale', 'minirocket')                 | 0.9354 | 0.0399 |
| ('emd', 'comp+log_scale', 'multirocket')                | 0.8885 | 0.0502 |
| ('imf_linear_a1.41', 'comp', 'minirocket')              | 0.9597 | 0.0277 |
| ('imf_linear_a1.41', 'comp', 'multirocket')             | 0.9191 | 0.0368 |
| ('imf_linear_a1.41', 'comp+log_scale', 'minirocket')    | 0.9649 | 0.0256 |
| ('imf_linear_a1.41', 'comp+log_scale', 'multirocket')   | 0.9264 | 0.0326 |
| ('imf_linear_a2', 'comp', 'minirocket')                 | 0.9552 | 0.0253 |
| ('imf_linear_a2', 'comp', 'multirocket')                | 0.9132 | 0.0335 |
| ('imf_linear_a2', 'comp+log_scale', 'minirocket')       | 0.9524 | 0.0316 |
| ('imf_linear_a2', 'comp+log_scale', 'multirocket')      | 0.9219 | 0.0340 |
| ('irmf_h1_a1.41_conv', 'comp', 'minirocket')            | 0.9608 | 0.0298 |
| ('irmf_h1_a1.41_conv', 'comp', 'multirocket')           | 0.9125 | 0.0348 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'minirocket')  | 0.9653 | 0.0310 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'multirocket') | 0.9201 | 0.0377 |
| ('irmf_h2_a1.41', 'comp', 'minirocket')                 | 0.9628 | 0.0234 |
| ('irmf_h2_a1.41', 'comp', 'multirocket')                | 0.9101 | 0.0455 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'minirocket')       | 0.9646 | 0.0248 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'multirocket')      | 0.9104 | 0.0350 |
| ('irmf_h2_a1.41_conv', 'comp', 'minirocket')            | 0.9622 | 0.0331 |
| ('irmf_h2_a1.41_conv', 'comp', 'multirocket')           | 0.9056 | 0.0408 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'minirocket')  | 0.9649 | 0.0234 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'multirocket') | 0.9177 | 0.0359 |
| ('irmf_h2_a2_conv', 'comp', 'minirocket')               | 0.9503 | 0.0300 |
| ('irmf_h2_a2_conv', 'comp', 'multirocket')              | 0.9052 | 0.0353 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'minirocket')     | 0.9625 | 0.0310 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'multirocket')    | 0.9149 | 0.0349 |
| ('irmf_h4_a1.41_conv', 'comp', 'minirocket')            | 0.9670 | 0.0241 |
| ('irmf_h4_a1.41_conv', 'comp', 'multirocket')           | 0.9135 | 0.0400 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'minirocket')  | 0.9635 | 0.0309 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'multirocket') | 0.9149 | 0.0336 |
| ('raw', 'comp', 'minirocket')                           | 0.9444 | 0.0393 |
| ('raw', 'comp', 'multirocket')                          | 0.8917 | 0.0439 |
| ('raw', 'comp+log_scale', 'minirocket')                 | 0.9462 | 0.0394 |
| ('raw', 'comp+log_scale', 'multirocket')                | 0.9031 | 0.0411 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | minirocket   | raw                | emd                |      0.0125 |  -0.0347 |    0.0597 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | imf_linear_a1.41   |     -0.0153 |  -0.0386 |    0.0081 |      1.0000 |            0.8188 |
| comp           | minirocket   | raw                | irmf_h2_a1.41      |     -0.0184 |  -0.0424 |    0.0056 |      1.0000 |            0.2754 |
| comp           | minirocket   | raw                | irmf_h2_a1.41_conv |     -0.0177 |  -0.0424 |    0.0070 |      1.0000 |            0.2754 |
| comp           | minirocket   | raw                | irmf_h2_a2_conv    |     -0.0059 |  -0.0277 |    0.0159 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | imf_linear_a2      |     -0.0108 |  -0.0336 |    0.0121 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h1_a1.41_conv |     -0.0163 |  -0.0462 |    0.0135 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw                | irmf_h4_a1.41_conv |     -0.0226 |  -0.0490 |    0.0039 |      1.0000 |            0.2080 |
| comp           | minirocket   | emd                | imf_linear_a1.41   |     -0.0278 |  -0.0624 |    0.0068 |      1.0000 |            0.4186 |
| comp           | minirocket   | emd                | irmf_h2_a1.41      |     -0.0309 |  -0.0700 |    0.0082 |      1.0000 |            0.2754 |
| comp           | minirocket   | emd                | irmf_h2_a1.41_conv |     -0.0302 |  -0.0696 |    0.0092 |      1.0000 |            0.5813 |
| comp           | minirocket   | emd                | irmf_h2_a2_conv    |     -0.0184 |  -0.0604 |    0.0236 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd                | imf_linear_a2      |     -0.0233 |  -0.0574 |    0.0108 |      1.0000 |            0.8750 |
| comp           | minirocket   | emd                | irmf_h1_a1.41_conv |     -0.0288 |  -0.0614 |    0.0038 |      1.0000 |            0.3217 |
| comp           | minirocket   | emd                | irmf_h4_a1.41_conv |     -0.0351 |  -0.0717 |    0.0016 |      1.0000 |            0.1047 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0031 |  -0.0202 |    0.0140 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0024 |  -0.0229 |    0.0180 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0094 |  -0.0076 |    0.0264 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | imf_linear_a2      |      0.0045 |  -0.0107 |    0.0197 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0010 |  -0.0149 |    0.0129 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0073 |  -0.0219 |    0.0073 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0007 |  -0.0171 |    0.0185 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0125 |  -0.0056 |    0.0306 |      1.0000 |            0.6445 |
| comp           | minirocket   | irmf_h2_a1.41      | imf_linear_a2      |      0.0076 |  -0.0060 |    0.0213 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h1_a1.41_conv |      0.0021 |  -0.0187 |    0.0229 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41      | irmf_h4_a1.41_conv |     -0.0042 |  -0.0177 |    0.0093 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0118 |  -0.0098 |    0.0334 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0069 |  -0.0122 |    0.0261 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |      0.0014 |  -0.0177 |    0.0205 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |     -0.0049 |  -0.0242 |    0.0145 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0049 |  -0.0233 |    0.0136 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0104 |  -0.0299 |    0.0090 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |     -0.0167 |  -0.0326 |   -0.0007 |      1.0000 |            0.0703 |
| comp           | minirocket   | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0056 |  -0.0256 |    0.0145 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0118 |  -0.0293 |    0.0057 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |     -0.0063 |  -0.0215 |    0.0090 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | emd                |      0.0108 |  -0.0389 |    0.0604 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | imf_linear_a1.41   |     -0.0187 |  -0.0525 |    0.0150 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a1.41      |     -0.0184 |  -0.0560 |    0.0192 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a1.41_conv |     -0.0187 |  -0.0555 |    0.0180 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h2_a2_conv    |     -0.0163 |  -0.0474 |    0.0147 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | imf_linear_a2      |     -0.0063 |  -0.0396 |    0.0271 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h1_a1.41_conv |     -0.0191 |  -0.0559 |    0.0177 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw                | irmf_h4_a1.41_conv |     -0.0174 |  -0.0525 |    0.0178 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | imf_linear_a1.41   |     -0.0295 |  -0.0700 |    0.0109 |      1.0000 |            0.8805 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a1.41      |     -0.0292 |  -0.0711 |    0.0128 |      1.0000 |            0.9558 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a1.41_conv |     -0.0295 |  -0.0773 |    0.0182 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h2_a2_conv    |     -0.0271 |  -0.0719 |    0.0178 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | imf_linear_a2      |     -0.0170 |  -0.0621 |    0.0280 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h1_a1.41_conv |     -0.0299 |  -0.0779 |    0.0182 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd                | irmf_h4_a1.41_conv |     -0.0281 |  -0.0748 |    0.0186 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0003 |  -0.0180 |    0.0187 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0000 |  -0.0175 |    0.0175 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0024 |  -0.0167 |    0.0216 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | imf_linear_a2      |      0.0125 |  -0.0046 |    0.0296 |      1.0000 |            0.6445 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0003 |  -0.0167 |    0.0160 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41   | irmf_h4_a1.41_conv |      0.0014 |  -0.0199 |    0.0227 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0003 |  -0.0135 |    0.0128 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0021 |  -0.0169 |    0.0211 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | imf_linear_a2      |      0.0122 |  -0.0089 |    0.0332 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0007 |  -0.0198 |    0.0184 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41      | irmf_h4_a1.41_conv |      0.0010 |  -0.0198 |    0.0219 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0024 |  -0.0155 |    0.0204 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0125 |  -0.0082 |    0.0332 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0003 |  -0.0164 |    0.0157 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0014 |  -0.0178 |    0.0206 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a2_conv    | imf_linear_a2      |      0.0101 |  -0.0068 |    0.0270 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0028 |  -0.0170 |    0.0115 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |     -0.0010 |  -0.0117 |    0.0096 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a2      | irmf_h1_a1.41_conv |     -0.0128 |  -0.0269 |    0.0012 |      1.0000 |            0.2754 |
| comp+log_scale | minirocket   | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0111 |  -0.0317 |    0.0095 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0017 |  -0.0147 |    0.0181 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | emd                |      0.0101 |  -0.0535 |    0.0736 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | imf_linear_a1.41   |     -0.0274 |  -0.0610 |    0.0061 |      1.0000 |            0.8114 |
| comp           | multirocket  | raw                | irmf_h2_a1.41      |     -0.0184 |  -0.0577 |    0.0209 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | irmf_h2_a1.41_conv |     -0.0139 |  -0.0591 |    0.0313 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | irmf_h2_a2_conv    |     -0.0135 |  -0.0496 |    0.0225 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | imf_linear_a2      |     -0.0215 |  -0.0753 |    0.0322 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | irmf_h1_a1.41_conv |     -0.0208 |  -0.0580 |    0.0164 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw                | irmf_h4_a1.41_conv |     -0.0219 |  -0.0549 |    0.0112 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | imf_linear_a1.41   |     -0.0375 |  -0.1031 |    0.0281 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | irmf_h2_a1.41      |     -0.0285 |  -0.0883 |    0.0314 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | irmf_h2_a1.41_conv |     -0.0240 |  -0.0878 |    0.0398 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | irmf_h2_a2_conv    |     -0.0236 |  -0.0699 |    0.0227 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | imf_linear_a2      |     -0.0316 |  -0.0731 |    0.0099 |      1.0000 |            0.5950 |
| comp           | multirocket  | emd                | irmf_h1_a1.41_conv |     -0.0309 |  -0.0867 |    0.0249 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd                | irmf_h4_a1.41_conv |     -0.0319 |  -0.0940 |    0.0302 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0090 |  -0.0263 |    0.0444 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0135 |  -0.0208 |    0.0479 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0139 |  -0.0194 |    0.0472 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41   | imf_linear_a2      |      0.0059 |  -0.0465 |    0.0583 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41   | irmf_h1_a1.41_conv |      0.0066 |  -0.0301 |    0.0433 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41   | irmf_h4_a1.41_conv |      0.0056 |  -0.0193 |    0.0304 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0045 |  -0.0151 |    0.0241 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0049 |  -0.0261 |    0.0358 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41      | imf_linear_a2      |     -0.0031 |  -0.0598 |    0.0535 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0024 |  -0.0347 |    0.0298 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41      | irmf_h4_a1.41_conv |     -0.0035 |  -0.0363 |    0.0293 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0003 |  -0.0277 |    0.0284 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0076 |  -0.0606 |    0.0453 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0069 |  -0.0349 |    0.0210 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |     -0.0080 |  -0.0363 |    0.0203 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0080 |  -0.0390 |    0.0230 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0073 |  -0.0308 |    0.0162 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |     -0.0083 |  -0.0329 |    0.0162 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a2      | irmf_h1_a1.41_conv |      0.0007 |  -0.0401 |    0.0415 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a2      | irmf_h4_a1.41_conv |     -0.0003 |  -0.0461 |    0.0454 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |     -0.0010 |  -0.0263 |    0.0242 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | emd                |      0.0146 |  -0.0483 |    0.0774 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | imf_linear_a1.41   |     -0.0233 |  -0.0589 |    0.0124 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | irmf_h2_a1.41      |     -0.0073 |  -0.0422 |    0.0276 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | irmf_h2_a1.41_conv |     -0.0146 |  -0.0562 |    0.0270 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | irmf_h2_a2_conv    |     -0.0118 |  -0.0480 |    0.0243 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | imf_linear_a2      |     -0.0187 |  -0.0560 |    0.0185 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | irmf_h1_a1.41_conv |     -0.0170 |  -0.0478 |    0.0137 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw                | irmf_h4_a1.41_conv |     -0.0118 |  -0.0370 |    0.0134 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | imf_linear_a1.41   |     -0.0378 |  -0.1083 |    0.0326 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | irmf_h2_a1.41      |     -0.0219 |  -0.0874 |    0.0437 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | irmf_h2_a1.41_conv |     -0.0292 |  -0.0954 |    0.0370 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | irmf_h2_a2_conv    |     -0.0264 |  -0.0924 |    0.0397 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | imf_linear_a2      |     -0.0333 |  -0.0937 |    0.0270 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | irmf_h1_a1.41_conv |     -0.0316 |  -0.0893 |    0.0261 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd                | irmf_h4_a1.41_conv |     -0.0264 |  -0.0877 |    0.0349 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0160 |  -0.0035 |    0.0355 |      1.0000 |            0.3785 |
| comp+log_scale | multirocket  | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0087 |  -0.0218 |    0.0392 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0115 |  -0.0169 |    0.0398 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41   | imf_linear_a2      |      0.0045 |  -0.0383 |    0.0473 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41   | irmf_h1_a1.41_conv |      0.0063 |  -0.0252 |    0.0377 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41   | irmf_h4_a1.41_conv |      0.0115 |  -0.0129 |    0.0358 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0073 |  -0.0308 |    0.0162 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0045 |  -0.0281 |    0.0191 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41      | imf_linear_a2      |     -0.0115 |  -0.0511 |    0.0282 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0097 |  -0.0315 |    0.0120 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41      | irmf_h4_a1.41_conv |     -0.0045 |  -0.0256 |    0.0166 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0028 |  -0.0238 |    0.0293 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0042 |  -0.0496 |    0.0412 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0024 |  -0.0291 |    0.0242 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |      0.0028 |  -0.0291 |    0.0346 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0069 |  -0.0408 |    0.0269 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |     -0.0052 |  -0.0270 |    0.0166 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |     -0.0000 |  -0.0331 |    0.0331 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a2      | irmf_h1_a1.41_conv |      0.0017 |  -0.0358 |    0.0393 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a2      | irmf_h4_a1.41_conv |      0.0069 |  -0.0328 |    0.0467 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0052 |  -0.0219 |    0.0323 |      1.0000 |            1.0000 |
