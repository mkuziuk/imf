# rocket_multi_pilot: birdclef2023 / species

10 fits on shared split IDs [0]; wall time 175.1 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                         |   mean |   std |
|:--------------------------------------------------------|-------:|------:|
| ('emd', 'comp', 'multirocket')                          | 0.4790 |   nan |
| ('emd', 'comp+log_scale', 'multirocket')                | 0.4855 |   nan |
| ('imf_linear_a1.41', 'comp', 'multirocket')             | 0.5061 |   nan |
| ('imf_linear_a1.41', 'comp+log_scale', 'multirocket')   | 0.5108 |   nan |
| ('irmf_h2_a1.41', 'comp', 'multirocket')                | 0.5252 |   nan |
| ('irmf_h2_a1.41', 'comp+log_scale', 'multirocket')      | 0.5251 |   nan |
| ('irmf_h2_a1.41_conv', 'comp', 'multirocket')           | 0.5180 |   nan |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'multirocket') | 0.5227 |   nan |
| ('raw', 'comp', 'multirocket')                          | 0.4788 |   nan |
| ('raw', 'comp+log_scale', 'multirocket')                | 0.4764 |   nan |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.
