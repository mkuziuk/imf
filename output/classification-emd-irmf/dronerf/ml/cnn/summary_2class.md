# dronerf_phase3_cnn: dronerf / 2class

70 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 642.7 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                            |   mean |    std |
|:-----------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'compact_resnet')                          | 0.7771 | 0.0778 |
| ('emd', 'comp+log_scale', 'compact_resnet')                | 0.8544 | 0.0425 |
| ('imf_linear_a1.41', 'comp', 'compact_resnet')             | 0.8699 | 0.0526 |
| ('imf_linear_a1.41', 'comp+log_scale', 'compact_resnet')   | 0.8745 | 0.0516 |
| ('imf_linear_a2', 'comp', 'compact_resnet')                | 0.8843 | 0.0261 |
| ('imf_linear_a2', 'comp+log_scale', 'compact_resnet')      | 0.8974 | 0.0320 |
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')                | 0.8897 | 0.0058 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet')      | 0.8669 | 0.0356 |
| ('irmf_h2_a1.41_conv', 'comp', 'compact_resnet')           | 0.8633 | 0.0442 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.8600 | 0.0335 |
| ('irmf_h2_a2_conv', 'comp', 'compact_resnet')              | 0.8714 | 0.0288 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'compact_resnet')    | 0.8750 | 0.0231 |
| ('raw', 'comp', 'compact_resnet')                          | 0.8685 | 0.0127 |
| ('raw', 'comp+log_scale', 'compact_resnet')                | 0.8681 | 0.0323 |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier     | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:---------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | compact_resnet | raw                | emd                |      0.0913 |  -0.0671 |    0.2498 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | imf_linear_a1.41   |     -0.0014 |  -0.1168 |    0.1140 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a1.41      |     -0.0213 |  -0.0469 |    0.0043 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a1.41_conv |      0.0052 |  -0.0942 |    0.1046 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a2_conv    |     -0.0029 |  -0.0368 |    0.0310 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | imf_linear_a2      |     -0.0158 |  -0.0709 |    0.0392 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | imf_linear_a1.41   |     -0.0928 |  -0.1495 |   -0.0360 |      0.4420 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a1.41      |     -0.1126 |  -0.2548 |    0.0296 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a1.41_conv |     -0.0861 |  -0.1846 |    0.0123 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a2_conv    |     -0.0942 |  -0.2560 |    0.0675 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | imf_linear_a2      |     -0.1072 |  -0.2774 |    0.0630 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0198 |  -0.1172 |    0.0775 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0066 |  -0.0625 |    0.0758 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0015 |  -0.1274 |    0.1244 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | imf_linear_a2      |     -0.0144 |  -0.1327 |    0.1039 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0265 |  -0.0488 |    0.1018 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0184 |  -0.0355 |    0.0722 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | imf_linear_a2      |      0.0054 |  -0.0537 |    0.0646 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0081 |  -0.1257 |    0.1095 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0210 |  -0.1421 |    0.1000 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0129 |  -0.0913 |    0.0654 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | emd                |      0.0137 |  -0.0944 |    0.1219 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | imf_linear_a1.41   |     -0.0064 |  -0.1424 |    0.1296 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a1.41      |      0.0012 |  -0.1130 |    0.1155 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a1.41_conv |      0.0081 |  -0.0988 |    0.1150 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a2_conv    |     -0.0069 |  -0.0783 |    0.0646 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | imf_linear_a2      |     -0.0293 |  -0.0786 |    0.0201 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | imf_linear_a1.41   |     -0.0201 |  -0.0947 |    0.0544 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a1.41      |     -0.0125 |  -0.1422 |    0.1172 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a1.41_conv |     -0.0056 |  -0.0557 |    0.0444 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a2_conv    |     -0.0206 |  -0.1414 |    0.1002 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | imf_linear_a2      |     -0.0430 |  -0.1457 |    0.0597 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0076 |  -0.1048 |    0.1200 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0145 |  -0.0324 |    0.0614 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0005 |  -0.1261 |    0.1251 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | imf_linear_a2      |     -0.0229 |  -0.1282 |    0.0824 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0069 |  -0.0877 |    0.1014 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0081 |  -0.0572 |    0.0409 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | imf_linear_a2      |     -0.0305 |  -0.1318 |    0.0708 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0150 |  -0.1144 |    0.0844 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0374 |  -0.1210 |    0.0462 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0224 |  -0.0987 |    0.0539 |      1.0000 |            1.0000 |
