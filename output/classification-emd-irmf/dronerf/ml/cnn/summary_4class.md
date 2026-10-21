# dronerf_phase3_cnn: dronerf / 4class

70 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 688.3 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                            |   mean |    std |
|:-----------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'compact_resnet')                          | 0.6272 | 0.0279 |
| ('emd', 'comp+log_scale', 'compact_resnet')                | 0.6992 | 0.0178 |
| ('imf_linear_a1.41', 'comp', 'compact_resnet')             | 0.7149 | 0.0208 |
| ('imf_linear_a1.41', 'comp+log_scale', 'compact_resnet')   | 0.7339 | 0.0702 |
| ('imf_linear_a2', 'comp', 'compact_resnet')                | 0.6663 | 0.0386 |
| ('imf_linear_a2', 'comp+log_scale', 'compact_resnet')      | 0.7286 | 0.0460 |
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')                | 0.6828 | 0.0411 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet')      | 0.7184 | 0.0558 |
| ('irmf_h2_a1.41_conv', 'comp', 'compact_resnet')           | 0.6734 | 0.0329 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.7383 | 0.0392 |
| ('irmf_h2_a2_conv', 'comp', 'compact_resnet')              | 0.6434 | 0.0190 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'compact_resnet')    | 0.7374 | 0.0239 |
| ('raw', 'comp', 'compact_resnet')                          | 0.6680 | 0.0220 |
| ('raw', 'comp+log_scale', 'compact_resnet')                | 0.7287 | 0.0340 |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier     | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:---------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | compact_resnet | raw                | emd                |      0.0408 |   0.0057 |    0.0759 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | imf_linear_a1.41   |     -0.0469 |  -0.1147 |    0.0209 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a1.41      |     -0.0148 |  -0.1021 |    0.0724 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a1.41_conv |     -0.0054 |  -0.0863 |    0.0754 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a2_conv    |      0.0246 |  -0.0319 |    0.0810 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | imf_linear_a2      |      0.0017 |  -0.1009 |    0.1043 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | imf_linear_a1.41   |     -0.0877 |  -0.1560 |   -0.0194 |      0.9871 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a1.41      |     -0.0556 |  -0.1249 |    0.0137 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a1.41_conv |     -0.0462 |  -0.1058 |    0.0134 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a2_conv    |     -0.0162 |  -0.0946 |    0.0623 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | imf_linear_a2      |     -0.0390 |  -0.1509 |    0.0728 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0321 |  -0.0453 |    0.1095 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0415 |  -0.0147 |    0.0977 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0715 |   0.0127 |    0.1303 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | imf_linear_a2      |      0.0486 |   0.0009 |    0.0964 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0094 |  -0.0263 |    0.0451 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0394 |  -0.0723 |    0.1511 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | imf_linear_a2      |      0.0165 |  -0.0877 |    0.1208 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0300 |  -0.0647 |    0.1248 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0071 |  -0.0850 |    0.0993 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0229 |  -0.1011 |    0.0554 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | emd                |      0.0295 |  -0.0407 |    0.0996 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | imf_linear_a1.41   |     -0.0052 |  -0.1206 |    0.1102 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a1.41      |      0.0103 |  -0.0834 |    0.1040 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a1.41_conv |     -0.0096 |  -0.0983 |    0.0790 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a2_conv    |     -0.0087 |  -0.0763 |    0.0588 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | imf_linear_a2      |      0.0001 |  -0.0664 |    0.0667 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | imf_linear_a1.41   |     -0.0347 |  -0.1811 |    0.1117 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a1.41      |     -0.0192 |  -0.1485 |    0.1101 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a1.41_conv |     -0.0391 |  -0.0911 |    0.0129 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a2_conv    |     -0.0382 |  -0.0736 |   -0.0028 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | imf_linear_a2      |     -0.0293 |  -0.0958 |    0.0371 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0155 |  -0.0415 |    0.0725 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0044 |  -0.1792 |    0.1704 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0035 |  -0.1676 |    0.1605 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | imf_linear_a2      |      0.0053 |  -0.1444 |    0.1551 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0199 |  -0.1813 |    0.1415 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0190 |  -0.1581 |    0.1201 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | imf_linear_a2      |     -0.0102 |  -0.1568 |    0.1364 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0009 |  -0.0418 |    0.0436 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0097 |  -0.0414 |    0.0609 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a2_conv    | imf_linear_a2      |      0.0089 |  -0.0587 |    0.0764 |      1.0000 |            1.0000 |
