# cnn_remaining: birdclef2023 / species

100 fits on shared split IDs [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 2485.7 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                            |   mean |    std |
|:-----------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'compact_resnet')                          | 0.5144 | 0.0305 |
| ('emd', 'comp+log_scale', 'compact_resnet')                | 0.5068 | 0.0256 |
| ('imf_linear_a1.41', 'comp', 'compact_resnet')             | 0.5250 | 0.0305 |
| ('imf_linear_a1.41', 'comp+log_scale', 'compact_resnet')   | 0.5232 | 0.0374 |
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')                | 0.5270 | 0.0330 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet')      | 0.5211 | 0.0304 |
| ('irmf_h2_a1.41_conv', 'comp', 'compact_resnet')           | 0.5202 | 0.0391 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.5194 | 0.0292 |
| ('raw', 'comp', 'compact_resnet')                          | 0.5240 | 0.0342 |
| ('raw', 'comp+log_scale', 'compact_resnet')                | 0.5210 | 0.0361 |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier     | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:---------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | compact_resnet | raw              | emd                |      0.0096 |  -0.0133 |    0.0325 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | imf_linear_a1.41   |     -0.0010 |  -0.0258 |    0.0239 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | irmf_h2_a1.41      |     -0.0030 |  -0.0350 |    0.0289 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | irmf_h2_a1.41_conv |      0.0038 |  -0.0171 |    0.0248 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | imf_linear_a1.41   |     -0.0106 |  -0.0436 |    0.0224 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | irmf_h2_a1.41      |     -0.0127 |  -0.0461 |    0.0207 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | irmf_h2_a1.41_conv |     -0.0058 |  -0.0350 |    0.0233 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0021 |  -0.0258 |    0.0216 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0048 |  -0.0263 |    0.0359 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0069 |  -0.0318 |    0.0455 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | emd                |      0.0142 |  -0.0266 |    0.0550 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | imf_linear_a1.41   |     -0.0022 |  -0.0446 |    0.0402 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | irmf_h2_a1.41      |     -0.0000 |  -0.0358 |    0.0358 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | irmf_h2_a1.41_conv |      0.0016 |  -0.0283 |    0.0315 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | imf_linear_a1.41   |     -0.0164 |  -0.0457 |    0.0129 |      1.0000 |            0.7422 |
| comp+log_scale | compact_resnet | emd              | irmf_h2_a1.41      |     -0.0142 |  -0.0529 |    0.0245 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | irmf_h2_a1.41_conv |     -0.0126 |  -0.0457 |    0.0205 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0022 |  -0.0367 |    0.0410 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0038 |  -0.0294 |    0.0369 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0016 |  -0.0156 |    0.0188 |      1.0000 |            1.0000 |
