# cnn: birdclef2023 / species

50 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 1131.5 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                            |   mean |    std |
|:-----------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'compact_resnet')                          | 0.5081 | 0.0321 |
| ('emd', 'comp+log_scale', 'compact_resnet')                | 0.5055 | 0.0230 |
| ('imf_linear_a1.41', 'comp', 'compact_resnet')             | 0.5296 | 0.0131 |
| ('imf_linear_a1.41', 'comp+log_scale', 'compact_resnet')   | 0.5237 | 0.0408 |
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')                | 0.5240 | 0.0094 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet')      | 0.5369 | 0.0237 |
| ('irmf_h2_a1.41_conv', 'comp', 'compact_resnet')           | 0.5244 | 0.0174 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.5257 | 0.0281 |
| ('raw', 'comp', 'compact_resnet')                          | 0.5309 | 0.0103 |
| ('raw', 'comp+log_scale', 'compact_resnet')                | 0.5156 | 0.0284 |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier     | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:---------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | compact_resnet | raw              | emd                |      0.0228 |  -0.0260 |    0.0717 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | imf_linear_a1.41   |      0.0013 |  -0.0176 |    0.0203 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | irmf_h2_a1.41      |      0.0069 |  -0.0104 |    0.0241 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | irmf_h2_a1.41_conv |      0.0065 |  -0.0162 |    0.0292 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | imf_linear_a1.41   |     -0.0215 |  -0.0789 |    0.0359 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | irmf_h2_a1.41      |     -0.0160 |  -0.0705 |    0.0386 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | irmf_h2_a1.41_conv |     -0.0164 |  -0.0704 |    0.0377 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0055 |  -0.0052 |    0.0163 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0051 |  -0.0056 |    0.0159 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0004 |  -0.0201 |    0.0194 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | emd                |      0.0101 |  -0.0130 |    0.0331 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | imf_linear_a1.41   |     -0.0081 |  -0.0326 |    0.0163 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | irmf_h2_a1.41      |     -0.0214 |  -0.0473 |    0.0046 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | irmf_h2_a1.41_conv |     -0.0102 |  -0.0404 |    0.0201 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | imf_linear_a1.41   |     -0.0182 |  -0.0596 |    0.0232 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | irmf_h2_a1.41      |     -0.0315 |  -0.0542 |   -0.0087 |      0.3723 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | irmf_h2_a1.41_conv |     -0.0203 |  -0.0605 |    0.0200 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0132 |  -0.0529 |    0.0265 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0020 |  -0.0439 |    0.0398 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0112 |  -0.0134 |    0.0358 |      1.0000 |            1.0000 |
