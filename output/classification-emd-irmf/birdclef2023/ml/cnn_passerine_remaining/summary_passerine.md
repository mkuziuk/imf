# cnn_passerine_remaining: birdclef2023 / passerine

40 fits on shared split IDs [1, 2, 3, 4]; wall time 1022.1 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                            |   mean |    std |
|:-----------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'compact_resnet')                          | 0.7814 | 0.0128 |
| ('emd', 'comp+log_scale', 'compact_resnet')                | 0.7958 | 0.0100 |
| ('imf_linear_a1.41', 'comp', 'compact_resnet')             | 0.8133 | 0.0157 |
| ('imf_linear_a1.41', 'comp+log_scale', 'compact_resnet')   | 0.7957 | 0.0097 |
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')                | 0.7893 | 0.0205 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet')      | 0.8030 | 0.0220 |
| ('irmf_h2_a1.41_conv', 'comp', 'compact_resnet')           | 0.7879 | 0.0041 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.7960 | 0.0128 |
| ('raw', 'comp', 'compact_resnet')                          | 0.8122 | 0.0100 |
| ('raw', 'comp+log_scale', 'compact_resnet')                | 0.8070 | 0.0132 |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier     | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:---------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | compact_resnet | raw              | emd                |      0.0308 |  -0.0193 |    0.0809 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | imf_linear_a1.41   |     -0.0010 |  -0.0235 |    0.0214 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | irmf_h2_a1.41      |      0.0229 |  -0.0122 |    0.0580 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | irmf_h2_a1.41_conv |      0.0243 |  -0.0019 |    0.0505 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | imf_linear_a1.41   |     -0.0318 |  -0.0954 |    0.0317 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | irmf_h2_a1.41      |     -0.0079 |  -0.0809 |    0.0651 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | irmf_h2_a1.41_conv |     -0.0065 |  -0.0317 |    0.0187 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0239 |   0.0038 |    0.0440 |      0.6441 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0254 |  -0.0171 |    0.0678 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0014 |  -0.0500 |    0.0529 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | emd                |      0.0112 |  -0.0145 |    0.0370 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | imf_linear_a1.41   |      0.0113 |  -0.0146 |    0.0372 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | irmf_h2_a1.41      |      0.0040 |  -0.0176 |    0.0257 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | irmf_h2_a1.41_conv |      0.0110 |  -0.0125 |    0.0345 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | imf_linear_a1.41   |      0.0001 |  -0.0363 |    0.0365 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | irmf_h2_a1.41      |     -0.0072 |  -0.0535 |    0.0391 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | irmf_h2_a1.41_conv |     -0.0003 |  -0.0383 |    0.0378 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0073 |  -0.0447 |    0.0301 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0003 |  -0.0105 |    0.0098 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0069 |  -0.0237 |    0.0376 |      1.0000 |            1.0000 |
