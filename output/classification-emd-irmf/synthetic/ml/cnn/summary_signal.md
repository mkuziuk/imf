# cnn: synthetic / signal

40 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 220.5 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                          |   mean |    std |
|:---------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'compact_resnet')                        | 0.2253 | 0.0749 |
| ('emd', 'comp+log_scale', 'compact_resnet')              | 0.2200 | 0.0567 |
| ('imf_linear_a1.41', 'comp', 'compact_resnet')           | 0.5580 | 0.1706 |
| ('imf_linear_a1.41', 'comp+log_scale', 'compact_resnet') | 0.5653 | 0.0634 |
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')              | 0.5013 | 0.1989 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet')    | 0.5093 | 0.1745 |
| ('raw', 'comp', 'compact_resnet')                        | 0.5720 | 0.0695 |
| ('raw', 'comp+log_scale', 'compact_resnet')              | 0.5760 | 0.0836 |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier     | method_a         | method_b         |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:---------------|:-----------------|:-----------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | compact_resnet | raw              | emd              |      0.3467 |   0.1844 |    0.5090 |      0.0486 |            0.7500 |
| comp           | compact_resnet | raw              | imf_linear_a1.41 |      0.0140 |  -0.3058 |    0.3338 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | irmf_h2_a1.41    |      0.0707 |  -0.2916 |    0.4329 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | imf_linear_a1.41 |     -0.3327 |  -0.6992 |    0.0339 |      0.4576 |            0.7500 |
| comp           | compact_resnet | emd              | irmf_h2_a1.41    |     -0.2760 |  -0.6930 |    0.1410 |      0.8400 |            0.7500 |
| comp           | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41    |      0.0567 |  -0.0001 |    0.1134 |      0.4514 |            0.7500 |
| comp+log_scale | compact_resnet | raw              | emd              |      0.3560 |   0.1266 |    0.5854 |      0.1256 |            0.7500 |
| comp+log_scale | compact_resnet | raw              | imf_linear_a1.41 |      0.0107 |  -0.1412 |    0.1626 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | irmf_h2_a1.41    |      0.0667 |  -0.4020 |    0.5353 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | imf_linear_a1.41 |     -0.3453 |  -0.5267 |   -0.1640 |      0.0676 |            0.7500 |
| comp+log_scale | compact_resnet | emd              | irmf_h2_a1.41    |     -0.2893 |  -0.5800 |    0.0013 |      0.4514 |            0.7500 |
| comp+log_scale | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41    |      0.0560 |  -0.3194 |    0.4314 |      1.0000 |            1.0000 |
