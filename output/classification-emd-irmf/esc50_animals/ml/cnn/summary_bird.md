# esc50_phase3_cnn: esc50_animals / bird

50 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 569.2 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                            |   mean |    std |
|:-----------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'compact_resnet')                          | 0.7198 | 0.0568 |
| ('emd', 'comp+log_scale', 'compact_resnet')                | 0.7443 | 0.0583 |
| ('imf_linear_a1.41', 'comp', 'compact_resnet')             | 0.7392 | 0.0472 |
| ('imf_linear_a1.41', 'comp+log_scale', 'compact_resnet')   | 0.7220 | 0.0540 |
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')                | 0.7591 | 0.0532 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet')      | 0.7493 | 0.0561 |
| ('irmf_h2_a1.41_conv', 'comp', 'compact_resnet')           | 0.7493 | 0.0702 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.7319 | 0.0585 |
| ('raw', 'comp', 'compact_resnet')                          | 0.7304 | 0.0463 |
| ('raw', 'comp+log_scale', 'compact_resnet')                | 0.7253 | 0.0637 |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier     | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:---------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | compact_resnet | raw              | emd                |      0.0106 |  -0.0859 |    0.1071 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | imf_linear_a1.41   |     -0.0088 |  -0.0391 |    0.0214 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | irmf_h2_a1.41      |     -0.0287 |  -0.1194 |    0.0621 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | irmf_h2_a1.41_conv |     -0.0189 |  -0.1182 |    0.0804 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | imf_linear_a1.41   |     -0.0194 |  -0.0892 |    0.0504 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | irmf_h2_a1.41      |     -0.0393 |  -0.0896 |    0.0111 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | irmf_h2_a1.41_conv |     -0.0295 |  -0.0658 |    0.0069 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0198 |  -0.0962 |    0.0565 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0101 |  -0.0846 |    0.0644 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0098 |  -0.0420 |    0.0615 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | emd                |     -0.0190 |  -0.1224 |    0.0844 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | imf_linear_a1.41   |      0.0033 |  -0.0162 |    0.0228 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | irmf_h2_a1.41      |     -0.0240 |  -0.1239 |    0.0759 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | irmf_h2_a1.41_conv |     -0.0065 |  -0.0642 |    0.0511 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | imf_linear_a1.41   |      0.0223 |  -0.0738 |    0.1185 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | irmf_h2_a1.41      |     -0.0050 |  -0.0506 |    0.0407 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | irmf_h2_a1.41_conv |      0.0125 |  -0.0880 |    0.1130 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0273 |  -0.1162 |    0.0616 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0099 |  -0.0563 |    0.0366 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0174 |  -0.0546 |    0.0895 |      1.0000 |            1.0000 |
