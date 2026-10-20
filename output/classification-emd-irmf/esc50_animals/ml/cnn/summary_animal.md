# esc50_phase3_cnn: esc50_animals / animal

50 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 1259.7 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                            |   mean |    std |
|:-----------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'compact_resnet')                          | 0.5146 | 0.0311 |
| ('emd', 'comp+log_scale', 'compact_resnet')                | 0.5434 | 0.0292 |
| ('imf_linear_a1.41', 'comp', 'compact_resnet')             | 0.5249 | 0.0367 |
| ('imf_linear_a1.41', 'comp+log_scale', 'compact_resnet')   | 0.5305 | 0.0351 |
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')                | 0.5084 | 0.0336 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet')      | 0.5596 | 0.0337 |
| ('irmf_h2_a1.41_conv', 'comp', 'compact_resnet')           | 0.5103 | 0.0305 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.5431 | 0.0562 |
| ('raw', 'comp', 'compact_resnet')                          | 0.5259 | 0.0389 |
| ('raw', 'comp+log_scale', 'compact_resnet')                | 0.5383 | 0.0469 |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier     | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:---------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | compact_resnet | raw              | emd                |      0.0113 |  -0.0498 |    0.0725 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | imf_linear_a1.41   |      0.0010 |  -0.0420 |    0.0441 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | irmf_h2_a1.41      |      0.0176 |  -0.0432 |    0.0783 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw              | irmf_h2_a1.41_conv |      0.0157 |  -0.0507 |    0.0821 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | imf_linear_a1.41   |     -0.0103 |  -0.0569 |    0.0363 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | irmf_h2_a1.41      |      0.0062 |  -0.0482 |    0.0607 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd              | irmf_h2_a1.41_conv |      0.0044 |  -0.0533 |    0.0620 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0165 |  -0.0075 |    0.0406 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0146 |  -0.0275 |    0.0568 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0019 |  -0.0446 |    0.0408 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | emd                |     -0.0051 |  -0.0781 |    0.0679 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | imf_linear_a1.41   |      0.0078 |  -0.0658 |    0.0813 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | irmf_h2_a1.41      |     -0.0213 |  -0.0737 |    0.0311 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw              | irmf_h2_a1.41_conv |     -0.0049 |  -0.0614 |    0.0516 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | imf_linear_a1.41   |      0.0129 |  -0.0382 |    0.0640 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | irmf_h2_a1.41      |     -0.0162 |  -0.0918 |    0.0594 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd              | irmf_h2_a1.41_conv |      0.0002 |  -0.0778 |    0.0783 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0291 |  -0.0820 |    0.0239 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0127 |  -0.0759 |    0.0506 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0164 |  -0.0603 |    0.0931 |      1.0000 |            1.0000 |
