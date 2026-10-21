# dronerf_phase3_cnn: dronerf / 10class

70 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 4924.1 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                            |   mean |    std |
|:-----------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'compact_resnet')                          | 0.2909 | 0.0251 |
| ('emd', 'comp+log_scale', 'compact_resnet')                | 0.3143 | 0.0221 |
| ('imf_linear_a1.41', 'comp', 'compact_resnet')             | 0.3164 | 0.0218 |
| ('imf_linear_a1.41', 'comp+log_scale', 'compact_resnet')   | 0.3367 | 0.0144 |
| ('imf_linear_a2', 'comp', 'compact_resnet')                | 0.3096 | 0.0253 |
| ('imf_linear_a2', 'comp+log_scale', 'compact_resnet')      | 0.3232 | 0.0149 |
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')                | 0.3083 | 0.0213 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet')      | 0.3389 | 0.0147 |
| ('irmf_h2_a1.41_conv', 'comp', 'compact_resnet')           | 0.2899 | 0.0212 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.3370 | 0.0141 |
| ('irmf_h2_a2_conv', 'comp', 'compact_resnet')              | 0.2967 | 0.0103 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'compact_resnet')    | 0.3141 | 0.0178 |
| ('raw', 'comp', 'compact_resnet')                          | 0.2880 | 0.0099 |
| ('raw', 'comp+log_scale', 'compact_resnet')                | 0.3162 | 0.0177 |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier     | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:---------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | compact_resnet | raw                | emd                |     -0.0028 |  -0.0403 |    0.0347 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | imf_linear_a1.41   |     -0.0284 |  -0.0835 |    0.0266 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a1.41      |     -0.0203 |  -0.0702 |    0.0297 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a1.41_conv |     -0.0018 |  -0.0546 |    0.0509 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a2_conv    |     -0.0087 |  -0.0328 |    0.0154 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | imf_linear_a2      |     -0.0216 |  -0.0743 |    0.0311 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | imf_linear_a1.41   |     -0.0256 |  -0.1113 |    0.0602 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a1.41      |     -0.0174 |  -0.0761 |    0.0412 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a1.41_conv |      0.0010 |  -0.0569 |    0.0589 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a2_conv    |     -0.0059 |  -0.0435 |    0.0318 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | imf_linear_a2      |     -0.0188 |  -0.0707 |    0.0332 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0081 |  -0.0451 |    0.0613 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0266 |  -0.0283 |    0.0815 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0197 |  -0.0325 |    0.0719 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | imf_linear_a2      |      0.0068 |  -0.0567 |    0.0703 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0184 |  -0.0001 |    0.0370 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0116 |  -0.0153 |    0.0385 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | imf_linear_a2      |     -0.0013 |  -0.0288 |    0.0262 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0069 |  -0.0375 |    0.0238 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0198 |  -0.0528 |    0.0133 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0129 |  -0.0468 |    0.0210 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | emd                |      0.0019 |  -0.0270 |    0.0308 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | imf_linear_a1.41   |     -0.0205 |  -0.0584 |    0.0173 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a1.41      |     -0.0227 |  -0.0727 |    0.0272 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a1.41_conv |     -0.0208 |  -0.0554 |    0.0137 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a2_conv    |      0.0021 |  -0.0110 |    0.0152 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | imf_linear_a2      |     -0.0070 |  -0.0452 |    0.0312 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | imf_linear_a1.41   |     -0.0224 |  -0.0724 |    0.0276 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a1.41      |     -0.0246 |  -0.0756 |    0.0264 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a1.41_conv |     -0.0227 |  -0.0598 |    0.0143 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a2_conv    |      0.0002 |  -0.0361 |    0.0365 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | imf_linear_a2      |     -0.0089 |  -0.0670 |    0.0492 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0022 |  -0.0327 |    0.0283 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0003 |  -0.0179 |    0.0173 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0226 |  -0.0135 |    0.0587 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | imf_linear_a2      |      0.0135 |  -0.0078 |    0.0348 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0019 |  -0.0219 |    0.0257 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0248 |  -0.0293 |    0.0789 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | imf_linear_a2      |      0.0157 |  -0.0283 |    0.0597 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0229 |  -0.0147 |    0.0605 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0138 |  -0.0219 |    0.0495 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0091 |  -0.0406 |    0.0224 |      1.0000 |            1.0000 |
