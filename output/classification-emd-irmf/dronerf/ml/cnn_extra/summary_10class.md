# dronerf_phase3_cnn_extra: dronerf / 10class

140 fits on shared split IDs [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 1509.8 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                            |   mean |    std |
|:-----------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'compact_resnet')                          | 0.2827 | 0.0239 |
| ('emd', 'comp+log_scale', 'compact_resnet')                | 0.3090 | 0.0154 |
| ('imf_linear_a1.41', 'comp', 'compact_resnet')             | 0.3013 | 0.0210 |
| ('imf_linear_a1.41', 'comp+log_scale', 'compact_resnet')   | 0.3411 | 0.0163 |
| ('imf_linear_a2', 'comp', 'compact_resnet')                | 0.2983 | 0.0226 |
| ('imf_linear_a2', 'comp+log_scale', 'compact_resnet')      | 0.3394 | 0.0234 |
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')                | 0.3030 | 0.0194 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet')      | 0.3326 | 0.0216 |
| ('irmf_h2_a1.41_conv', 'comp', 'compact_resnet')           | 0.2957 | 0.0138 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.3289 | 0.0251 |
| ('irmf_h2_a2_conv', 'comp', 'compact_resnet')              | 0.2994 | 0.0130 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'compact_resnet')    | 0.3262 | 0.0159 |
| ('raw', 'comp', 'compact_resnet')                          | 0.2898 | 0.0163 |
| ('raw', 'comp+log_scale', 'compact_resnet')                | 0.3123 | 0.0267 |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier     | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:---------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | compact_resnet | raw                | emd                |      0.0071 |  -0.0362 |    0.0504 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | imf_linear_a1.41   |     -0.0115 |  -0.0377 |    0.0147 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a1.41      |     -0.0132 |  -0.0328 |    0.0065 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a1.41_conv |     -0.0059 |  -0.0239 |    0.0121 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a2_conv    |     -0.0096 |  -0.0361 |    0.0169 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | imf_linear_a2      |     -0.0085 |  -0.0391 |    0.0221 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | imf_linear_a1.41   |     -0.0186 |  -0.0648 |    0.0275 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a1.41      |     -0.0203 |  -0.0600 |    0.0194 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a1.41_conv |     -0.0131 |  -0.0522 |    0.0261 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a2_conv    |     -0.0167 |  -0.0489 |    0.0155 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | imf_linear_a2      |     -0.0156 |  -0.0584 |    0.0271 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0017 |  -0.0179 |    0.0145 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0056 |  -0.0186 |    0.0298 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0019 |  -0.0312 |    0.0350 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | imf_linear_a2      |      0.0030 |  -0.0334 |    0.0394 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0073 |  -0.0132 |    0.0277 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0036 |  -0.0243 |    0.0315 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | imf_linear_a2      |      0.0047 |  -0.0295 |    0.0389 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0037 |  -0.0325 |    0.0252 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0026 |  -0.0383 |    0.0332 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a2_conv    | imf_linear_a2      |      0.0011 |  -0.0280 |    0.0302 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | emd                |      0.0034 |  -0.0413 |    0.0480 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | imf_linear_a1.41   |     -0.0288 |  -0.0745 |    0.0169 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a1.41      |     -0.0203 |  -0.0756 |    0.0350 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a1.41_conv |     -0.0166 |  -0.0664 |    0.0333 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a2_conv    |     -0.0139 |  -0.0483 |    0.0205 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | imf_linear_a2      |     -0.0271 |  -0.0845 |    0.0303 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | imf_linear_a1.41   |     -0.0322 |  -0.0463 |   -0.0180 |      0.0257 |            0.0820 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a1.41      |     -0.0237 |  -0.0630 |    0.0157 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a1.41_conv |     -0.0199 |  -0.0492 |    0.0093 |      1.0000 |            0.7812 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a2_conv    |     -0.0173 |  -0.0435 |    0.0090 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | imf_linear_a2      |     -0.0305 |  -0.0697 |    0.0087 |      1.0000 |            0.5605 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0085 |  -0.0259 |    0.0429 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0123 |  -0.0159 |    0.0404 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0149 |  -0.0142 |    0.0440 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | imf_linear_a2      |      0.0017 |  -0.0361 |    0.0395 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0037 |  -0.0369 |    0.0444 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0064 |  -0.0335 |    0.0463 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | imf_linear_a2      |     -0.0068 |  -0.0408 |    0.0271 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0027 |  -0.0353 |    0.0406 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0106 |  -0.0580 |    0.0369 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0132 |  -0.0512 |    0.0247 |      1.0000 |            1.0000 |
