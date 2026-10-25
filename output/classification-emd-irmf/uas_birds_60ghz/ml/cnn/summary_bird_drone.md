# uas_phase3_cnn: uas_birds_60ghz / bird_drone

270 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 947.8 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                            |   mean |    std |
|:-----------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'compact_resnet')                          | 0.9497 | 0.0382 |
| ('emd', 'comp+log_scale', 'compact_resnet')                | 0.9379 | 0.0328 |
| ('imf_linear_a1.41', 'comp', 'compact_resnet')             | 0.9523 | 0.0350 |
| ('imf_linear_a1.41', 'comp+log_scale', 'compact_resnet')   | 0.9593 | 0.0287 |
| ('imf_linear_a2', 'comp', 'compact_resnet')                | 0.9643 | 0.0259 |
| ('imf_linear_a2', 'comp+log_scale', 'compact_resnet')      | 0.9593 | 0.0307 |
| ('irmf_h1_a1.41_conv', 'comp', 'compact_resnet')           | 0.9517 | 0.0337 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.9558 | 0.0298 |
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')                | 0.9500 | 0.0328 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet')      | 0.9452 | 0.0338 |
| ('irmf_h2_a1.41_conv', 'comp', 'compact_resnet')           | 0.9463 | 0.0401 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.9503 | 0.0365 |
| ('irmf_h2_a2_conv', 'comp', 'compact_resnet')              | 0.9563 | 0.0305 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'compact_resnet')    | 0.9572 | 0.0231 |
| ('irmf_h4_a1.41_conv', 'comp', 'compact_resnet')           | 0.9509 | 0.0408 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.9581 | 0.0278 |
| ('raw', 'comp', 'compact_resnet')                          | 0.9508 | 0.0246 |
| ('raw', 'comp+log_scale', 'compact_resnet')                | 0.9552 | 0.0271 |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier     | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:---------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | compact_resnet | raw                | emd                |      0.0011 |  -0.0378 |    0.0399 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | imf_linear_a1.41   |     -0.0015 |  -0.0446 |    0.0416 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a1.41      |      0.0008 |  -0.0368 |    0.0384 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a1.41_conv |      0.0045 |  -0.0349 |    0.0439 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a2_conv    |     -0.0055 |  -0.0394 |    0.0284 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | imf_linear_a2      |     -0.0135 |  -0.0466 |    0.0195 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h1_a1.41_conv |     -0.0010 |  -0.0388 |    0.0369 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h4_a1.41_conv |     -0.0001 |  -0.0465 |    0.0462 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | imf_linear_a1.41   |     -0.0026 |  -0.0359 |    0.0307 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a1.41      |     -0.0003 |  -0.0294 |    0.0289 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a1.41_conv |      0.0034 |  -0.0190 |    0.0258 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h2_a2_conv    |     -0.0066 |  -0.0516 |    0.0385 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | imf_linear_a2      |     -0.0146 |  -0.0488 |    0.0195 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h1_a1.41_conv |     -0.0020 |  -0.0363 |    0.0322 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | irmf_h4_a1.41_conv |     -0.0012 |  -0.0401 |    0.0377 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0023 |  -0.0262 |    0.0308 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0060 |  -0.0226 |    0.0346 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0040 |  -0.0351 |    0.0271 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | imf_linear_a2      |     -0.0120 |  -0.0383 |    0.0142 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h1_a1.41_conv |      0.0005 |  -0.0269 |    0.0279 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h4_a1.41_conv |      0.0014 |  -0.0227 |    0.0254 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0037 |  -0.0186 |    0.0259 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0063 |  -0.0352 |    0.0226 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | imf_linear_a2      |     -0.0143 |  -0.0407 |    0.0120 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0018 |  -0.0214 |    0.0179 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h4_a1.41_conv |     -0.0009 |  -0.0314 |    0.0295 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0100 |  -0.0462 |    0.0263 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0180 |  -0.0515 |    0.0155 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0055 |  -0.0296 |    0.0187 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |     -0.0046 |  -0.0344 |    0.0252 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0081 |  -0.0364 |    0.0202 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |      0.0045 |  -0.0282 |    0.0373 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0053 |  -0.0207 |    0.0314 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a2      | irmf_h1_a1.41_conv |      0.0126 |  -0.0153 |    0.0404 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a2      | irmf_h4_a1.41_conv |      0.0134 |  -0.0159 |    0.0427 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0008 |  -0.0332 |    0.0348 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | emd                |      0.0174 |  -0.0046 |    0.0393 |      1.0000 |            0.2184 |
| comp+log_scale | compact_resnet | raw                | imf_linear_a1.41   |     -0.0041 |  -0.0346 |    0.0264 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a1.41      |      0.0100 |  -0.0269 |    0.0469 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a1.41_conv |      0.0049 |  -0.0298 |    0.0396 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a2_conv    |     -0.0019 |  -0.0257 |    0.0218 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | imf_linear_a2      |     -0.0040 |  -0.0371 |    0.0291 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h1_a1.41_conv |     -0.0006 |  -0.0271 |    0.0260 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h4_a1.41_conv |     -0.0028 |  -0.0291 |    0.0234 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | imf_linear_a1.41   |     -0.0215 |  -0.0505 |    0.0076 |      1.0000 |            0.4099 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a1.41      |     -0.0074 |  -0.0449 |    0.0301 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a1.41_conv |     -0.0124 |  -0.0440 |    0.0191 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a2_conv    |     -0.0193 |  -0.0516 |    0.0130 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | imf_linear_a2      |     -0.0214 |  -0.0542 |    0.0114 |      1.0000 |            0.7256 |
| comp+log_scale | compact_resnet | emd                | irmf_h1_a1.41_conv |     -0.0179 |  -0.0477 |    0.0119 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | irmf_h4_a1.41_conv |     -0.0202 |  -0.0431 |    0.0027 |      1.0000 |            0.1230 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0141 |  -0.0235 |    0.0517 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0090 |  -0.0156 |    0.0337 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0022 |  -0.0245 |    0.0288 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | imf_linear_a2      |      0.0001 |  -0.0259 |    0.0260 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h1_a1.41_conv |      0.0035 |  -0.0218 |    0.0289 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h4_a1.41_conv |      0.0013 |  -0.0196 |    0.0221 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0051 |  -0.0401 |    0.0299 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0119 |  -0.0482 |    0.0243 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | imf_linear_a2      |     -0.0140 |  -0.0483 |    0.0203 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0106 |  -0.0431 |    0.0219 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h4_a1.41_conv |     -0.0128 |  -0.0416 |    0.0159 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0069 |  -0.0371 |    0.0234 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0090 |  -0.0320 |    0.0141 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0055 |  -0.0322 |    0.0212 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |     -0.0078 |  -0.0252 |    0.0097 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0021 |  -0.0289 |    0.0247 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |      0.0014 |  -0.0241 |    0.0268 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |     -0.0009 |  -0.0245 |    0.0227 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a2      | irmf_h1_a1.41_conv |      0.0035 |  -0.0203 |    0.0273 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a2      | irmf_h4_a1.41_conv |      0.0012 |  -0.0175 |    0.0199 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |     -0.0023 |  -0.0236 |    0.0190 |      1.0000 |            1.0000 |
