# uas_phase3_cnn: uas_birds_60ghz / target3

270 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 1472.0 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                            |   mean |    std |
|:-----------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'compact_resnet')                          | 0.8559 | 0.0566 |
| ('emd', 'comp+log_scale', 'compact_resnet')                | 0.8587 | 0.0502 |
| ('imf_linear_a1.41', 'comp', 'compact_resnet')             | 0.9090 | 0.0376 |
| ('imf_linear_a1.41', 'comp+log_scale', 'compact_resnet')   | 0.9184 | 0.0391 |
| ('imf_linear_a2', 'comp', 'compact_resnet')                | 0.9271 | 0.0374 |
| ('imf_linear_a2', 'comp+log_scale', 'compact_resnet')      | 0.9347 | 0.0391 |
| ('irmf_h1_a1.41_conv', 'comp', 'compact_resnet')           | 0.9115 | 0.0392 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.9156 | 0.0416 |
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')                | 0.9101 | 0.0453 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet')      | 0.9139 | 0.0428 |
| ('irmf_h2_a1.41_conv', 'comp', 'compact_resnet')           | 0.9059 | 0.0397 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.9094 | 0.0434 |
| ('irmf_h2_a2_conv', 'comp', 'compact_resnet')              | 0.9219 | 0.0330 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'compact_resnet')    | 0.9181 | 0.0490 |
| ('irmf_h4_a1.41_conv', 'comp', 'compact_resnet')           | 0.9174 | 0.0408 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'compact_resnet') | 0.9153 | 0.0461 |
| ('raw', 'comp', 'compact_resnet')                          | 0.9056 | 0.0332 |
| ('raw', 'comp+log_scale', 'compact_resnet')                | 0.9125 | 0.0375 |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier     | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:---------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | compact_resnet | raw                | emd                |      0.0497 |  -0.0100 |    0.1093 |      1.0000 |            0.1139 |
| comp           | compact_resnet | raw                | imf_linear_a1.41   |     -0.0035 |  -0.0388 |    0.0318 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a1.41      |     -0.0045 |  -0.0402 |    0.0312 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a1.41_conv |     -0.0003 |  -0.0349 |    0.0342 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h2_a2_conv    |     -0.0163 |  -0.0702 |    0.0376 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | imf_linear_a2      |     -0.0215 |  -0.0498 |    0.0068 |      1.0000 |            0.7545 |
| comp           | compact_resnet | raw                | irmf_h1_a1.41_conv |     -0.0059 |  -0.0419 |    0.0300 |      1.0000 |            1.0000 |
| comp           | compact_resnet | raw                | irmf_h4_a1.41_conv |     -0.0118 |  -0.0543 |    0.0307 |      1.0000 |            1.0000 |
| comp           | compact_resnet | emd                | imf_linear_a1.41   |     -0.0531 |  -0.1186 |    0.0124 |      1.0000 |            0.1878 |
| comp           | compact_resnet | emd                | irmf_h2_a1.41      |     -0.0542 |  -0.1104 |    0.0021 |      1.0000 |            0.0964 |
| comp           | compact_resnet | emd                | irmf_h2_a1.41_conv |     -0.0500 |  -0.1040 |    0.0040 |      1.0000 |            0.0088 |
| comp           | compact_resnet | emd                | irmf_h2_a2_conv    |     -0.0660 |  -0.1472 |    0.0153 |      1.0000 |            0.4324 |
| comp           | compact_resnet | emd                | imf_linear_a2      |     -0.0712 |  -0.1286 |   -0.0137 |      1.0000 |            0.0088 |
| comp           | compact_resnet | emd                | irmf_h1_a1.41_conv |     -0.0556 |  -0.1192 |    0.0081 |      1.0000 |            0.0964 |
| comp           | compact_resnet | emd                | irmf_h4_a1.41_conv |     -0.0615 |  -0.1252 |    0.0023 |      1.0000 |            0.0998 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0010 |  -0.0289 |    0.0268 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0031 |  -0.0276 |    0.0338 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0128 |  -0.0641 |    0.0384 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | imf_linear_a2      |     -0.0181 |  -0.0439 |    0.0077 |      1.0000 |            0.5416 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h1_a1.41_conv |     -0.0024 |  -0.0371 |    0.0322 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a1.41   | irmf_h4_a1.41_conv |     -0.0083 |  -0.0379 |    0.0212 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0042 |  -0.0231 |    0.0314 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0118 |  -0.0696 |    0.0460 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | imf_linear_a2      |     -0.0170 |  -0.0457 |    0.0116 |      1.0000 |            0.6982 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0014 |  -0.0371 |    0.0343 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41      | irmf_h4_a1.41_conv |     -0.0073 |  -0.0391 |    0.0245 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0160 |  -0.0758 |    0.0439 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0212 |  -0.0581 |    0.0157 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0056 |  -0.0443 |    0.0332 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |     -0.0115 |  -0.0571 |    0.0342 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0052 |  -0.0568 |    0.0464 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |      0.0104 |  -0.0519 |    0.0727 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0045 |  -0.0432 |    0.0523 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a2      | irmf_h1_a1.41_conv |      0.0156 |  -0.0257 |    0.0569 |      1.0000 |            1.0000 |
| comp           | compact_resnet | imf_linear_a2      | irmf_h4_a1.41_conv |      0.0097 |  -0.0175 |    0.0369 |      1.0000 |            1.0000 |
| comp           | compact_resnet | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |     -0.0059 |  -0.0457 |    0.0339 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | emd                |      0.0538 |  -0.0012 |    0.1088 |      1.0000 |            0.0777 |
| comp+log_scale | compact_resnet | raw                | imf_linear_a1.41   |     -0.0059 |  -0.0438 |    0.0320 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a1.41      |     -0.0014 |  -0.0367 |    0.0339 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a1.41_conv |      0.0031 |  -0.0402 |    0.0464 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h2_a2_conv    |     -0.0056 |  -0.0541 |    0.0430 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | imf_linear_a2      |     -0.0222 |  -0.0622 |    0.0178 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h1_a1.41_conv |     -0.0031 |  -0.0386 |    0.0323 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | raw                | irmf_h4_a1.41_conv |     -0.0028 |  -0.0387 |    0.0331 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | emd                | imf_linear_a1.41   |     -0.0597 |  -0.1190 |   -0.0004 |      1.0000 |            0.2205 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a1.41      |     -0.0552 |  -0.1050 |   -0.0054 |      1.0000 |            0.0964 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a1.41_conv |     -0.0507 |  -0.1157 |    0.0143 |      1.0000 |            0.2230 |
| comp+log_scale | compact_resnet | emd                | irmf_h2_a2_conv    |     -0.0594 |  -0.1113 |   -0.0074 |      1.0000 |            0.0544 |
| comp+log_scale | compact_resnet | emd                | imf_linear_a2      |     -0.0760 |  -0.1272 |   -0.0249 |      0.4714 |            0.0450 |
| comp+log_scale | compact_resnet | emd                | irmf_h1_a1.41_conv |     -0.0569 |  -0.1084 |   -0.0055 |      1.0000 |            0.0964 |
| comp+log_scale | compact_resnet | emd                | irmf_h4_a1.41_conv |     -0.0566 |  -0.0976 |   -0.0156 |      0.7345 |            0.0450 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0045 |  -0.0392 |    0.0483 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0090 |  -0.0532 |    0.0713 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0003 |  -0.0534 |    0.0541 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | imf_linear_a2      |     -0.0163 |  -0.0689 |    0.0362 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h1_a1.41_conv |      0.0028 |  -0.0358 |    0.0413 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a1.41   | irmf_h4_a1.41_conv |      0.0031 |  -0.0372 |    0.0434 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0045 |  -0.0293 |    0.0384 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0042 |  -0.0458 |    0.0375 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | imf_linear_a2      |     -0.0208 |  -0.0627 |    0.0211 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h1_a1.41_conv |     -0.0017 |  -0.0317 |    0.0282 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41      | irmf_h4_a1.41_conv |     -0.0014 |  -0.0259 |    0.0231 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0087 |  -0.0661 |    0.0487 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0253 |  -0.0730 |    0.0224 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv |     -0.0063 |  -0.0499 |    0.0374 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv |     -0.0059 |  -0.0528 |    0.0410 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0167 |  -0.0530 |    0.0196 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a2_conv    | irmf_h1_a1.41_conv |      0.0024 |  -0.0474 |    0.0523 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h2_a2_conv    | irmf_h4_a1.41_conv |      0.0028 |  -0.0380 |    0.0435 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a2      | irmf_h1_a1.41_conv |      0.0191 |  -0.0227 |    0.0609 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | imf_linear_a2      | irmf_h4_a1.41_conv |      0.0194 |  -0.0206 |    0.0595 |      1.0000 |            1.0000 |
| comp+log_scale | compact_resnet | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0003 |  -0.0285 |    0.0292 |      1.0000 |            1.0000 |
