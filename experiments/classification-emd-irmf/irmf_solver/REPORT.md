# IRMF convergence and solver audit

Measured on 2026-10-20 in the supplied worktree, pimf 0.4.0, Python 3.12. At most two computational processes ran concurrently, each with one native thread. Code, shared study files, and the installed pimf package were left unchanged outside this folder. No full-dataset decomposition was run.

## Existing-cache diagnosis

These tables use every successful channel in the original caches. DroneRF has two channels per frame; all other datasets have one. Times sum channels within a frame. Reconstruction does not establish convergence, because the residual telescopes for any sequence of fitted components.

| dataset       | n_frames   | n_channels   | unconverged_stage_pct   | any_unconverged_frame_pct   | wall_s_per_frame_mean   | cpu_s_per_frame_mean   |
|:--------------|:-----------|:-------------|:------------------------|:----------------------------|:------------------------|:-----------------------|
| dronerf       | 12000      | 24000        | 6.700                   | 16.642                      | 1.301                   | 1.279                  |
| birdclef2023  | 12000      | 12000        | 14.049                  | 34.267                      | 1.343                   | 1.320                  |
| esc50_animals | 5100       | 5100         | 21.566                  | 55.020                      | 1.789                   | 1.759                  |
| synthetic     | 1500       | 1500         | 4.907                   | 65.600                      | 0.101                   | 0.101                  |

Stage failure percentages in canonical fine-to-coarse order:

|   window |   birdclef2023 |   dronerf |   esc50_animals |   synthetic |
|---------:|---------------:|----------:|----------------:|------------:|
|     5.00 |           0.61 |      5.71 |            0.71 |        3.07 |
|     7.00 |           5.87 |      8.18 |            6.35 |       65.60 |
|     9.00 |           3.73 |      6.91 |            4.08 |        3.73 |
|    13.00 |           9.41 |      7.88 |            8.84 |        0.93 |
|    17.00 |          13.47 |      7.93 |           15.20 |        0.27 |
|    25.00 |          16.63 |      7.71 |           21.49 |        0.00 |
|    33.00 |          19.65 |      7.41 |           29.96 |        0.00 |
|    47.00 |          20.50 |      7.20 |           38.14 |        0.00 |
|    65.00 |          21.59 |      7.00 |           39.45 |        0.00 |
|    93.00 |          21.28 |      6.85 |           36.67 |        0.00 |
|   129.00 |          20.84 |      6.69 |           34.80 |        0.00 |
|   183.00 |          18.82 |      6.51 |           30.39 |        0.00 |
|   257.00 |          16.67 |      6.29 |           26.37 |        0.00 |
|   365.00 |          14.66 |      6.01 |           22.76 |        0.00 |
|   513.00 |          12.88 |      5.69 |           19.92 |        0.00 |
|   727.00 |          11.53 |      5.25 |           16.55 |      nan    |
|  1025.00 |          10.70 |      4.65 |           14.94 |      nan    |

Iteration distributions over all stages, and the stopping ratio r = final_max_delta / [1e-6 (1 + max |fitted component|)]. The denominator uses cached components, rather than input amplitude. Float32 caching reproduces every convergence flag exactly. r > 1 denotes a capped unconverged stage.

| dataset       | iterations_q05   | iterations_q50   | iterations_q95   | iterations_max   | delta_ratio_q50   | delta_ratio_q95   | delta_ratio_max   | failed_delta_ratio_q50   |
|:--------------|:-----------------|:-----------------|:-----------------|:-----------------|:------------------|:------------------|:------------------|:-------------------------|
| dronerf       | 7.000            | 9.000            | 60.000           | 60.000           | 0.525             | 1918.421          | 33837.745         | 4572.644                 |
| birdclef2023  | 7.000            | 20.000           | 60.000           | 60.000           | 0.783             | 177.411           | 32176.954         | 43.403                   |
| esc50_animals | 7.000            | 29.000           | 60.000           | 60.000           | 0.850             | 559.777           | 29146.000         | 50.374                   |
| synthetic     | 9.000            | 12.000           | 59.000           | 60.000           | 0.632             | 0.999             | 15893.891         | 1641.075                 |

Wall seconds per frame:

| dataset       | wall_seconds_q05   | wall_seconds_q50   | wall_seconds_q95   | wall_seconds_max   |
|:--------------|:-------------------|:-------------------|:-------------------|:-------------------|
| dronerf       | 0.853              | 0.946              | 3.662              | 6.396              |
| birdclef2023  | 0.483              | 1.027              | 3.062              | 4.268              |
| esc50_animals | 0.645              | 1.617              | 3.126              | 4.198              |
| synthetic     | 0.083              | 0.098              | 0.131              | 0.174              |

Primary-task class failure percentages below count stage-channel fits. All secondary tasks, iteration quantiles by window, and delta-ratio quantiles by window are in the external CSV files.

| dataset       | class     | n_channels   | unconverged_stage_pct   | any_unconverged_channel_pct   | H_median   |
|:--------------|:----------|:-------------|:------------------------|:------------------------------|:-----------|
| dronerf       | 00000     | 2400         | 1.645                   | 3.042                         | 2.121      |
| dronerf       | 10000     | 2400         | 7.480                   | 9.833                         | 2.020      |
| dronerf       | 10001     | 2400         | 7.265                   | 8.833                         | 2.066      |
| dronerf       | 10010     | 2400         | 7.176                   | 8.792                         | 2.020      |
| dronerf       | 10011     | 2400         | 7.081                   | 8.542                         | 2.052      |
| dronerf       | 10100     | 2400         | 4.941                   | 6.042                         | 2.061      |
| dronerf       | 10101     | 2400         | 6.127                   | 7.375                         | 2.041      |
| dronerf       | 10110     | 2400         | 8.505                   | 10.042                        | 1.979      |
| dronerf       | 10111     | 2400         | 6.681                   | 7.708                         | 2.086      |
| dronerf       | 11000     | 2400         | 10.093                  | 13.750                        | 2.072      |
| birdclef2023  | barswa    | 750          | 13.608                  | 36.667                        | 1.501      |
| birdclef2023  | blakit1   | 750          | 4.220                   | 12.667                        | 1.065      |
| birdclef2023  | cohmar1   | 750          | 20.275                  | 45.200                        | 1.423      |
| birdclef2023  | combul2   | 750          | 11.294                  | 27.467                        | 1.042      |
| birdclef2023  | combuz1   | 750          | 3.812                   | 8.933                         | 0.979      |
| birdclef2023  | comsan    | 750          | 9.647                   | 21.867                        | 1.718      |
| birdclef2023  | eaywag1   | 750          | 11.271                  | 23.333                        | 1.841      |
| birdclef2023  | egygoo    | 750          | 7.325                   | 24.667                        | 0.740      |
| birdclef2023  | eubeat1   | 750          | 16.855                  | 31.200                        | 0.774      |
| birdclef2023  | grecor    | 750          | 19.388                  | 61.867                        | 0.484      |
| birdclef2023  | greegr    | 750          | 28.447                  | 71.067                        | 0.602      |
| birdclef2023  | hoopoe    | 750          | 28.965                  | 67.067                        | 0.271      |
| birdclef2023  | litegr    | 750          | 12.455                  | 41.867                        | 0.621      |
| birdclef2023  | thrnig1   | 750          | 16.478                  | 35.467                        | 0.970      |
| birdclef2023  | wlwwar    | 750          | 4.337                   | 8.800                         | 1.456      |
| birdclef2023  | woosan    | 750          | 16.408                  | 30.133                        | 1.301      |
| esc50_animals | cat       | 510          | 6.367                   | 20.392                        | 0.667      |
| esc50_animals | cow       | 510          | 30.669                  | 86.667                        | 0.263      |
| esc50_animals | crow      | 510          | 3.379                   | 11.961                        | 0.703      |
| esc50_animals | dog       | 510          | 44.037                  | 89.412                        | 0.337      |
| esc50_animals | frog      | 510          | 36.194                  | 81.373                        | 0.756      |
| esc50_animals | hen       | 510          | 25.594                  | 61.961                        | 0.464      |
| esc50_animals | insects   | 510          | 23.656                  | 67.451                        | 0.328      |
| esc50_animals | pig       | 510          | 28.385                  | 57.059                        | 0.466      |
| esc50_animals | rooster   | 510          | 5.156                   | 17.647                        | 0.626      |
| esc50_animals | sheep     | 510          | 12.226                  | 56.275                        | 0.517      |
| synthetic     | am        | 300          | 4.956                   | 65.667                        | 1.176      |
| synthetic     | chirp     | 300          | 4.933                   | 66.667                        | 1.127      |
| synthetic     | fm        | 300          | 5.200                   | 66.000                        | 1.184      |
| synthetic     | tone      | 300          | 4.711                   | 64.000                        | 1.186      |
| synthetic     | transient | 300          | 4.733                   | 65.667                        | 1.137      |

## Why gradient descent stops late

The installed `_gd_fit_windows` starts at the **unweighted** window median, although the objective uses squared-triangle weights. It sets step = 0.95 H / sqrt(2/pi), so |delta| <= 1.190648 H. Small H relative to a burst or a bimodal window can therefore impose many transit steps. More generally, the local convergence factor is approximately 1 - 0.95 sum(w psi') / [sqrt(2/pi)/H]; it approaches one in flat score regions.

The single global maximum-delta criterion waits for the slowest of all L windows. Every row keeps being evaluated until that happens. Its tolerance uses the largest fitted value anywhere in the stage, so a small last update is also not a uniform bound on distance to each root.

The study computes H once as 2 median(|diff(y)|) / (0.6745 sqrt(2)), on the robustly normalized input channel, then reuses it through the cascade. This is an uncentered median absolute difference, not median(|diff(y) - median(diff(y))|). The mean-absolute-difference fallback does not occur in these caches. Smooth, high-amplitude signals can have a small difference-based H despite large local spread.

At window 7, radius 2 sqrt(2), the center weight is 0.49816692. A window with center 100 and all other values 0, H=1, has score root 2.68259364. GD60 returns 2.43134204, an error of 0.251252; GD1000 still has error 7.79841e-07. This verifies the flat-curvature mechanism without relying on a transit-distance explanation.

The next table uses five hardest subset frames per dataset. Roots are solved on the exact inputs seen by each default stage, before its residual is updated. It reports the largest initial median-to-root distance in H units, the strongest transit lower bound, the smallest relative curvature, and the median percentage of still-active rows in failed stages.

| dataset       | distance_H_max   | move_cap_min_steps_max   | relative_curvature_min   | failed_active_row_pct_median   |
|:--------------|:-----------------|:-------------------------|:-------------------------|:-------------------------------|
| dronerf       | 180.71429        | 151.77805                | 4.142e-16                | 25.48828                       |
| birdclef2023  | 421.27099        | 353.81644                | 0.000e+00                | 14.94141                       |
| esc50_animals | 27.73763         | 23.29624                 | 1.178e-03                | 18.16406                       |

## Subset sensitivity

Seed 20261020, ten frames per primary class sampled across groups, first channel: 100 DroneRF, 160 BirdCLEF, 100 ESC-50. GD60, GD300, and GD1000 call unchanged `emdirmf.decompose.irmf_components`. Every setting except the iteration cap is identical. Newton uses the same H, kernel, wrap boundary, schedule, and storage order.

The fresh GD60 decompositions reproduce the stored float32 cache bit for bit:

| dataset       | n   | bitwise_matches_float32   | cache_max_abs_difference   |
|:--------------|:----|:--------------------------|:---------------------------|
| dronerf       | 100 | 100                       | 0.000                      |
| birdclef2023  | 160 | 160                       | 0.000                      |
| esc50_animals | 100 | 100                       | 0.000                      |

Unconverged stage counts and current CPU seconds per channel:

| dataset       | method   | n   | unconverged_stage_pct   | frames_any_unconverged_pct   | cpu_seconds_mean   | wall_seconds_mean   |
|:--------------|:---------|:----|:------------------------|:-----------------------------|:-------------------|:--------------------|
| birdclef2023  | gd1000   | 160 | 0.735                   | 3.125                        | 1.410              | 1.486               |
| birdclef2023  | gd300    | 160 | 1.765                   | 7.500                        | 1.106              | 1.163               |
| birdclef2023  | gd60     | 160 | 12.022                  | 31.250                       | 0.846              | 0.886               |
| birdclef2023  | newton   | 160 | 0.000                   | 0.000                        | 0.048              | 0.051               |
| dronerf       | gd1000   | 100 | 6.882                   | 12.000                       | 2.389              | 2.646               |
| dronerf       | gd300    | 100 | 9.294                   | 12.000                       | 1.225              | 1.333               |
| dronerf       | gd60     | 100 | 12.000                  | 14.000                       | 0.552              | 0.611               |
| dronerf       | newton   | 100 | 0.000                   | 0.000                        | 0.047              | 0.052               |
| esc50_animals | gd1000   | 100 | 0.412                   | 2.000                        | 1.881              | 2.066               |
| esc50_animals | gd300    | 100 | 2.176                   | 12.000                       | 1.690              | 1.864               |
| esc50_animals | gd60     | 100 | 23.000                  | 57.000                       | 1.305              | 1.434               |
| esc50_animals | newton   | 100 | 0.000                   | 0.000                        | 0.056              | 0.061               |

Component relative L2 errors, each component including the residual counted separately. The input-relative column divides by the input frame norm and limits the influence of tiny components.

| dataset       | method   | reference   | component_rel_l2_median   | component_rel_l2_p95   | component_rel_l2_max   | component_pct_above_1pct   | input_rel_l2_p95   |
|:--------------|:---------|:------------|:--------------------------|:-----------------------|:-----------------------|:---------------------------|:-------------------|
| dronerf       | gd1000   | newton      | 0.000001                  | 0.085582               | 155.666464             | 7.277778                   | 0.005552           |
| dronerf       | gd300    | gd1000      | 0.000000                  | 0.249649               | 17.811100              | 8.333333                   | 0.009864           |
| dronerf       | gd300    | newton      | 0.000001                  | 0.293136               | 313.675286             | 8.444444                   | 0.012568           |
| dronerf       | gd60     | gd1000      | 0.000000                  | 0.782936               | 73.127915              | 11.000000                  | 0.038626           |
| dronerf       | gd60     | newton      | 0.000001                  | 0.788529               | 479.977682             | 11.000000                  | 0.046587           |
| birdclef2023  | gd1000   | newton      | 0.000003                  | 0.000085               | 1.839987               | 0.972222                   | 0.000001           |
| birdclef2023  | gd300    | gd1000      | 0.000000                  | 0.000017               | 17.015044              | 1.840278                   | 0.000001           |
| birdclef2023  | gd300    | newton      | 0.000003                  | 0.000173               | 17.024795              | 1.909722                   | 0.000002           |
| birdclef2023  | gd60     | gd1000      | 0.000000                  | 0.017306               | 161.873084             | 5.625000                   | 0.000661           |
| birdclef2023  | gd60     | newton      | 0.000004                  | 0.017510               | 161.967328             | 5.659722                   | 0.000665           |
| esc50_animals | gd1000   | newton      | 0.000005                  | 0.000044               | 0.064578               | 0.388889                   | 0.000002           |
| esc50_animals | gd300    | gd1000      | 0.000000                  | 0.000265               | 0.624703               | 1.722222                   | 0.000011           |
| esc50_animals | gd300    | newton      | 0.000005                  | 0.000261               | 0.625461               | 1.722222                   | 0.000011           |
| esc50_animals | gd60     | gd1000      | 0.000000                  | 0.039031               | 1.156692               | 8.333333                   | 0.002076           |
| esc50_animals | gd60     | newton      | 0.000014                  | 0.039049               | 1.156692               | 8.333333                   | 0.002077           |

Whole component-stack relative L2 errors against Newton:

| dataset       | method   | median   | p95      | maximum   |
|:--------------|:---------|:---------|:---------|:----------|
| birdclef2023  | gd1000   | 0.000001 | 0.000016 | 0.215669  |
| birdclef2023  | gd300    | 0.000001 | 0.000782 | 0.685870  |
| birdclef2023  | gd60     | 0.000001 | 0.053937 | 1.290549  |
| dronerf       | gd1000   | 0.000000 | 0.107401 | 0.641341  |
| dronerf       | gd300    | 0.000000 | 0.299402 | 1.128265  |
| dronerf       | gd60     | 0.000000 | 0.909763 | 1.577453  |
| esc50_animals | gd1000   | 0.000003 | 0.000016 | 0.019081  |
| esc50_animals | gd300    | 0.000003 | 0.001689 | 0.085034  |
| esc50_animals | gd60     | 0.000015 | 0.067525 | 0.383580  |

At max_iter=60, the 95th-percentile whole-stack relative errors are birdclef2023 5.39%, dronerf 90.98%, esc50_animals 6.75%. The fractions of feature columns with Spearman correlation below .95 are dronerf 18.9%, birdclef2023 0.0%, esc50_animals 1.7%. Exact reconstruction therefore does not imply stable components or stable feature values.

Features are the unchanged 20 emdirmf values per slot. Spearman correlations compare each feature across subset frames. NRMSE divides feature RMSE by that feature's across-frame reference SD. Constant reference columns are omitted. The shared log_scale is identical by construction. The complete per-feature MAE, maximum error, and rank correlation are in the external cache.

| dataset       | method   | reference   | feature_set   | rho_median   | rho_p05   | rho_min   | nrmse_median   | nrmse_p95   | features_rho_below_0.95_pct   |
|:--------------|:---------|:------------|:--------------|:-------------|:----------|:----------|:---------------|:------------|:------------------------------|
| dronerf       | gd1000   | newton      | components    | 1.000        | 0.950     | 0.803     | 0.064          | 0.635       | 5.000                         |
| dronerf       | gd1000   | newton      | +scale        | 1.000        | 0.950     | 0.803     | 0.062          | 0.635       | 4.986                         |
| dronerf       | gd300    | gd1000      | components    | 0.999        | 0.931     | 0.794     | 0.125          | 0.748       | 6.944                         |
| dronerf       | gd300    | gd1000      | +scale        | 0.999        | 0.932     | 0.794     | 0.124          | 0.744       | 6.925                         |
| dronerf       | gd300    | newton      | components    | 0.999        | 0.912     | 0.749     | 0.154          | 0.966       | 8.333                         |
| dronerf       | gd300    | newton      | +scale        | 0.999        | 0.912     | 0.749     | 0.154          | 0.962       | 8.310                         |
| dronerf       | gd60     | gd1000      | components    | 0.996        | 0.827     | 0.663     | 0.259          | 1.085       | 20.000                        |
| dronerf       | gd60     | gd1000      | +scale        | 0.996        | 0.827     | 0.663     | 0.259          | 1.084       | 19.945                        |
| dronerf       | gd60     | newton      | components    | 0.996        | 0.837     | 0.527     | 0.255          | 1.219       | 18.889                        |
| dronerf       | gd60     | newton      | +scale        | 0.996        | 0.838     | 0.527     | 0.254          | 1.218       | 18.837                        |
| birdclef2023  | gd1000   | newton      | components    | 1.000        | 0.998     | 0.964     | 0.005          | 0.320       | 0.000                         |
| birdclef2023  | gd1000   | newton      | +scale        | 1.000        | 0.998     | 0.964     | 0.005          | 0.319       | 0.000                         |
| birdclef2023  | gd300    | gd1000      | components    | 1.000        | 0.995     | 0.968     | 0.023          | 0.412       | 0.000                         |
| birdclef2023  | gd300    | gd1000      | +scale        | 1.000        | 0.995     | 0.968     | 0.023          | 0.412       | 0.000                         |
| birdclef2023  | gd300    | newton      | components    | 1.000        | 0.993     | 0.964     | 0.033          | 0.560       | 0.000                         |
| birdclef2023  | gd300    | newton      | +scale        | 1.000        | 0.993     | 0.964     | 0.033          | 0.558       | 0.000                         |
| birdclef2023  | gd60     | gd1000      | components    | 1.000        | 0.983     | 0.949     | 0.085          | 0.721       | 0.278                         |
| birdclef2023  | gd60     | gd1000      | +scale        | 1.000        | 0.983     | 0.949     | 0.085          | 0.721       | 0.277                         |
| birdclef2023  | gd60     | newton      | components    | 0.999        | 0.982     | 0.959     | 0.082          | 0.800       | 0.000                         |
| birdclef2023  | gd60     | newton      | +scale        | 0.999        | 0.982     | 0.959     | 0.081          | 0.799       | 0.000                         |
| esc50_animals | gd1000   | newton      | components    | 1.000        | 1.000     | 1.000     | 0.000          | 0.003       | 0.000                         |
| esc50_animals | gd1000   | newton      | +scale        | 1.000        | 1.000     | 1.000     | 0.000          | 0.003       | 0.000                         |
| esc50_animals | gd300    | gd1000      | components    | 1.000        | 0.999     | 0.955     | 0.003          | 0.049       | 0.000                         |
| esc50_animals | gd300    | gd1000      | +scale        | 1.000        | 0.999     | 0.955     | 0.003          | 0.049       | 0.000                         |
| esc50_animals | gd300    | newton      | components    | 1.000        | 0.999     | 0.955     | 0.003          | 0.050       | 0.000                         |
| esc50_animals | gd300    | newton      | +scale        | 1.000        | 0.999     | 0.955     | 0.003          | 0.050       | 0.000                         |
| esc50_animals | gd60     | gd1000      | components    | 1.000        | 0.990     | 0.896     | 0.029          | 0.245       | 1.667                         |
| esc50_animals | gd60     | gd1000      | +scale        | 1.000        | 0.990     | 0.896     | 0.029          | 0.244       | 1.662                         |
| esc50_animals | gd60     | newton      | components    | 1.000        | 0.990     | 0.896     | 0.029          | 0.245       | 1.667                         |
| esc50_animals | gd60     | newton      | +scale        | 1.000        | 0.990     | 0.896     | 0.029          | 0.244       | 1.662                         |

## Classification sensitivity

Exploratory fixed shrinkage LDA, preprocessing learned in each training fold, same saved 5-fold x 3-seed group splits intersected with the subset. DroneRF uses its available binary study split and labels; BirdCLEF uses species and ESC-50 uses animal labels. Some subset test folds lack classes. Balanced accuracy uses present test classes; macro-F1 follows the shared metric implementation. The small sample cannot establish whether IRMF outperforms EMD.

Subset fold support:

| dataset       | n_train_min   | n_train_max   | n_test_min   | n_test_max   | test_classes_min   | test_classes_max   |
|:--------------|:--------------|:--------------|:-------------|:-------------|:-------------------|:-------------------|
| birdclef2023  | 118           | 137           | 23           | 42           | 10                 | 16                 |
| dronerf       | 76            | 85            | 15           | 24           | 1                  | 2                  |
| esc50_animals | 74            | 84            | 16           | 26           | 7                  | 10                 |

| dataset       | feature_set    | method   | balanced_accuracy   | macro_f1   | accuracy   |
|:--------------|:---------------|:---------|:--------------------|:-----------|:-----------|
| birdclef2023  | comp           | gd1000   | 0.326               | 0.265      | 0.306      |
| birdclef2023  | comp           | gd300    | 0.328               | 0.265      | 0.306      |
| birdclef2023  | comp           | gd60     | 0.310               | 0.246      | 0.290      |
| birdclef2023  | comp           | newton   | 0.332               | 0.272      | 0.313      |
| birdclef2023  | comp+log_scale | gd1000   | 0.328               | 0.263      | 0.309      |
| birdclef2023  | comp+log_scale | gd300    | 0.329               | 0.264      | 0.310      |
| birdclef2023  | comp+log_scale | gd60     | 0.308               | 0.237      | 0.289      |
| birdclef2023  | comp+log_scale | newton   | 0.332               | 0.265      | 0.313      |
| dronerf       | comp           | gd1000   | 0.889               | 0.798      | 0.918      |
| dronerf       | comp           | gd300    | 0.885               | 0.788      | 0.911      |
| dronerf       | comp           | gd60     | 0.889               | 0.781      | 0.916      |
| dronerf       | comp           | newton   | 0.914               | 0.809      | 0.932      |
| dronerf       | comp+log_scale | gd1000   | 0.887               | 0.792      | 0.914      |
| dronerf       | comp+log_scale | gd300    | 0.885               | 0.788      | 0.911      |
| dronerf       | comp+log_scale | gd60     | 0.883               | 0.774      | 0.908      |
| dronerf       | comp+log_scale | newton   | 0.914               | 0.824      | 0.936      |
| esc50_animals | comp           | gd1000   | 0.423               | 0.351      | 0.412      |
| esc50_animals | comp           | gd300    | 0.425               | 0.355      | 0.415      |
| esc50_animals | comp           | gd60     | 0.439               | 0.364      | 0.408      |
| esc50_animals | comp           | newton   | 0.423               | 0.351      | 0.412      |
| esc50_animals | comp+log_scale | gd1000   | 0.439               | 0.374      | 0.427      |
| esc50_animals | comp+log_scale | gd300    | 0.436               | 0.372      | 0.426      |
| esc50_animals | comp+log_scale | gd60     | 0.442               | 0.368      | 0.413      |
| esc50_animals | comp+log_scale | newton   | 0.439               | 0.374      | 0.427      |

Paired balanced-accuracy differences are method_a minus method_b. Nadeau-Bengio corrected resampled t-test and its corrected 95% CI are primary, Wilcoxon secondary. Holm adjustment covers all 30 rows of this table. Separate macro-F1 and accuracy families, with the same 30-row correction, are retained in classification_comparisons.csv.

| dataset       | feature_set    | method_a   | method_b   | mean_diff   | ci_low   | ci_high   | p_nb_holm   | p_wilcoxon_holm   |
|:--------------|:---------------|:-----------|:-----------|:------------|:---------|:----------|:------------|:------------------|
| birdclef2023  | comp           | gd60       | gd1000     | -0.016      | -0.048   | 0.016     | 1.000       | 1.000             |
| birdclef2023  | comp           | gd300      | gd1000     | 0.002       | -0.014   | 0.018     | 1.000       | 1.000             |
| birdclef2023  | comp           | gd60       | newton     | -0.022      | -0.047   | 0.003     | 1.000       | 0.113             |
| birdclef2023  | comp           | gd300      | newton     | -0.004      | -0.019   | 0.010     | 1.000       | 1.000             |
| birdclef2023  | comp           | gd1000     | newton     | -0.006      | -0.027   | 0.015     | 1.000       | 1.000             |
| birdclef2023  | comp+log_scale | gd60       | gd1000     | -0.020      | -0.046   | 0.006     | 1.000       | 0.113             |
| birdclef2023  | comp+log_scale | gd300      | gd1000     | 0.001       | -0.005   | 0.008     | 1.000       | 1.000             |
| birdclef2023  | comp+log_scale | gd60       | newton     | -0.024      | -0.048   | 0.000     | 1.000       | 0.059             |
| birdclef2023  | comp+log_scale | gd300      | newton     | -0.003      | -0.018   | 0.013     | 1.000       | 1.000             |
| birdclef2023  | comp+log_scale | gd1000     | newton     | -0.004      | -0.017   | 0.009     | 1.000       | 1.000             |
| dronerf       | comp           | gd60       | gd1000     | 0.000       | -0.027   | 0.028     | 1.000       | 1.000             |
| dronerf       | comp           | gd300      | gd1000     | -0.004      | -0.020   | 0.013     | 1.000       | 1.000             |
| dronerf       | comp           | gd60       | newton     | -0.025      | -0.109   | 0.060     | 1.000       | 1.000             |
| dronerf       | comp           | gd300      | newton     | -0.028      | -0.122   | 0.065     | 1.000       | 1.000             |
| dronerf       | comp           | gd1000     | newton     | -0.025      | -0.112   | 0.062     | 1.000       | 1.000             |
| dronerf       | comp+log_scale | gd60       | gd1000     | -0.004      | -0.023   | 0.015     | 1.000       | 1.000             |
| dronerf       | comp+log_scale | gd300      | gd1000     | -0.001      | -0.015   | 0.012     | 1.000       | 1.000             |
| dronerf       | comp+log_scale | gd60       | newton     | -0.031      | -0.116   | 0.054     | 1.000       | 0.844             |
| dronerf       | comp+log_scale | gd300      | newton     | -0.029      | -0.123   | 0.065     | 1.000       | 1.000             |
| dronerf       | comp+log_scale | gd1000     | newton     | -0.027      | -0.114   | 0.059     | 1.000       | 1.000             |
| esc50_animals | comp           | gd60       | gd1000     | 0.016       | -0.056   | 0.087     | 1.000       | 1.000             |
| esc50_animals | comp           | gd300      | gd1000     | 0.002       | -0.008   | 0.013     | 1.000       | 1.000             |
| esc50_animals | comp           | gd60       | newton     | 0.016       | -0.056   | 0.087     | 1.000       | 1.000             |
| esc50_animals | comp           | gd300      | newton     | 0.002       | -0.008   | 0.013     | 1.000       | 1.000             |
| esc50_animals | comp           | gd1000     | newton     | 0.000       | 0.000    | 0.000     | 1.000       | 1.000             |
| esc50_animals | comp+log_scale | gd60       | gd1000     | 0.003       | -0.072   | 0.079     | 1.000       | 1.000             |
| esc50_animals | comp+log_scale | gd300      | gd1000     | -0.003      | -0.028   | 0.023     | 1.000       | 1.000             |
| esc50_animals | comp+log_scale | gd60       | newton     | 0.003       | -0.072   | 0.079     | 1.000       | 1.000             |
| esc50_animals | comp+log_scale | gd300      | newton     | -0.003      | -0.028   | 0.023     | 1.000       | 1.000             |
| esc50_animals | comp+log_scale | gd1000     | newton     | 0.000       | 0.000    | 0.000     | 1.000       | 1.000             |

For GD60 versus Newton, 0 of six balanced-accuracy comparisons reject after correction. A nonsignificant result on this small subset is not evidence of equivalence. Full-study classification sensitivity remains to be measured.

## Prototype accuracy and speed

`NewtonSmoothAbs` uses pimf's public closed-form-solver hook to preserve the existing driver. It starts at the weighted mean, brackets each score root, uses the exact lookup-table slope for Newton, and falls back to bisection for unsafe steps or every eighth iteration. Independent row stopping and a fused single-threaded Numba loop avoid repeated work and large broadcast temporaries. A row failing its stopping test raises an error.

Important numerical distinction: lookup=True's score is piecewise linear and is not the exact derivative of the interpolated loss returned by SmoothAbs.__call__. This prototype matches the score equation actually used by pimf. In saturated floating-point tails it can have flat root intervals; the ideal analytic SmoothAbs objective is strictly convex. No equivalence to the literal interpolated loss minimizer is claimed.

Independent SciPy brentq checks cover 240 windows at H=.01, 1, 10 and widths 5, 7, 65, 1025, including constants and center outliers. Maximum absolute root discrepancy 1.76e-13.

Timing signals are fixed noisy and smooth/bursty examples at every L, plus a bimodal stress case at L=1024. Its 20000-iteration failure motivated stopping an in-progress L=2048 bimodal repeat and omitting larger stress repeats; they are not accuracy checks. Tight-request pimf uses max_iter=20000 and tol=1e-12, prototype root_tol=1e-12. JIT compilation is excluded. Each cell is one measured run, so speed ratios are descriptive; the real-frame subset above supplies a larger timing sample. CPU times reduce interference from concurrent machine work. A tight-GD run with any unconverged stages is not an accuracy reference; its runtime is a lower bound on the time to convergence. Such stress cases remain in the table, with failure counts, rather than being silently dropped.

| L    | signal   | gd60_cpu_s   | tight_cpu_s   | newton_cpu_s   | speedup_vs_gd60   | speedup_vs_tight   | newton_rel_l2   | max_component_rel_l2   | tight_unconverged   |
|:-----|:---------|:-------------|:--------------|:---------------|:------------------|:-------------------|:----------------|:-----------------------|:--------------------|
| 1024 | bimodal  | 0.5033       | 45.7719       | 0.0169         | 29.7875           | 2709.1962          | 7.302e-07       | 3.438e-05              | 1                   |
| 1024 | bursty   | 0.4361       | 5.1841        | 0.0147         | 29.6934           | 352.9993           | 1.555e-11       | 7.923e-11              | 0                   |
| 1024 | noisy    | 0.0912       | 0.1626        | 0.0119         | 7.6566            | 13.6541            | 2.927e-13       | 2.735e-12              | 0                   |
| 2048 | bursty   | 1.8866       | 21.3598       | 0.0517         | 36.4575           | 412.7652           | 1.552e-11       | 3.102e-10              | 0                   |
| 2048 | noisy    | 0.3848       | 0.6728        | 0.0454         | 8.4724            | 14.8121            | 2.445e-13       | 2.522e-12              | 0                   |
| 4096 | bursty   | 8.1806       | 87.5672       | 0.1880         | 43.5033           | 465.6715           | 1.316e-11       | 4.881e-10              | 0                   |
| 4096 | noisy    | 1.5640       | 2.8261        | 0.1727         | 9.0573            | 16.3669            | 2.794e-13       | 3.947e-12              | 0                   |

Across the 6 fully converged tight-GD references, maximum Newton stack relative L2 discrepancy is 1.55e-11; maximum individual-component relative discrepancy is 4.88e-10. Only those rows validate agreement with tightly converged pimf.

Largest reconstruction absolute error in the benchmark: 3.55e-15.

## Recommendation and full-study cost

Keep the default GD60 tag as the protocol's primary result. Add a fully converged solver as `irmf_h2_a1.41_conv`, with separate provenance and convergence diagnostics, then repeat the full paired classification analysis with and without log_scale. GD1000 is not an unconditional accuracy reference. Class-dependent solver error can itself become a classification feature. The proposed `_conv` tag needs dispatcher/parser support before use; those shared modules were intentionally left untouched under this task's scope restriction.

For an upstream pimf change, expose a public lookup-compatible score derivative, retain the reference solver, and make safeguarded Newton an explicit option. Numba acceleration should be optional. Existing StageInfo reports iterations=1 on the solve hook; use the prototype's own per-stage diagnostics when recording Newton work. No installed or source pimf files were changed.

Cost estimate uses mean measured Newton CPU seconds, 12,000 two-channel DroneRF frames, 12,000 mono BirdCLEF, and 5,100 mono ESC-50, all L=2048. DroneRF's second-channel estimate comes from 34 additional H-band frames. Parallel estimates assume linear scaling and exclude normalization, feature extraction, I/O, JIT startup, and scheduling contention.

| dataset       | frames   | cpu_s_per_frame   | one_core_minutes   | two_core_minutes   | six_core_minutes   |
|:--------------|:---------|:------------------|:-------------------|:-------------------|:-------------------|
| dronerf       | 12000    | 0.090             | 18.066             | 9.033              | 3.011              |
| birdclef2023  | 12000    | 0.048             | 9.582              | 4.791              | 1.597              |
| esc50_animals | 5100     | 0.056             | 4.794              | 2.397              | 0.799              |

Total: 32.4 CPU-minutes, 16.2 minutes on two cores or 5.4 minutes on six cores under ideal scaling. Allowing 2x for overhead and load gives a conservative planning range. The full float32 decomposition cache would occupy about 6.06 GB decimal, before metadata.

## Files and reproducibility

Code: `newton.py`, `run_study.py`, `write_report.py`; instructions: `README.md`. The README lists exact reproduction commands. No git commit was made. All numerical tables and checkpoints remain under `/Users/mikhail/Datasets/imf-classification/solver_audit`. Top-level files include diagnosis_overview.csv, diagnosis_stage.csv, diagnosis_class.csv, diagnosis_class_stage.csv, component_summary.csv, feature_summary.csv, stack_differences.csv, cache_reproduction.csv, classification_folds.csv, classification_comparisons.csv, classification_summary.csv, benchmark.csv, benchmark_summary.csv, mechanism.csv, validation.json, and full_cost_estimate.csv. Each dataset subfolder holds selection.json, frame NPZ checkpoints, feature and component differences, costs.csv, and features.npz.
