# ML cost estimates

| dataset       | runner   | classifier     |   target_folds |   total_minutes |   low_minutes |   high_minutes |   pilot_5fold_minutes |
|:--------------|:---------|:---------------|---------------:|----------------:|--------------:|---------------:|----------------------:|
| birdclef2023  | cnn      | compact_resnet |              5 |           262.7 |         131.4 |          525.4 |                  53.5 |
| birdclef2023  | rocket   | minirocket     |             15 |           538.1 |         269.0 |         1076.1 |                  11.4 |
| birdclef2023  | rocket   | multirocket    |             15 |           444.2 |         222.1 |          888.4 |                   9.3 |
| birdclef2023  | trees    | extratrees     |             15 |             9.6 |           4.8 |           19.1 |                   0.7 |
| birdclef2023  | trees    | lightgbm       |             15 |           329.4 |         164.7 |          658.8 |                   9.3 |
| birdclef2023  | trees    | xgboost        |             15 |           541.8 |         270.9 |         1083.6 |                  21.2 |
| dronerf       | cnn      | compact_resnet |              5 |           290.3 |         145.2 |          580.6 |                  59.1 |
| dronerf       | rocket   | minirocket     |             15 |           533.3 |         266.7 |         1066.6 |                  11.5 |
| dronerf       | rocket   | multirocket    |             15 |           387.6 |         193.8 |          775.2 |                   8.3 |
| dronerf       | trees    | extratrees     |             15 |            19.1 |           9.6 |           38.2 |                   1.4 |
| dronerf       | trees    | lightgbm       |             15 |           411.8 |         205.9 |          823.5 |                  11.6 |
| dronerf       | trees    | xgboost        |             15 |           677.2 |         338.6 |         1354.5 |                  26.6 |
| esc50_animals | cnn      | compact_resnet |              5 |           111.6 |          55.8 |          223.3 |                  22.8 |
| esc50_animals | rocket   | minirocket     |             15 |           182.7 |          91.4 |          365.4 |                   3.9 |
| esc50_animals | rocket   | multirocket    |             15 |           140.2 |          70.1 |          280.4 |                   3.0 |
| esc50_animals | trees    | extratrees     |             15 |             4.1 |           2.0 |            8.1 |                   0.3 |
| esc50_animals | trees    | lightgbm       |             15 |            87.5 |          43.7 |          175.0 |                   2.5 |
| esc50_animals | trees    | xgboost        |             15 |           143.9 |          72.0 |          287.8 |                   5.6 |

Planning estimates only; no target dataset arrays or feature tables were read.

All four main tags including raw, both log_scale variants; one primary task per dataset.

pilot_5fold_minutes scales the validated quick budgets and first-seed five folds. CNN pilot budgets six selection plus six refit epochs; tree early-stop patterns are assumed to match synthetic.

Trees/ROCKET: 15 folds. CNN: 5 folds, 30 selection plus 30 refit epochs as a budget ceiling. 15-fold CNN costs three times the shown amount; early stopping can reduce it substantially.

Trees extrapolate to 500 boosting rounds / 300 ExtraTrees trees, including inner selection/refit, with linear N, channel/slot and booster class-count scaling. Synthetic observed rounds are the reference.

CPU transforms/trees use 16 cores at 50% relative scaling efficiency; serial logistic-head updates get no core-count speedup. MPS gets no assumed speedup from more CPU cores.

ROCKET uses 10080 features, 256 bias-fit samples and 15 SGD epochs; transform scales with N, L and sqrt(channel count), head with N and class count. Feature/epoch scaling uses the saved synthetic-size head calibration when available, otherwise linear scaling. Shared transforms are counted once.

CNN uses observed median refit epoch time, scales with N*L, and allocates 10% of time to input-channel scaling. Validation forward cost is approximated as one-third of training forward/backward cost.

Ranges are 0.5x-2x the model estimate, not statistical confidence intervals. Numba compilation, thermal load, disk bandwidth, class count and different early stopping can change actual times.

Decomposition and feature-cache generation are excluded. Existing decomposition jobs must finish first.
