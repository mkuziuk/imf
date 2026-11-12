# Reproduce BirdCLEF phase 3

All commands run from `/Users/mikhail/.t3/worktrees/imf/t3-44ed3594` with absolute paths. Data root is `/Users/mikhail/Datasets/imf-classification`. No cache, split, feature, or decomposition generation is part of this run.

The exact tree commands are saved in `tree_commands_species.json`. The first full LightGBM fold is in `trees_lightgbm_fold0`; `trees_lightgbm_remaining` contains split IDs 1 through 14. They use identical model settings and together cover all 15 shared splits. Other tree families use all 15 splits in one run. Combining the two LightGBM exports recomputes statistics and weights scale importance by individual fitted folds.

Use the root `paired_all_metrics_<task>` tables for the combined analysis. Comparison files inside individual runner folders describe their own split subsets and use their local Holm family.

The pilot used split 0 for every method, both log-scale variants, and two CPU threads per runner. `pilot/trees` used the quick 60-round boosting and 64-tree ExtraTrees budgets. `pilot/rocket` used 672 features and five head epochs. `pilot/cnn` used six selection epochs, patience two, and MPS. A second LightGBM pilot allowed the full 500-round ceiling. Its exported fits are retained as final split 0.

The initial tree pilot took 204.4 s in total (fit/predict sums: LightGBM 81.0 s, XGBoost 109.4 s, ExtraTrees 13.4 s). MiniRocket took 275.0 s and the CNN 119.9 s. These supported full tree budgets and an initial five-fold ceiling for stack models; full-budget LightGBM split 0 took 89.9 s. The final MiniRocket budget remained below the shared runner's 10,080-feature default. CNN coverage was increased only after full-budget timing demonstrated room, using identical settings for every method.

Final common settings:

| Model | Shared splits | Budget |
|---|---|---|
| LightGBM | 0 through 14 | 500 rounds maximum, 15 leaves, learning rate .05, patience 12 |
| ExtraTrees | 0 through 14 | 300 trees, inner selection of leaf size 1 or 3 |
| XGBoost | 0 through 14 | 500 rounds maximum, depth 4, learning rate .05, patience 12 |
| MiniRocket | 0 through 4 | 2,016 transform features, 256 bias-fit samples, 10 SGD epochs |
| MultiRocket | 0 through 4 | 1,344 transform features, 256 bias-fit samples, 10 SGD epochs |
| Compact 1-D ResNet | 0 through 14 | width 16, 30 selection epochs maximum, patience 5, MPS |
| Raw log-STFT 2-D ResNet reference | 0 through 14 | width 16, 30 selection epochs maximum, patience 5, MPS |

Both log-scale variants and all five methods use the same settings within each family. Raw has one stack channel, EMD nine, and the IMF methods 18. All 2,048 time samples are retained. The converged tag was complete before the pilots and appears in every runner. Hyperparameters and epoch counts are selected on grouped inner validation, then refitted on the complete outer training fold.

CNN coverage was expanded from five to 15 folds after observing the per-fit times in the first seed. Its additional split IDs 5 through 14 are exported separately under `cnn_remaining`, then combined by the analysis script. The architecture, training ceiling and stopping rule do not change.

The dataset-local wrapper `experiments/classification-emd-irmf/datasets/birdclef2023/run_phase3.py` delegates all fitting to the shared `ml/` runners. It adds fold checkpoints and confidence exports only. Tree confidence uses native probabilities; CNN confidence uses softmax logits. ROCKET confidence uses a stable softmax of SGD decision scores because native one-versus-rest sigmoid probabilities can saturate and lose the ranking used by `predict`. This confidence is uncalibrated and used only for auxiliary recording aggregation. Frame predictions are unchanged. No shared `ml/` code is edited.

For example, MiniRocket was run with:

```bash
env IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python -u \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/birdclef2023/run_phase3.py rocket \
  --run-name rocket_mini --dataset birdclef2023 --task species \
  --methods raw emd imf_linear_a1.41 irmf_h2_a1.41 irmf_h2_a1.41_conv \
  --classifiers minirocket --features 2016 --head-epochs 10 --folds 5 --workers 2 \
  --out /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/output/classification-emd-irmf/birdclef2023/ml/rocket_mini
```

For the CNN, replace `rocket` with `cnn`, use `--run-name cnn`, remove the ROCKET-specific options, and add `--device mps --epochs 30 --patience 5`. The corresponding output directory is `ml/cnn`.

After all fits, run:

```bash
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/birdclef2023/analyse_phase3.py species
```

The analysis verifies saved scores against predictions, shared test membership, exclusion of recordists across training and test, unique rows, finite normalized confidence, prediction/confidence agreement, and identical shared log-scale values. Recording aggregation averages confidence within each original recording inside each outer test fold. It preserves recordist splits and evaluates both individual folds and pooled out-of-fold recordings per seed.

The analysis combines recording predictions across all runner shards before computing a pooled seed metric. It asserts a unique recording/method/model/log-variant/seed key and exactly 1,280 recordings for each complete seed. An early reporting bug averaged the two partial LightGBM seed-0 exports; this was corrected and checked independently for all 30 LightGBM method/log-variant/seed cells before the final report. Frame scores and paired fold tests were unaffected.

Primary statistics use the Nadeau-Bengio corrected resampled t-test and its 95% interval. Wilcoxon is secondary. Holm correction covers all method pairs, classifiers, and log variants within each task/metric table, separately for frame and recording analyses. Five-fold models have less inferential power than 15-fold models. No inference compares classifier families selected by their test scores. Naming a best model for the recording summary is descriptive.

Model cost totals use each run's measured wall clock under concurrent load. ROCKET shares transform fitting and application between the two log-scale variants; each variant's individual timing row includes that shared work, so summing both rows double-counts transforms. Actual transform dimensions are recorded in its fold tables. Cached decomposition CPU times describe the supplied caches, rather than a new isolated timing benchmark.

Large frame and recording prediction exports, including checkpoints, remain under `DATA_ROOT/birdclef2023/results/`. Repository artifacts contain only code, small tables, reports, and figures. No commit was created.

The exploratory feature checks use all 15 species folds with the same LightGBM ceiling and inner selection. Nine fixed ranks, evenly spaced in true fine-to-coarse order, leave exactly 180 IRMF component features, matching EMD's count. This restricts slots; it does not match their frequency bands. Fusion concatenates all 180 EMD and 360 IRMF features, adding the shared log-scale value once. Both default and converged IRMF are checked. Their 16 prespecified contrasts form a separate Holm family per metric.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python -u \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/birdclef2023/run_feature_checks.py
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/birdclef2023/run_feature_checks.py --analyse
```

`run_stft_reference.py` uses normalized raw frames and fixed log-power STFT images, 129 frequency bins by 15 time bins. The periodic Hann window has 256 samples, hop 128, and no padding. A 2-D residual CNN uses 3x3 kernels, widths 16/32/64, GroupNorm and frequency/time mean/max pooling. It shares the 1-D runner's grouped selection/refit loop, optimizer and stopping rule. It is a raw-only reference, excluded from the decomposition-pair Holm family. Its own audit checks all frame metrics and recording probability means. `--audit-only` repeats that check without training. Its results live in `stft_reference/` and its prediction stem is `cnn_stft`, so it cannot overwrite the 1-D predictions.

The STFT pilot used split 0, six epochs and patience two for both log variants, and took 49.9 s. The full reference uses all 15 folds, 30 epochs and patience five. MPS has one host thread for this reference. The CPU training jobs each use two threads, with at most three CPU model jobs running alongside the separate MPS job.

MultiRocket's one-fold pilot used 672 features and five head epochs, with all five inputs and both scale variants, and took 175.1 s. The final budget was the largest of 2,016/1,344/672 satisfying `pilot_seconds * 5 * feature_ratio * 1.3` before the three-hour deadline, with 15 minutes reserved for reporting. This selected 1,344 features. The same budget and split IDs 0 through 4 apply to every input. Its output directory and prediction stem are `rocket_multi`; the runner invocation matches MiniRocket with `--classifiers multirocket --features 1344`.

The passerine runs use the same prepared frames and their own saved grouped splits. LightGBM uses all 15 folds. MiniRocket uses split IDs 0 through 4 and the same 2,016-feature/10-epoch budget. The CNN uses 30 epochs/patience five with one MPS host thread; its full-budget first-fold pilot took 328.9 s and is retained in `cnn_passerine_fold0`, followed by split IDs 1 through 4 in `cnn_passerine_remaining`. The raw-STFT passerine pilot took 85.9 s for split 0 at the full 30-epoch budget. A 30% timing margin and 15-minute report reserve allowed all 15 folds, exported under `stft_reference` with task `passerine`.

```bash
env IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python -u \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/birdclef2023/run_stft_reference.py \
  --dataset birdclef2023 --task species --methods raw --folds 15 \
  --epochs 30 --patience 5 --workers 1 --device mps \
  --out /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/output/classification-emd-irmf/birdclef2023/ml/stft_reference
```


Passerine ExtraTrees also uses all 15 folds and 300 trees. The one-fold passerine XGBoost pilot took 58.3 s at the full 500-round ceiling. Choosing the largest of 15/5 folds with a 30% timing margin and a 15-minute report reserve allowed all 15 folds (projection 1,136 s). This budget was fixed before the full run and applied to every method and scale variant. Outputs are task-specific files under trees_extratrees/ and trees_xgboost/; nested pilot outputs are excluded from final analysis.

For the final audit and report, repeat analyse_phase3.py for passerine, then execute render_phase3.py and write_phase3_report.py from the same dataset-local directory. report.html is a self-contained interactive score/paired-interval browser; paired_irmf_emd.png and scale_importance.png are standalone figures. All source scripts are syntax-checked. No shared ml/ fix was necessary: confidence export and pooled-shard corrections are limited to the dataset-local helpers.

Final coverage is 700 species and 550 passerine method/fold/variant evaluations in the combined model tables, 120 species feature checks, and 30 raw-STFT evaluations per task: 1,430 final evaluations, excluding discarded nested pilots. Model fitting finished at 22:07 UTC, about 153 minutes after starting. All saved-prediction audits passed. The primary and secondary Holm families contain 120 and 100 method-pair/variant/model comparisons per metric, respectively. The report generator enforces a 500-word limit, including separated table cells.
