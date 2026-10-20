# Phase-3 reproduction and choices

Run from `/Users/mikhail/.t3/worktrees/imf/t3-44ed3594`. Python is
`/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python`.
Data root is `/Users/mikhail/Datasets/imf-classification`.

## Budgets fixed after timing

The pilot uses animal split 0, raw and default IRMF, both log-scale variants,
one CPU thread per runner, with three concurrent runners. Trees take 45.9 s
with 60 boosting rounds and 64 ExtraTrees trees. CNN takes 24.5 s with two
epochs, width 8, downsampling 2. MiniRocket takes 69.0 s at 1,008 features,
five head passes and 128 bias-fit frames. The multichannel transform dominates
MiniRocket; the IRMF fold takes about 58 s including prediction and shared
transform work. Pilot scores did not select methods, folds or hyperparameters.

Main budgets apply to all five inputs and both feature sets:

| Model | Outer folds | Fixed budget |
|---|---:|---|
| LightGBM | 15 | at most 150 rounds, 15 leaves, patience 12 |
| XGBoost | 15 | at most 60 rounds, depth 4, patience 12 |
| ExtraTrees | 15 | 64 trees, inner leaf grid 1/3 |
| MiniRocket | 15 | 1,008 features, five head passes, 128 bias-fit frames |
| Compact residual CNN | 5 | seed 0, width 8, at most 20 epochs, patience 4 |

Trees use the existing component feature parquets. MiniRocket preserves the
2,048 time samples and the total transformed feature budget is independent of
component count. CNN applies the same anti-aliased downsampling by 2 to every
method and channel. All other architecture, optimizer and training settings
are those in `ml/README.md`. These are restricted-budget experiments, not
fully tuned accuracy ceilings.

The five main methods are raw, EMD, linear IMF, default GD60 IRMF and converged
Newton IRMF, the last three at a=sqrt(2), H=2 where applicable. The converged
H=2 and H=1 data and features were complete at the starting checkpoint.
Other sensitivity tags are checked again after the main jobs.

Main jobs initially use two CPU runners and one MPS CNN, each with one CPU
thread. After CNN finishes, its CPU slot runs supplemental models and its MPS
slot runs the raw log-STFT reference. At most three CPU runners plus one MPS
runner are active. Folds are serial and DataLoader uses zero child workers.
Each queue runs animal then bird. Other datasets share the machine and MPS.

`worker_plan.json`, when present, reallocates CPU threads between completed
queues before the next runner/task begins. Effective workers are saved in each
run JSON. This changes hardware allocation only, never model capacity, folds,
features or tuning. The plan is populated only when the other main CPU/CNN
jobs and MultiRocket have finished, leaving one supplemental CPU worker.

Supplemental MultiRocket uses split IDs 0-4 for every method, target 1,008
features, five head passes and 128 bias-fit frames. aeon rounds this to 672
actual output features. It has its own 20-test Holm family per task/metric.
The raw log-STFT reference uses five folds, both log-scale variants, width 16,
20 epochs and patience 4. Its 129x33 images use a 256-sample Hann window,
64-sample hop, zero boundary padding, and log1p squared STFT magnitude after
the shared frame normalization. Three 2-D convolution/GroupNorm/ReLU/pooling
blocks train from scratch with the same grouped validation and refit loop.
This is a separate architecture/representation reference, not a paired
decomposition contrast.

## Commands

```bash
PY=/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python
RUN=/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/esc50_animals/run_phase3.py
OUT=/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/output/classification-emd-irmf/esc50_animals/ml
for TASK in animal bird; do
  "$PY" "$RUN" trees --dataset esc50_animals --task "$TASK" \
    --methods raw emd imf_linear_a1.41 irmf_h2_a1.41 irmf_h2_a1.41_conv \
    --folds 15 --workers 1 --out "$OUT/trees" --lightgbm-rounds 150 \
    --cheap-extra-trees --classifiers lightgbm xgboost extratrees
done
for TASK in animal bird; do
  "$PY" "$RUN" rocket --dataset esc50_animals --task "$TASK" \
    --methods raw emd imf_linear_a1.41 irmf_h2_a1.41 irmf_h2_a1.41_conv \
    --folds 15 --workers 1 --out "$OUT/rocket" --features 1008 \
    --head-epochs 5 --fit-samples 128
done
for TASK in animal bird; do
  "$PY" "$RUN" cnn --dataset esc50_animals --task "$TASK" \
    --methods raw emd imf_linear_a1.41 irmf_h2_a1.41 irmf_h2_a1.41_conv \
    --folds 5 --workers 1 --out "$OUT/cnn" --epochs 20 --patience 4 \
    --device mps --downsample 2 --width 8
done
LOCAL=/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/esc50_animals
for TASK in animal bird; do
  "$PY" "$RUN" rocket --dataset esc50_animals --task "$TASK" \
    --methods raw emd imf_linear_a1.41 irmf_h2_a1.41 irmf_h2_a1.41_conv \
    --folds 5 --workers 1 --out "$OUT/multirocket" \
    --run-name esc50_phase3_multirocket --classifiers multirocket \
    --features 1008 --head-epochs 5 --fit-samples 128
done
for TASK in animal bird; do
  "$PY" "$LOCAL/run_spectrogram.py" --dataset esc50_animals --task "$TASK" \
    --methods raw --folds 5 --workers 1 --out "$OUT/spectrogram" \
    --run-name esc50_phase3_spectrogram --epochs 20 --patience 4 --width 16 --device mps
done
"$PY" "$LOCAL/run_supplemental.py" --workers 1
"$PY" "$LOCAL/analyse_phase3.py"
"$PY" "$LOCAL/analyse_sensitivity.py"
"$PY" "$LOCAL/make_overview.py"
```

## Local extensions and inference

No shared ML runner or `emdirmf` implementation is changed by this task.
The dataset-local `run_phase3.py` exports native probabilities and checkpoints,
caps LightGBM rounds, and applies quick budgets only to XGBoost/ExtraTrees.
All train-only imputation, class weights, grouped validation and outer refits
remain in the shared runners. Selected rounds/epochs are recorded per fit.

SGD's one-vs-rest logistic probabilities can saturate into ties. Its decision
score argmax can then differ from the first probability argmax. The export
check verifies that the predicted class attains the maximum probability,
allowing a tied maximum. The classifier labels and probabilities are unmodified.
The final audit reports the number of such labels. CNN and trees use their
native softmax/probability outputs.

`paired_all_metrics.csv` makes one primary table per task and metric, including
all ten method pairs, both feature sets and all five classifier families.
Holm covers 100 tests in each table, separately for Nadeau-Bengio and secondary
Wilcoxon. Corrected 95% intervals use the mean fold test/train ratio and 14 df
for CPU models or 4 df for CNN. These are ordinary corrected intervals, not
simultaneous Holm intervals. The individual runner tables also retain their
original runner-specific Holm families. Headline inference uses the combined
100-test table. `paired_requested.csv` orients the requested contrasts with
IRMF or a decomposition as numerator.

Clip aggregation averages native probabilities within each held-out recording,
then pools all 400 clips per seed. Scores average over three seeds for CPU
models and one for CNN. Every clip belongs to one uploader and outer test fold.
All model families are exported so the best-model clip summary can be shown
without rerunning or changing a classifier. Selecting a best model by outer
BA is exploratory and is not a new confirmatory test.

Scale importance sums normalized tree importance across the same component's
20 statistics. IMF's final storage residual is the finest scale. Split count,
gain and impurity importance are compared only within their model families.
Costs separate cached decomposition seconds/frame from fitting; MiniRocket's
two feature-set timing rows double-count shared transforms, so use runner wall
time for totals. Concurrent jobs and shared MPS affect wall-clock comparisons.

Large predictions and probability checkpoints stay in
`DATA_ROOT/esc50_animals/results/esc50_phase3_*`; pilot files use separate names.
No decomposition is regenerated. At the matched-count readiness check the
linear a=2 feature cache is already present; the missing GD60 IRMF a=2 feature
parquet is generated from its completed decomposition in 6.3 s, with zero
failed/non-finite rows. All three extra converged tags are ready at the
20:23:33 UTC sensitivity checkpoint. Exact commands and stage timings are in
`supplemental_manifest.json`. No commit is made.

The nine-slot ablation retains evenly spaced true scale ranks, rounded to
nearest integer, independently of any scores. The selected IMF/IRMF storage
slots are residual_fine, w7, w13, w25, w47, w129, w257, w513 and w1025.
It keeps 180 component statistics plus optionally one log-scale value, matching
EMD's predictor count. Raw and EMD references reuse their unchanged main
LightGBM fits. This restricts the tree input, not decomposition computation.
The a=2 methods have ten source slots and 200/201 predictors. Fusion has
27 source slots and 540/541 predictors; shared log-scale appears once.
Input-feature counts are exported and checked. All supplemental LightGBM
fits have the same 150-round cap, leaf settings, inner validation and 15 folds
as their main references. Supplemental comparison tables have their own
declared Holm family sizes. Main inference keeps the predeclared 100-test family.
