#!/bin/zsh
set -eu
ROOT=/Users/mikhail/.t3/worktrees/imf/t3-44ed3594
PY="$ROOT/.venv/bin/python"
RUN="$ROOT/experiments/classification-emd-irmf/datasets/uas_birds_60ghz/run_phase3.py"
OUT="$ROOT/output/classification-emd-irmf/uas_birds_60ghz/ml"
cd "$ROOT"
export IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
RUNNER="$1"
METHODS=(raw emd imf_linear_a1.41 irmf_h2_a1.41 irmf_h2_a1.41_conv irmf_h2_a2_conv imf_linear_a2 irmf_h1_a1.41_conv irmf_h4_a1.41_conv)
EXTRA=()
case "$RUNNER" in
 trees) EXTRA=(--classifiers lightgbm xgboost extratrees) ;;
 rocket) EXTRA=(--classifiers minirocket multirocket --features 1008 --head-epochs 5 --fit-samples 128) ;;
 cnn) EXTRA=(--epochs 20 --patience 4 --width 8 --device mps) ;;
 *) exit 2 ;;
esac
for TASK in bird_drone target3; do
 "$PY" "$RUN" "$RUNNER" --dataset uas_birds_60ghz --task "$TASK" \
  --methods "${METHODS[@]}" --folds 15 --workers 1 --out "$OUT/$RUNNER" \
  --run-name "uas_phase3_$RUNNER" "${EXTRA[@]}" > "$OUT/logs/${RUNNER}_${TASK}.log" 2>&1
done
