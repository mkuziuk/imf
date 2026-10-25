# Impulsive-contamination experiment

Read [the pre-registration](PROTOCOL_robustness.md). This experiment writes only
new robustness outputs and reads prepared frames, clean caches and saved splits.
The excluded pilot subset, seed mapping and analysis families are fixed there.

Run from `experiments/classification-emd-irmf/`:

```bash
export IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=../../.venv/bin/python
$PY -m pytest -q tests robustness/tests
$PY -m robustness.pilot --dataset synthetic --workers 6
$PY -m robustness.pilot --dataset dronerf --workers 6
$PY -m robustness.pilot --dataset birdclef2023 --workers 6
$PY -m robustness.pilot --dataset esc50_animals --workers 6
```

The implementation adds these resumable commands after pre-registration:

```bash
$PY -m robustness.run --dataset synthetic --workers 6
$PY -m robustness.run --dataset dronerf --workers 6
$PY -m robustness.run --dataset birdclef2023 --workers 6
$PY -m robustness.run --dataset esc50_animals --workers 6
$PY -m robustness.analyse
```

For a batch independent of the invoking terminal, run `$PY -m robustness.run_real
--detach`; it prints its PID and log path. After that batch completes,
`$PY -m robustness.run_to_report` runs the required tests, validates a full
report copy, then writes the final report. Both commands use file-backed logs
under `dronerf/robustness/runs/` when launched as background processes.

Completed outputs are reused without replacement. A failed/incompatible checkpoint
requires inspection; no overwrite flag is provided. Full component stacks exist
only for the 100 fixed pilot/figure frames per condition. Numeric NPZ files use
allow_pickle=False; estimators are never pickled. Small report tables and figures
will live in `output/classification-emd-irmf/robustness/`.

The active feature provenance files are `manifest_v2.json` and
`clean_verification_v2.json`. Each `features/<condition>/part_<start>.npz`
contains all four methods' numeric arrays. The failed first raw-precision audit
is preserved separately; it produced no contaminated outputs. The report builder
verifies exact clean reproduction and recomputes all per-split metrics from the
saved Unicode predictions. `run_real` runs the three real datasets serially.

The 2026-10-21 amendment adds all 87 synthetic contrasts in one separate Holm
family. Real-data family membership remains as pre-registered. Every summary
figure includes all four datasets. `analyse --out <new-directory>` can render
another report copy without replacing an existing artifact directory.

## Run notes, 2026-10-21

The first real-data batch used a pipe held by the agent session. A provider
interruption made that pipe a risk for the long run. A detached watcher waited
until all six in-flight linear-IMF fits had published checkpoints, then stopped
the old process group (PIDs 21371 and 21374) and launched the remaining batch
as PID 19215. The batch resumed with 66 completed classifier fits. Its log
shows clean verification passing and all nine already-complete DroneRF feature
conditions being skipped. The completed fit checkpoint hashes were unchanged.

The [restart audit](/Users/mikhail/Datasets/imf-classification/dronerf/robustness/runs/restarts/20261021T072742_706258Z.json)
records 3,226.42 s of interrupted classification wall time and 3,204.93 s of
worker CPU outside completed-fit timers, including partial new fits at the
handoff. The detached [batch log](/Users/mikhail/Datasets/imf-classification/dronerf/robustness/runs/detached/20261021T072744_137930Z.log)
is under the dataset's robustness directory to respect the new-data location
constraint. A separate detached completion process (PID 12822) waits for the
three real datasets, then runs the required tests and publishes the report;
its [log](/Users/mikhail/Datasets/imf-classification/dronerf/robustness/runs/report_supervisor.log)
is in the same directory.

At 2026-10-21 15:02 UTC the user requested concurrent real-dataset runs. The
serial launcher (PID 19215) and its old report waiter (PID 12822) were stopped;
the existing DroneRF child (PID 19217) and its six workers continued. BirdCLEF
started as detached PID 1882 with four workers and ESC-50 as detached PID 1883
with two workers. Each new launcher holds a dataset-specific kernel lock and
writes to its own file-backed log. The [scheduling record](/Users/mikhail/Datasets/imf-classification/dronerf/robustness/runs/parallel_schedule.json)
contains the timestamps, PIDs, log paths and pinned dependency hashes. Thus
the robustness total is at most 12 single-thread workers while the separate
job uses four cores. Each dataset completes features before any classifiers;
shift classical fits for all methods and both feature sets precede LightGBM
and matched fits. The final report lists this scheduling deviation.
