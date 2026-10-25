# Decomposition features for classification

Start with the [cross-dataset synthesis](../../output/classification-emd-irmf/SUMMARY.md)
and [contrast forest plot](../../output/classification-emd-irmf/figures/contrasts_forest.png).
The study compares PyEMD EMD, pimf linear IMF, default GD60 IRMF, converged Newton
IRMF and normalized raw frames. All experiments are complete; data and large
caches remain outside the repository.

## Layout

- [PROTOCOL.md](PROTOCOL.md) defines normalization, decompositions, features,
  grouped splits and paired tests. [LITERATURE.md](LITERATURE.md) records the
  literature search and differences from published evaluation protocols.
- [datasets/](datasets/) contains preparation, diagnostics and dataset-specific
  experiment wrappers. Each dataset README records sources, licenses and choices.
- [emdirmf/](emdirmf/README.md) provides loading, validation, decomposition,
  features, grouped CV, statistics and component-stack readers.
- [baselines/](baselines/) contains the classical runner; [ablations/](baselines/ablations/)
  adds checkpointed fits and verified reuse of historical phase-2 results.
- [irmf_solver/](irmf_solver/README.md) contains the solver audit and integration
  tools. [REPORT.md](irmf_solver/REPORT.md) diagnoses GD60 nonconvergence;
  [INTEGRATION.md](irmf_solver/INTEGRATION.md) records the completed Newton caches.
- [ml/](ml/README.md) contains trees, ROCKET and CNN runners. [tests/](tests/)
  uses temporary synthetic data. [plot_contrasts.py](plot_contrasts.py) rebuilds
  the forest PNG and its small source CSV from committed result tables.
- [output/classification-emd-irmf/](../../output/classification-emd-irmf/) contains
  reports, aggregate CSV/JSON tables and figures, including each dataset's
  `classical/`, `classical_ablations/` and `ml/` results.

## Environment

Run from this worktree with Python 3.12. On a fresh checkout:

```bash
study_root=/Users/mikhail/.t3/worktrees/imf/t3-44ed3594
study_dir="$study_root/experiments/classification-emd-irmf"
cd "$study_root"
uv venv --python 3.12 "$study_root/.venv"
uv pip install --python "$study_root/.venv/bin/python" -r "$study_dir/requirements.txt"
export IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
study_python="$study_root/.venv/bin/python"
```

The pinned [requirements.txt](requirements.txt) includes preparation and plotting
dependencies as well as models. DroneRF extraction also needs `bsdtar` with RAR
support. CNN runs used Apple's MPS; CPU is supported but timing and numerical
results may differ. Do not recreate the supplied `.venv` just to inspect results.

## Data

`IMF_CLS_DATA` selects the external root, defaulting to
`/Users/mikhail/Datasets/imf-classification`. For each dataset:

```text
<DATA_ROOT>/<dataset>/raw/                 original downloads
<DATA_ROOT>/<dataset>/prepared/            frames.npy, meta.parquet, dataset.json
<DATA_ROOT>/<dataset>/decomp/              component arrays and solver diagnostics
<DATA_ROOT>/<dataset>/features/            per-frame feature parquets
<DATA_ROOT>/<dataset>/splits_<task>.json    immutable grouped folds
<DATA_ROOT>/<dataset>/results/             OOF predictions, probabilities, checkpoints
```

No raw signals, prepared frames, component/feature caches or per-frame predictions
belong in the repository. Small aggregate result CSVs are intentional artifacts.

| Dataset | Primary task and prepared population | Group unit | Preparation/source instructions |
|---|---|---|---|
| DroneRF | 10 BUI classes; 12,000 two-view frames, L=2,048 | 227 segments; sessions remain shared | [Data README](datasets/dronerf/README.md) |
| BirdCLEF 2023 | 16 species; 12,000 mono frames, L=2,048 | 451 recordists | [Data README](datasets/birdclef2023/README.md) |
| ESC-50 animals | 10 animals; 5,100 mono frames, L=2,048 | 157 uploaders | [Data README](datasets/esc50_animals/README.md) |
| 60 GHz public subset | UAS/bird; 784 I/Q frames, L=1,000 | 45 target-plus-pose groups, 49 recordings | [Data README](datasets/uas_birds_60ghz/README.md) |
| Synthetic | Five signal classes; 1,500 mono frames, L=1,024 | 150 generated recordings | [Generator README](datasets/synthetic/README.md) |

Only the **full 60 GHz IEEE DataPort release** needs the user's IEEE login. Its
documented archive is `Data.zip`, 15.21 GB, placed at
`<DATA_ROOT>/uas_birds_60ghz/raw/dataport/Data.zip`. It was not used in the reported
experiments. The [dataset instructions](datasets/uas_birds_60ghz/README.md#user-action-getting-the-full-release)
describe download, conversion and the untested full-cube adapter. A wider crop may
contain the same campaign rather than new independent recordings. Keep it in a
separate external root to preserve the public-subset caches.

## Reproduction pipeline

Follow prepare → validate → decompose → features → classical → ablations → ML.
The commands below describe a new run with isolated outputs. Exact historical
grids, budgets, seeds, checkpoints and analyses are recorded in the dataset
reproduction files linked below; generic defaults do not reproduce every local
budget decision. Start with `synthetic` to check the installation.

1. Prepare the selected dataset. Use its data README for downloads and prerequisite
   conversions. Run preparation with `-I` from outside downloaded folders. Radar
   requires the audited [numeric-only converter](datasets/uas_birds_60ghz/convert_pickles.py)
   before preparation. For example:

   ```bash
   "$study_python" -I "$study_dir/datasets/synthetic/prepare.py"
   ```

2. Validate prepared values and metadata. Set the task to the dataset's primary
   task from the table above, using its code name from `dataset.json`.

   ```bash
   study_dataset=synthetic
   study_task=signal
   study_run=reproduction_20261021
   export PYTHONPATH="$study_dir"
   "$study_python" -m emdirmf.validate "$study_dataset"
   ```

3. Decompose the identical normalized frames. The default process cap is six;
   `IMF_CLS_MAX_WORKERS` can raise it to twelve when the machine is free. Each
   process uses one native thread. Partial caches resume; completed caches reuse
   their recorded parameters.

   ```bash
   "$study_python" -m emdirmf.run_decompose --dataset "$study_dataset" \
       --method emd imf_linear_a1.41 irmf_h2_a1.41 --workers 6
   ```

4. Extract the shared per-component features, including the normalized raw control.

   ```bash
   "$study_python" -m emdirmf.run_features --dataset "$study_dataset" \
       --method raw emd imf_linear_a1.41 irmf_h2_a1.41 --workers 6
   ```

5. Run classical models. This creates/reuses shared grouped splits and evaluates
   both `log_scale` settings. `--out` isolates small tables; `--results-tag`
   independently isolates predictions. Native probabilities use `prob_<class>`
   columns; linear SVM has NaN because it has no native probability output.

   ```bash
   "$study_python" "$study_dir/baselines/run_classical.py" \
       --dataset "$study_dataset" --task "$study_task" --workers 6 \
       --save-probabilities --results-tag "${study_run}_main" \
       --out "$study_root/output/classification-emd-irmf/$study_dataset/$study_run/classical"
   ```

6. Prepare ablation decompositions/features, then fit them. Use the same two
   `emdirmf` commands above with tags `irmf_h2_a1.41_conv`,
   `irmf_h1_a1.41_conv`, `irmf_h4_a1.41_conv`, `irmf_h2_a2_conv` and
   `imf_linear_a2`. Run the classical driver with all main and ablation tags,
   `--classifiers lda logreg linsvm`, a distinct results tag and a distinct output
   directory. Default comparisons now include every tag pair. Use the Python
   statistics API's explicit `pairs` argument to restrict the test family.
   For exact historical reuse, follow [classical ablation reports](../../output/classification-emd-irmf/SUMMARY.md#dataset-reports).
   Those wrappers expect their original prediction paths and verify source hashes;
   use the recorded experiment revision/checkpoints for historical audits. Current
   shared-runner edits intentionally invalidate old source fingerprints. Do not
   rewrite manifests to bypass those checks.

7. Run ML only after all requested caches and shared splits exist. For example:

   ```bash
   "$study_python" "$study_dir/ml/run_trees.py" --dataset "$study_dataset" \
       --task "$study_task" --methods raw emd imf_linear_a1.41 irmf_h2_a1.41 \
       irmf_h2_a1.41_conv --workers 2
   ```

   Follow [ML runner instructions](ml/README.md) for ROCKET/CNN, grouped validation
   and CPU/MPS choices. Shared ML runners still use fixed prediction filenames;
   `--results-tag` belongs to the classical runner. Use dataset-local wrappers and
   their documented run names to reproduce the isolated ML exports.

| Dataset | Historical classical reproduction | Historical ML reproduction |
|---|---|---|
| DroneRF | [Methods](../../output/classification-emd-irmf/dronerf/classical/METHODS.md) | [Run notes](../../output/classification-emd-irmf/dronerf/ml/RUN_NOTES.md) |
| BirdCLEF | [Reproduce](../../output/classification-emd-irmf/birdclef2023/classical/REPRODUCE.md) | [Reproduce](../../output/classification-emd-irmf/birdclef2023/ml/REPRODUCE.md) |
| ESC-50 | [Run notes](../../output/classification-emd-irmf/esc50_animals/classical/RUN_NOTES.md) | [Run notes](../../output/classification-emd-irmf/esc50_animals/ml/RUN_NOTES.md) |
| 60 GHz | [Report commands](../../output/classification-emd-irmf/uas_birds_60ghz/classical/REPORT.md#reproduce-and-inspect) | [Run notes](../../output/classification-emd-irmf/uas_birds_60ghz/ml/RUN_NOTES.md) |
| Synthetic | [Classical run](../../output/classification-emd-irmf/synthetic/classical/run_signal.json) | [Validation commands](../../output/classification-emd-irmf/synthetic/ml/VALIDATION.md) |

## Compute and checks

Full-cache CPU seconds/frame for GD60/Newton are DroneRF 1.2787/.1076,
BirdCLEF 1.3205/.0579, ESC-50 1.7586/.0644 and radar .3547/.0314.
EMD and linear IMF are much cheaper. See the
[source-linked cost table](../../output/classification-emd-irmf/SUMMARY.md#decomposition-cost).
Measured historical ML elapsed times were 176.3, 154.0 and 102.4 minutes for
DroneRF, BirdCLEF and ESC-50, respectively, under concurrent CPU/MPS load;
their run notes above specify coverage and budgets. These are not isolated speed
benchmarks or accuracy ceilings. ROCKET transform timings shared across scale
variants must not be summed twice. Preprocessing and model fits are separate costs.

Rebuild only the synthesis figure from small committed CSVs and run the full suite:

```bash
"$study_python" "$study_dir/plot_contrasts.py"
"$study_python" -m pytest "$study_dir/tests" -q
```

Runner defaults preserve the original label-only prediction path and two-item
`fit_predict_split` result. Opt in to native probabilities and tagged directories
for new classical batches. Existing result artifacts are unchanged until rerun;
the synthesis reads committed tables without launching any experiment.
