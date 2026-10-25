"""Verify completed shared caches and regenerate INTEGRATION.md and CSV tables."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emdirmf import decompose as dec
from emdirmf.irmf_newton import solver_params
from emdirmf.paths import data_root
from run_integration import CONV_TAGS, DATASETS

RUN_DIR = data_root() / "solver_integration"


def read_json(path):
    return json.loads(path.read_text())


def describe_cache(dataset, tag):
    directory = data_root() / dataset
    summary = read_json(directory / "decomp" / f"{tag}.json")
    features = read_json(directory / "features" / f"{tag}.json")
    frames = np.load(directory / "prepared" / "frames.npy", mmap_mode="r")
    components = np.load(directory / "decomp" / f"{tag}.npy", mmap_mode="r")
    meta = pd.read_parquet(directory / "decomp" / f"{tag}.meta.parquet")
    expected = (*frames.shape[:2], dec.n_slots(tag, frames.shape[-1]), frames.shape[-1])
    assert components.shape == expected and components.dtype == np.float32
    assert not summary["partial"] and summary["n_failed"] == 0 and meta["ok"].all()
    assert features["n_frames"] == len(frames) and features["n_failed_frames"] == 0
    assert features["n_nonfinite_rows_among_ok"] == 0
    assert summary["params"] == dec.parse_method(tag).params()
    reconstruction_abs_max = 0.0
    reconstruction_relative_max = 0.0
    for start in range(0, len(frames), 32):
        stop = start + 32
        normalized, _ = dec.normalize_frame(frames[start:stop])
        recovered = components[start:stop].astype(np.float64).sum(axis=2)
        error = recovered - normalized
        reconstruction_abs_max = max(reconstruction_abs_max, float(np.abs(error).max()))
        relative = np.linalg.norm(error, axis=-1) / np.maximum(np.linalg.norm(normalized, axis=-1), 1e-12)
        reconstruction_relative_max = max(reconstruction_relative_max, float(relative.max()))
    assert reconstruction_relative_max < 5e-7
    record = {
        "dataset": dataset, "tag": tag, "n_frames": len(frames),
        "n_channels": frames.shape[1], "n_stages": components.shape[2] - 1,
        "decomposition_wall_seconds": summary["wall_seconds_total"],
        "decomposition_cpu_seconds": summary["seconds_per_frame"]["cpu_total"],
        "cpu_seconds_per_frame": summary["seconds_per_frame"]["cpu_mean"],
        "feature_wall_seconds": features["seconds"],
        "failed_channels": summary["n_failed"],
        "reconstruction_abs_max_float32": reconstruction_abs_max,
        "reconstruction_relative_l2_max_float32": reconstruction_relative_max,
    }
    if dec.parse_method(tag).converged:
        converged = np.stack(meta["stage_converged"])
        iterations = np.stack(meta["stage_iterations"])
        means = np.stack(meta["stage_iterations_mean"])
        fallbacks = np.stack(meta["stage_bisection_fallbacks"])
        fallback_max = np.stack(meta["stage_bisection_fallbacks_max"])
        scores = np.stack(meta["stage_max_score_residual"])
        assert converged.all() and converged.shape == (len(meta), record["n_stages"])
        assert iterations.max() <= solver_params()["root_max_iter"]
        n_roots = int(converged.size * frames.shape[-1])
        record.update(
            all_stages_converged=bool(converged.all()),
            n_channel_stages=int(converged.size), n_window_roots=n_roots,
            row_iterations_mean=float(means.mean()),
            stage_max_iterations_median=float(np.median(iterations)),
            stage_max_iterations_p95=float(np.quantile(iterations, 0.95)),
            stage_max_iterations_max=int(iterations.max()),
            bisection_fallbacks_total=int(fallbacks.sum()),
            bisection_fallbacks_per_root=float(fallbacks.sum() / n_roots),
            bisection_fallbacks_per_root_max=int(fallback_max.max()),
            max_score_residual=float(scores.max()),
        )
    return record


def component_differences(dataset):
    """All successful matched channels; denominator is the converged component."""
    directory = data_root() / dataset / "decomp"
    gd_tag, conv_tag = "irmf_h2_a1.41", "irmf_h2_a1.41_conv"
    gd = np.load(directory / f"{gd_tag}.npy", mmap_mode="r")
    conv = np.load(directory / f"{conv_tag}.npy", mmap_mode="r")
    assert gd.shape == conv.shape
    gd_meta = pd.read_parquet(directory / f"{gd_tag}.meta.parquet")
    conv_meta = pd.read_parquet(directory / f"{conv_tag}.meta.parquet")
    np.testing.assert_array_equal(gd_meta[["frame_id", "channel"]], conv_meta[["frame_id", "channel"]])
    ok = (gd_meta["ok"].to_numpy() & conv_meta["ok"].to_numpy()).reshape(gd.shape[:2])
    errors = []
    for start in range(0, len(gd), 32):
        stop = start + 32
        reference = conv[start:stop].astype(np.float64)
        delta = gd[start:stop].astype(np.float64) - reference
        relative = np.linalg.norm(delta, axis=-1) / np.maximum(np.linalg.norm(reference, axis=-1), 1e-12)
        errors.append(relative[ok[start:stop]])
    errors = np.concatenate(errors)
    labels = dec.slot_labels(conv_tag, gd.shape[-1])
    return [{"dataset": dataset, "gd_tag": gd_tag, "conv_tag": conv_tag,
             "slot": slot, "label": label, "n_matched_channels": len(errors),
             "relative_l2_median": float(np.median(errors[:, slot])),
             "relative_l2_p95": float(np.quantile(errors[:, slot], 0.95)),
             "relative_l2_max": float(errors[:, slot].max())}
            for slot, label in enumerate(labels)]


def write_note(caches, differences, invocations):
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total_wall = sum(record["elapsed_seconds"] for record in invocations if record["status"] == "completed")
    conv = caches[caches["tag"].isin(CONV_TAGS)]
    lines = ["# Converged IRMF integration", "", f"Verified {timestamp}. pimf 0.4.0, Python 3.12.", "",
             "`emdirmf/irmf_newton.py` now owns the audited lookup-compatible safeguarded Newton solver. "
             "The prototype module is a thin import shim. The public pimf solve hook preserves normalization, "
             "H estimation and fallbacks, squared-triangle kernel, wrap boundary, bandwidth schedule and slot order. "
             "No installed pimf files or PROTOCOL.md were changed. No git commit was made.", "",
             "Tags are `irmf_h2_a1.41_conv` and the ablations `irmf_h1_a1.41_conv`, "
             "`irmf_h4_a1.41_conv`, `irmf_h2_a2_conv`. `_conv` excludes `_it{N}`. "
             "Solver version 1.0.0 uses root tolerance 1e-12, a 100-iteration cap, and bisection every eighth "
             "update or when a Newton candidate leaves its bracket. The stopping test bounds the root step "
             "or bracket width by `1e-12 * (1 + abs(root))`. Parameters record this provenance. "
             "Numba warms one strided-window signature per worker; it uses no parallel Numba loop.", "",
             "## Validation and outputs", "",
             "The complete suite passed: **98 tests**. It covers tag canonicalization, suffix exclusion, exact "
             "float64 reconstruction, tight-GD agreement at max_iter=20000 and tol=1e-12 within 1e-8 relative "
             "L2 per component, identical stage order, failure propagation to NaN caches/features, multiprocessing "
             "metadata, JIT signature reuse, and byte-identical float32 GD cache slices and stage diagnostics. "
             "GD regression checks use frames 0 and 1 of all five datasets plus ESC-50's a=2 cache. "
             "Existing GD parameter dictionaries match the stored manifests exactly.", "",
             "Every listed output has a full float32 decomposition, meta parquet, JSON, and feature parquet/JSON "
             "under DATA_ROOT. Features report zero failed frames and zero nonfinite rows. "
             "Full-cache reconstruction was checked after float32 storage.", "",
             "| Dataset | h2/a1.41 conv | h1/a1.41 conv | h4/a1.41 conv | h2/a2 conv | linear a2 |",
             "|---|---|---|---|---|---|"]
    for dataset in DATASETS:
        linear = "complete" if dataset != "synthetic" else "outside requested scope"
        lines.append(f"| {dataset} | complete | complete | complete | complete | {linear} |")
    lines += ["", "BirdCLEF, DroneRF and ESC-50 linear-a2 decompositions were reused. "
              "Their missing features were generated, including ESC-50. UAS linear-a2 decomposition and features "
              "were generated. The interrupted ESC-50 `irmf_h1_a1.41` GD cache was left alone.", "",
              "## Convergence and timings", "",
              "All channel-stages converged for every convergence tag. There were no decomposition failures. "
              "Per-stage arrays in the meta parquet record maximum/mean row iterations, total/maximum row "
              "bisection fallbacks, maximum absolute weighted score at returned roots, and convergence. "
              "`stage_iterations` is the maximum over window rows, rather than pimf's solve-hook placeholder of 1.", "",
              "All runs used two worker processes with one native thread each, sequential datasets and tags. "
              "Other phase-2 processes were active. Decomposition wall times include process startup, worker JIT "
              "warmup and cache I/O. CPU time sums measured channel fits and excludes worker warmup. "
              "Feature times are computation wall times reported by the shared CLI, before table serialization.", "",
              "| Dataset | Tag | Decomp wall s | Fit CPU s | Features s | Mean row iterations | Stage max iterations, p50/p95/max | Bisections total, per root | Max score residual |",
              "|---|---|---:|---:|---:|---:|---|---|---:|"]
    for record in conv.to_dict("records"):
        lines.append(f"| {record['dataset']} | {record['tag']} | {record['decomposition_wall_seconds']:.1f} "
                     f"| {record['decomposition_cpu_seconds']:.1f} | {record['feature_wall_seconds']:.1f} "
                     f"| {record['row_iterations_mean']:.3f} "
                     f"| {record['stage_max_iterations_median']:g}/{record['stage_max_iterations_p95']:g}/{record['stage_max_iterations_max']:g} "
                     f"| {record['bisection_fallbacks_total']:g}, {record['bisection_fallbacks_per_root']:.5g} "
                     f"| {record['max_score_residual']:.3g} |")
    lines += ["", f"Completed CLI invocations took {total_wall:.1f} s in total. "
              f"Convergence-tag fit CPU time sums to {conv['decomposition_cpu_seconds'].sum():.1f} s. "
              f"Largest float32 reconstruction relative L2 error was {caches['reconstruction_relative_l2_max_float32'].max():.3g}. "
              "`cache_summary.csv` retains counts, per-root maximum fallback counts, timings and reconstruction "
              "errors at full precision. All run logs and invocation records are under "
              f"`{RUN_DIR}`.", "",
              "## GD60 versus convergence", "",
              "The table compares the unchanged primary GD60 tag `irmf_h2_a1.41` against "
              "`irmf_h2_a1.41_conv` on every successful matched channel. Each cell is median / 95th percentile "
              "of `norm(GD60_component - conv_component) / max(norm(conv_component), 1e-12)`, computed from "
              "the float32 caches in float64. The residual is included separately. Relative errors can be large "
              "when the converged component has little energy. These are component differences, not classification results.", "",
              "| Slot/window | " + " | ".join(DATASETS) + " |",
              "|---|" + "---:|" * len(DATASETS)]
    labels = list(dict.fromkeys(differences["label"]))
    labels.sort(key=lambda label: int(label[1:]) if label.startswith("w") else float("inf"))
    for label in labels:
        values = []
        for dataset in DATASETS:
            selected = differences[(differences["dataset"] == dataset) & (differences["label"] == label)]
            if selected.empty:
                values.append("n/a")
            else:
                row = selected.iloc[0]
                values.append(f"{row['relative_l2_median']:.3g} / {row['relative_l2_p95']:.3g}")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += ["", "Exact per-slot statistics and matched-channel counts are in "
              "`solver_integration/gd60_conv_component_differences.csv` under DATA_ROOT.", "",
              "## Reproduction and suggested protocol wording", "", "```bash",
              ".venv/bin/python -m pytest experiments/classification-emd-irmf/tests -q",
              ".venv/bin/python -I experiments/classification-emd-irmf/irmf_solver/run_integration.py",
              ".venv/bin/python -I experiments/classification-emd-irmf/irmf_solver/summarize_integration.py",
              "```", "",
              "Suggested addition to PROTOCOL.md, which remains unchanged:", "",
              "> Keep the default GD60 IRMF tag as the primary analysis. Add `irmf_h2_a1.41_conv` "
              "as the convergence sensitivity analysis, with corresponding H-ratio and a=2 ablations. "
              "The `_conv` tag uses safeguarded Newton roots of pimf 0.4.0's lookup score, solver version 1.0.0, "
              "root tolerance 1e-12 and at most 100 iterations per window. It shares GD's normalization, "
              "H rule, kernel, boundary, schedule and slot order. `_conv` and `_it{N}` are mutually exclusive. "
              "Record per-stage iteration counts, bisection fallbacks, maximum score residual and convergence. "
              "A stage that misses convergence marks its channel as failed. Repeat paired classification "
              "comparisons with and without log_scale using the same grouped splits.", ""]
    (Path(__file__).parent / "INTEGRATION.md").write_text("\n".join(lines))


def main():
    records, differences = [], []
    for dataset in DATASETS:
        for tag in CONV_TAGS:
            print(f"verify {dataset}/{tag}", flush=True)
            records.append(describe_cache(dataset, tag))
        if dataset != "synthetic":
            records.append(describe_cache(dataset, "imf_linear_a2"))
        differences.extend(component_differences(dataset))
    caches = pd.DataFrame(records)
    differences = pd.DataFrame(differences)
    caches.to_csv(RUN_DIR / "cache_summary.csv", index=False)
    differences.to_csv(RUN_DIR / "gd60_conv_component_differences.csv", index=False)
    invocations = read_json(RUN_DIR / "invocations.json")
    write_note(caches, differences, invocations)
    print(f"Verified {len(caches)} caches and feature tables; wrote INTEGRATION.md", flush=True)


if __name__ == "__main__":
    main()
