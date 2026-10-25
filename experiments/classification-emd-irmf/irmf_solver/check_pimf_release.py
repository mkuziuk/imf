"""Compare a new pimf release with the pimf 0.4.0 decompositions the study used.

Four modes, run on the same deterministic frame sample:

- ``reference``: run with the study venv (pimf 0.4.0). Decomposes each sampled
  channel with the default gradient descent (GD60) and with the Numba Newton
  prototype (``emdirmf.irmf_newton``).
- ``candidate``: run with an environment holding the new release. Decomposes the
  same channels with the release's gradient descent and its ``solver="newton"``.
- ``compare``: either interpreter. Writes a CSV for one candidate version.
- ``time``: times only the Newton solve of whichever interpreter runs it (the
  prototype under 0.4.0, the release otherwise) and appends per-frame CPU
  seconds and output hashes to timing.jsonl. Run it alternately with each
  environment, round by round, so every version sees the same machine load;
  ``compare --timing`` then summarizes the rounds.

Arrays stay under DATA_ROOT/solver_audit/pimf_release/; only the summary goes to
output/classification-emd-irmf/pimf_release/<version>/.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))

import pimf  # noqa: E402

from emdirmf.decompose import (  # noqa: E402
    H1,
    SQRT2,
    _imf_stack,
    imf_h_min,
    irmf_components,
    noise_sigma_hat,
    normalize_frame,
)
from emdirmf.io import load_frames  # noqa: E402
from emdirmf.paths import data_root  # noqa: E402

DATASETS = ["synthetic", "dronerf", "birdclef2023", "esc50_animals"]
SEED = 20261021
H_RATIO = 2.0


def scratch_dir():
    path = data_root() / "solver_audit" / "pimf_release"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_ids(name, n_frames):
    n_total = load_frames(name).shape[0]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, DATASETS.index(name), 7]))
    return np.sort(rng.choice(n_total, size=min(n_frames, n_total), replace=False))


def install_source():
    """Version plus, for VCS installs, the commit the package was built from."""
    dist = importlib.metadata.distribution("pimf")
    source = {"version": dist.version}
    direct = dist.read_text("direct_url.json")
    if direct:
        info = json.loads(direct)
        source["url"] = info.get("url")
        source["commit"] = info.get("vcs_info", {}).get("commit_id")
    return source


def newton_release_components(y):
    """The release's Newton solver with the study's H rule and schedule."""
    sigma_hat, _ = noise_sigma_hat(y)
    model = pimf.IMF(pimf.SmoothAbs(H_RATIO * sigma_hat, lookup=True), solver="newton", workers=1)
    result = model.decompose(y, h1=H1, a=SQRT2, k_max=None, h_min=imf_h_min(y.size))
    stages = result.stages
    info = {
        "converged": all(stage.converged for stage in stages),
        "iterations_max": max(int(stage.iterations) for stage in stages),
        "unconverged_windows": sum(int(getattr(stage, "unconverged_windows", 0)) for stage in stages),
        "bisection_fallbacks": sum(int(getattr(stage, "bisection_fallbacks", 0)) for stage in stages),
        "max_abs_score": max(float(getattr(stage, "max_abs_score", np.nan)) for stage in stages),
    }
    return _imf_stack(result), info


def candidate_prefix(version):
    return f"candidate-{version}"


def run_methods(mode):
    if mode == "reference":
        from emdirmf.irmf_newton import warm_up

        warm_up()
        return {
            "gd60": lambda y: (irmf_components(y)[0], {}),
            "newton": lambda y: (irmf_components(y, converged=True)[0], {}),
        }
    return {"gd60": lambda y: (irmf_components(y)[0], {}), "newton": newton_release_components}


def decompose_sample(mode, n_frames):
    if mode == "reference" and pimf.__version__ != "0.4.0":
        raise SystemExit(f"reference mode needs pimf 0.4.0, found {pimf.__version__}")
    if mode == "candidate" and pimf.__version__ == "0.4.0":
        raise SystemExit("candidate mode needs the new pimf release, found 0.4.0")
    methods = run_methods(mode)
    prefix = mode if mode == "reference" else candidate_prefix(pimf.__version__)
    meta = {"mode": mode, "pimf": install_source(), "n_frames": n_frames, "datasets": {}}
    for name in DATASETS:
        frames = load_frames(name)
        ids = sample_ids(name, n_frames)
        arrays, seconds, diagnostics = {}, {}, {}
        for method, fn in methods.items():
            stacks, times, infos, n_warnings = [], [], [], 0
            for frame_id in ids:
                channels, frame_seconds = [], 0.0
                for channel in np.asarray(frames[frame_id], dtype=np.float64):
                    y, _ = normalize_frame(channel)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        start = time.process_time()
                        stack, info = fn(y)
                        frame_seconds += time.process_time() - start
                    n_warnings += len(caught)
                    channels.append(stack)
                    if info:
                        infos.append(info)
                stacks.append(np.stack(channels))
                times.append(frame_seconds)
            arrays[method] = np.stack(stacks)
            seconds[method] = times
            diagnostics[method] = {"warnings": n_warnings}
            if infos:
                diagnostics[method].update({
                    "all_converged": all(info["converged"] for info in infos),
                    "iterations_max": max(info["iterations_max"] for info in infos),
                    "unconverged_windows": sum(info["unconverged_windows"] for info in infos),
                    "bisection_fallbacks": sum(info["bisection_fallbacks"] for info in infos),
                    "max_abs_score": max(info["max_abs_score"] for info in infos),
                })
            print(f"{mode} {name} {method}: {np.median(times):.4f} CPU s/frame median", flush=True)
        np.savez(scratch_dir() / f"{prefix}_{name}.npz", ids=ids, **arrays,
                 **{f"seconds_{m}": np.asarray(s) for m, s in seconds.items()})
        meta["datasets"][name] = {"frame_ids": ids.tolist(), "diagnostics": diagnostics}
    (scratch_dir() / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2))


def time_newton(round_id, datasets, n_frames):
    """Append per-frame Newton CPU seconds for this interpreter's solver."""
    if pimf.__version__ == "0.4.0":
        from emdirmf.irmf_newton import warm_up

        warm_up()
        label, fn = "prototype", lambda y: irmf_components(y, converged=True)[0]
    else:
        label, fn = f"pimf {pimf.__version__}", lambda y: newton_release_components(y)[0]
    with (scratch_dir() / "timing.jsonl").open("a") as log:
        for name in datasets:
            frames = load_frames(name)
            for frame_id in sample_ids(name, n_frames):
                seconds, digest = 0.0, hashlib.sha256()
                for channel in np.asarray(frames[frame_id], dtype=np.float64):
                    y, _ = normalize_frame(channel)
                    start = time.process_time()
                    stack = fn(y)
                    seconds += time.process_time() - start
                    digest.update(np.ascontiguousarray(stack, dtype=np.float64).tobytes())
                log.write(json.dumps({"label": label, "version": pimf.__version__, "round": round_id,
                                      "dataset": name, "frame_id": int(frame_id), "cpu_seconds": seconds,
                                      "sha256": digest.hexdigest()}) + "\n")
            print(f"time round {round_id} {label} {name} done", flush=True)


def timing_summary(output_root, version, base):
    import pandas as pd

    log = pd.read_json(scratch_dir() / "timing.jsonl", lines=True)
    labels = ["prototype", f"pimf {base}", f"pimf {version}"]
    log = log[log["label"].isin(labels)]
    rows = []
    for name in DATASETS:
        part = log[log["dataset"] == name]
        if part.empty:
            continue
        # Median over rounds per frame, then over frames.
        per_frame = part.groupby(["label", "frame_id"])["cpu_seconds"].median().unstack("label")
        hashes = part.groupby(["label", "frame_id"])["sha256"].nunique()
        row = {"dataset": name, "frames": int(per_frame.shape[0]),
               "rounds": int(part["round"].nunique()),
               "outputs_repeatable": bool((hashes == 1).all())}
        for label in labels:
            row[f"cpu_s_{label.replace(' ', '_')}"] = float(per_frame[label].median())
        new, old, proto = per_frame[f"pimf {version}"], per_frame[f"pimf {base}"], per_frame["prototype"]
        row[f"speedup_vs_{base}"] = float((old / new).median())
        row["slowdown_vs_prototype"] = float((new / proto).median())
        same = part[part["label"].isin(labels[1:])].groupby(["frame_id", "label"])["sha256"].first().unstack("label")
        row[f"outputs_identical_to_{base}"] = bool((same[f"pimf {base}"] == same[f"pimf {version}"]).all())
        rows.append(row)
    table = pd.DataFrame(rows)
    out = Path(output_root) / version
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "timing.csv", index=False)
    print(table.to_string(index=False))
    return table


def relative_l2(a, b, axes):
    return np.linalg.norm(a - b, axis=axes) / np.maximum(np.linalg.norm(b, axis=axes), 1e-300)


def compare(output_root, version):
    import pandas as pd

    prefix = candidate_prefix(version)
    ref_meta = json.loads((scratch_dir() / "reference_meta.json").read_text())
    cand_meta = json.loads((scratch_dir() / f"{prefix}_meta.json").read_text())
    rows = []
    for name in DATASETS:
        ref = np.load(scratch_dir() / f"reference_{name}.npz", allow_pickle=False)
        cand = np.load(scratch_dir() / f"{prefix}_{name}.npz", allow_pickle=False)
        if not np.array_equal(ref["ids"], cand["ids"]):
            raise SystemExit(f"{name}: frame samples differ between modes")
        # Stacks are (frames, channels, slots, L); compare per channel stack.
        gd_diff = np.abs(cand["gd60"] - ref["gd60"])
        new_vs_proto = relative_l2(cand["newton"], ref["newton"], axes=(2, 3))
        abs_diff = np.abs(cand["newton"] - ref["newton"])
        tolerance = 1e-9 * (1 + np.abs(ref["newton"]))
        newton_vs_gd = relative_l2(ref["gd60"], ref["newton"], axes=(2, 3))
        diag = cand_meta["datasets"][name]["diagnostics"]
        rows.append({
            "dataset": name,
            "frames": int(ref["ids"].size),
            "channels": int(new_vs_proto.size),
            "gd60_bitwise_identical": bool(np.all(gd_diff == 0)),
            "gd60_max_abs_diff": float(gd_diff.max()),
            "gd60_warnings": diag["gd60"]["warnings"],
            "newton_rel_l2_median": float(np.median(new_vs_proto)),
            "newton_rel_l2_max": float(new_vs_proto.max()),
            "newton_max_abs_diff": float(abs_diff.max()),
            "newton_samples_beyond_1e-9": int(np.count_nonzero(abs_diff > tolerance)),
            "newton_channels_beyond_1e-9": int(np.count_nonzero((abs_diff > tolerance).any(axis=(2, 3)))),
            "newton_all_converged": diag["newton"].get("all_converged"),
            "newton_iterations_max": diag["newton"].get("iterations_max"),
            "newton_bisection_fallbacks": diag["newton"].get("bisection_fallbacks"),
            "newton_max_abs_score": diag["newton"].get("max_abs_score"),
            "newton_warnings": diag["newton"]["warnings"],
            "gd60_vs_newton_rel_l2_p95": float(np.percentile(newton_vs_gd, 95)),
            "cpu_s_gd60_release": float(np.median(cand["seconds_gd60"])),
            "cpu_s_newton_prototype": float(np.median(ref["seconds_newton"])),
            "cpu_s_newton_release": float(np.median(cand["seconds_newton"])),
        })
    table = pd.DataFrame(rows)
    out = Path(output_root) / version
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison.csv", index=False)
    (out / "sources.json").write_text(json.dumps(
        {"reference": ref_meta["pimf"], "candidate": cand_meta["pimf"],
         "n_frames": ref_meta["n_frames"], "seed": SEED}, indent=2))
    print(table.to_string(index=False))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["reference", "candidate", "compare", "time"], required=True)
    parser.add_argument("--frames", type=int, default=40, help="frames sampled per dataset")
    parser.add_argument("--output", default=str(STUDY.parent.parent / "output/classification-emd-irmf/pimf_release"))
    parser.add_argument("--version", help="candidate version for compare")
    parser.add_argument("--timing", metavar="BASE", help="compare: summarize timing.jsonl against this base version")
    parser.add_argument("--round", type=int, default=0, help="time: round number")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS, help="time: datasets")
    args = parser.parse_args()
    if args.mode == "compare":
        if not args.version:
            parser.error("compare needs --version")
        if args.timing:
            timing_summary(args.output, args.version, args.timing)
        else:
            compare(args.output, args.version)
    elif args.mode == "time":
        time_newton(args.round, args.datasets, args.frames)
    else:
        decompose_sample(args.mode, args.frames)


if __name__ == "__main__":
    main()
