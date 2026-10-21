"""Label-free timing and direct-cache verification, before classification."""

import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from components.construction import (
    DEFAULTS, EMD_K, NATIVE_K, TRUNC_K, direct_truncation, fold_emd,
    native_components, relative_error, truncate_imf,
)
from components.integrity import DATA_ROOT, DATASETS, check_pimf, record_entry
from emdirmf.decompose import emd_components, normalize_frame
from emdirmf.irmf_newton import warm_up
from emdirmf.parallel import limit_threads, single_thread_env


def initialize():
    limit_threads()
    check_pimf()
    warm_up()


def timing_job(dataset, index):
    check_pimf()
    frames = np.load(DATA_ROOT / dataset / "prepared/frames.npy", mmap_mode="r", allow_pickle=False)
    normalized, _ = normalize_frame(frames[index])
    rows = []
    for family in ("linear", "irmf"):
        for k in NATIVE_K:
            cpu = 0.0
            caps = 0
            for y in normalized:
                _, info = native_components(y, family, k)
                cpu += info["cpu_seconds"]
                caps += sum(bool(a["cap_stages"]) for a in info["attempts"])
            rows.append({"dataset": dataset, "frame_id": int(index), "family": family,
                         "k": k, "cpu_seconds": cpu, "cap_hits": caps})
    return rows


def verify_dataset(dataset, indices):
    record_entry(dataset, "pilot_verification")
    root = DATA_ROOT / dataset
    frames = np.load(root / "prepared/frames.npy", mmap_mode="r", allow_pickle=False)
    sources = {tag: np.load(root / "decomp" / f"{tag}.npy", mmap_mode="r", allow_pickle=False)
               for tag in ("emd", *DEFAULTS.values(), "irmf_h2_a1.41")}
    errors = []
    for index in indices:
        for channel, y in enumerate(normalize_frame(frames[index])[0]):
            for k in EMD_K:
                folded = fold_emd(sources["emd"][index, channel], k).astype(np.float32)
                direct, _ = emd_components(y, k_emd=k)
                err = relative_error(folded, direct)
                assert err < 2e-7, (dataset, index, channel, k, err)
                errors.append({"dataset": dataset, "frame_id": int(index), "channel": channel,
                               "family": "emd", "k": k, "relative_error": err})
            for family, tag in (*DEFAULTS.items(), ("gd60", "irmf_h2_a1.41")):
                for k in (TRUNC_K if family != "gd60" else (8,)):
                    folded = truncate_imf(sources[tag][index, channel], k, tag).astype(np.float32)
                    direct = direct_truncation(y, family, k)
                    err = relative_error(folded, direct)
                    assert err < 2e-7, (dataset, index, channel, family, k, err)
                    errors.append({"dataset": dataset, "frame_id": int(index), "channel": channel,
                                   "family": family, "k": k, "relative_error": err})
    return errors


def main():
    single_thread_env()
    limit_threads()
    started = time.perf_counter()
    jobs = []
    selections = {}
    for dataset in DATASETS:
        record_entry(dataset, "pilot")
        frames = np.load(DATA_ROOT / dataset / "prepared/frames.npy", mmap_mode="r", allow_pickle=False)
        selections[dataset] = np.sort(np.random.default_rng(20261021).choice(len(frames), 50, replace=False))
        jobs.extend((dataset, int(index)) for index in selections[dataset])
    with ProcessPoolExecutor(max_workers=6, initializer=initialize) as pool:
        futures = [pool.submit(timing_job, *job) for job in jobs]
        timing = [row for future in futures for row in future.result()]
    for dataset in DATASETS:
        directory = DATA_ROOT / dataset / "components/pilot"
        directory.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(timing).query("dataset == @dataset")
        with (directory / "timing.csv").open("x") as handle:
            table.to_csv(handle, index=False)
        verification = verify_dataset(dataset, selections[dataset])
        with (directory / "verification.csv").open("x") as handle:
            pd.DataFrame(verification).to_csv(handle, index=False)
        check_pimf()
        print(f"{dataset}: timing and verification complete", flush=True)
    print(json.dumps({"pilot_wall_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
