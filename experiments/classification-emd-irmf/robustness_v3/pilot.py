"""Time the exact fresh-clean plus four-condition path on excluded pilot IDs."""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from emdirmf.paths import data_root
from robustness_v3 import processing as proc
from robustness_v3.classification import CLASSICAL, expected_cost
from robustness_v3.construction import METHODS, parse_variant, variants
from robustness_v3.contamination import CONDITIONS, DATASETS, subset_ids
from robustness_v3.estimators import await_decision, calibrate_h
from robustness_v3.run import initialize
from robustness_v3.storage import digest, save_json


def time_frame(dataset, frame_id, fixed_h):
    frames = np.load(data_root()/dataset/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
    cpu, wall = time.process_time(), time.perf_counter()
    nbytes, failed, caps = 0, 0, 0
    for condition in CONDITIONS:
        result, diagnostic = proc.extract_all(frames[frame_id], dataset, frame_id, condition, fixed_h)
        for variant, r in result.items():
            failed += bool(r['error'])
            caps += int(r['cap'].sum())
            nbytes += r['values'].nbytes + r['stack'].nbytes
    return dict(dataset=dataset, frame_id=frame_id, cpu_seconds=time.process_time()-cpu,
                seconds=time.perf_counter()-wall, uncompressed_bytes=nbytes, failed=failed, caps=caps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--frames', type=int, default=100, help='first N excluded pilot IDs per dataset')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not 1 <= args.frames <= 100:
        raise ValueError('pilot frame count must be 1..100')
    await_decision()
    rows, projections = [], []
    started = time.perf_counter()
    calibrations = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        pending = []
        for d in DATASETS:
            frames = np.load(data_root()/d/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
            ids = subset_ids(d, len(frames))
            calibrations[d] = calibrate_h(frames, ids)
            pending += [pool.submit(time_frame, d, int(i), calibrations[d]['H_d']) for i in ids[:args.frames]]
        for i, future in enumerate(as_completed(pending), 1):
            rows.append(future.result())
            if i % 25 == 0:
                print(f'pilot {i}/{len(pending)}', flush=True)
    table = pd.DataFrame(rows)
    for d in DATASETS:
        frames = np.load(data_root()/d/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
        cell = table[table.dataset == d]
        # Production contamination also recomputes a clean stack for L2, hence 6/5.
        feature_cpu = float(cell.cpu_seconds.mean()*len(frames)*6/5)
        fit_cpu = sum(expected_cost(d, clf, v) * 15 for v in variants(frames.shape[-1])
                      for clf in (CLASSICAL[d], 'lightgbm'))
        fit_cpu += sum(max(expected_cost(d, CLASSICAL[d], v, 'matched')
                          for v in variants(frames.shape[-1]) if parse_variant(v)[0] == m)*15 for m in METHODS)
        # Conservative uncompressed accounting for all pilot stacks, features, diagnostics, matrices.
        feature_width = sum((parse_variant(v)[1]+1)*20*frames.shape[1] for v in variants(frames.shape[-1]))
        projected_bytes = (float(cell.uncompressed_bytes.mean())*100 + len(frames)*5*feature_width*8
                           + len(frames)*5*16*frames.shape[1]*14*8 + 200_000_000)
        projections.append(dict(dataset=d, n_frames=len(frames), pilot_frames=len(cell),
            feature_cpu_seconds=feature_cpu, fit_cpu_seconds_prior_estimate=fit_cpu,
            wall_hours_14_workers=(feature_cpu+fit_cpu)/14/3600,
            projected_bytes=projected_bytes, failed=int(cell.failed.sum()), caps=int(cell.caps.sum())))
    args.out.mkdir(parents=True)
    table.to_csv(args.out/'frame_timings.csv', index=False, mode='x')
    pd.DataFrame(projections).to_csv(args.out/'projection.csv', index=False, mode='x')
    save_json(args.out/'timing.json', dict(workers=args.workers, wall_seconds=time.perf_counter()-started,
        calibration=calibrations, projection=projections, source_sha256=digest(Path(__file__)),
        note='feature projection includes 6/5 clean-reference recomputation; fit projection uses prior Laplace timings'))
    print(pd.DataFrame(projections).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
