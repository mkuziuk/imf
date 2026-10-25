"""Timing-only decompositions of the fixed 100-frame excluded pilot subset."""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from emdirmf.decompose import decompose_signal, normalize_frame
from emdirmf.irmf_newton import newton_components, warm_up
from emdirmf.parallel import limit_threads, single_thread_env
from emdirmf.paths import data_root
from robustness.contamination import DATASETS, contaminate, subset_ids
from robustness.environment import verify

METHODS = ('raw', 'emd', 'imf_linear_a1.41', 'irmf_h2_a1.41_conv')


def solve(y, method):
    if method != METHODS[-1]:
        stack, info = decompose_signal(y, method)
        return stack, info, False
    try:
        stack, info = newton_components(y)
        hit = max(info['stage_iterations']) >= 100
        if not hit:
            return stack, info, False
    except RuntimeError as error:
        if 'Newton failed to converge' not in str(error):
            raise
    stack, info = newton_components(y, root_max_iter=1000)
    return stack, info, True


def initialize():
    limit_threads()
    warm_up()


def job(args):
    dataset, frame_id, condition = args
    frames = np.load(data_root()/dataset/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
    rows = []
    for method in METHODS:
        cpu, wall = time.process_time(), time.perf_counter()
        cap, failed, error = 0, 0, ''
        for channel, x in enumerate(frames[frame_id]):
            y, _ = normalize_frame(contaminate(x, dataset, frame_id, channel, condition))
            try:
                _, info, hit = solve(y, method)
                cap += hit
            except RuntimeError as exc:
                failed += 1
                error = str(exc)
        rows.append(dict(dataset=dataset, frame_id=int(frame_id), condition=condition, method=method,
                         cpu_seconds=time.process_time()-cpu, seconds=time.perf_counter()-wall,
                         cap_hits=cap, failed_channels=failed, error=error))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError('workers must be 1..6')
    verify()
    root = data_root()/args.dataset
    out = root/'robustness/pilot'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'costs.csv').exists():
        print('pilot already complete', args.dataset)
        return
    frames = np.load(root/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
    ids = subset_ids(args.dataset, len(frames))
    single_thread_env()
    started = time.perf_counter()
    jobs = [(args.dataset, int(i), c) for i in ids for c in ('impulse2_0.04', 'gauss_0.04')]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        rows = [r for block in pool.map(job, jobs, chunksize=2) for r in block]
    pd.DataFrame(rows).to_csv(out/'costs.csv', index=False, mode='x')
    with (out/'run.json').open('x') as f:
        json.dump(dict(dataset=args.dataset, ids=ids.tolist(), workers=args.workers,
                       wall_seconds=time.perf_counter()-started, pimf=verify()), f, indent=2)
    print(args.dataset, pd.DataFrame(rows).groupby('method').cpu_seconds.mean().to_dict(), flush=True)


if __name__ == '__main__':
    main()
