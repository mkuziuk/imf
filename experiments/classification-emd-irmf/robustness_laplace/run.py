"""Run all four datasets in one worker pool.

Order: clean verification, contaminated feature chunks, matrices, then fits in
descending expected cost. A matched fit waits for its clean-trained shift fit.
Completed checkpoints are reused, so the run resumes after an interruption.

    $PY -m robustness_laplace.run --workers 14 [--detach]
"""
import argparse
import fcntl
import heapq
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone

import numpy as np

from emdirmf.irmf_newton import warm_up
from emdirmf.parallel import limit_threads, single_thread_env
from emdirmf.paths import data_root
from robustness.environment import verify
from robustness_laplace import classification as cls
from robustness_laplace import processing as proc
from robustness_laplace.contamination import DATASETS
from robustness_laplace.storage import check_disk, root, save_json

MAX_WORKERS = 15


def runs_dir():
    return data_root()/'robustness_laplace_runs'


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    limit_threads()
    verify()
    warm_up()


def execute(kind, args):
    started, cpu = time.perf_counter(), time.process_time()
    if kind == 'verify':
        result = proc.verification_job(*args)
    elif kind == 'features':
        result = proc.chunk_job(*args)
    elif kind == 'matrices':
        result = cls.materialize(*args)
    else:
        result = cls.job(*args)
    return kind, args, result, time.perf_counter()-started, time.process_time()-cpu


def verify_clean(pool, datasets):
    pending = {}
    for d in datasets:
        if not (root(d)/proc.VERIFICATION).exists():
            n = len(np.load(data_root()/d/'prepared/frames.npy', mmap_mode='r', allow_pickle=False))
            pending[d] = [pool.submit(execute, 'verify', (d, int(i))) for i in np.linspace(0, n-1, 50, dtype=int)]
    for d, futures in pending.items():
        rows = [r for f in futures for r in f.result()[2]]
        save_json(root(d)/proc.VERIFICATION, dict(rows=rows, ok=all(r['ok'] for r in rows)))
    for d in datasets:
        check = json.loads((root(d)/proc.VERIFICATION).read_text())
        if not check['ok']:
            raise RuntimeError(f'{d}: clean features do not reproduce; stopping before contamination')
        exact = sum(r['exact'] for r in check['rows'])
        print(f'{d}: clean verification passed, {exact}/{len(check["rows"])} exact', flush=True)


def schedule(pool, workers, datasets):
    """Dependency-aware longest-first scheduler over every remaining job."""
    ready, blocked, inflight = [], {}, {}
    order = iter(range(10**9))
    stages = {d: dict(features=None, classification=None, started=time.perf_counter()) for d in datasets}
    remaining_chunks, remaining_fits = {}, {}
    totals = dict(features=0, fits=0)

    def push(priority, kind, args):
        heapq.heappush(ready, (-priority, next(order), kind, args))

    def unlock_fits(d):
        jobs = cls.fit_jobs(d)
        remaining_fits[d] = len(jobs)
        totals['fits'] += len(jobs)
        pending = {args for _, args in jobs}
        for cost, args in jobs:
            dependency = cls.shift_dependency(args)
            if dependency is not None and dependency in pending:
                blocked.setdefault(dependency, []).append((cost, args))
            else:
                push(cost, 'fit', args)
        if not jobs:
            stages[d]['classification'] = 0.0

    for d in datasets:
        jobs = proc.feature_jobs(d)
        remaining_chunks[d] = len(jobs)
        totals['features'] += len(jobs)
        # Feature chunks first, the largest datasets' first.
        for args in jobs:
            push(1e9 + (args[3]-args[2]) * {'dronerf': 4, 'birdclef2023': 2}.get(d, 1), 'features', args)
        if not jobs:
            push(2e9, 'matrices', (d,))
    done = dict(features=0, fits=0)
    print(f'{now()} scheduling {totals["features"]} feature chunks', flush=True)
    while ready or inflight:
        while ready and len(inflight) < workers:
            _, _, kind, args = heapq.heappop(ready)
            inflight[pool.submit(execute, kind, args)] = (kind, args)
        finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
        for future in finished:
            kind, args = inflight.pop(future)
            _, _, result, seconds, _ = future.result()
            d = args[0] if kind in ('features', 'matrices') else args[1]
            if kind == 'features':
                done['features'] += 1
                remaining_chunks[d] -= 1
                if remaining_chunks[d] == 0:
                    stages[d]['features'] = time.perf_counter()-stages[d]['started']
                    print(f'{now()} {d}: features complete', flush=True)
                    push(2e9, 'matrices', (d,))
                if done['features'] % 200 == 0:
                    check_disk()
                    print(f'{now()} features {done["features"]}/{totals["features"]}', flush=True)
            elif kind == 'matrices':
                print(f'{now()} {d}: matrices ready, {seconds:.1f} s', flush=True)
                cls.write_manifest(d)
                unlock_fits(d)
                stages[d]['fits_started'] = time.perf_counter()-stages[d]['started']
            else:
                done['fits'] += 1
                remaining_fits[d] -= 1
                for cost, child in blocked.pop(args, []):
                    push(cost, 'fit', child)
                if remaining_fits[d] == 0:
                    stages[d]['classification'] = time.perf_counter()-stages[d]['started']-stages[d]['fits_started']
                    print(f'{now()} {d}: classification complete', flush=True)
                if done['fits'] % 20 == 0 or not (ready or inflight):
                    check_disk()
                    print(f'{now()} fits {done["fits"]}/{totals["fits"]} (last {args[0]} {d} {args[2]} '
                          f'{args[3]} {args[4]} split {args[5]}, {seconds:.0f} s)', flush=True)
    if blocked:
        raise RuntimeError(f'unresolved dependencies: {list(blocked)[:3]}')
    return stages


def main_run(workers, datasets):
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f'workers must be 1..{MAX_WORKERS}')
    pimf = verify()
    runs_dir().mkdir(parents=True, exist_ok=True)
    with (runs_dir()/'active.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('a robustness_laplace run is already active') from error
        for d in datasets:
            proc.write_manifest(d)
        single_thread_env()
        limit_threads()
        started, stamp = time.perf_counter(), now()
        print(f'{stamp} start, {workers} workers, pimf {pimf["version"]}', flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
            verify_clean(pool, datasets)
            stages = schedule(pool, workers, datasets)
        verify()
        save_json(runs_dir()/f'run_{stamp.replace(":", "")}.json',
                  dict(started_utc=stamp, finished_utc=now(), wall_seconds=time.perf_counter()-started,
                       workers=workers, datasets=list(datasets), stages=stages, pimf=verify()))
        print(f'{now()} {", ".join(datasets)} complete', flush=True)


def detach(workers, datasets):
    verify()
    runs_dir().mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_path = runs_dir()/f'{stamp}.log'
    command = [sys.executable, '-m', 'robustness_laplace.run', '--workers', str(workers), '--datasets', *datasets]
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    record = dict(pid=process.pid, utc=now(), command=command, cwd=os.getcwd(), log=str(log_path))
    save_json(runs_dir()/f'{stamp}.json', record)
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    if args.detach:
        detach(args.workers, args.datasets)
    else:
        main_run(args.workers, args.datasets)


if __name__ == '__main__':
    main()
