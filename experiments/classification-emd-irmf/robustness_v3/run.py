"""One pool for verification, fresh clean features, contamination and dependent fits."""
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
from robustness_v3 import classification as cls
from robustness_v3 import processing as proc
from robustness_v3.construction import METHODS, parse_variant
from robustness_v3.contamination import DATASETS
from robustness_v3.estimators import await_decision
from robustness_v3.storage import check_disk, root, save_json

MAX_WORKERS = 14


def runs_dir():
    return data_root()/'robustness_v3_runs'


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    single_thread_env()
    os.environ['NUMBA_NUM_THREADS'] = '1'
    limit_threads()
    verify()
    warm_up()


def execute(kind, args):
    started, cpu = time.perf_counter(), time.process_time()
    if kind == 'verify':
        result = proc.verification_job(*args)
    elif kind == 'clean':
        result = proc.chunk_job(*args, clean=True)
    elif kind == 'features':
        result = proc.chunk_job(*args)
    elif kind == 'clean_matrices':
        result = proc.materialize_condition(*args, 'clean')
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
        save_json(root(d)/proc.VERIFICATION, dict(version=2, raw_stack_dtype='float64', rows=rows, ok=all(r['ok'] for r in rows)))
    for d in datasets:
        value = json.loads((root(d)/proc.VERIFICATION).read_text())
        if not value['ok']:
            raise RuntimeError(f'{d}: clean verification failed; STOP before contamination')
        print(f'{d}: clean verification passed, {sum(r["exact"] for r in value["rows"])}/100 exact', flush=True)


def schedule(pool, workers, datasets):
    """Longest-first queue; matched groups unlock only after every K x split."""
    ready, inflight, enqueued, unlocked = [], {}, set(), set()
    order = iter(range(10**9))
    started = time.perf_counter()
    stages = {d: {} for d in datasets}
    remaining = {}
    done = {'clean': 0, 'features': 0, 'fit': 0}

    def push(priority, kind, args):
        key = kind, args
        if key not in enqueued:
            enqueued.add(key)
            heapq.heappush(ready, (-priority, next(order), kind, args))

    def unlock_selected(d):
        for method in METHODS:
            for clf in (cls.CLASSICAL[d], 'lightgbm'):
                value = cls.maybe_select(d, method, clf)
                if value is not None and (d, method, clf) not in unlocked:
                    unlocked.add((d, method, clf))
                    print(f'{now()} {d} {clf} {method}: K*={value["k"]}, clean BA={value["mean_clean_ba"]:.6f}', flush=True)
                    if clf == cls.CLASSICAL[d]:
                        for cost, args in cls.matched_jobs(d, method):
                            push(cost, 'fit', args)
        cls.write_k_summary(d)

    def add_clean(d):
        jobs = proc.chunk_jobs(d, clean=True)
        remaining[d, 'clean'] = len(jobs)
        for args in jobs:
            push(3e9 + (args[2]-args[1]) * (2 if d == 'dronerf' else 1), 'clean', args)
        if not jobs:
            push(4e9, 'clean_matrices', (d,))

    for d in datasets:
        add_clean(d)
    print(f'{now()} scheduling fresh clean chunks: {sum(remaining.values())}', flush=True)
    while ready or inflight:
        while ready and len(inflight) < workers:
            _, _, kind, args = heapq.heappop(ready)
            inflight[pool.submit(execute, kind, args)] = kind, args
        finished, _ = wait(inflight, timeout=30, return_when=FIRST_COMPLETED)
        if not finished:
            print(f'{now()} active={len(inflight)} queued={len(ready)} completed={done}', flush=True)
        for future in finished:
            kind, args = inflight.pop(future)
            _, _, _, seconds, _ = future.result()
            d = args[1] if kind == 'fit' else args[0]
            if kind in ('clean', 'features'):
                done[kind] += 1
                remaining[d, kind] -= 1
                if remaining[d, kind] == 0:
                    stages[d][kind+'_seconds_from_start'] = time.perf_counter()-started
                    print(f'{now()} {d}: {kind} complete', flush=True)
                    push(4e9, 'clean_matrices' if kind == 'clean' else 'matrices', (d,))
                if done[kind] % 25 == 0:
                    print(f'{now()} {kind} chunks completed={done[kind]}, last={d}/{args[1]}, {seconds:.1f}s', flush=True)
                    check_disk()
            elif kind == 'clean_matrices':
                jobs = proc.chunk_jobs(d)
                remaining[d, 'features'] = len(jobs)
                for a in jobs:
                    push(2e9 + (a[2]-a[1]) * (2 if d == 'dronerf' else 1), 'features', a)
                if not jobs:
                    push(4e9, 'matrices', (d,))
                print(f'{now()} {d}: clean matrices ready; contaminated chunks={len(jobs)}', flush=True)
            elif kind == 'matrices':
                cls.write_manifest(d)
                for cost, a in cls.fit_jobs(d):
                    push(cost, 'fit', a)
                unlock_selected(d)
                print(f'{now()} {d}: all matrices ready', flush=True)
            else:
                done['fit'] += 1
                unlock_selected(d)
                if done['fit'] % 20 == 0 or not (ready or inflight):
                    print(f'{now()} fits completed={done["fit"]}, last={args}, {seconds:.1f}s', flush=True)
                    check_disk()
    for d in datasets:
        cls.write_k_summary(d)
        if not (root(d)/'k_star.json').exists():
            raise RuntimeError(f'{d}: K selection incomplete')
        expected = len(cls.dataset_variants(d))*2*15 + 4*15
        actual = len(list((root(d)/'classification').rglob('complete.json')))
        if actual != expected:
            raise RuntimeError(f'{d}: {actual} fits instead of {expected}')
    return stages


def main_run(workers, datasets, verify_only=False):
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f'workers must be 1..{MAX_WORKERS}')
    pimf = verify()
    runs_dir().mkdir(parents=True, exist_ok=True)
    with (runs_dir()/'active.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('a robustness_v3 run is already active') from error
        check_disk()
        decision = await_decision()
        print(f'H decision: {json.dumps(decision)}', flush=True)
        for d in datasets:
            proc.write_manifest(d)
        initialize()
        started, stamp = time.perf_counter(), now()
        print(f'{stamp} start, {workers} workers, pimf {pimf["version"]}', flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
            verify_clean(pool, datasets)
            if verify_only:
                return
            stages = schedule(pool, workers, datasets)
        verify()
        save_json(runs_dir()/f'run_{stamp.replace(":", "")}.json',
            dict(started_utc=stamp, finished_utc=now(), wall_seconds=time.perf_counter()-started,
                 workers=workers, datasets=list(datasets), stages=stages, pimf=verify()))
        print(f'{now()} {", ".join(datasets)} complete', flush=True)


def detach(workers, datasets):
    verify()
    single_thread_env()
    os.environ['NUMBA_NUM_THREADS'] = '1'
    runs_dir().mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log_path = runs_dir()/f'{stamp}.log'
    command = [sys.executable, '-m', 'robustness_v3.run', '--workers', str(workers), '--datasets', *datasets]
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
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    if args.detach:
        detach(args.workers, args.datasets)
    else:
        main_run(args.workers, args.datasets, args.verify_only)


if __name__ == '__main__':
    main()
