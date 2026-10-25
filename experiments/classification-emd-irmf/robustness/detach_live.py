"""One-time, checkpoint-boundary handoff from an inherited pipe to a detached run."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

from robustness.environment import verify
from robustness.storage import digest, root, save_json


def checkpoints():
    return sorted((root('dronerf')/'classification').rglob('complete.json'))


def now():
    return datetime.now(timezone.utc).isoformat()


def old_processes(parent_pid, child_pid):
    parent, child = psutil.Process(parent_pid), psutil.Process(child_pid)
    if child.ppid() != parent_pid or os.getpgid(parent_pid) != parent_pid or os.getpgid(child_pid) != parent_pid:
        raise RuntimeError('Original batch process identity changed')
    workers = [p for p in child.children() if 'spawn_main' in ' '.join(p.cmdline())]
    if len(workers) != 6:
        raise RuntimeError(f'Expected six live classifier workers, found {len(workers)}')
    return parent, child, workers


def handoff(parent_pid, child_pid, minimum):
    verify()
    folder = root('dronerf')/'runs/restarts'
    folder.mkdir(parents=True, exist_ok=True)
    while True:
        old_processes(parent_pid, child_pid)
        n = len(checkpoints())
        if n >= minimum:
            break
        print(f'{now()} waiting for current fit checkpoints: {n}/{minimum}', flush=True)
        time.sleep(15)
        verify()

    # These six workers have all published one more immutable fit. Freeze the
    # process group to take a consistent ledger before it starts further jobs.
    os.killpg(parent_pid, signal.SIGSTOP)
    time.sleep(0.3)
    try:
        _, _, workers = old_processes(parent_pid, child_pid)
        paths = checkpoints()
        if len(paths) < minimum:
            raise RuntimeError('Checkpoint count fell after freeze')
        cpu_total = sum(p.cpu_times().user + p.cpu_times().system for p in workers)
        complete_cpu = 0.0
        for path in paths:
            value = json.loads(path.read_text())
            complete_cpu += value['fit_cpu_seconds'] + value['predict_cpu_seconds']
        if cpu_total < complete_cpu:
            raise RuntimeError('Worker CPU ledger is inconsistent')
        manifest = root('dronerf')/'classification_manifest.json'
        snapshot = dict(utc=now(), old_parent_pid=parent_pid, old_child_pid=child_pid,
            old_worker_pids=[p.pid for p in workers], completed_fit_checkpoints=len(paths),
            completed_sha256={str(p.relative_to(root('dronerf'))): digest(p) for p in paths},
            classification_segment_wall_seconds=time.time()-manifest.stat().st_mtime,
            wall_origin='classification_manifest.json modification time; upper bound by initialization time',
            original_worker_cpu_seconds=cpu_total,
            completed_fit_and_predict_cpu_seconds=complete_cpu,
            uncheckpointed_worker_cpu_seconds=cpu_total-complete_cpu,
            pimf=verify(), reason='Detach the original pipe-backed batch after provider interruption')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        save_json(folder/'preflight'/f'{stamp}.json', snapshot)
    except BaseException:
        os.killpg(parent_pid, signal.SIGCONT)
        raise

    os.killpg(parent_pid, signal.SIGTERM)
    os.killpg(parent_pid, signal.SIGCONT)
    deadline = time.monotonic()+60
    while time.monotonic() < deadline:
        live = [pid for pid in [parent_pid, child_pid, *snapshot['old_worker_pids']]
                if psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE]
        if not live:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError(f'Old batch did not stop; refusing duplicate launch: {live}')

    result = subprocess.run([sys.executable, '-m', 'robustness.run_real', '--detach'],
                            check=True, text=True, capture_output=True)
    launched = json.loads(result.stdout)
    time.sleep(2)
    if not psutil.pid_exists(launched['pid']):
        raise RuntimeError(f'Detached batch exited during startup: {launched}')
    for relative, expected in snapshot['completed_sha256'].items():
        if digest(root('dronerf')/relative) != expected:
            raise RuntimeError(f'Completed checkpoint changed after restart: {relative}')
    snapshot.update(detached=launched, resumed_checkpoint_count=len(checkpoints()),
                    stopped_old_group=True, completed_checkpoints_unchanged=True)
    save_json(folder/f'{stamp}.json', snapshot)
    print(json.dumps({k:snapshot[k] for k in ('utc','completed_fit_checkpoints',
        'classification_segment_wall_seconds','uncheckpointed_worker_cpu_seconds','detached')},
        indent=2), flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--child-pid', type=int, required=True)
    parser.add_argument('--minimum', type=int, required=True)
    args=parser.parse_args()
    handoff(args.parent_pid, args.child_pid, args.minimum)


if __name__=='__main__':
    main()
