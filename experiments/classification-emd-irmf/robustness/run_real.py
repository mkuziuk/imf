"""Run real datasets serially, with an optional session-independent launcher."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from robustness.contamination import DATASETS
from robustness.environment import verify
from robustness.storage import root, save_json


def launch_detached():
    """Give the batch its own session and file-backed standard streams."""
    verify()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder = root('dronerf')/'runs/detached'
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder/f'{stamp}.log'
    command = [sys.executable, '-m', 'robustness.run_real']
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   close_fds=True)
    record = dict(pid=process.pid, utc=datetime.now(timezone.utc).isoformat(),
                  command=command, cwd=os.getcwd(), log=str(log_path), pimf=verify())
    save_json(folder/f'{stamp}.json', record)
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detach', action='store_true', help='launch with file-backed stdout/stderr')
    args = parser.parse_args()
    if args.detach:
        launch_detached()
        return
    verify()
    folder = root('dronerf')/'runs'
    folder.mkdir(parents=True, exist_ok=True)
    # Kernel-released lock prevents two batch launchers from creating two pools.
    with (folder/'run_real.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('A robustness real-data batch is already running') from error
        started = time.perf_counter()
        for dataset in DATASETS[1:]:
            print(f'Starting {dataset}', flush=True)
            subprocess.run([sys.executable, '-m', 'robustness.run', '--dataset', dataset,
                            '--workers', '6'], check=True)
        path = folder/'real_batch.json'
        if not path.exists():
            save_json(path, dict(wall_seconds=time.perf_counter()-started, pimf=verify()))


if __name__ == '__main__':
    main()
