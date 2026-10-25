"""Run one dataset under a kernel-released, dataset-specific lock."""
import argparse
import fcntl
import subprocess
import sys

from robustness.contamination import DATASETS
from robustness.environment import verify
from robustness.storage import root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--workers', required=True, type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError('workers must be 1..6')
    verify()
    folder = root(args.dataset)/'runs'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'active.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'{args.dataset} robustness run is already active') from error
        subprocess.run([sys.executable, '-m', 'robustness.run', '--dataset',
                        args.dataset, '--workers', str(args.workers)], check=True)
    verify()


if __name__ == '__main__':
    main()
