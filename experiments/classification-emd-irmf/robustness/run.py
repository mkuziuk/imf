"""Execute a dataset without modifying shared code, data or old results."""
import argparse
import time

from robustness.contamination import DATASETS
from robustness.environment import verify
from robustness.storage import root, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--stage', choices=('features', 'classify', 'all'), default='all')
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError('workers must be 1..6')
    started = time.perf_counter()
    verify()
    if args.stage in ('features', 'all'):
        from robustness.processing import prepare
        prepare(args.dataset, args.workers)
    if args.stage in ('classify', 'all'):
        from robustness.classification import run
        run(args.dataset, args.workers)
    path = root(args.dataset)/'runs'/f'{args.stage}.json'
    if not path.exists():
        save_json(path, dict(wall_seconds=time.perf_counter()-started, pimf=verify()))


if __name__ == '__main__':
    main()
