"""Process-pool helpers: one BLAS/OpenMP thread per worker process."""

import os

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
MAX_WORKERS = int(os.environ.get("IMF_CLS_MAX_WORKERS", "6"))


def single_thread_env():
    """Set thread env vars to 1 so spawned children start single-threaded."""
    for name in THREAD_ENV_VARS:
        os.environ[name] = "1"


def limit_threads():
    """Limit already-loaded BLAS/OpenMP pools in this process (used in initializers)."""
    single_thread_env()
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass


def check_workers(workers):
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers > MAX_WORKERS:
        print(f"note: capping workers at {MAX_WORKERS} (set IMF_CLS_MAX_WORKERS to change)")
        workers = MAX_WORKERS
    return workers
