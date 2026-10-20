#!/usr/bin/env python3
"""LightGBM fusion ablation from existing EMD and IRMF feature parquets."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))

from emdirmf.paths import features_dir
from ml import run_trees
import run_phase3


def main():
    original_setup = run_trees.setup
    original_read = pd.read_parquet
    original_order = run_trees.scale_order
    original_labels = run_trees.slot_labels
    original_finish = run_trees.finish

    def setup(args, runner):
        result = list(original_setup(args, runner))
        if any(not method.startswith("irmf_") for method in result[5]):
            raise ValueError("--methods must name the IRMF feature sets to concatenate with EMD")
        result[5] = [f"emd__{method}" for method in result[5]]
        result[0]["methods"] = result[5]
        return tuple(result)

    def read_parquet(path, *extra, **kwargs):
        path = Path(path)
        if path.parent.name != "features" or not path.stem.startswith("emd__"):
            return original_read(path, *extra, **kwargs)
        import re

        tag = path.stem[5:]
        emd = original_read(features_dir("esc50_animals") / "emd.parquet")
        irmf = original_read(features_dir("esc50_animals") / f"{tag}.parquet")
        np.testing.assert_array_equal(emd.frame_id, irmf.frame_id)
        np.testing.assert_array_equal(emd.ch0_log_scale, irmf.ch0_log_scale)
        assert emd.decomp_ok.all() and irmf.decomp_ok.all()
        columns = run_trees.feature_columns(irmf.columns, include_log_scale=False)
        n_emd = len(original_order("emd", 2048))
        renamed = {column: re.sub(r"_c(\d+)_", lambda match: f"_c{int(match.group(1)) + n_emd}_",
                                  column) for column in columns}
        return pd.concat([emd, irmf[columns].rename(columns=renamed)], axis=1)

    def scale_order(tag, length):
        if not tag.startswith("emd__"):
            return original_order(tag, length)
        emd = original_order("emd", length)
        return emd + [slot + len(emd) for slot in original_order(tag[5:], length)]

    def slot_labels(tag, length):
        if not tag.startswith("emd__"):
            return original_labels(tag, length)
        return [f"emd/{label}" for label in original_labels("emd", length)] + [
            f"irmf/{label}" for label in original_labels(tag[5:], length)]

    def finish(context, out, rows, predictions, notes=(), extra=None):
        return original_finish(context, out, rows, predictions, notes=[
            *notes, "EMD and IRMF component statistics are concatenated; shared log_scale appears once.",
            "No new decomposition or feature parquet is generated. Identical grouped tuning/refit and boosting budget.",
            "Fusion importance order is two blocks, EMD then IRMF, each in its true scale order.",
        ], extra=extra)

    run_trees.setup = setup
    pd.read_parquet = read_parquet
    run_trees.scale_order = scale_order
    run_trees.slot_labels = slot_labels
    run_trees.finish = finish
    sys.argv.insert(1, "trees")
    return run_phase3.main()


if __name__ == "__main__":
    sys.exit(main())
