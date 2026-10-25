#!/usr/bin/env python3
"""Run shared ML code with probability exports and per-fit checkpoints.

Only DroneRF output is extended. Model fitting and tuning stay in the shared
runners. Probability columns permit segment-level probability means.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))

import numpy as np
import pandas as pd


def main():
    runner, *arguments = sys.argv[1:]
    if runner not in {"trees", "rocket", "cnn"}:
        raise ValueError("first argument must be trees, rocket, or cnn")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-name", default=f"dronerf_phase3_{runner}")
    parser.add_argument("--lightgbm-rounds", type=int, default=None)
    parser.add_argument("--cheap-extra-trees", action="store_true")
    parser.add_argument("--l-only", action="store_true", help="use only the L-band view")
    parser.add_argument("--cache-stacks", action="store_true", help="cache exactly the shared CNN reader's tensors")
    parser.add_argument("--slots", type=int, default=None,
                        help="tree ablation: retain at most K evenly spaced true scale ranks")
    parser.add_argument("--worker-plan", default=str(STUDY.parents[1] /
                        "output/classification-emd-irmf/dronerf/ml/worker_plan.json"))
    options, arguments = parser.parse_known_args(arguments)
    module = importlib.import_module(f"ml.run_{runner}")
    state = {}
    original_setup = module.setup
    original_record = module.record

    def setup(args, name):
        plan = Path(options.worker_plan)
        if plan.exists():
            allocation = json.loads(plan.read_text()).get(f"{options.run_name}/{args.task}")
            if allocation is not None:
                args.workers = int(allocation)
        result = original_setup(args, name)
        result[0]["runner"] = options.run_name
        state["out"] = result[-1]
        state["task"] = result[0]["task"]
        result[0]["local_budget"] = vars(options)
        return result

    module.setup = setup
    if options.cache_stacks:
        if runner != "cnn" or options.l_only:
            raise ValueError("stack caching is used by the full-view CNN only")
        from stack_cache import CachedStackReader

        module.StackReader = CachedStackReader
    if options.l_only and runner == "trees":
        base_columns = module.feature_columns

        def l_columns(columns, include_log_scale=False):
            return [c for c in base_columns(columns, include_log_scale=include_log_scale)
                    if c.startswith("ch0_")]

        module.feature_columns = l_columns
    elif options.l_only:
        from emdirmf.stacks import StackReader

        class LBandReader(StackReader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.frames = self.frames[:, :1]
                if self.source is not None:
                    self.source = self.source[:, :1]
                self.shape = (self.shape[0], len(self.order), self.shape[2])

        module.StackReader = LBandReader
    if runner == "trees":
        original_fit = module.fit_tree
        original_estimator = module.estimator

        def estimator(name, args, seed, n_estimators, *extra, **kwargs):
            if name == "lightgbm" and options.lightgbm_rounds is not None:
                n_estimators = min(n_estimators, options.lightgbm_rounds)
            return original_estimator(name, args, seed, n_estimators, *extra, **kwargs)

        module.estimator = estimator
        if options.slots is not None:
            if options.slots < 1:
                raise ValueError("slots must be positive")
            original_read = pd.read_parquet
            original_columns = module.feature_columns

            def read_parquet(path, *extra, **kwargs):
                path = Path(path)
                if path.parent.name == "features":
                    state["feature_method"] = path.stem
                return original_read(path, *extra, **kwargs)

            def feature_columns(columns, include_log_scale=False):
                import re

                selected = original_columns(columns, include_log_scale=include_log_scale)
                order = module.scale_order(state["feature_method"], 2048)
                ranks = np.unique(np.rint(np.linspace(0, len(order) - 1,
                                                      min(options.slots, len(order)))).astype(int))
                slots = {order[rank] for rank in ranks}
                return [column for column in selected if "log_scale" in column or
                        int(re.match(r"ch\d+_c(\d+)_", column).group(1)) in slots]

            pd.read_parquet = read_parquet
            module.feature_columns = feature_columns

        def fit_tree(name, args, X, y, groups, split):
            previous_quick = args.quick
            if options.cheap_extra_trees and name != "lightgbm":
                args.quick = True
            try:
                result = original_fit(name, args, X, y, groups, split)
            finally:
                args.quick = previous_quick
            model, imputer, _ = result
            if model.n_features_in_ != X.shape[1]:
                raise AssertionError("fitted model feature count does not match the design matrix")
            state["input_features"] = int(X.shape[1])
            state["probabilities"] = model.predict_proba(imputer.transform(X[split.test]))
            return result

        module.fit_tree = fit_tree
    elif runner == "rocket":
        original_predict = module.predict_head

        def predict_head(values, scales, indices, head, args):
            prediction = original_predict(values, scales, indices, head, args)
            scaler, model = head
            state["probabilities"] = np.concatenate([
                model.predict_proba(scaler.transform(module.input_features(values, block, scales)))
                for block in module.blocks(indices, args.head_batch_size)
            ])
            return prediction

        module.predict_head = predict_head
    else:
        original_epoch = module.epoch_pass

        def epoch_pass(model, loader, loss_fn, device, optimizer=None):
            if optimizer is not None or loss_fn.weight is not None:
                return original_epoch(model, loader, loss_fn, device, optimizer)
            batches = []

            def capture(_model, _inputs, output):
                batches.append(output.detach().softmax(dim=1).cpu().numpy())

            hook = model.register_forward_hook(capture)
            try:
                result = original_epoch(model, loader, loss_fn, device, optimizer)
            finally:
                hook.remove()
            state["probabilities"] = np.concatenate(batches)
            return result

        module.epoch_pass = epoch_pass

    def record(rows, predictions, split, y, pred, *args, **kwargs):
        if runner == "trees":
            kwargs["n_input_features"] = state.pop("input_features")
        original_record(rows, predictions, split, y, pred, *args, **kwargs)
        probabilities = state.pop("probabilities")
        labels = np.unique(y)
        if probabilities.shape != (len(split.test), len(labels)):
            raise AssertionError("probability dimensions do not match test rows/classes")
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-5)
        # SGD uses decision-score argmax; saturated one-vs-rest probabilities
        # can tie, so their first argmax need not reproduce the predicted label.
        positions = np.searchsorted(labels, pred)
        np.testing.assert_allclose(probabilities[np.arange(len(pred)), positions],
                                   probabilities.max(axis=1), atol=2e-6)
        for index, label in enumerate(labels):
            predictions[-1][f"prob_{label}"] = probabilities[:, index].astype(np.float32)
        pd.DataFrame(rows).to_csv(state["out"] / f"checkpoint_{state['task']}.csv", index=False)
        # Keep large frame-level predictions outside the repository.
        from emdirmf.paths import results_dir

        destination = results_dir("dronerf")
        destination.mkdir(parents=True, exist_ok=True)
        parts = destination / f"{options.run_name}_{state['task']}_parts"
        parts.mkdir(parents=True, exist_ok=True)
        row = rows[-1]
        filename = "__".join(str(row[key]) for key in
                             ("method", "feature_set", "classifier", "split_id")) + ".parquet"
        predictions[-1].to_parquet(parts / filename, index=False)

    module.record = record
    return module.main(arguments)


if __name__ == "__main__":
    sys.exit(main())
