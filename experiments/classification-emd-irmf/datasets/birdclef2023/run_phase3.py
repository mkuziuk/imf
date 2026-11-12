#!/usr/bin/env python3
"""Run shared ML code with probability exports and per-fit checkpoints.

Only the BirdCLEF output is extended. Model fitting and tuning stay in the
shared runners. Probability columns permit recording-level probability means.
"""

import argparse
import importlib
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))

import numpy as np
import pandas as pd
from scipy.special import softmax


def main():
    runner, *arguments = sys.argv[1:]
    if runner not in {"trees", "rocket", "cnn"}:
        raise ValueError("first argument must be trees, rocket, or cnn")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-name", default=runner)
    options, arguments = parser.parse_known_args(arguments)
    module = importlib.import_module(f"ml.run_{runner}")
    state = {}
    original_setup = module.setup
    original_record = module.record

    def setup(args, name):
        if args.dataset != "birdclef2023":
            raise ValueError("this probability/checkpoint wrapper is BirdCLEF-specific")
        result = original_setup(args, name)
        result[0]["runner"] = options.run_name
        result[0]["recording_confidence"] = (
            "softmax of SGD decision scores, auxiliary and uncalibrated" if runner == "rocket"
            else "native class probabilities")
        state["out"] = result[-1]
        state["task"] = result[0]["task"]
        return result

    module.setup = setup
    if runner == "trees":
        original_fit = module.fit_tree

        def fit_tree(name, args, X, y, groups, split):
            result = original_fit(name, args, X, y, groups, split)
            model, imputer, _ = result
            state["probabilities"] = model.predict_proba(imputer.transform(X[split.test]))
            return result

        module.fit_tree = fit_tree
    elif runner == "rocket":
        original_predict = module.predict_head

        def predict_head(values, scales, indices, head, args):
            prediction = original_predict(values, scales, indices, head, args)
            scaler, model = head
            probabilities = []
            for block in module.blocks(indices, args.head_batch_size):
                scores = model.decision_function(
                    scaler.transform(module.input_features(values, block, scales)))
                if scores.ndim == 1:
                    scores = np.column_stack([np.zeros(len(scores)), scores])
                # OvR sigmoid probabilities can saturate and lose the ordering
                # used by predict. Softmax retains that ordering without
                # changing the fitted classifier or its frame predictions.
                probabilities.append(softmax(scores, axis=1))
            state["probabilities"] = np.concatenate(probabilities)
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
        original_record(rows, predictions, split, y, pred, *args, **kwargs)
        probabilities = state.pop("probabilities")
        labels = np.unique(y)
        if probabilities.shape != (len(split.test), len(labels)):
            raise AssertionError("probability dimensions do not match test rows/classes")
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-5)
        np.testing.assert_array_equal(labels[probabilities.argmax(axis=1)], pred)
        for index, label in enumerate(labels):
            predictions[-1][f"prob_{label}"] = probabilities[:, index].astype(np.float32)
        pd.DataFrame(rows).to_csv(state["out"] / f"checkpoint_{state['task']}.csv", index=False)
        # Keep large frame-level predictions outside the repository.
        from emdirmf.paths import results_dir

        destination = results_dir("birdclef2023")
        destination.mkdir(parents=True, exist_ok=True)
        pd.concat(predictions, ignore_index=True).to_parquet(
            destination / f"{options.run_name}_{state['task']}_checkpoint.parquet", index=False)

    module.record = record
    return module.main(arguments)


if __name__ == "__main__":
    sys.exit(main())
