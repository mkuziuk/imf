import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV

from components.analysis import paired_tests, registered_pairs, survival_reading
from components.classification import CLASSICAL, classical_estimator
from components.construction import all_tags
from components.integrity import DATASETS, check_pimf
from emdirmf.stats import holm


def test_registered_family_sizes_and_orientations():
    assert [len(registered_pairs(family)) for family in ("A", "B", "C")] == [12, 17, 8]
    assert registered_pairs("A")[0][2:] == ("irmf_native_k3", "emd_k3")
    assert registered_pairs("C")[-2][2:] == ("gd60_trunc_k8", "irmf_h2_a1.41")


def test_holm_boundaries_and_complete_pairing():
    rows = []
    rng = np.random.default_rng(70)
    for dataset in DATASETS:
        for classifier in (CLASSICAL[dataset], "lightgbm"):
            for feature_set in ("comp", "comp+log_scale"):
                for method in all_tags():
                    for split_id in range(15):
                        score = .6 + rng.normal(scale=.03)
                        rows.append({"dataset": dataset, "classifier": classifier,
                                     "feature_set": feature_set, "method": method,
                                     "split_id": split_id, "n_train": 800, "n_test": 200,
                                     "balanced_accuracy": score, "macro_f1": score - .02,
                                     "accuracy": score + .02})
    scores = pd.DataFrame(rows)
    tests = paired_tests(scores)
    for _, cell in tests.groupby("holm_family"):
        expected = {"A": 48, "B": 17, "C": 8}[cell.iloc[0].family]
        assert len(cell) == expected
        np.testing.assert_array_equal(cell.p_nb_holm, holm(cell.p_nb))
    reading = survival_reading(tests)
    assert len(reading) == 4 and not reading.survives_matched_counts.any()
    # Silently accepting incomplete pairs would change the pre-registered family.
    with pytest.raises(AssertionError):
        paired_tests(scores.drop(scores.index[0:30]))


def test_classical_models_preserve_historical_tuning():
    drone = classical_estimator("dronerf")
    assert drone.named_steps["clf"].C == .1
    assert drone.named_steps["clf"].max_iter == 5000
    for dataset in ("synthetic", "birdclef2023", "esc50_animals"):
        model = classical_estimator(dataset)
        assert isinstance(model, GridSearchCV) and model.n_jobs == 1
        assert model.cv.n_splits == 3
    check_pimf()
