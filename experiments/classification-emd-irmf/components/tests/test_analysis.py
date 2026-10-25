import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV

from components.analysis import paired_tests, registered_pairs, survival_reading
from components.classification import CLASSICAL, classical_estimator
from components.budget import eligible, selection
from components.construction import all_tags
from components.integrity import DATASETS, check_pimf
from emdirmf.stats import holm


def test_registered_family_sizes_and_orientations():
    assert [len(registered_pairs(family)) for family in ("A", "B", "C")] == [12, 17, 8]
    assert registered_pairs("A")[0][2:] == ("irmf_native_k3", "emd_k3")
    assert registered_pairs("C")[-2][2:] == ("gd60_trunc_k8", "irmf_h2_a1.41")


def fake_scores():
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
                                     "split_id": split_id, "seed": split_id // 5,
                                     "n_train": 800, "n_test": 200,
                                     "balanced_accuracy": score, "macro_f1": score - .02,
                                     "accuracy": score + .02})
    return pd.DataFrame(rows)


def test_holm_boundaries_and_complete_pairing():
    scores = fake_scores()
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


def test_tree_seed_cut_preserves_primary_tests_and_corrects_five_folds():
    scores = fake_scores()
    full = paired_tests(scores)
    plan = selection(1)
    selected = scores[scores.apply(lambda row: eligible(row, plan), axis=1)]
    cut = paired_tests(selected, plan)
    keys = ["family", "variant", "dataset", "metric", "method_a", "method_b"]
    first = full[full.classifier != "lightgbm"].set_index(keys).sort_index()
    second = cut[cut.classifier != "lightgbm"].set_index(keys).sort_index()
    for column in ("mean_diff", "ci_low", "ci_high", "p_nb", "p_nb_holm"):
        np.testing.assert_array_equal(first[column], second[column])
    trees = cut[cut.classifier == "lightgbm"]
    assert (trees.n_splits == 5).all() and (trees.df == 4).all()
    for _, cell in trees.groupby("holm_family"):
        assert len(cell) == (48 if cell.iloc[0].family == "A" else 8)
        np.testing.assert_array_equal(cell.p_nb_holm, holm(cell.p_nb))


def test_last_tree_cut_uses_only_pre_specified_available_pairs():
    scores = fake_scores()
    plan = selection(3)
    scores = scores[scores.apply(lambda row: eligible(row, plan), axis=1)]
    tests = paired_tests(scores, plan)
    trees = tests[tests.classifier == "lightgbm"]
    assert set(trees.feature_set) == {"comp"}
    for _, cell in trees.groupby("holm_family"):
        assert len(cell) == (24 if cell.iloc[0].family == "A" else 4)


def test_classical_models_preserve_historical_tuning():
    drone = classical_estimator("dronerf")
    assert drone.named_steps["clf"].C == .1
    assert drone.named_steps["clf"].max_iter == 5000
    for dataset in ("synthetic", "birdclef2023", "esc50_animals"):
        model = classical_estimator(dataset)
        assert isinstance(model, GridSearchCV) and model.n_jobs == 1
        assert model.cv.n_splits == 3
    check_pimf()
