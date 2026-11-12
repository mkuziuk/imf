from components.budget import eligible, project, selection
from components.construction import BASE_TAGS, all_tags


def test_cuts_apply_in_order_without_cutting_classical():
    hour_cpu = 6 * 3600 / 1.2
    for fixed_hours, tree_hours, expected in (
        (8, [2, 1, .5, .25], 0),
        (8, [6, 2, 1, .5], 1),
        (9, [8, 4, 2, 1], 2),
        (11, [5, 2, 1.2, .5], 3),
    ):
        plan = project(0, fixed_hours * hour_cpu, [t * hour_cpu for t in tree_hours])
        assert plan["level"] == expected
        assert len(plan["projections"]) == expected + 1
        assert all(p["future_classical_cpu_seconds"] == fixed_hours * hour_cpu for p in plan["projections"])
        for tag in all_tags():
            for seed in (0, 1, 2):
                for feature_set in ("comp", "comp+log_scale"):
                    assert eligible({"classifier": "linsvm", "seed": seed,
                                     "feature_set": feature_set, "method": tag}, plan)


def test_budget_never_discards_defaults_and_only_last_tier_cuts_counts():
    for level in (0, 1, 2, 3):
        plan = selection(level)
        assert set(BASE_TAGS) <= set(plan["lightgbm_tags"])
        assert "gd60_trunc_k8" in plan["lightgbm_tags"]
        if level < 3:
            assert plan["lightgbm_tags"] == all_tags()
    assert selection(1)["lightgbm_seeds"] == [0]
    assert selection(2)["lightgbm_feature_sets"] == ["comp"]
    assert "linear_native_k4" in selection(3)["lightgbm_tags"]
    assert "irmf_native_k6" in selection(3)["lightgbm_tags"]
    assert "linear_native_k3" not in selection(3)["lightgbm_tags"]
    # Exhausting the allowed cuts never removes mandatory classical jobs.
    plan = project(0, 13 * 6 * 3600 / 1.2, [1, 1, 1, 1])
    assert plan["level"] == 3 and plan["projected_total_wall_hours"] > 12
