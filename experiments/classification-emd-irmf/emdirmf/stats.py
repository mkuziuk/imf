"""Paired method comparisons over the shared CV splits (PROTOCOL.md, "Evaluation").

All tests take per-split differences d_j = score_A - score_B on the same J
splits (J = 5 folds x 3 seeds = 15).

- Nadeau-Bengio corrected resampled t-test (Nadeau & Bengio 2003; the repeated
  k-fold form of Bouckaert & Frank 2004): var = (1/J + n_test/n_train) s^2,
  t = mean(d) / sqrt(var), df = J - 1. The 95% CI uses the same corrected
  standard error, so it is wider than the naive paired interval.
- Wilcoxon signed-rank test on d (zero differences dropped; p = 1 if all are 0).
- Holm step-down adjustment within each comparison table.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .decompose import MAIN_TAGS, parse_method

DEFAULT_FAMILY_PAIRS = (
    ("irmf", "emd"),
    ("irmf", "imf_linear"),
    ("imf_linear", "emd"),
    ("irmf", "raw"),
    ("imf_linear", "raw"),
    ("emd", "raw"),
)


def corrected_resampled_ttest(differences, n_train, n_test, *, confidence=0.95):
    """Nadeau-Bengio corrected t-test on paired per-split differences."""
    d = np.asarray(differences, dtype=float)
    J = d.size
    if J < 2:
        raise ValueError("need at least two paired splits")
    ratio = float(np.mean(np.asarray(n_test, dtype=float) / np.asarray(n_train, dtype=float)))
    mean = float(d.mean())
    variance = float(d.var(ddof=1))
    se = np.sqrt((1.0 / J + ratio) * variance)
    df = J - 1
    if se > 0:
        t = mean / se
        p = float(2 * scipy_stats.t.sf(abs(t), df))
    else:
        t = 0.0 if mean == 0 else np.copysign(np.inf, mean)
        p = 1.0 if mean == 0 else 0.0
    half = float(scipy_stats.t.ppf(0.5 + confidence / 2, df) * se)
    return {
        "mean_diff": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "se_corrected": float(se),
        "t": float(t),
        "df": df,
        "p_nb": p,
        "n_test_over_n_train": ratio,
    }


def wilcoxon_signed_rank(differences):
    d = np.asarray(differences, dtype=float)
    nonzero = d[d != 0]
    if nonzero.size == 0:
        return {"wilcoxon_stat": 0.0, "p_wilcoxon": 1.0}
    result = scipy_stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
    return {"wilcoxon_stat": float(result.statistic), "p_wilcoxon": float(result.pvalue)}


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (NaNs are left out and stay NaN)."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(p[valid], kind="stable")]
    m = order.size
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_comparison(scores_a, scores_b, n_train, n_test):
    """All statistics for one pair of aligned per-split score vectors."""
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    d = a - b
    out = {"n_splits": d.size, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "wins_a": int((d > 0).sum()), "ties": int((d == 0).sum()), "wins_b": int((d < 0).sum())}
    out.update(corrected_resampled_ttest(d, n_train, n_test))
    out.update(wilcoxon_signed_rank(d))
    return out


def representative_tags(methods):
    """Map family -> tag used in default comparisons (main tag if present, else first seen)."""
    chosen = {}
    for tag in methods:
        family = parse_method(tag).family
        if family not in chosen or tag in MAIN_TAGS:
            chosen[family] = tag
    return chosen


def default_pairs(methods):
    """Keep the primary pair order, then include every other distinct tag pair.

    Ablation tags must not disappear behind one representative of their family.
    Cross-family contrasts retain DEFAULT_FAMILY_PAIRS orientation; same-family
    contrasts use input order. The four-method default is unchanged.
    """
    methods = list(dict.fromkeys(methods))
    chosen = representative_tags(methods)
    pairs = [(chosen[a], chosen[b]) for a, b in DEFAULT_FAMILY_PAIRS if a in chosen and b in chosen]
    seen = {frozenset(pair) for pair in pairs}
    for a, b in combinations(methods, 2):
        if frozenset((a, b)) in seen:
            continue
        families = (parse_method(a).family, parse_method(b).family)
        if families[::-1] in DEFAULT_FAMILY_PAIRS:
            a, b = b, a
        pairs.append((a, b))
        seen.add(frozenset((a, b)))
    return pairs


def to_long(per_split, metrics=("balanced_accuracy", "macro_f1", "accuracy"), id_columns=None):
    """Wide per-split results (one metric per column) -> long (metric, value) table."""
    id_columns = [column for column in per_split.columns if column not in metrics] if id_columns is None \
        else list(id_columns)
    return per_split.melt(id_vars=id_columns, value_vars=list(metrics), var_name="metric", value_name="value")


def compare_methods(
    results,
    *,
    metric="balanced_accuracy",
    pairs=None,
    by=("classifier",),
    method_column="method",
    split_column="split_id",
):
    """Pairwise method comparisons from a long results table.

    results needs columns method, split_id, metric, value, n_train, n_test and the
    `by` columns (e.g. classifier, feature_set). Each row of the output compares
    method_a - method_b within one `by` cell on matched splits. Holm adjustment is
    applied over all rows of the returned table, separately for both tests.
    """
    table = results[results["metric"] == metric]
    by = list(by)
    methods = list(dict.fromkeys(table[method_column]))
    pairs = default_pairs(methods) if pairs is None else pairs
    rows = []
    cells = table.groupby(by, sort=False) if by else [((), table)]
    for key, cell in cells:
        key = key if isinstance(key, tuple) else (key,)
        for method_a, method_b in pairs:
            a = cell[cell[method_column] == method_a].set_index(split_column)
            b = cell[cell[method_column] == method_b].set_index(split_column)
            common = a.index.intersection(b.index)
            if len(common) < 2:
                continue
            a, b = a.loc[common], b.loc[common]
            row = dict(zip(by, key))
            row.update(metric=metric, method_a=method_a, method_b=method_b)
            row.update(paired_comparison(a["value"], b["value"], a["n_train"], a["n_test"]))
            rows.append(row)
    comparison = pd.DataFrame(rows)
    if len(comparison):
        comparison["p_nb_holm"] = holm(comparison["p_nb"])
        comparison["p_wilcoxon_holm"] = holm(comparison["p_wilcoxon"])
    return comparison
