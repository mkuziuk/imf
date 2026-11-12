"""Shared package for the EMD vs IMF vs IRMF classification study.

Put experiments/classification-emd-irmf on sys.path, then `import emdirmf`.
See README.md in this folder and ../PROTOCOL.md.
"""

from .cv import Split, evaluate, fit_predict_split, load_splits, make_splits, split_metrics
from .decompose import (
    ABLATION_TAGS,
    K_EMD,
    MAIN_TAGS,
    MethodSpec,
    canonical_tag,
    decompose_frame,
    decompose_signal,
    emd_components,
    imf_h_min,
    imf_linear_components,
    imf_schedule,
    irmf_components,
    n_slots,
    noise_sigma_hat,
    normalize_frame,
    parse_method,
    scale_order,
    slot_labels,
)
from .features import FEATURE_NAMES, component_features, feature_columns, frame_features
from .io import load_prepared, validate_prepared
from .paths import data_root
from .stats import compare_methods, corrected_resampled_ttest, holm, wilcoxon_signed_rank

# The CLI modules load lazily so `python -m emdirmf.run_*` does not import them twice.
_LAZY = {
    "load_decomposition": "run_decompose",
    "run_decompose": "run_decompose",
    "load_features": "run_features",
    "run_features": "run_features",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ABLATION_TAGS",
    "FEATURE_NAMES",
    "K_EMD",
    "MAIN_TAGS",
    "MethodSpec",
    "Split",
    "canonical_tag",
    "compare_methods",
    "component_features",
    "corrected_resampled_ttest",
    "data_root",
    "decompose_frame",
    "decompose_signal",
    "emd_components",
    "evaluate",
    "feature_columns",
    "fit_predict_split",
    "frame_features",
    "holm",
    "imf_h_min",
    "imf_linear_components",
    "imf_schedule",
    "irmf_components",
    "load_decomposition",
    "load_features",
    "load_prepared",
    "load_splits",
    "make_splits",
    "n_slots",
    "noise_sigma_hat",
    "normalize_frame",
    "parse_method",
    "run_decompose",
    "run_features",
    "scale_order",
    "slot_labels",
    "split_metrics",
    "validate_prepared",
    "wilcoxon_signed_rank",
]
