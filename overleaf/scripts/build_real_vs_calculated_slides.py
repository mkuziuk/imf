"""Recompute paired linear/robust IMF comparisons using recursive RMSE only.

Run from any directory with the repository's Python requirements installed.
Use --plots-only to redraw the saved data without rerunning the experiment.
Source notebooks and the review's original figures are never overwritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "experiments" / "real-vs-calculated"
OUTPUT = ROOT / "overleaf" / "figures" / "real_vs_calculated_slides"
SINGLE_CELLS = [2, 4, 6, 8, 9]
SWEEP_CELLS = [3, 5, 7, 9, 10, 14, 16]
SINGLE_SIGMA = 0.6
SINGLE_P = 0.2
SINGLE_CONTAMINATION_SCALE = 2.0
SINGLE_SEED = 777
HEATMAP_SIGMAS = [0.4, 0.6, 0.8]
HEATMAP_PROBABILITIES = np.round(np.arange(0.1, 0.301, 0.05), 2)
HEATMAP_SCALES = np.arange(1.0, 3.01, 0.5)
WAVEFORM_REFERENCE = "clean_linear_1"
BLUE = "#2563A6"
ORANGE = "#C65B32"
INK = "#172B3A"
TEAL = "#087F8C"


def execute_notebook(filename, indices):
    """Use the original numerical cells, retaining their assertions and seeds."""
    path = SOURCES / filename
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "slide_source", "__file__": str(path)}
    print(f"Executing {filename}", flush=True)
    for index in indices:
        cell = notebook["cells"][index]
        if cell["cell_type"] != "code":
            raise ValueError(f"Expected code at {filename}, cell {index}")
        exec(compile("".join(cell["source"]), f"{filename}:cell-{index}", "exec"), namespace)
        namespace["display"] = lambda *args, **kwargs: None
        plt.close("all")
    return namespace


def configure_plots():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.titleweight": "normal",
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 14,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#A7B5BF",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDE5E9",
        "grid.alpha": 0.8,
        "axes.axisbelow": True,
        "lines.linewidth": 2.6,
        "lines.markersize": 6,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })



METHODS = [("linear", BLUE), ("robust", ORANGE)]
MODEL_LABELS = {"additive": "Additive contamination", "masked": "Masked pure-noise contamination"}


def decompose(ns, method, signal, sigma):
    if method == "linear":
        result = ns["linear_imf_with_history"](signal, ns["window_sizes"])
    else:
        result = ns["robust_gd_imf_with_history"](
            signal, ns["window_sizes"], H=2 * sigma, grid=ns["lookup_grid"],
            max_iter=60, tol=1e-6, max_workers=ns["DEFAULT_MAX_WORKERS"],
        )
    np.testing.assert_allclose(result["reconstruction"], signal, rtol=0, atol=1e-10)
    return result


def stage_rows(ns, reference, calculated, metadata):
    values = np.sqrt(np.mean((calculated["imfs"] - reference["imfs"]) ** 2, axis=1))
    assert np.isfinite(values).all()
    return [dict(metadata, stage=k, window_size=w, rmse=float(value))
            for k, w, value in zip(range(1, 10), ns["window_sizes"], values)]


def write_csv(frame, name):
    frame.to_csv(OUTPUT / name, index=False, float_format="%.12g")


def shared_reference_rmse(traces):
    rows = []
    for method, _ in METHODS:
        rmse = np.sqrt(np.mean((traces[f"masked_{method}_1"] - traces[WAVEFORM_REFERENCE]) ** 2))
        rows.append({"model": "masked", "method": method, "stage": 1,
                     "reference": WAVEFORM_REFERENCE, "rmse": float(rmse)})
    return pd.DataFrame(rows)


def recompute_data():
    single_sources = {"additive": "gd_imf_real_vs_calculated.ipynb",
                      "masked": "gd_imf_real_vs_calculated_noise_only.ipynb"}
    sweep_sources = {"additive": "gd_imf_real_vs_calculated_limits.ipynb",
                     "masked": "gd_imf_real_vs_calculated_limits_noise_only.ipynb"}
    rows, traces = [], None
    for model, filename in single_sources.items():
        ns = execute_notebook(filename, SINGLE_CELLS)
        observed, noise_info = ns["generate_observation"](
            ns["x_clean"], sigma=SINGLE_SIGMA, contamination_prob=SINGLE_P,
            contamination_scale=SINGLE_CONTAMINATION_SCALE,
            rng=np.random.default_rng(SINGLE_SEED),
        )
        if traces is None:
            traces = pd.DataFrame({"t": ns["t"], "clean": ns["x_clean"],
                "contaminated": noise_info["contamination_mask"].astype(int)})
        else:
            np.testing.assert_allclose(traces["clean"], ns["x_clean"], rtol=0, atol=0)
            np.testing.assert_array_equal(traces["contaminated"], noise_info["contamination_mask"])
        traces[f"{model}_observation"] = observed
        for method, _ in METHODS:
            ref = decompose(ns, method, ns["x_clean"], SINGLE_SIGMA)
            calc = decompose(ns, method, observed, SINGLE_SIGMA)
            rows.extend(stage_rows(ns, ref, calc, {"model": model, "method": method,
                "sigma": SINGLE_SIGMA, "contamination_probability": SINGLE_P,
                "contamination_scale": SINGLE_CONTAMINATION_SCALE}))
            if model == "additive":
                traces[f"clean_{method}_1"] = ref["imfs"][0]
            else:
                np.testing.assert_allclose(traces[f"clean_{method}_1"], ref["imfs"][0], rtol=0, atol=1e-12)
            traces[f"{model}_{method}_1"] = calc["imfs"][0]
    np.testing.assert_allclose(traces["additive_observation"] - traces["masked_observation"],
                              traces["clean"] * traces["contaminated"], rtol=0, atol=1e-12)
    current = pd.DataFrame(rows)
    assert len(current) == 36
    assert current.groupby(["model", "method"]).size().eq(9).all()
    # Check stored traces against the stage-1 errors using method-matched references.
    for model in single_sources:
        for method, _ in METHODS:
            trace_rmse = np.sqrt(np.mean(
                (traces[f"{model}_{method}_1"] - traces[f"clean_{method}_1"]) ** 2))
            value = current.loc[current["model"].eq(model) & current["method"].eq(method)
                                & current["stage"].eq(1), "rmse"].item()
            np.testing.assert_allclose(value, trace_rmse, rtol=0, atol=1e-12)
    write_csv(current, "paired_stage_rmse.csv")
    write_csv(traces, "paired_traces.csv")
    write_csv(shared_reference_rmse(traces), "masked_first_stage_shared_rmse.csv")
    rows = []
    for model, filename in sweep_sources.items():
        ns = execute_notebook(filename, SWEEP_CELLS)
        refs = {m: decompose(ns, m, ns["x_clean"], 0.2) for m, _ in METHODS}
        for ratio in [2, 4, 6, 8, 10, 12]:
            for repeat in range(3):
                observed, _ = ns["generate_coupled_observation"](ns["x_clean"], sigma=0.2,
                    contamination_prob=0.2, contamination_scale=0.2 * ratio, repeat=repeat)
                for method, _ in METHODS:
                    calc = decompose(ns, method, observed, 0.2)
                    rows.extend(stage_rows(ns, refs[method], calc, {"model": model, "method": method,
                        "sigma": 0.2, "contamination_probability": 0.2, "contamination_scale": 0.2 * ratio,
                        "scale_ratio": ratio, "repeat": repeat}))
            print(f"Completed {model}, contamination scale / sigma = {ratio}", flush=True)
    stages = pd.DataFrame(rows)
    assert len(stages) == 648
    assert stages.groupby(["model", "method", "scale_ratio", "repeat"]).size().eq(9).all()
    write_csv(stages, "contamination_stage_rmse.csv")
    aggregates = stages.groupby(["model", "method", "sigma", "contamination_probability",
        "contamination_scale", "scale_ratio", "repeat"], as_index=False)["rmse"].mean()
    write_csv(aggregates.rename(columns={"rmse": "mean_rmse"}), "contamination_mean_rmse.csv")
    recompute_heatmap_data()
    recompute_heatmap_data("masked")
    manifest = {
        "source_sha256": {name: hashlib.sha256((SOURCES / name).read_bytes()).hexdigest()
                          for name in [*single_sources.values(), *sweep_sources.values()]},
        "single_run_cells": SINGLE_CELLS, "sweep_definition_cells": SWEEP_CELLS,
        "metric": "sqrt(mean((calculated_imf - same_method_clean_imf)**2)) over time",
        "aggregate": "arithmetic mean of nine stage RMSE values, per repeat",
        "presentation_setting": {"sigma": SINGLE_SIGMA, "p": SINGLE_P,
                                 "contamination_scale": SINGLE_CONTAMINATION_SCALE,
                                 "seed": SINGLE_SEED},
        "sweep_setting": {"sigma": 0.2, "p": 0.2, "scale_ratios": [2, 4, 6, 8, 10, 12],
                          "repeats": 3, "seeds": [777000, 777001, 777002], "trials_per_model": 18},
        "heatmap_setting": heatmap_setting(),
        "waveform_reference": WAVEFORM_REFERENCE,
        "component_definition": {"source": "IMF.pdf", "section": "2.2", "page": 3,
                                 "equivalent_equation": "3.2, page 9",
                                 "sha256": hashlib.sha256((ROOT / "IMF.pdf").read_bytes()).hexdigest()},
        "robust_H": "2 * sigma", "kernel": "normalized (1 - abs(u))_+^2; periodic wrap boundaries",
        "validation": "Decomposition reconstructions agree with inputs to atol 1e-10.",
    }
    (OUTPUT / "rmse_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


def heatmap_setting():
    return {"models": ["additive", "masked"], "sigmas": HEATMAP_SIGMAS,
            "probabilities": HEATMAP_PROBABILITIES.tolist(),
            "contamination_scales": HEATMAP_SCALES.tolist(), "repeats": 3,
            "seeds": [777000, 777001, 777002],
            "trials_per_model": len(HEATMAP_SIGMAS) * len(HEATMAP_PROBABILITIES) * len(HEATMAP_SCALES) * 3,
            "color": "one logarithmic scale of unscaled mean RMSE shared by both models"}


def recompute_heatmap_data(model="additive"):
    """Evaluate a grid around the default setting using raw recursive RMSE."""
    filenames = {"additive": "gd_imf_real_vs_calculated_limits.ipynb",
                 "masked": "gd_imf_real_vs_calculated_limits_noise_only.ipynb"}
    ns = execute_notebook(filenames[model], SWEEP_CELLS)
    rows = []
    for sigma in HEATMAP_SIGMAS:
        refs = {m: decompose(ns, m, ns["x_clean"], sigma) for m, _ in METHODS}
        for probability in HEATMAP_PROBABILITIES:
            for scale in HEATMAP_SCALES:
                for repeat in range(3):
                    observed, _ = ns["generate_coupled_observation"](
                        ns["x_clean"], sigma=sigma, contamination_prob=probability,
                        contamination_scale=scale, repeat=repeat,
                    )
                    for method, _ in METHODS:
                        calc = decompose(ns, method, observed, sigma)
                        rows.extend(stage_rows(ns, refs[method], calc, {
                            "model": model, "method": method, "sigma": sigma,
                            "contamination_probability": probability,
                            "contamination_scale": scale, "scale_ratio": scale / sigma,
                            "repeat": repeat,
                        }))
            print(f"Completed {model} heatmap sigma={sigma}, p={probability}", flush=True)
    stages = pd.DataFrame(rows)
    assert len(stages) == heatmap_setting()["trials_per_model"] * 2 * 9
    groups = ["model", "method", "sigma", "contamination_probability",
              "contamination_scale", "scale_ratio", "repeat"]
    assert stages.groupby(groups).size().eq(9).all()
    aggregate = stages.groupby(groups, as_index=False)["rmse"].mean()
    default = aggregate.loc[aggregate["sigma"].eq(SINGLE_SIGMA)
                            & aggregate["contamination_probability"].eq(SINGLE_P)
                            & aggregate["contamination_scale"].eq(SINGLE_CONTAMINATION_SCALE)]
    assert len(default) == 6  # Both methods, with three paired repeats at the default.
    prefix = "" if model == "additive" else "masked_"
    write_csv(stages, f"{prefix}heatmap_stage_rmse.csv")
    write_csv(aggregate.rename(columns={"rmse": "mean_rmse"}), f"{prefix}heatmap_mean_rmse.csv")


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def method_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)


def draw_observations(traces):
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.2), sharex=True, sharey=True, layout="constrained")
    mask = traces["contaminated"].astype(bool)
    for ax, model in zip(axes, ["additive", "masked"]):
        observed = traces[f"{model}_observation"]
        ax.plot(traces["t"], observed, color="#A7B5BF", lw=0.65)
        ax.scatter(traces.loc[mask, "t"], observed[mask], s=9, color=ORANGE, alpha=0.65)
        ax.plot(traces["t"], traces["clean"], color=INK, lw=1.8, label="Clean signal")
        ax.set_title(MODEL_LABELS[model], loc="left", fontsize=15)
        ax.set_ylabel("Signal")
        ax.set_xlim(0, 1)
    axes[0].legend(frameon=False, loc="upper right", fontsize=12)
    axes[1].set_xlabel("Time t")
    save(fig, "paired_observations")


def draw_current_rmse(current):
    for model in ["additive", "masked"]:
        fig, ax = plt.subplots(figsize=(12.5, 4.1), layout="constrained")
        for method, color in METHODS:
            part = current.loc[current["model"].eq(model) & current["method"].eq(method)]
            ax.plot(part["stage"], part["rmse"], "o-", color=color, label=method.capitalize())
        ax.set_xlabel("Stage k, window shrinks from 501 to 31 points")
        ax.set_ylabel("Recursive component RMSE")
        ax.set_xticks(range(1, 10))
        ax.set_ylim(bottom=0)
        method_legend(fig, ax)
        save(fig, f"{model}_method_rmse")


def draw_masked_first_stage(traces):
    fig, ax = plt.subplots(figsize=(13, 4.1), layout="constrained")
    for method, color in METHODS:
        ax.plot(traces["t"], traces[f"masked_{method}_1"], color=color,
                label=f"Calculated {method}")
    write_csv(shared_reference_rmse(traces), "masked_first_stage_shared_rmse.csv")
    ax.plot(traces["t"], traces[WAVEFORM_REFERENCE], color=INK, ls="--", lw=2,
            label="Clean IMF reference")
    ax.set_xlabel("Time t")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Stage 1 IMF component")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    save(fig, "masked_first_stage_methods")


def plot_bands(ax, frame, x, value, method, color):
    part = frame.loc[frame["method"].eq(method)].groupby(x)[value].agg(
        median="median", q25=lambda v: v.quantile(0.25), q75=lambda v: v.quantile(0.75)).reset_index()
    ax.plot(part[x], part["median"], "o-", color=color, label=method.capitalize())
    ax.fill_between(part[x], part["q25"], part["q75"], color=color, alpha=0.14)


def draw_magnitude(aggregate):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True, layout="constrained")
    for ax, model in zip(axes, ["additive", "masked"]):
        frame = aggregate.loc[aggregate["model"].eq(model)]
        for method, color in METHODS:
            plot_bands(ax, frame, "scale_ratio", "mean_rmse", method, color)
        ax.set_title(MODEL_LABELS[model], loc="left")
        ax.set_xlabel(r"Contamination scale / $\sigma$")
        ax.set_xticks(range(2, 13, 2))
    axes[0].set_ylabel("Mean RMSE over nine stages")
    axes[0].set_ylim(bottom=0)
    method_legend(fig, axes[0])
    save(fig, "contamination_magnitude_rmse")


def draw_strong_contamination(stages):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True, layout="constrained")
    for ax, model in zip(axes, ["additive", "masked"]):
        frame = stages.loc[stages["model"].eq(model) & stages["scale_ratio"].eq(12)]
        for method, color in METHODS:
            plot_bands(ax, frame, "stage", "rmse", method, color)
        ax.set_title(MODEL_LABELS[model], loc="left")
        ax.set_xlabel("Stage k")
        ax.set_xticks(range(1, 10))
    axes[0].set_ylabel("Recursive component RMSE")
    axes[0].set_ylim(bottom=0)
    method_legend(fig, axes[0])
    save(fig, "strong_contamination_stage_rmse")


def draw_masked_magnitude(stages, aggregate):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
    panels = [(stages.loc[stages["stage"].eq(1)], "rmse", "Stage 1 RMSE"),
              (aggregate, "mean_rmse", "Mean RMSE over nine stages")]
    for ax, (frame, value, title) in zip(axes, panels):
        part = frame.loc[frame["model"].eq("masked")]
        for method, color in METHODS:
            plot_bands(ax, part, "scale_ratio", value, method, color)
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"Contamination scale / $\sigma$")
        ax.set_ylabel("RMSE")
        ax.set_xticks(range(2, 13, 2))
        ax.set_ylim(bottom=0)
    method_legend(fig, axes[0])
    save(fig, "masked_magnitude_rmse")


def draw_sweep_heatmaps(aggregate, model, norm):
    surface = aggregate.groupby(["method", "sigma", "contamination_probability",
                                 "contamination_scale"], as_index=False)["mean_rmse"].median()
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 5), sharex=True, sharey=True,
                             layout="constrained")
    for col, sigma in enumerate(HEATMAP_SIGMAS):
        for row, (method, _) in enumerate(METHODS):
            panel = surface.loc[surface["sigma"].eq(sigma) & surface["method"].eq(method)]
            grid = panel.pivot(index="contamination_probability", columns="contamination_scale",
                               values="mean_rmse")
            ax = axes[row, col]
            ax.grid(False)
            mesh = ax.pcolormesh(np.arange(0.75, 3.26, 0.5), np.arange(0.075, 0.326, 0.05),
                                 grid.to_numpy(), cmap="viridis", norm=norm, shading="flat")
            if sigma == SINGLE_SIGMA:
                ax.scatter([SINGLE_CONTAMINATION_SCALE], [SINGLE_P], marker="*", s=220,
                           facecolor="white", edgecolor=INK, linewidth=0.9, zorder=3)
            if row == 0:
                ax.set_title(rf"$\sigma={sigma:g}$")
            if col == 0:
                ax.set_ylabel(f"{method.capitalize()}\np")
            if row == 1:
                ax.set_xlabel("Contamination scale")
            ax.set_xticks(HEATMAP_SCALES)
            ax.set_yticks([0.1, 0.2, 0.3])
    colorbar = fig.colorbar(mesh, ax=axes, shrink=0.9, pad=0.02)
    colorbar.set_label("Median mean RMSE over nine stages")
    save(fig, "sweep_heatmaps" if model == "additive" else "masked_sweep_heatmaps")


def draw_all():
    configure_plots()
    current = pd.read_csv(OUTPUT / "paired_stage_rmse.csv")
    traces = pd.read_csv(OUTPUT / "paired_traces.csv")
    stages = pd.read_csv(OUTPUT / "contamination_stage_rmse.csv")
    aggregate = pd.read_csv(OUTPUT / "contamination_mean_rmse.csv")
    draw_observations(traces)
    draw_current_rmse(current)
    draw_masked_first_stage(traces)
    draw_magnitude(aggregate)
    draw_strong_contamination(stages)
    draw_masked_magnitude(stages, aggregate)
    heatmaps = {"additive": pd.read_csv(OUTPUT / "heatmap_mean_rmse.csv"),
                "masked": pd.read_csv(OUTPUT / "masked_heatmap_mean_rmse.csv")}
    medians = pd.concat(heatmaps.values()).groupby([
        "model", "method", "sigma", "contamination_probability", "contamination_scale"
    ])["mean_rmse"].median()
    norm = LogNorm(vmin=medians.min(), vmax=medians.max())
    for model, frame in heatmaps.items():
        draw_sweep_heatmaps(frame, model, norm)
    print(f"Saved RMSE figures to {OUTPUT.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--plots-only", action="store_true")
    options.add_argument("--data-only", action="store_true")
    options.add_argument("--heatmap-only", action="store_true",
                         help="Recompute the heatmap using existing main and sweep data, then redraw figures.")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    if args.heatmap_only:
        manifest_path = OUTPUT / "rmse_provenance.json"
        manifest = json.loads(manifest_path.read_text())
        for filename, digest in manifest["source_sha256"].items():
            assert hashlib.sha256((SOURCES / filename).read_bytes()).hexdigest() == digest
        recompute_heatmap_data()
        recompute_heatmap_data("masked")
        manifest["heatmap_setting"] = heatmap_setting()
        manifest["waveform_reference"] = WAVEFORM_REFERENCE
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    elif not args.plots_only:
        recompute_data()
    if not args.data_only:
        draw_all()


if __name__ == "__main__":
    main()
