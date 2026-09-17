"""Fit descriptive power laws and draw measured full-decomposition timings."""
from __future__ import annotations

import hashlib
import json
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

SPECS = [("quantlet", 1, "QuantLet · 1 thread", "#f0b56b", "s"),
         ("gd", 1, "My GD · 1 thread", "#8ab4f8", "o"),
         ("gd", 4, "My GD · 4 threads", "#80cbc4", "^"),
         ("gd", 12, "My GD · 12 threads", "#ce9cf2", "D")]


def fit_power_law(n, seconds):
    """Unweighted least squares in log space: T = a * (n / 1000) ** p.

    The normalization makes a the fitted time at 1000 points. This describes
    the observed range, without claiming an asymptotic complexity or an error bar.
    """
    n, seconds = np.asarray(n, dtype=float), np.asarray(seconds, dtype=float)
    if (n.ndim != 1 or n.shape != seconds.shape or len(n) < 3 or
            len(np.unique(n)) != len(n) or not np.isfinite(n).all() or
            not np.isfinite(seconds).all() or (n <= 0).any() or (seconds <= 0).any()):
        raise ValueError("At least three unique, positive sizes and finite positive times are required.")
    x, y = np.log(n / 1000), np.log(seconds)
    power, intercept = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    residual = y - (intercept + power * x)
    total = np.sum((y - y.mean()) ** 2)
    return {"power": float(power), "seconds_at_1000": float(np.exp(intercept)),
            "doubling_factor": float(2 ** power),
            "r_squared_log": float(1 - np.sum(residual ** 2) / total) if total > 0 else None,
            "point_count": len(n), "n_min": int(n.min()), "n_max": int(n.max())}


def fit_scaling(result):
    frame = pd.DataFrame(result["runs"])
    common_sizes = set.intersection(*(
        set(frame[(frame.method == method) & (frame.workers == workers)].n)
        for method, workers, *_ in SPECS))
    fits = []
    for method, workers, label, *_ in SPECS:
        group = frame[(frame.method == method) & (frame.workers == workers)].sort_values("n")
        fit = fit_power_law(group.n, group.seconds)
        common = group[group.n.isin(common_sizes)]
        fit.update(method=method, workers=workers, label=label,
                   shared_range_fit=fit_power_law(common.n, common.seconds))
        fits.append(fit)
    timing_data = [{"n": int(row.n), "method": row.method, "workers": int(row.workers),
                    "seconds": float(row.seconds)}
                   for row in frame.sort_values(["n", "method", "workers"]).itertuples()]
    return {"model": "T_seconds = seconds_at_1000 * (n / 1000) ** power",
            "fit_method": "Unweighted ordinary least squares of log(time) on log(n / 1000), one point per measured size.",
            "scope": "Descriptive fit over each measured range, with a second fit restricted to the shared range. No extrapolation, asymptotic claim or timing uncertainty estimate.",
            "timings_sha256": hashlib.sha256(json.dumps(timing_data, sort_keys=True).encode()).hexdigest(),
            "fits": fits}


def save_fits(fitted, output):
    (output / "scaling_fits.json").write_text(json.dumps(fitted, indent=2, allow_nan=False) + "\n")
    rows = []
    for fit in fitted["fits"]:
        row = {key: value for key, value in fit.items() if key != "shared_range_fit"}
        row.update({"shared_" + key: value for key, value in fit["shared_range_fit"].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "scaling_fits.csv", index=False)


def normalized_timings(result):
    """Divide each configuration's timings by its measured time at 1000 points."""
    frame = pd.DataFrame(result["runs"])
    baseline = frame[frame.n == 1000][["method", "workers", "seconds"]].rename(
        columns={"seconds": "baseline_seconds"})
    frame = frame.merge(baseline, on=["method", "workers"], how="left", validate="many_to_one")
    if frame.baseline_seconds.isna().any() or (frame.baseline_seconds <= 0).any():
        raise ValueError("Every configuration needs a positive measured time at 1000 points.")
    frame["time_multiple"] = frame.seconds / frame.baseline_seconds
    return frame.sort_values(["n", "method", "workers"])


def draw_scaling(result, save):
    assert result["metadata"].get("complete"), "The size sweep must finish before plotting."
    frame = normalized_timings(result)
    quantlet_limit = result["metadata"].get("quantlet_max_n", max(result["sizes"]))
    expected = {(n, method, workers) for n in result["sizes"] for method, workers, *_ in SPECS
                if method != "quantlet" or n <= quantlet_limit}
    assert set(zip(frame.n, frame.method, frame.workers)) == expected
    assert len(frame) == len(expected)
    assert (frame.seconds > 0).all()
    fitted = fit_scaling(result)
    for units, scale, show_fit in product(["seconds", "relative"], ["linear", "loglog"], [False, True]):
        fig, ax = plt.subplots(figsize=(10.5, 5.6), layout="constrained")
        for (method, workers, label, color, marker), fit in zip(SPECS, fitted["fits"]):
            group = frame[(frame.method == method) & (frame.workers == workers)].sort_values("n")
            values = group.time_multiple if units == "relative" else group.seconds
            ax.plot(group.n, values, color=color, marker=marker, markersize=5,
                    linestyle="none" if show_fit else "-", linewidth=2.2, label=label, zorder=3)
            if show_fit:
                n = np.geomspace(group.n.min(), group.n.max(), 160)
                # Normalize each fitted curve by its own fitted baseline,
                # so the existing power is preserved and the line starts at 1.
                fitted_values = (n / 1000) ** fit["power"]
                if units == "seconds":
                    fitted_values *= fit["seconds_at_1000"]
                ax.plot(n, fitted_values,
                        color=color, linestyle=(0, (5, 4)), linewidth=2, alpha=0.9, zorder=2)
        ax.set(xlabel="Number of points" + (" · log scale" if scale == "loglog" else ""),
               ylabel=("Time / own time at 1,000 points" if units == "relative" else
                       "Seconds for the full decomposition") + (" · log scale" if scale != "linear" else ""),
               title=("Relative growth · each method starts at 1×" if units == "relative" else
                      "My default settings at every signal length"))
        ax.set_yscale("linear" if scale == "linear" else "log")
        if scale == "linear":
            ax.set_ylim(bottom=0)
        if scale == "loglog":
            ax.set_xscale("log")
            ax.set_xticks([1000, 2000, 5000, 10000, 20000])
            ax.set_xlim(900, 23000)
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
        else:
            ax.set_xticks([1000, 5000, 10000, 15000, 20000])
            ax.set_xlim(600, 20400)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        if units == "relative":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
            ax.axhline(1, color="#b2b8c0", linewidth=0.8, linestyle=":", alpha=0.65)
        ax.grid(True, axis="both", which="major")
        if show_fit:
            ax.plot([], [], color="#b2b8c0", linestyle="--",
                    label="Growth from fitted power" if units == "relative" else "Fitted power law")
        ax.legend(loc="upper left", ncol=2, fontsize=10.5)
        save(fig, "scaling_" + ("relative_" if units == "relative" else "") + scale +
             ("_fit" if show_fit else ""))
    return frame.sort_values(["n", "method", "workers"]), fitted
