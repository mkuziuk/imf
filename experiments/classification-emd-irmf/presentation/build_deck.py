"""Rebuild the content slides of the IRMF classification Keynote deck.

Drives /Applications/Keynote.app through osascript. Slide 1 (the reference
deck's title slide) is kept and retitled; every later slide is deleted and
rebuilt from the deck's "Content Slide Enumeration" master, so the script can
be rerun after new results arrive:

    .venv/bin/python experiments/classification-emd-irmf/presentation/make_deck_charts.py \
        --laplace output/classification-emd-irmf/robustness_laplace
    .venv/bin/python experiments/classification-emd-irmf/presentation/build_deck.py \
        --laplace output/classification-emd-irmf/robustness_laplace

Tables and slide text are filled from the same result CSVs as the charts; the
Laplace slides read ``--laplace DIR`` and need all four datasets there; the
notebook-scale spike slides read ``--v3 DIR`` (default
output/classification-emd-irmf/robustness_v3). Add
``--images /tmp/irmf_deck/final --close`` to export slide PNGs and close the
document; a PDF is written next to the deck unless ``--no-pdf``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "output" / "classification-emd-irmf"
DECK_DIR = Path.home() / "Documents" / "IRMF decks"
DECK_NAME = "20261022 Spo LI Hae Kuz IRMF classification and robustness"
CHARTS = DECK_DIR / "charts"
MASTER = "Content Slide Enumeration"
MINUS = "\u2212"
DS = {"dronerf": "DroneRF", "birdclef2023": "BirdCLEF 2023", "esc50_animals": "ESC-50 animals",
      "uas_birds_60ghz": "60 GHz radar", "synthetic": "Synthetic"}
ORDER = ["dronerf", "birdclef2023", "esc50_animals", "synthetic"]
CLF = {"dronerf": "linsvm", "birdclef2023": "logreg", "esc50_animals": "linsvm", "synthetic": "logreg"}

# Slide geometry, in slide points (1024 x 768), measured on the reference deck.
TITLE_Y = 60
BODY_X, BODY_W = 62, 900
CONTENT_BOTTOM = 708


def s(v, d=1):
    """Signed number with the true minus sign."""
    return (MINUS if v < 0 else "+") + f"{abs(v):.{d}f}"


def n(v, d=1):
    return (MINUS if v < 0 else "") + f"{abs(v):.{d}f}"


def p_text(p):
    if p < 0.001:
        return "p < 0.001"
    if p >= 0.9995:
        return "p = 1"
    return f"p = {p:.3f}".rstrip("0").rstrip(".") if p >= 0.01 else f"p = {p:.3f}"


def p_cell(p):
    if p >= 0.9995:
        return "1"
    if p < 0.001:
        mant, exp = f"{p:.1e}".split("e")
        return f"{mant}e{MINUS}{int(exp[1:])}"
    return f"{p:.3f}"


def aq(text: str) -> str:
    """AppleScript string expression; newlines become paragraph returns."""
    text = str(text).replace(" = ", "\u00a0=\u00a0").replace(" × ", "\u00a0×\u00a0").replace(" < ", "\u00a0<\u00a0")
    text = text.replace(" ≤ ", "\u00a0≤\u00a0").replace(" pp", "\u00a0pp")
    parts = text.replace("\\", "\\\\").replace('"', '\\"').split("\n")
    return " & return & ".join(f'"{p}"' for p in parts)


# --------------------------------------------------------------------------
# Data for text and tables
# --------------------------------------------------------------------------

def load():
    data = {}
    data["main"] = pd.read_csv(CHARTS / "main_contrasts_values.csv")
    data["manifest"] = json.loads((CHARTS / "charts_manifest.json").read_text())
    data["family_a"] = pd.read_csv(OUT_ROOT / "components" / "family_a.csv")
    data["timing"] = pd.read_csv(OUT_ROOT / "pimf_release" / "0.5.1" / "timing.csv")
    data["full050"] = pd.read_csv(OUT_ROOT / "pimf_release" / "0.5.0" / "full_data_comparison.csv")
    data["paired050"] = pd.read_csv(OUT_ROOT / "pimf_release" / "0.5.0" / "paired_classical.csv")
    data["imp_tests"] = pd.read_csv(OUT_ROOT / "robustness" / "paired_tests.csv")
    data["imp_scores"] = pd.read_csv(OUT_ROOT / "robustness" / "score_summary.csv")
    data["stability"] = pd.read_csv(OUT_ROOT / "robustness" / "stability_metrics.csv")
    return data


def main_cell(tab, ds, contrast, model="classical", fs="comp"):
    r = tab[(tab.dataset == ds) & (tab.contrast == contrast) & (tab.model == model) & (tab.feature_set == fs)].iloc[0]
    return r


def cost_rows():
    text = (OUT_ROOT / "SUMMARY.md").read_text()
    block = text.split("### Decomposition cost", 1)[1].split("## ", 1)[0]
    rows = []
    for ln in block.splitlines():
        if ln.startswith("| ") and "CSV" in ln and "---" not in ln and not ln.startswith("| Dataset"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            name = cells[0].split(";")[0].replace("BirdCLEF", "BirdCLEF 2023")
            rows.append([name, *cells[1:]])
    return rows


# --------------------------------------------------------------------------
# Slide specification
# --------------------------------------------------------------------------

def slides(data, laplace_dir: Path, v3_dir: Path):
    tab = data["main"]
    man = data["manifest"]
    out = []

    def add(**kw):
        out.append(kw)

    IE, IL, LE = (f"IRMF {MINUS} EMD", f"IRMF {MINUS} linear IMF", f"Linear IMF {MINUS} EMD")

    add(section=None, title="Outline", body=[
        "The question and the datasets",
        "Methods and what we compared",
        "Main study: IRMF, EMD, linear IMF and raw frames",
        "Number of components",
        "pimf 0.5.0 and 0.5.1 releases",
        "Robustness: impulsive spikes and Laplace noise",
        "Notebook-scale spikes at each method's best number of components",
        "Conclusions and backup",
    ], notes="Order: design first (data, methods, comparisons), then the results as plots.")

    add(section="Design", title="The Question", body=[
        "Does a robust multiscale decomposition (IRMF) give better classification features than EMD, "
        "linear IMF and raw frames?",
        "Does it hold up better when the test signals are contaminated?",
        "Every input sees the same normalized frames, the same 20 features per component, "
        "the same classifiers and the same grouped folds",
        "Linear IMF is the same recursive IMF with a quadratic contrast, so IRMF − linear IMF isolates "
        "the robust contrast",
    ], size=22, notes="Linear IMF is the key control: same schedule and kernel, only the contrast changes. "
                      "Raw frames go through the same feature functions as a single component.")

    add(section="Design", title="Datasets", body=[
        "Examples: frame 0 of each dataset, channel 0 (DroneRF L band), after median/MAD normalization",
        "Groups never split across train and test; every comparison uses the same frames",
        "60 GHz radar: main study only, excluded from all follow-ups (next slide)",
    ], size=16, body_y=588, body_h=118, image="dataset_frames.png", image_box=(BODY_X, 392, BODY_W), table=dict(
        header=["Dataset", "Signal", "Classes", "Frame length", "Frames", "Grouped by"],
        rows=[
            ["DroneRF", "RF, 2 bands (L, H)", "10 drone × flight mode", "2,048", "12,000", "227 segments"],
            ["BirdCLEF 2023", "bird audio, mono", "16 species", "2,048", "12,000", "451 recordists"],
            ["ESC-50 animals", "animal audio, mono", "10 animals", "2,048", "5,100", "157 uploaders"],
            ["Synthetic", "generated, mono", "5 signal classes", "1,024", "1,500", "150 recordings"],
            ["60 GHz radar", "radar I/Q, 2 channels", "bird vs drone", "1,000", "784", "45 target-poses"],
        ],
        widths=[160, 190, 220, 110, 90, 130], font=15, y=132, h=250),
        notes="Frame counts are the prepared populations. The examples are the first frame of each prepared "
              "dataset, not chosen: DroneRF background (no drone), a Barn Swallow, a cat, a synthetic tone. "
              "DroneRF classes are drone type times flight mode; "
              "each class comes from one acquisition session, so folds still share a session. "
              "Synthetic: two thirds of its recordings carry impulsive noise by design; it is a pipeline check. "
              "The robustness runs exclude 100 fixed pilot frames per dataset.")

    add(section="Design", title="Datasets: 60 GHz Radar (Main Study Only)", body=[
        "Public subset of Kang et al. (IEEE VTC2024-Fall; IEEE DataPort, doi 10.21227/c8r5-6h79): one indoor "
        "campaign, 60 GHz radar, I and Q channels",
        "Targets mounted on a turntable or tripod in front of absorbers: DJI Mavic, DJI Phantom 3 Pro and a "
        "flapping-wing toy bird; task bird vs drone",
        "49 moving recordings, 784 frames of 1,000 samples (320 bird, 464 drone), 45 target-plus-pose groups; "
        "16 range-angle cells per recording, clutter removed by subtracting each recording's mean",
        "Main study: best balanced accuracy per input 96.5 to 99.3%; with the fixed LDA, IRMF − EMD +1.75 pp "
        "and IRMF − linear IMF +0.83 pp, both Holm p = 1",
        "Left out of the follow-ups: few groups and accuracy near the ceiling cannot separate the methods, "
        "and mounted targets in one session are far from field conditions",
    ], size=17, body_h=318, image="radar_frames.png", image_box=(BODY_X, 462, BODY_W),
        notes="Source: github.com/sk8053/drone_bird_detection_with_radar (no license; research use only) and "
              "the VTC2024-Fall paper. The pulse repetition frequency is undocumented, so frequencies are in "
              "cycles per sample. The full 15 GB DataPort release was not used. Examples: the first bird and the "
              "first drone frame, I channel, median/MAD normalized. Best balanced accuracy per input (main study): "
              "raw 96.53, EMD 97.60, linear IMF 98.48, IRMF 99.32 (60-step gradient descent).")

    add(section="Design", title="Methods: Decompositions", body=[
        "Shared input: subtract the median, divide by 1.4826 MAD; the per-frame log scale is a separate "
        "feature, results with and without it",
        "EMD: PyEMD defaults, 8 IMFs plus residue (zero padded when fewer)",
        "Linear IMF: pimf.IMF(Quadratic()), window schedule h1 = 0.25, ratio a = √2, down to a 3 to 5 sample window",
        "IRMF: the same recursive IMF with the robust SmoothAbs(H) contrast, H = 2σ, "
        "σ = median|Δy| / (0.6745 √2) estimated on each normalized channel",
        "IRMF solved to convergence by a safeguarded Newton solver in Numba (tolerance 1e−12), pimf 0.4.0 as driver",
        "Main-study tables use pimf's default 60 gradient steps; the converged solver changes BA by "
        "+0.83, +0.07 and −0.31 pp (DroneRF, BirdCLEF, ESC-50)",
    ], size=19, notes="Both IMF methods use one schedule, h_min = 1.5/L, 17 stages at L = 2,048 and 15 at 1,024. "
                      "The pimf default gradient solver leaves 16.6 to 55.0% of frames with a capped stage; "
                      "Newton solves the same lookup objective at 11.9 to 27.3 times lower cost on the "
                      "RF and audio data. Follow-ups all use the converged solver.")

    add(section="Design", title="Methods: Features, Classifiers, Tests", body=[
        "20 features per component and channel: energy share, log energy, SD, mean |x|, skewness, kurtosis, "
        "crest, zero-crossing and extrema rates, Hilbert amplitude and frequency, spectral shape, "
        "Hjorth parameters, permutation entropy",
        "Classifiers on these features: a fixed classical model per dataset and LightGBM (next slide)",
        "5 grouped folds × 3 seeds; all scaling and tuning inside training folds",
        "Paired Nadeau-Bengio corrected t-tests and 95% intervals, Holm correction within each family",
        "Primary metric: balanced accuracy",
    ], size=19, notes="Holm families differ between studies; each slide uses the family of its source table.")

    add(section="Design", title="Methods: Classical Models and LightGBM", body=[
        "Classical models are standard statistical classifiers on the per-component features. All share one "
        "pipeline, fitted on the training fold only: median imputation, drop constant features, standardize, classify",
        "Fixed classical model: per dataset, the family with the best average over raw, EMD, linear IMF and IRMF "
        "in the main study, then held fixed for every input, so differences come from the features",
        "Tuned parameters are chosen by grouped 3-fold CV inside the training fold",
    ], size=16, body_h=190, table=dict(
        header=["Model", "What it is", "Tuned", "Fixed model for"],
        rows=[
            ["LDA", "linear discriminant, Ledoit-Wolf shrinkage", "none", "60 GHz radar"],
            ["QDA", "quadratic discriminant, covariance shrinkage", "shrinkage", ""],
            ["Gaussian NB", "naive Bayes, Gaussian per feature", "none", ""],
            ["Logistic regression", "multinomial, L2 penalty", "C: 0.001 to 10", "BirdCLEF, synthetic"],
            ["kNN", "k nearest neighbours", "k: 1, 5, 15, 35", ""],
            ["Linear SVM", "one-vs-rest, squared hinge (liblinear)", "C: 0.001 to 1", "DroneRF, ESC-50"],
            ["LightGBM", "gradient-boosted trees, 15 leaves, rate 0.05", "rounds (≤ 500, early stop)",
             "second model, all datasets"],
        ],
        widths=[180, 330, 190, 200], font=14, y=335, h=330),
        notes="Classical = the six statistical families of the main study's classical baselines "
              "(baselines/run_classical.py). The main study also ran ML models on the same features or on the "
              "frames (XGBoost, ExtraTrees, ROCKET variants, a 1-D CNN); the follow-ups use LightGBM as the "
              "ML model. LightGBM picks its number of rounds by early stopping on a grouped validation split "
              "of the training fold and uses balanced class weights.")

    add(section="Design", title="What We Compared", body=[
        "(a) Main study: IRMF vs EMD vs linear IMF vs raw frames",
        "(b) Number of components: K = 3 to 17, pre-registered equal-count test, pimf's default k_max = 8",
        "(c) pimf 0.5.0 and 0.5.1 releases vs our Newton prototype, on every frame",
        "(d) Impulsive spikes on 0.2%, 1% and 4% of test samples (amplitude A = 10)",
        "(e) Laplace noise on 10% to 40% of test samples (scale = 1 robust SD); also trained and tested at 40%",
        "(f) Notebook-scale spikes (scale 3σ, about 1.5 H) on 10% to 40% of test samples, IMF methods stopped "
        "at 31 samples, each method at its best K; also trained and tested at 40%",
        "(d) to (f): models trained on clean data, tested on the same folds clean and contaminated",
    ], size=20, notes="(b), (d), (e) and (f) were pre-registered. Synthetic was added to (d) after pre-registration "
                      "at our request and has its own Holm family. (f) has a dated amendment, made before any "
                      "result: IRMF uses a fixed H per dataset.")

    # ---------------- Main study
    d_ie = {ds: main_cell(tab, ds, IE) for ds in ["dronerf", "birdclef2023", "esc50_animals"]}
    d_il = main_cell(tab, "dronerf", IL)
    add(section="Main study", title="Main Study: Paired Contrasts", body=[
        f"{IE}: {s(d_ie['dronerf'].d)} pp DroneRF, {s(d_ie['birdclef2023'].d)} BirdCLEF, "
        f"{s(d_ie['esc50_animals'].d)} ESC-50 (fixed classical model, Holm p < 0.001)",
        f"{IL}: {s(d_il.d)} pp on DroneRF (Holm {p_text(d_il.p)}), near zero elsewhere; radar not significant",
    ], size=18, image="main_contrasts.png", notes="Features without log_scale. Fixed classical models as on the "
        "methods slide, LDA on radar. Holm p from each dataset's original phase-2 or ML table (12 to 120 tests). "
        f"LightGBM shrinks IRMF − EMD to {s(main_cell(tab, 'dronerf', IE, 'lightgbm').d)}, "
        f"{s(main_cell(tab, 'birdclef2023', IE, 'lightgbm').d)} and {s(main_cell(tab, 'esc50_animals', IE, 'lightgbm').d)} pp; "
        "only DroneRF keeps Holm significance. IRMF here is the pimf default solver.")

    add(section="Main study", title="Main Study: Best Balanced Accuracy per Input", body=[
        "Classical models: linear IMF and IRMF well above EMD and raw frames on the three large datasets",
        "DroneRF: linear IMF best (57.5% classical, 69.5% LightGBM); BirdCLEF: a CNN on raw frames reaches 52.6% "
        "vs 52.9% for IRMF",
    ], size=18, image="main_best_ba.png", notes="Best outer-CV balanced accuracy among the models tried on each "
        "input, chosen separately per input. Descriptive, not a paired test. Converged IRMF (classical: "
        "DroneRF 50.36, BirdCLEF 49.57, ESC-50 65.29%) is within one point of the default. Synthetic ML used five "
        "quick folds.")

    # ---------------- Components
    fa = data["family_a"]
    fa = fa[(fa.variant == "classical/comp") & (fa.metric == "balanced_accuracy")]

    def fa_row(ds, k, c):
        return fa[(fa.dataset == ds) & (fa.k == k) & (fa.contrast == c)].iloc[0]

    add(section="Components", title="Component Count: Balanced Accuracy vs K", body=[
        "IMF methods gain from more components; EMD is best with 3 or 4 IMFs and declines after that",
        "Fixed classical model, features without log_scale; diamonds mark the study defaults "
        "(17 IMF stages, 15 on synthetic; EMD 8 IMFs)",
    ], size=18, image="comp_ba_vs_k.png", notes="For IMF and IRMF, K stages span the same range of scales as the "
        "default 17 stages, so fewer stages means wider spacing. EMD keeps its first K IMFs; the rest stays in the "
        "residue. Best EMD count improves on EMD K = 8 by +1.9 pp on DroneRF and +2.5 pp on BirdCLEF (Holm "
        "within dataset). Converged Newton IRMF throughout.")

    r5 = {ds: fa_row(ds, 5, "IRMF - EMD") for ds in ORDER}
    add(section="Components", title="Component Count: Equal Numbers of Components", body=[
        "Pre-registered test (both IMF methods beat EMD at K = 5 and 6, Holm): met on DroneRF and synthetic, "
        "not on ESC-50 (3 of 4) or BirdCLEF (0 of 4)",
        f"{IE} at K = 5: {s(pp_(r5['dronerf']))} pp DroneRF, {s(pp_(r5['birdclef2023']))} BirdCLEF, "
        f"{s(pp_(r5['esc50_animals']))} ESC-50 (Holm {p_text(r5['esc50_animals'].p_nb_holm)})",
    ], size=18, image="comp_equal_k.png", notes="Holm across all 48 tests. Matched counts control feature "
        "dimension, not scale bands. DroneRF still favors linear IMF over IRMF at every K. With default counts the "
        "IRMF − EMD gap was 11.6, 6.1, 10.7 and 34.3 pp (DroneRF, BirdCLEF, ESC-50, synthetic).")

    add(section="Components", title="Component Count: pimf Default k_max = 8", body=[
        "Keeping only the 8 coarsest stages costs 8.8 to 10.4 pp on all four datasets (Holm p < 0.001)",
        f"With pimf defaults, {IE} is +0.4 pp DroneRF and +1.2 ESC-50 (not significant), "
        "−4.2 BirdCLEF (Holm p < 0.001)",
        "pimf defaults = 60-step gradient descent with k_max = 8, i.e. pimf.IMF(SmoothAbs(H)).decompose(y)",
    ], size=18, image="comp_kmax8.png", notes="With k_max = 8 pimf extracts the coarsest stages first and lumps "
        "every finer stage into the residual. Holm within each dataset across eight contrasts. Synthetic keeps "
        "+25.7 pp over EMD.")

    # ---------------- pimf release
    full = data["full050"]
    n_ch = int(full.channels.sum())
    within = 1 - full["channels_rel_l2_above_1e-6"].sum() / n_ch
    dr = full[full.dataset == "dronerf"].iloc[0]
    dr_within = 1 - dr["channels_rel_l2_above_1e-6"] / dr.channels
    pc = data["paired050"]
    rp = pc[pc.contrast == "release - prototype"]
    max_all = rp.mean_diff.abs().max()
    timing = data["timing"]
    add(section="pimf release", title="pimf 0.5.0 and 0.5.1: Results Unchanged", body=[
        "0.5.0 gradient descent is bit-for-bit identical to 0.4.0 on all four datasets",
        f"0.5.0 Newton vs our prototype: {within * 100:.1f}% of {n_ch:,} channels within 1e−6 "
        f"(DroneRF {dr_within * 100:.1f}%); the rest differ where the score is flat",
        f"Classification, release − prototype: at most 0.18 pp (fixed classical), {max_all:.2f} pp over all "
        "classifiers; Holm p = 1 everywhere",
        f"0.5.1: features byte-identical to 0.5.0 on all {n_ch:,} channels; Newton "
        f"{timing['speedup_vs_0.5.0'].min():.1f}–{timing['speedup_vs_0.5.0'].max():.1f}× faster than 0.5.0",
    ], size=17, body_h=210, table=dict(
        header=["CPU s per frame", "Prototype", "pimf 0.5.0", "pimf 0.5.1", "0.5.1 vs 0.5.0"],
        rows=[[DS[ds], f"{r.cpu_s_prototype:.3f}", f"{r['cpu_s_pimf_0.5.0']:.3f}",
               f"{r['cpu_s_pimf_0.5.1']:.3f}", f"{r['speedup_vs_0.5.0']:.2f}× faster"]
              for ds, r in timing.set_index("dataset").loc[ORDER].iterrows()],
        widths=[200, 150, 150, 150, 200], font=16, y=365, h=230),
        notes="Speed is from a 40-frame sample per dataset, 3 rounds. 0.5.1 is still "
              f"{timing.slowdown_vs_prototype.min():.1f}–{timing.slowdown_vs_prototype.max():.1f}× slower than "
              "our Numba prototype. 0.5.0 gradient descent now warns when a stage hits its cap. "
              "Linear IMF from 0.5.0 matched the cache on every channel.")

    # ---------------- Impulse robustness
    sc = data["imp_scores"]
    sc = sc[(sc.scenario == "shift") & (sc.feature_set == "comp")]

    def ba(ds, method, cond):
        clf = "linsvm" if ds in ("dronerf", "esc50_animals") else "logreg"
        return 100 * float(sc[(sc.dataset == ds) & (sc.classifier == clf) & (sc.method == method)
                              & (sc.condition == cond)].balanced_accuracy_mean.iloc[0])

    I, L = "irmf_h2_a1.41_conv", "imf_linear_a1.41"
    c4 = "impulse2_0.04"
    add(section="Robustness", title="Impulsive Spikes: Balanced Accuracy", body=[
        "Trained on clean data, tested on the same folds with spikes: on real data every input falls close to chance",
        f"Two-sided spikes on 4% of samples, IRMF vs linear IMF: synthetic {ba('synthetic', I, c4):.1f} vs "
        f"{ba('synthetic', L, c4):.1f}%, DroneRF {ba('dronerf', I, c4):.1f} vs {ba('dronerf', L, c4):.1f}%, "
        f"BirdCLEF {ba('birdclef2023', I, c4):.1f} vs {ba('birdclef2023', L, c4):.1f}%, "
        f"ESC-50 {ba('esc50_animals', I, c4):.1f} vs {ba('esc50_animals', L, c4):.1f}%",
    ], size=18, image="imp_ba_curves.png", notes="Fixed classical model, features without log_scale. One-sided "
        "spikes give almost the same curves. Gaussian noise of the same average power is the control. "
        "H is re-estimated from each contaminated channel.")

    t = data["imp_tests"]
    prim = t[(t.family == "Primary") & (t.contrast == "DiD")]

    def did(ds):
        x = prim[prim.dataset == ds]
        return 100 * x.mean_diff.min(), 100 * x.mean_diff.max()

    dr_lo, dr_hi = did("dronerf")
    bc_lo, bc_hi = did("birdclef2023")
    sy_lo, sy_hi = did("synthetic")
    add(section="Robustness", title=f"Impulsive Spikes: Change in the IRMF {MINUS} Linear IMF Gap", body=[
        f"DroneRF {s(dr_lo)} to {s(dr_hi)} pp at every rate (Holm p < 0.001); BirdCLEF {s(bc_lo)} to {s(bc_hi)} pp, "
        "significant at 0.2% and 1% only; ESC-50 no gain",
        "Pre-registered rule (positive and Holm significant at 4%, smaller under Gaussian noise): met on DroneRF "
        f"and synthetic ({s(sy_lo)} to {s(sy_hi)} pp, added after registration)",
    ], size=18, image="imp_did_linear.png", notes="Difference in differences: (IRMF − linear IMF) contaminated "
        "minus (IRMF − linear IMF) clean, per paired split. Positive means contamination hurt linear IMF more. "
        "Holm across the 18 real-data tests; synthetic has its own 87-test family. A positive DiD can coexist "
        "with lower absolute IRMF accuracy, as on DroneRF.")

    # ---------------- Laplace
    lap = man.get("laplace_status", {})
    if not lap.get("final"):
        raise SystemExit("Laplace results incomplete: rerun make_deck_charts.py on the final directory")
    lap_scores = pd.read_csv(laplace_dir / "score_summary.csv")
    lap_tests = pd.read_csv(laplace_dir / "paired_tests.csv")
    reading = pd.read_csv(laplace_dir / "prespecified_reading.csv").set_index("dataset")
    lap_ba = lap_scores[lap_scores.feature_set == "comp"]
    levels = [f"laplace_0.{k}" for k in range(1, 5)]

    def lba(ds, method, cond, scen="shift"):
        r = lap_ba[(lap_ba.dataset == ds) & (lap_ba.classifier == CLF[ds]) & (lap_ba.method == method)
                   & (lap_ba.condition == cond) & (lap_ba.scenario == scen)]
        assert len(r) == 1, (ds, method, cond, scen)
        return 100 * float(r.balanced_accuracy_mean.iloc[0])

    def ltest(ds, family, comparator, cond):
        r = lap_tests[(lap_tests.dataset == ds) & (lap_tests.family == family) & (lap_tests.comparator == comparator)
                      & (lap_tests.condition == cond) & (lap_tests.feature_set == "comp")
                      & (lap_tests.classifier == CLF[ds]) & (lap_tests.contrast == "DiD")]
        assert len(r) == 1, (ds, family, comparator, cond)
        return r.iloc[0]

    def lrange(ds, family, comparator, conds=levels):
        rows = [ltest(ds, family, comparator, c) for c in conds]
        return (min(100 * r.mean_diff for r in rows), max(100 * r.mean_diff for r in rows),
                max(r.p_nb_holm for r in rows))

    add(section="Robustness", title="Laplace Noise: Balanced Accuracy", body=[
        "Clean-trained models drop sharply on the real datasets, already at 10% noise",
        f"DroneRF at 10%: linear IMF {lba('dronerf', L, 'clean'):.1f} → {lba('dronerf', L, 'laplace_0.1'):.1f}%, "
        f"IRMF {lba('dronerf', I, 'clean'):.1f} → {lba('dronerf', I, 'laplace_0.1'):.1f}%",
        f"Synthetic degrades gradually: at 40%, IRMF {lba('synthetic', I, 'clean'):.1f} → "
        f"{lba('synthetic', I, 'laplace_0.4'):.1f}%, linear IMF {lba('synthetic', L, 'clean'):.1f} → "
        f"{lba('synthetic', L, 'laplace_0.4'):.1f}%",
        f"BirdCLEF at 40%: IRMF {lba('birdclef2023', I, 'laplace_0.4'):.1f}, linear IMF "
        f"{lba('birdclef2023', L, 'laplace_0.4'):.1f}, EMD {lba('birdclef2023', 'emd', 'laplace_0.4'):.1f}, "
        f"raw {lba('birdclef2023', 'raw', 'laplace_0.4'):.1f}%",
    ], size=17, image="lap_ba_curves.png",
        notes="Laplace(0, b) noise on a share p of test samples, b = 1 robust SD of the clean channel, before "
              "the median/MAD normalization. Same design as the spike run: clean-trained fixed classical model, "
              "features without log_scale, H re-estimated from each contaminated channel. The noisy samples are "
              "nested across p. No Gaussian control in this run. Source: robustness_laplace/score_summary.csv.")

    dr_lo, dr_hi, dr_p = lrange("dronerf", "Primary", L)
    others = {ds: lrange(ds, "Primary", L) for ds in ("birdclef2023", "esc50_animals")}
    sy40 = ltest("synthetic", "Primary", L, "laplace_0.4")
    assert dr_p < 0.05 and all(v[2] >= 0.05 for v in others.values())
    assert all(ltest("synthetic", "Primary", L, c).p_nb_holm >= 0.05 for c in levels)
    verdicts = reading.verdict.to_dict()
    assert verdicts["dronerf"] == "IRMF more robust"
    assert all(verdicts[d] == "no significant difference" for d in ("birdclef2023", "esc50_animals", "synthetic"))
    e_dr = lrange("dronerf", "S1", "emd")
    e_bc = lrange("birdclef2023", "S1", "emd", levels[1:])
    assert e_dr[2] < 0.05 and e_bc[2] < 0.05 and e_dr[1] < 0 and e_bc[0] > 0
    bc, es = others["birdclef2023"], others["esc50_animals"]
    add(section="Robustness", title="Laplace Noise: Change in the IRMF Gap", body=[
        f"{IL}: DroneRF {s(dr_lo)} to {s(dr_hi)} pp at every rate (Holm p ≤ {math.ceil(dr_p * 1000) / 1000:.3f}), "
        "the only pre-registered win",
        f"Yet DroneRF ends near chance at 40% (IRMF {lba('dronerf', I, 'laplace_0.4'):.1f}, linear IMF "
        f"{lba('dronerf', L, 'laplace_0.4'):.1f}%, chance 10%): linear IMF had more to lose",
        f"BirdCLEF {s(bc[0])} to {s(bc[1])} pp, ESC-50 {s(es[0])} to {s(es[1])} pp, synthetic "
        f"{s(100 * sy40.mean_diff)} pp at 40%: none survives Holm",
        f"{IE} gap: {s(e_dr[0])} to {s(e_dr[1])} pp on DroneRF, {s(e_bc[0])} to {s(e_bc[1])} pp on BirdCLEF "
        "at 20 to 40% (Holm p < .05)",
    ], size=16, image="lap_did.png",
        notes="Difference in differences: the IRMF gap under noise minus the gap on clean data, per paired split. "
              "The DroneRF gain mostly reflects how much more linear IMF's higher clean score had to lose. "
              "Pre-registered reading at 40%: IRMF more robust on DroneRF; no significant difference on BirdCLEF, "
              "ESC-50 or synthetic. "
              "Diamonds: primary family, 12 real-data tests against linear IMF. Circles: S1 against EMD, its own "
              "12-test family. Synthetic has its own 36-test family. Rule: IRMF more robust if the 40% DiD is "
              "positive with Holm p < .05. Sources: primary_tests.csv, paired_tests.csv, "
              "prespecified_reading.csv.")

    def recovered(ds, method):
        c, sh, m = lba(ds, method, "clean"), lba(ds, method, "laplace_0.4"), lba(ds, method, "laplace_0.4", "matched")
        return 100 * (m - sh) / (c - sh)

    def matched_row(ds):
        return " / ".join(f"{lba(ds, m, 'laplace_0.4', 'matched'):.1f}" for m in (L, I, "emd", "raw"))

    s5 = {ds: ltest(ds, "S5", L, "laplace_0.4") for ds in ORDER}
    assert s5["dronerf"].p_nb_holm < 0.001 and all(s5[d].p_nb_holm >= 0.05 for d in ORDER[1:])
    add(section="Robustness", title="Laplace Noise: Training at 40%", body=[
        f"Training and testing at 40% wins back {recovered('dronerf', L):.0f}% of linear IMF's loss on DroneRF, "
        f"{recovered('birdclef2023', L):.0f}% on BirdCLEF and {recovered('esc50_animals', L):.0f}% on ESC-50",
        f"Balanced accuracy at 40%, linear IMF / IRMF / EMD / raw: DroneRF {matched_row('dronerf')}%, "
        f"BirdCLEF {matched_row('birdclef2023')}%, ESC-50 {matched_row('esc50_animals')}%",
        f"Matched vs clean, change in the {IL} gap (S5): {s(100 * s5['dronerf'].mean_diff)} pp on DroneRF "
        f"(Holm {p_text(s5['dronerf'].p_nb_holm)}), not significant elsewhere",
    ], size=16, body_h=125, image="lap_matched.png",
        notes="Matched models are trained and tested at 40% with the full grid search, on the same folds. "
              "Share won back = (matched − clean-trained) / (clean − clean-trained), all at 40%. "
              f"Synthetic recovers least: {recovered('synthetic', L):.0f}% for linear IMF, "
              f"{recovered('synthetic', I):.0f}% for IRMF. S5 is its own 12-test family. "
              "Source: robustness_laplace/score_summary.csv and paired_tests.csv.")

    # ---------------- Robustness v3: notebook-scale spikes at each method's K*
    if "v3_status" not in man:
        raise SystemExit("v3 charts missing: rerun make_deck_charts.py with --v3")
    bvb = pd.read_csv(v3_dir / "best_vs_best_ba.csv")
    kstar = pd.read_csv(v3_dir / "k_star.csv")
    v3t = pd.read_csv(v3_dir / "paired_tests.csv")
    every_k = pd.read_csv(v3_dir / "ba_every_k_condition.csv")
    v3_reading = pd.read_csv(v3_dir / "prespecified_reading.csv").set_index("dataset")
    protocol = (REPO / "experiments" / "classification-emd-irmf" / "robustness_v3" / "PROTOCOL_v3.md").read_text()
    V3 = [f"nb3_0.{k}" for k in range(1, 5)]
    FAM = ("irmf", "linear", "emd", "raw")

    def kk(ds, method, clf=None):
        r = kstar[(kstar.dataset == ds) & (kstar.classifier == (clf or CLF[ds])) & (kstar.method == method)]
        assert len(r) == 1, (ds, method, clf)
        return int(r.k.iloc[0])

    def vba(ds, family, cond, clf=None, scen="shift"):
        clf = clf or CLF[ds]
        r = bvb[(bvb.dataset == ds) & (bvb.classifier == clf) & (bvb.family == family) & (bvb.condition == cond)
                & (bvb.scenario == scen)]
        assert len(r) == 1 and int(r.k.iloc[0]) == kk(ds, family, clf), (ds, family, cond, clf, scen)
        return 100 * float(r["mean"].iloc[0])

    def vtest(ds, family, comparator, cond, clf=None):
        clf = clf or CLF[ds]
        r = v3t[(v3t.dataset == ds) & (v3t.family == family) & (v3t.comparator == comparator)
                & (v3t.classifier == clf) & (v3t.condition == cond) & (v3t.contrast == "DiD")]
        assert len(r) == 1, (ds, family, comparator, cond, clf)
        r = r.iloc[0]
        assert int(r.irmf_k) == kk(ds, "irmf", clf) and int(r.comparator_k) == kk(ds, comparator, clf)
        return r

    def vrange(ds, family, comparator, conds=V3, clf=None):
        rows = [vtest(ds, family, comparator, c, clf) for c in conds]
        return (min(100 * r.mean_diff for r in rows), max(100 * r.mean_diff for r in rows),
                max(r.p_nb_holm for r in rows), min(r.p_nb_holm for r in rows))

    def p_le(p):
        return f"Holm p ≤ {math.ceil(p * 1000) / 1000:.3f}"

    def slash(ds, cond, clf=None, scen="shift", fams=FAM):
        return " / ".join(f"{vba(ds, f, cond, clf, scen):.1f}" for f in fams)

    # design facts from the protocol and the run itself
    grids = {(ds, fam): sorted(every_k[(every_k.dataset == ds) & (every_k.family == fam)].k.unique())
             for ds in ORDER for fam in ("emd", "linear", "irmf")}
    for ds in ORDER:
        assert grids[(ds, "emd")] == [2, 3, 4, 6, 8], ds
        imf_grid = [3, 4, 6, 8, 9] if ds == "synthetic" else [3, 4, 6, 8, 11]
        assert grids[(ds, "linear")] == imf_grid and grids[(ds, "irmf")] == imf_grid, ds
    assert "b = 3 * sigma_noise(x_clean)" in protocol and "There is no Gaussian" in protocol
    windows = [ln for ln in (v3_dir / "REPORT.md").read_text().splitlines() if "Linear and IRMF windows" in ln]
    lists = [re.findall(r"K=\d+: ([\d,]+)", w) for w in windows]
    assert len(windows) == 4 and all(len(x) == 5 and all(v.endswith(",31") for v in x) for x in lists)
    assert "every E2/E1 error ratio was between 0.996 and 1.005" in protocol
    assert "IRMF now uses a fixed H per dataset" in protocol

    add(section="Robustness", title="Notebook-Scale Spikes: Design", body=[
        "The Laplace run scaled the noise to the signal (b = 1 robust SD), below IRMF's threshold H; "
        "the notebooks put spikes at about 1.5 H",
        "Spikes: signed exponential with scale b = 3σ of the clean channel on 10% to 40% of test samples; "
        "H = 2σ, so b is about 1.5 H. No Gaussian term",
        "Linear IMF and IRMF stop at a 31-sample window instead of 3 to 5 samples; finer content stays in "
        "the residual",
        "Each method at its best number of components on clean data (K*), per dataset and classifier, "
        "fixed at every noise level",
        "K grids: EMD 2, 3, 4, 6, 8; linear IMF and IRMF 3, 4, 6, 8, 11 (synthetic 3, 4, 6, 8, 9)",
        "IRMF uses one fixed H per dataset, 2σ from the 100 excluded pilot frames, for clean and noisy inputs",
        "An IQR-based σ was tested on the pilot frames and not adopted: component errors within ±0.5% of "
        "the current estimator",
    ], size=18, notes="σ = median|Δy| / (0.6745 √2), the estimator behind H = 2σ. b is set per channel from the "
        "clean raw channel; noise is added before the median/MAD normalization, with the same random positions "
        "as the Laplace run, nested across rates. The Laplace run had b/H of about 0.4 to 1. The 31-sample stop "
        "follows the notebook audit, which found IRMF's component errors larger at the finest windows. The IQR "
        "pilot rule needed lower error in 9 of 12 real cells; it got 8, with every error ratio between 0.996 "
        "and 1.005. The fixed H lowered component error in all 12 real cells and was adopted in a dated "
        "amendment before any result. Source: robustness_v3/PROTOCOL_v3.md.")

    # best K
    def k_cell(ds, fam, clf=None):
        v = vba(ds, fam, "clean", clf)
        return f"{v:.1f}" if fam == "raw" else f"{v:.1f} (K = {kk(ds, fam, clf)})"

    def k_rows(clf=None):
        return [[DS[ds], *(k_cell(ds, f, clf) for f in ("emd", "linear", "irmf", "raw"))] for ds in ORDER]

    edge_emd = [ds for ds in ORDER if kk(ds, "emd") == 2]
    assert edge_emd == ["dronerf", "birdclef2023"]
    top_imf = sum(kk(ds, f) == max(grids[(ds, f)]) for ds in ORDER for f in ("linear", "irmf"))
    assert top_imf == 6, top_imf  # of 8; DroneRF IRMF picks 8 and synthetic linear IMF picks 6
    comp_sc = pd.read_csv(OUT_ROOT / "components" / "score_summary.csv")
    comp_sc = comp_sc[comp_sc.feature_set == "comp"]
    cost = []
    for ds in ("dronerf", "birdclef2023", "esc50_animals"):
        c = comp_sc[(comp_sc.dataset == ds) & (comp_sc.classifier == CLF[ds])]
        for fam, pre, dflt in (("linear", "linear_native_k", "imf_linear_a1.41"),
                               ("irmf", "irmf_native_k", "irmf_h2_a1.41_conv")):
            best = 100 * c[c.method.str.startswith(pre) | (c.method == dflt)].balanced_accuracy_mean.max()
            cost.append(best - vba(ds, fam, "clean"))
    assert all(x > 0 for x in cost)
    add(section="Robustness", title="Best Number of Components", body=[
        "K* = the K with the best mean clean balanced accuracy over the 15 splits, ties to the smaller K",
        "EMD picks K = 2, the bottom of its grid, on DroneRF and BirdCLEF; the IMF methods mostly pick the top",
        f"With the 31-sample stop, clean IMF scores are {min(cost):.1f} to {max(cost):.1f} pp below the "
        "component-count study's best on real data",
    ], size=16, body_h=100, tables=[
        dict(header=["Fixed classical", "EMD", "Linear IMF", "IRMF", "Raw"], rows=k_rows(),
             widths=[190, 180, 180, 180, 120], font=14, y=240, h=200),
        dict(header=["LightGBM", "EMD", "Linear IMF", "IRMF", "Raw"], rows=k_rows("lightgbm"),
             widths=[190, 180, 180, 180, 120], font=14, y=470, h=200)],
        notes="Clean balanced accuracy in %, mean of 15 grouped splits, at each method's K*. Selecting K on the "
              "same clean outer-fold scores makes clean scores slightly optimistic for every method alike. "
              "A K* at the edge of a grid means a better K may lie outside it. The cost of the 31-sample stop is "
              "descriptive: the component-count study is a separate run that went down to 3 to 5 sample windows "
              "with up to 17 stages; IRMF there also re-estimated H per input. "
              "Source: robustness_v3/k_star.csv and best_vs_best_ba.csv.")

    # balanced accuracy curves
    vdr40 = {f: vba("dronerf", f, "nb3_0.4") for f in FAM}
    assert all(v < 14 for v in vdr40.values())
    assert all(vba("dronerf", "irmf", c) < 11 for c in V3)
    add(section="Robustness", title="Notebook-Scale Spikes: Balanced Accuracy", body=[
        "Clean-trained fixed classical model, each method at its K* (panel titles)",
        "Real-data accuracy falls steeply at 10% already; DroneRF IRMF is within 1 pp of chance from 10% on",
        f"At 40%, IRMF / linear IMF / EMD / raw: DroneRF {slash('dronerf', 'nb3_0.4')}%, "
        f"BirdCLEF {slash('birdclef2023', 'nb3_0.4')}%, ESC-50 {slash('esc50_animals', 'nb3_0.4')}%, "
        f"synthetic {slash('synthetic', 'nb3_0.4')}%",
    ], size=17, image="v3_ba_curves.png",
        notes="Fixed classical model as before (linear SVM on DroneRF and ESC-50, logistic regression on "
              "BirdCLEF and synthetic), features without log_scale. K* is chosen on clean data and kept at every "
              "rate. Chance: 10% DroneRF and ESC-50, 6.25% BirdCLEF, 20% synthetic. "
              "Source: robustness_v3/best_vs_best_ba.csv.")

    # gap changes
    v_dr = vrange("dronerf", "Primary", "linear")
    v_bc = vrange("birdclef2023", "Primary", "linear")
    v_es = vrange("esc50_animals", "Primary", "linear")
    v_sy = vrange("synthetic", "Primary", "linear")
    e_dr = vrange("dronerf", "S1", "emd")
    e_bc = vrange("birdclef2023", "S1", "emd")
    e_es = vrange("esc50_animals", "S1", "emd")
    assert v_dr[0] > 0 and v_dr[2] < 0.05
    assert v_bc[3] >= 0.05 and v_es[3] >= 0.05 and v_sy[3] >= 0.05
    assert e_bc[0] > 0 and e_es[0] > 0 and max(e_bc[2], e_es[2]) < 0.05
    assert e_dr[1] < 0 and e_dr[2] < 0.05
    assert v3_reading.verdict["dronerf"] == "IRMF more robust"
    assert all(v3_reading.verdict[d].startswith("does not reject") for d in ("birdclef2023", "esc50_animals",
                                                                             "synthetic"))
    add(section="Robustness", title="Notebook-Scale Spikes: Change in the Gaps", body=[
        f"{IL}: DroneRF {s(v_dr[0])} to {s(v_dr[1])} pp at every rate ({p_le(v_dr[2])}), the only "
        f"pre-specified win; yet at 40% IRMF {vdr40['irmf']:.1f}%, linear IMF {vdr40['linear']:.1f}%, chance 10%",
        f"BirdCLEF {s(v_bc[0])} to {s(v_bc[1])} pp, ESC-50 {s(v_es[0])} to {s(v_es[1])} pp, synthetic "
        f"{s(v_sy[0])} to {s(v_sy[1])} pp: not significant",
        f"{IE}: BirdCLEF {s(e_bc[0])} to {s(e_bc[1])} pp, ESC-50 {s(e_es[0])} to {s(e_es[1])} pp "
        f"({p_le(max(e_bc[2], e_es[2]))}), EMD collapses; DroneRF {s(e_dr[0])} to {s(e_dr[1])} pp "
        f"({p_le(e_dr[2])})",
    ], size=16, image="v3_did.png",
        notes="Difference in differences at K*: the IRMF gap under spikes minus the gap on clean data, per paired "
              "split; each arm at its own K* (panel titles). Positive means the comparator lost more. The DroneRF "
              "gain comes from linear IMF's higher clean score (50.5 vs 42.9%) falling to chance. "
              "Pre-specified reading at 40%: IRMF more robust on DroneRF; the test does not reject on BirdCLEF, "
              "ESC-50 or synthetic, which is not equivalence. Holm families: primary 12 real-data tests, S1 12, "
              "synthetic its own 18. Source: robustness_v3/paired_tests.csv and prespecified_reading.csv.")

    # LightGBM and training at 40%
    lg = "lightgbm"
    g_es = vrange("esc50_animals", "S3", "linear", V3[1:], lg)
    g_es10 = vtest("esc50_animals", "S3", "linear", "nb3_0.1", lg)
    g_dr = vrange("dronerf", "S3", "linear", V3, lg)
    g_bc = vrange("birdclef2023", "S3", "linear", V3, lg)
    assert g_es[0] > 0 and g_es[2] < 0.05 and g_es10.p_nb_holm >= 0.05
    assert g_dr[0] > 0 and g_dr[2] < 0.001 and g_bc[3] >= 0.05
    es40 = {f: vba("esc50_animals", f, "nb3_0.4", lg) for f in FAM}
    assert es40["irmf"] > es40["linear"] > es40["emd"]
    m40 = {ds: {f: vba(ds, f, "nb3_0.4", scen="matched") for f in FAM} for ds in ORDER}
    dec = ("irmf", "linear", "emd")
    s5 = {ds: vtest(ds, "S5", "linear", "nb3_0.4") for ds in ORDER}
    assert s5["dronerf"].p_nb_holm < 0.001 and all(s5[d].p_nb_holm >= 0.05 for d in ORDER[1:])
    dr_clean_gap = vba("dronerf", "irmf", "clean") - vba("dronerf", "linear", "clean")
    add(section="Robustness", title="Notebook-Scale Spikes: LightGBM and Training at 40%", body=[
        f"LightGBM on ESC-50 (K: IRMF {kk('esc50_animals', 'irmf', lg)}, linear "
        f"{kk('esc50_animals', 'linear', lg)}, EMD {kk('esc50_animals', 'emd', lg)}): IRMF loses "
        f"{g_es[0]:.1f} to {g_es[1]:.1f} pp less than linear IMF at 20 to 40% ({p_le(g_es[2])}); at 40% "
        f"IRMF {es40['irmf']:.1f}%, linear IMF {es40['linear']:.1f}%, EMD {es40['emd']:.1f}%",
        f"LightGBM on DroneRF (K: IRMF {kk('dronerf', 'irmf', lg)}, linear {kk('dronerf', 'linear', lg)}): "
        f"{s(g_dr[0])} to {s(g_dr[1])} pp vs linear IMF (Holm p < 0.001), but only "
        f"{vba('dronerf', 'irmf', 'nb3_0.4', lg):.1f} vs {vba('dronerf', 'linear', 'nb3_0.4', lg):.1f}% at 40%; "
        "BirdCLEF not significant",
        f"Trained and tested at 40%, classical model at K*: ESC-50 back to {m40['esc50_animals']['irmf']:.1f}% "
        f"IRMF and {m40['esc50_animals']['linear']:.1f}% linear IMF; BirdCLEF "
        f"{min(m40['birdclef2023'][f] for f in dec):.1f} to {max(m40['birdclef2023'][f] for f in dec):.1f}%; "
        f"DroneRF {min(m40['dronerf'].values()):.1f} to {max(m40['dronerf'].values()):.1f}% for every input",
    ], size=16, image="v3_lgbm_matched.png",
        notes="LightGBM picks its own K* on clean data (best-components slide). S3 is its own 24-test Holm family "
              "over both comparators. At 10% the ESC-50 LightGBM gain is "
              f"{s(100 * g_es10.mean_diff)} pp, not significant. Trained and tested at 40%, BirdCLEF IRMF / "
              f"linear IMF / EMD / raw: {slash('birdclef2023', 'nb3_0.4', scen='matched')}%; synthetic "
              f"{slash('synthetic', 'nb3_0.4', scen='matched')}%. Matched vs clean, change in the IRMF − linear "
              f"IMF gap (S5): {s(100 * s5['dronerf'].mean_diff)} pp on DroneRF "
              f"(Holm {p_text(s5['dronerf'].p_nb_holm)}), because the clean gap of {s(dr_clean_gap)} pp "
              "closes; not significant elsewhere. Source: robustness_v3/best_vs_best_ba.csv and paired_tests.csv.")

    # ---------------- Conclusions
    imp4 = prim[(prim.dataset == "dronerf") & prim.condition.str.endswith("0.04")]
    dr40 = ltest("dronerf", "Primary", L, "laplace_0.4")
    assert v_dr[1] == 100 * vtest("dronerf", "Primary", "linear", "nb3_0.4").mean_diff
    assert min(e_bc[0], e_es[0]) == e_es[0] and max(e_bc[1], e_es[1]) == e_bc[1]
    for ds in ("dronerf", "birdclef2023", "esc50_animals"):  # training at 40% beats any choice of input
        shift40 = {f: vba(ds, f, "nb3_0.4") for f in FAM}
        assert min(m40[ds][f] - shift40[f] for f in FAM) > max(shift40.values()) - min(shift40.values()), ds
    add(section="Summary", title="Conclusions", body=[
        f"IRMF beats EMD by {s(d_ie['birdclef2023'].d)} to {s(d_ie['esc50_animals'].d)} pp with a fixed classical "
        "model on DroneRF, BirdCLEF and ESC-50 (Holm p < 0.001)",
        f"Linear IMF gains as much over EMD as IRMF on BirdCLEF and ESC-50, and {n(-d_il.d)} pp more on DroneRF",
        "The two differ only in the local fit, so the gain over EMD comes from the recursive IMF itself, "
        "partly from its larger number of components; on clean data the robust fit adds nothing",
        "At equal component counts the gain over EMD holds on DroneRF and synthetic only; pimf's k_max = 8 "
        "costs about 10 pp",
        "pimf 0.5.0/0.5.1 Newton reproduces our results; 0.5.1 is 1.8–2.5× faster than 0.5.0",
        f"With test-time spikes IRMF loses less than linear IMF on DroneRF ({s(100 * imp4.mean_diff.min())} to "
        f"{s(100 * imp4.mean_diff.max())} pp at 4%), at light rates on BirdCLEF, not on ESC-50",
        f"Under Laplace noise IRMF loses less than linear IMF only on DroneRF ({s(100 * dr40.mean_diff)} pp "
        "at 40%), and both end near chance there",
        f"Notebook-scale spikes at each method's best K: IRMF loses less than linear IMF on DroneRF "
        f"({s(v_dr[1])} pp at 40%, both near chance) and, with LightGBM, on ESC-50 ({es40['irmf']:.1f} vs "
        f"{es40['linear']:.1f}% at 40%); not significant on BirdCLEF and ESC-50 with the classical model",
        f"In that run IRMF loses {e_es[0]:.1f} to {e_bc[1]:.1f} pp less than EMD on BirdCLEF and ESC-50, more on "
        "DroneRF; training at 40% recovers more than any choice of decomposition",
    ], size=17, notes="Real-data accuracy under spikes or Laplace noise is close to chance for every clean-trained "
        "input, so the relative gains do not mean IRMF classifies contaminated signals well. Training on noisy "
        "data recovers far more than any choice of decomposition. The notebook-scale run compares each method at "
        "its own clean-selected K; the IRMF − EMD numbers there are classical-model gap changes.")

    # ---------------- Backup
    def contrast_table(fs):
        rows = []
        for ds in ["dronerf", "birdclef2023", "esc50_animals", "uas_birds_60ghz", "synthetic"]:
            row = [DS[ds]]
            for c in (IE, IL, LE):
                r = main_cell(tab, ds, c, fs=fs)
                row.append(f"{s(r.d, 2)} [{s(r.lo, 2)}, {s(r.hi, 2)}]\nHolm p {p_cell(r.p)}")
            rows.append(row)
        return rows

    for fs, lab in (("comp", "Without log_scale"), ("comp+log_scale", "With log_scale")):
        add(section="Backup", title=f"Backup: Main Study Contrasts, {lab}", body=[
            "Fixed classical model, mean Δ balanced accuracy in pp [95% corrected CI]; Holm p from the source table",
        ], size=16, table=dict(header=["Dataset", IE, IL, LE], rows=contrast_table(fs),
                               widths=[180, 240, 240, 240], font=14, y=200, h=440),
            notes="Sources: dataset classical CSVs linked in output/classification-emd-irmf/SUMMARY.md. "
                  "IRMF is the pimf default solver here.")

    add(section="Backup", title="Backup: Synthetic Main Contrasts", body=[
        "Synthetic data carry impulsive noise by design; its ML runs used five quick folds",
    ], size=18, image="main_contrasts_synthetic.png", notes="Shown on a separate scale because the effects are "
        "three to four times larger than on real data.")

    rows = []
    for ds in ORDER:
        for k in (5, 6):
            row = [f"{DS[ds]}, K = {k}"]
            for c in ("IRMF - EMD", "Linear IMF - EMD", "IRMF - Linear IMF"):
                r = fa_row(ds, k, c)
                row.append(f"{s(100 * r.mean_diff, 2)} (Holm p {p_cell(r.p_nb_holm)})")
            rows.append(row)
    add(section="Backup", title="Backup: Equal-Count Contrasts at K = 5 and 6", body=[
        "Fixed classical model, features without log_scale, Holm across 48 tests",
    ], size=16, table=dict(header=["Dataset", IE, LE, IL], rows=rows, widths=[220, 225, 225, 225], font=14,
                           y=190, h=470),
        notes="Source: output/classification-emd-irmf/components/family_a.csv.")

    add(section="Backup", title=f"Backup: Spikes, Change in the IRMF {MINUS} EMD Gap", body=[
        "The IRMF − EMD gap shrinks under spikes on DroneRF and ESC-50 (all Holm significant), not on BirdCLEF",
    ], size=18, image="imp_did_emd.png", notes="Secondary family S1, 18 real-data tests with its own Holm "
        "correction; synthetic in its own family.")

    add(section="Backup", title=f"Backup: Spikes vs Laplace Noise, IRMF {MINUS} Linear IMF", body=[
        "Same primary contrast in both runs: DroneRF gains under both; BirdCLEF under light spikes only; "
        "ESC-50 under neither; synthetic under spikes, not under Laplace noise",
    ], size=18, image="spike_vs_laplace_did.png", notes="Change in the IRMF − linear IMF gap, clean-trained fixed "
        "classical model. Holm within each run's primary family; synthetic has its own family in both. The "
        "spikes have amplitude 10 robust SD; the Laplace noise has scale 1 robust SD on many more samples. "
        "Sources: robustness/primary_tests.csv and robustness_laplace/primary_tests.csv.")

    st = data["stability"]
    st = st[(st.measure == "feature_drift") & (st.feature_set == "comp") & (st.slot == "all")
            & (st.condition == "impulse2_0.04")]
    order_m = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41_conv"]
    rows = []
    for ds in ORDER:
        row = [DS[ds]]
        for m in order_m:
            r = st[(st.dataset == ds) & (st.method == m)].iloc[0]
            row.append(f"{r['median']:.2f} / {r['p95']:.2f}")
        rows.append(row)
    add(section="Backup", title="Backup: Spikes, Feature Drift", body=[
        "Two-sided spikes on 4% of samples; median / 95th percentile change of each frame's features, "
        "in clean-population SDs",
    ], size=16, table=dict(header=["Dataset", "Raw", "EMD", "Linear IMF", "IRMF"], rows=rows,
                           widths=[220, 170, 170, 170, 170], font=16, y=200, h=300),
        notes="IRMF features move least on every dataset, yet real-data accuracy still falls close to chance. "
              "Source: output/classification-emd-irmf/robustness/stability_metrics.csv.")

    cost = cost_rows()
    add(section="Backup", title="Backup: Decomposition Cost", body=[
        "CPU seconds per frame from the full cached runs, all channels; concurrent runs, not a benchmark",
    ], size=16, table=dict(header=["Dataset", "EMD", "Linear IMF", "IRMF, pimf default", "IRMF, Newton",
                                   "Default / Newton"], rows=cost, widths=[180, 120, 140, 170, 150, 140],
                           font=16, y=200, h=300),
        notes="Source: output/classification-emd-irmf/SUMMARY.md, decomposition cost table.")
    return out


def pp_(r):
    return 100 * r.mean_diff


# --------------------------------------------------------------------------
# AppleScript generation
# --------------------------------------------------------------------------

AS_HEADER = """
on styleText(itm, fname, fsize, r, g, b)
  tell application "/Applications/Keynote.app"
    set font of object text of itm to fname
    set size of object text of itm to fsize
    set color of object text of itm to {r, g, b}
  end tell
end styleText

on replaceText(txt, old, new)
  set saved to AppleScript's text item delimiters
  set AppleScript's text item delimiters to old
  set parts to text items of txt
  set AppleScript's text item delimiters to new
  set txt to parts as text
  set AppleScript's text item delimiters to saved
  return txt
end replaceText
"""


def png_size(path: Path):
    with open(path, "rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def slide_script(i: int, sp: dict) -> str:
    L = []
    L.append(f'  set s to make new slide at end of slides of d with properties {{base slide:master slide "{MASTER}" of d}}')
    L.append(f"  set object text of default title item of s to {aq(sp['title'])}")
    body = sp.get("body") or []
    size = sp.get("size", 20)
    line_h = size * 1.8
    side = sp.get("side_text")
    # rough wrap estimate: Helvetica Neue Light averages ~0.5 em per character
    width = 330 if side else BODY_W - 30
    chars_per_line = max(20, int(width / (0.45 * size)))
    n_lines = sum(max(1, -(-len(b) // chars_per_line)) for b in body)
    body_h = sp.get("body_h") or (n_lines * line_h + 10)
    if not sp.get("image") and not sp.get("table") and not sp.get("tables"):
        body_h = CONTENT_BOTTOM - 132
    if body:
        L.append(f"  set object text of default body item of s to {aq(chr(10).join(body))}")
        L.append(f"  set size of object text of default body item of s to {size}")
        if sp.get("body_y"):
            L.append(f"  set position of default body item of s to {{{BODY_X}, {sp['body_y']}}}")
            L.append(f"  set height of default body item of s to {int(body_h)}")
        elif side:
            L.append(f"  set position of default body item of s to {{{BODY_X + 560}, 140}}")
            L.append("  set width of default body item of s to 340")
            L.append(f"  set height of default body item of s to {CONTENT_BOTTOM - 150}")
        elif sp.get("body_pos") == "bottom":
            L.append(f"  set position of default body item of s to {{{BODY_X}, {sp['table']['y'] + sp['table']['h'] + 25}}}")
            L.append(f"  set height of default body item of s to {CONTENT_BOTTOM - sp['table']['y'] - sp['table']['h'] - 30}")
        else:
            L.append(f"  set position of default body item of s to {{{BODY_X}, 132}}")
            L.append(f"  set height of default body item of s to {int(body_h)}")
        if sp.get("red_first"):
            L.append("  set color of paragraph 1 of object text of default body item of s to {50934, 0, 4082}")
        if sp.get("placeholder_last"):
            npar = len(body)
            L.append(f"  set color of paragraph {npar} of object text of default body item of s to {{50934, 0, 4082}}")
    else:
        L.append("  set body showing of s to false")
    if sp.get("section"):
        L.append(f"  tell s to set lbl to make new text item with properties {{object text:{aq(sp['section'])}}}")
        L.append('  my styleText(lbl, "HelveticaNeue-Thin", 20, 18492, 19685, 19412)')
        L.append("  set width of lbl to 300")
        L.append("  set position of lbl to {59, 7}")
    if sp.get("image"):
        path = CHARTS / sp["image"]
        pw, ph = png_size(path)
        if sp.get("image_box"):
            x0, y0, w = sp["image_box"]
            h = w * ph / pw
        elif side:
            x0, w = sp.get("image_x", 62), sp.get("image_w", 540)
            h = w * ph / pw
            y0 = 150 + (CONTENT_BOTTOM - 150 - h) / 2
        else:
            y0 = 132 + (body_h if body else 0) + 6
            avail_h = CONTENT_BOTTOM - y0
            w = min(BODY_W, avail_h * pw / ph)
            h = w * ph / pw
            x0 = BODY_X + (BODY_W - w) / 2
        L.append(f'  tell s to set img to make new image with properties {{file:(POSIX file "{path}")}}')
        L.append(f"  set width of img to {w:.1f}")
        L.append(f"  set height of img to {h:.1f}")
        L.append(f"  set position of img to {{{x0:.1f}, {y0:.1f}}}")
    for t in ([sp["table"]] if sp.get("table") else []) + sp.get("tables", []):
        nr, nc = len(t["rows"]) + 1, len(t["header"])
        tw = sum(t["widths"])
        L.append(f"  tell s to set tb to make new table with properties {{row count:{nr}, column count:{nc}, "
                 f"header row count:1, header column count:1, footer row count:0}}")
        for j, wcol in enumerate(t["widths"], start=1):
            L.append(f"  set width of column {j} of tb to {wcol}")
        for r_i, row in enumerate([t["header"], *t["rows"]], start=1):
            for c_i, val in enumerate(row, start=1):
                L.append(f"  set value of cell {c_i} of row {r_i} of tb to {aq(val)}")
        L.append(f"  set font size of every cell of tb to {t['font']}")
        L.append("  set font name of every cell of tb to \"HelveticaNeue-Light\"")
        L.append("  set font name of every cell of row 1 of tb to \"HelveticaNeue\"")
        L.append("  set font name of every cell of column 1 of tb to \"HelveticaNeue\"")
        L.append("  set alignment of every cell of tb to center")
        L.append("  set alignment of every cell of column 1 of tb to left")
        row_h = (t["h"]) / nr
        L.append(f"  set height of every row of tb to {row_h:.1f}")
        L.append(f"  set position of tb to {{{BODY_X + (BODY_W - tw) / 2:.0f}, {t['y']}}}")
    if sp.get("notes"):
        L.append(f"  set presenter notes of s to {aq(sp['notes'])}")
    return "\n".join(L)


def run_as(script: str, timeout=600):
    with tempfile.NamedTemporaryFile("w", suffix=".applescript", delete=False, encoding="utf-8") as f:
        f.write(script)
        name = f.name
    res = subprocess.run(["osascript", name], capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"osascript failed ({name}):\n{res.stderr}")
    return res.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laplace", type=Path, default=OUT_ROOT / "robustness_laplace")
    ap.add_argument("--v3", type=Path, default=OUT_ROOT / "robustness_v3")
    ap.add_argument("--deck", type=Path, default=DECK_DIR / f"{DECK_NAME}.key")
    ap.add_argument("--dry-run", action="store_true", help="print the slide list only")
    ap.add_argument("--images", type=Path, help="also export slide PNGs to this directory")
    ap.add_argument("--no-pdf", action="store_true", help="skip the PDF export next to the deck")
    ap.add_argument("--close", action="store_true", help="save and close the document at the end")
    args = ap.parse_args()
    if "Mobile Documents" in str(args.deck):
        raise SystemExit("refusing to write into an iCloud folder")
    spec = slides(load(), args.laplace, args.v3)
    for i, sp in enumerate(spec, start=2):
        print(f"{i:2d} [{sp.get('section') or ''}] {sp['title']}{'  (placeholder)' if sp.get('placeholder') else ''}")
    if args.dry_run:
        return
    deck = args.deck.resolve()
    head = f"""
with timeout of 600 seconds
tell application "/Applications/Keynote.app"
  set d to missing value
  repeat with doc in documents
    if (POSIX path of (file of doc as alias)) starts with "{deck}" then set d to doc
  end repeat
  if d is missing value then set d to open (POSIX file "{deck}")
"""
    tail = "\nend tell\nend timeout\n"
    # 1. title slide and cleanup
    setup = head + f"""
  set t1 to slide 1 of d
  set object text of iWork item 1 of t1 to "IRMF Classification and Robustness"
  set object text of iWork item 3 of t1 to "{DECK_NAME}.key"
  set width of iWork item 3 of t1 to 351
  set position of iWork item 3 of t1 to {{648, 710}}
  set object text of iWork item 4 of t1 to "{DECK_DIR}"
  repeat with itm in iWork items of t1
    try
      set txt to object text of itm
      if txt contains "Volodia" or txt contains "Kuzyuk" then
        set txt to my replaceText(txt, "Volodia", "Vladimir")
        set txt to my replaceText(txt, "Kuzyuk", "Kuziuk")
        set object text of itm to txt
      end if
    end try
  end repeat
  set m1 to base slide of t1
  repeat with j from (count of iWork items of m1) to 1 by -1
    try
      if (object text of iWork item j of m1) contains "Humboldt-Universit" then set object text of iWork item j of m1 to ""
    end try
  end repeat
  repeat while (count of slides of d) > 1
    delete slide (count of slides of d) of d
  end repeat
  return count of slides of d
""" + tail
    print("slides after cleanup:", run_as(AS_HEADER + setup))
    for i, sp in enumerate(spec, start=2):
        run_as(AS_HEADER + head + slide_script(i, sp) + "\n  return \"ok\"" + tail)
        print(f"built slide {i}: {sp['title']}")
    print(run_as(head + "  save d\n  return \"saved \" & (count of slides of d) & \" slides\"" + tail))
    if args.images:
        args.images.mkdir(parents=True, exist_ok=True)
        run_as(head + f'  export d to (POSIX file "{args.images}") as slide images with properties '
               '{image format:PNG, skipped slides:true}' + tail)
        print("slide images in", args.images)
    if not args.no_pdf:
        pdf = deck.with_suffix(".pdf")
        run_as(head + f'  export d to (POSIX file "{pdf}") as PDF with properties {{PDF image quality:Best}}' + tail)
        print("wrote", pdf)
    if args.close:
        print(run_as(head + "  close d saving yes\n  return \"closed\"" + tail))


if __name__ == "__main__":
    main()
