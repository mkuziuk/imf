# Robust IMF presentation

The 16-slide Beamer deck, including two backup slides, compares linear and robust
IMF using recursive component RMSE in the original signal units. Every method
comparison uses the same observation, and every stage-comparison plot includes
all nine stages. The RMSE curves and heatmaps use each method's own clean
reference. The first-component waveform slide uses one shared clean linear
IMF reference for both calculated curves and its two displayed RMSE values.

The main results cover the additive and masked single-run examples, the first
masked component, and the increased advantage under large contamination. The
first masked component overlays both calculated curves and one clean
reference on the same axes. The masked model's response to contamination
magnitude compares linear and robust IMF using both stage-1 RMSE and the mean
RMSE over nine stages. A separate slide gives a possible explanation for that
response. Two blue-to-yellow heatmaps compare both methods on the same grid
near the default setting, first with additive contamination and then with
masked contamination. Both slides share one color scale. The comparison of
all nine stages at `12 sigma` follows these two heatmap slides.
The opening method slide uses the IMF component definition and
notation from `IMF.pdf`, section 2.2 on page 3 and equation (3.2) on page 9.
It defines the contrast and its derivative, including `E|r + H xi|`
with standard Gaussian `xi`. Main slides have no subtitles or source notes;
references and reproduction details remain in the appendix.

## Compile

From the repository root, with a standard TeX installation including Beamer:

```bash
latexmk -pdf -cd overleaf/gd_imf_real_vs_calculated_slides.tex
```

The supplied figures are sufficient to compile the deck. In Overleaf, upload
the `overleaf` directory and select `gd_imf_real_vs_calculated_slides.tex` as the
main document.

## Regenerate the data and figures

Use the repository's Python environment and `requirements.txt`. The script
needs NumPy, pandas, Matplotlib and IPython.

```bash
python overleaf/scripts/build_real_vs_calculated_slides.py
```

The script loads the original numerical definitions without modifying the
notebooks. It evaluates both methods on the same contaminated observation at
`sigma=0.6`, `p=0.2`, contamination scale `2.0`, `H=1.2`, and seed `777`, for
each model. These are the presentation settings; the review's original
single-run settings remain unchanged in the source notebooks.

At these settings, robust IMF has lower RMSE at all nine stages in both models.
Mean RMSE falls from 0.0611 to 0.0394 (35.5%) for additive contamination and
from 0.0654 to 0.0437 (33.2%) for masked contamination. On the waveform slide,
masked stage-1 RMSE against the shared clean linear IMF component falls from
0.1234 to 0.0898 (27.3%). The stage-comparison curves continue to use
method-matched references, giving robust stage-1 RMSE 0.0904. The two clean
references differ by only 0.00142 RMSE, but are not treated as identical.

It also recomputes a slice of each model's contamination sweep at `sigma=0.2`
and `p=0.2`. Six contamination magnitudes and three coupled repeats give 18
trials per model, with both methods evaluated in every trial. The slide showing
all nine stages at `12 sigma` uses this separate sweep, with `sigma=0.2`,
`p=0.2`, contamination scale `2.4`, and medians across three paired repeats.
The two slides labeled "default setting" each show one observation at the
main parameters above, one slide per observation model.

For each heatmap, the script recomputes 225 trials at `sigma` values `0.4`,
`0.6`, and `0.8`, five contamination probabilities `0.1`, `0.15`, `0.2`,
`0.25`, and `0.3`, five absolute contamination scales `1.0`, `1.5`, `2.0`,
`2.5`, and `3.0`, and three coupled repeats. Both observation models use the
same grid and draws. Stars mark the exact default in the central panels.
The figures retain the six-panel layout and Viridis palette. Displayed values
are unscaled mean RMSE over all nine stages, with one logarithmic color scale
shared by all twelve panels across the additive and masked slides.
The full numerical run takes several minutes.

To redraw the saved data without repeating the numerical calculations:

```bash
python overleaf/scripts/build_real_vs_calculated_slides.py --plots-only
```

Use `--data-only` to stop after the numerical exports.
Use `--heatmap-only` to recompute both heatmaps while retaining the saved main
and magnitude-sweep data, then redraw the figures. This checks the saved
notebook hashes before reusing those numerical results.

## Outputs and definitions

The directory `overleaf/figures/real_vs_calculated_slides/` contains:

- `paired_stage_rmse.csv`: 36 rows covering both methods, both single-run
  observation models and all nine stages.
- `paired_traces.csv`: the observations and first-stage components used in the
  waveform comparisons, including each method's clean reference.
- `masked_first_stage_shared_rmse.csv`: two waveform-comparison RMSE values,
  both measured against the same clean linear IMF component.
- `contamination_stage_rmse.csv`: 648 stage-level RMSE values from the two
  18-trial magnitude slices, with both methods per trial.
- `contamination_mean_rmse.csv`: the mean of nine stage RMSE values for each
  method, model, magnitude and repeat.
- `heatmap_stage_rmse.csv`: 4,050 stage-level RMSE values from the 225 additive
  trials, covering both methods and all nine stages.
- `heatmap_mean_rmse.csv`: 450 mean RMSE values, one per method and trial.
- `masked_heatmap_stage_rmse.csv` and `masked_heatmap_mean_rmse.csv`: the
  corresponding 4,050 stage values and 450 mean values for the masked model.
- `rmse_provenance.json`: source hashes, numerical cell indices, configurations
  and validation details.
- Nine vector PDF figures used by the slides.
- `presentation_preview.png`: a rendered masked-grid slide for pull request review.

Regenerate the preview after compiling the deck:

```bash
pdftoppm -f 10 -l 10 -r 110 -singlefile -png \
  overleaf/gd_imf_real_vs_calculated_slides.pdf \
  overleaf/figures/real_vs_calculated_slides/presentation_preview
```

RMSE is the square root of the mean squared component difference over the time
grid. The mean RMSE summary first averages the nine stage RMSE values within
one trial. Sweep plots then show medians and interquartile ranges across the
three repeats. Percentage gains use ratios of these medians. Stage-specific
claims use medians at each stage.

The script checks reconstruction identities, the paired observation-model
relationship, full stage coverage and agreement between the plotted traces
and exported stage-1 RMSE. It also verifies complete heatmap stage coverage
and inclusion of three paired repeats at the exact default. If the notebooks
change, review the numerical claims written in the LaTeX file as well as the
new figures.

The review calls the kernel Epanechnikov. The source code normalizes
`(1 - abs(u))_+^2` and uses periodic wrap boundaries. The deck preserves this
implementation and records the distinction in a backup slide.
