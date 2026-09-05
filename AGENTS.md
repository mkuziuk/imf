# Repository Guidelines

## Project structure and module organization

This repository contains reproducible research on intrinsic multiscale filtering. Experimental notebooks live under `experiments/`, grouped by study: baseline IMF/IRMF comparisons, robust gradient descent, and real-versus-calculated error analyses. Publication-ready LaTeX and referenced plots belong in `overleaf/` and `overleaf/figures/`. The focused audit in `research/first-imf-recursive-error/` contains notes, a companion notebook, a standalone diagnostic script, and its CSV/JSON results. `output/jupyter-notebook/` stores selected generated analyses. Root PDFs and links are source references, not build inputs.

## Build, test, and development commands

Create an isolated environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Execute a changed notebook without overwriting it:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
  experiments/gd-irmf/gd_irmf.ipynb --output-dir=/tmp --output=gd_irmf.executed
```

Run the standalone numerical audit with:

```bash
.venv/bin/python research/first-imf-recursive-error/diagnostics/run_numerical_diagnosis.py
```

When a local TeX installation is available, compile a report with `latexmk -pdf -cd overleaf/gd_irmf_review.tex`.

## Coding style and naming conventions

Use four-space indentation and standard Python conventions: `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for constants, and descriptive names for experiment parameters. Keep notebooks readable by introducing each non-obvious computation with a short Markdown cell. Use NumPy vectorization where practical and fixed `numpy.random.default_rng` seeds for reproducible experiments. Name new notebooks and generated data with lowercase `snake_case`.

## Testing guidelines

There is no automated unit-test suite or coverage threshold. Validate changes by executing every touched notebook from a clean kernel and checking its reconstruction/error assertions and final plots. For diagnostic changes, rerun `run_numerical_diagnosis.py` and review the resulting CSV/JSON diff. Compile changed TeX and check that figures, links, and equations render correctly.

## Commit and pull request guidelines

Recent commits use short, imperative subjects such as `Add IMF residual distribution experiments`, `Move observation-model comparison notebook`, and `Prove IMF Lemma 4.2`. Keep one research change per commit. Pull requests should explain the question being tested, list reproducibility commands, and identify regenerated artifacts. Link the relevant issue when one exists, and include before/after plots or screenshots for figure and rendering changes. Never commit `.venv/` or incidental notebook output.
