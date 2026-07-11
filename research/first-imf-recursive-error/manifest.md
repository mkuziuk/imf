# First-IMF Recursive-Error Investigation

- Objective: explain the unusually large stage-1 recursive error in [`gd_imf_real_vs_calculated.ipynb`](../../experiments/real-vs-calculated/gd_imf_real_vs_calculated.ipynb) for the linear and robust decompositions, and determine exactly what theory does and does not imply about scaled-error constancy.
- Mode: research / diagnostic investigation.
- Artifact root: `research/first-imf-recursive-error/`.
- Constraints: do not modify existing project code or notebooks; generated diagnostics stay under this artifact root or the notebook-output directory.
- Assumptions: the current working-tree version of the notebook is the experiment to explain; `IMF.pdf` is a candidate primary theory source until its provenance and statements are verified.
- Source boundaries: local notebooks, project notes/PDFs, reproducible numerical checks, and primary mathematical sources where needed.
- Stages: static audit; numerical reproduction and ablations; theory verification; independent cross-check; synthesis.
- Status: complete. The original notebook and existing project code were not modified.
