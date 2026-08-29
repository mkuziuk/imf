# Demo website

Interactive companion page for the IMF project: live linear and robust (gradient-descent)
intrinsic multiscale filtering, computed in the visitor's browser.

- `imf_core.py` — the canonical NumPy implementation (window schedule, kernels, contrasts,
  gradient descent, decompositions). It runs unmodified in the browser via Pyodide and can
  equally be imported from the notebooks; the definitions mirror the notebook versions
  verbatim and reproduce the numbers in
  `research/first-imf-recursive-error/diagnostics/summary.json`.
- `imf_worker.js` — Web Worker that boots Pyodide + NumPy and calls `imf_core.run_demo`.
- `index.html`, `main.js`, `style.css` — the page. No build step; charts via Plotly.js,
  math via KaTeX (both from CDN).

Run locally with any static server, e.g. `python3 -m http.server` from this directory.
To publish: enable GitHub Pages for the repository with source "Deploy from a branch",
branch `main`, folder `/docs`.
