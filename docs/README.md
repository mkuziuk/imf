# IMF demo

Compare linear and robust intrinsic multiscale filtering in the browser.

The page starts with signal and noise controls beside the observed signal. The
filtered signal comes next, followed by components and stage errors. Signal shape,
window sizes, and the error table expand when needed. Solver settings and
convergence diagnostics stay out of the interface.

- `imf_core.py` contains the NumPy implementation, including the robust solver.
- `imf_worker.js` loads Python and NumPy through Pyodide and calls `run_demo`.
- `index.html`, `main.js`, and `style.css` define the page. Plotly draws the charts;
  KaTeX renders the formulas. Both load from a CDN.

No build step is needed. Run `python3 -m http.server` from this directory to preview.
GitHub Pages serves the `/docs` folder on `main` using "Deploy from a branch".

The script and stylesheet URLs in `index.html` include the first 12 characters of
each file's SHA-256 hash. After editing either file, run
`sha256sum docs/main.js docs/style.css` from the repository root and update its
`?v=` value. This prevents an updated page from loading a cached script that still
expects removed controls.
