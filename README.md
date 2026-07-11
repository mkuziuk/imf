# Intrinsic Multiscale Filtering

This project explores intrinsic multiscale filtering (IMF) and robust IMF variants for
one-dimensional signals with Gaussian noise and one-sided exponential contamination.

## Repository layout

- [`experiments/imf-irmf-comparison/`](experiments/imf-irmf-comparison/): baseline
  IMF comparison using local mean, median, and weighted median filters.
- [`experiments/robust-gradient-descent/`](experiments/robust-gradient-descent/):
  standalone robust local fitting, gradient descent, and parallel execution.
- [`experiments/gd-irmf/`](experiments/gd-irmf/): lookup-grid gradient-descent IRMF
  experiment.
- [`experiments/real-vs-calculated/`](experiments/real-vs-calculated/): clean-reference
  comparisons, observation-model variants, limit sweeps, and error-scaling studies.
- [`overleaf/`](overleaf/): publication-ready TeX sources and their figures.
- [`research/first-imf-recursive-error/`](research/first-imf-recursive-error/): audit
  of the first recursive IMF error, including its companion notebook and diagnostics.
- [`IMF.pdf`](IMF.pdf): reference note for intrinsic robust multiscale filtering.
- `requirements.txt`: Python dependencies needed to run the notebooks.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

To execute a notebook from the command line:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute experiments/robust-gradient-descent/robust_gradient_descent_imf.ipynb --output-dir=/tmp --output=robust_gradient_descent_imf.executed
```

## Overleaf

Upload the contents of [`overleaf/`](overleaf/) to the root of an Overleaf project.
The three `.tex` files stay at project root, and the nested paths below the single
`figures/` directory must be preserved. The TeX sources link to the notebooks in this
GitHub repository and use figure paths relative to the Overleaf project root.

## Robust Gradient-Descent IMF

The baseline IMF decomposition is iterative. At stage $k$, a local smoother is applied
to the current residual, producing IMF $S_k$; then the residual is updated:

$$
r_{k+1} = r_k - S_k .
$$

Here $r_k$ is the residual signal before stage $k$, $S_k$ is the IMF extracted at
stage $k$, and $r_{k+1}$ is the residual passed to the next stage.

The robust notebook replaces the local mean or median smoother with a smooth robust
location fit. For each time index $t$, it solves:

$$
\widehat S_k(t)
= \arg\min_x \sum_u w_{t,u}\,\rho_H\!\left(r_k(u) - x\right).
$$

Here $\widehat S_k(t)$ is the fitted IMF value at time $t$, $x$ is the scalar local
location being optimized, $u$ indexes observations inside the local window, $w_{t,u}$
are normalized Epanechnikov window weights, and $\rho_H$ is the robust contrast.

The smooth absolute-value contrast from the PDF is:

$$
\rho_H(r)
= r\,\operatorname{erf}\!\left(\frac{r}{\sqrt{2}\,H}\right)
  + \sqrt{\frac{2}{\pi}}\,H
    \exp\!\left(-\frac{r^2}{2H^2}\right).
$$

Here $r$ is a local residual difference, $H > 0$ controls the transition from quadratic
near zero to absolute-value behavior in the tails, and $\operatorname{erf}$ is the
Gaussian error function. Smaller $H$ makes the fit more median-like; larger $H$ makes it
closer to a local mean.

Its score is:

$$
\psi_H(r)
= \rho_H'(r)
= \operatorname{erf}\!\left(\frac{r}{\sqrt{2}\,H}\right).
$$

Here $\psi_H$ is the derivative of the robust contrast. It is bounded between $-1$ and
$1$, which limits the influence of large contaminated observations.

The local gradient-descent update for one window is:

$$
x^{(m+1)}
= x^{(m)}
  + \eta \sum_u w_{t,u}\,
    \psi_H\!\left(r_k(u) - x^{(m)}\right).
$$

Here $m$ is the gradient iteration index, $x^{(m)}$ is the current local location
estimate, $x^{(m+1)}$ is the next estimate, and $\eta$ is the step size.

The sign is positive because the objective is written in terms of
$\rho_H(r_k(u) - x)$, so the derivative with respect to $x$ is the negative of
the weighted score.

Implementation details:

- Initial value: local median of the window.
- Step size: $\eta = 0.95\,H / \sqrt{2 / \pi}$, based on the global curvature bound of the
  smoothed absolute loss with normalized weights.
- Stabilization: each update is clipped to the local window range.
- Stopping: stop when the max update is below
  $\mathrm{tol}\,(1 + \max |x|)$, or after `max_iter`.
- Tuning: the notebook evaluates $H$ on
  $\{0.25, 0.5, 1.0, 2.0\}\sigma$.

## How Parallelization Works

The robust decomposition has two levels of work:

1. IMF stages are sequential.
2. Local fits inside a stage are independent.

Stage $k+1$ depends on the residual produced by stage $k$, so the full IMF chain cannot
be freely reordered for the robust method. However, once $r_k$ is fixed, every
local fit $S_k(t)$ solves its own one-dimensional optimization problem and can be run
independently.

The notebook uses this structure in two places:

- `local_robust_gd_filter`: splits all local windows into chunks and runs those chunks
  with `ThreadPoolExecutor`.
- the benchmark section: runs independent contamination cases in parallel.

Within each chunk, the gradient iterations are vectorized with NumPy. That means one
chunk updates many local windows at once using array operations, instead of looping
over time points in Python.

The default worker count is:

```python
DEFAULT_ROBUST_MAX_WORKERS = min(4, os.cpu_count() or 1)
```

For small signals, threading may only give a modest speedup because NumPy vectorization
already removes most Python-loop overhead. The parallel structure is still useful as
the number of time points, windows, benchmark cases, or repeats grows.

## Linear Operator Note

The PDF proves that the linear mean-filter operators $W^{(k)}$ and residual operators
$A^{(k)}$ commute for a regular wrap design. This is useful for the linear baseline.

That shortcut is not used for robust gradient-descent IMF because the robust smoother
is nonlinear: each local fit depends on the current residual through an optimization
problem, not through a fixed linear matrix.

## Validation

The robust notebook checks:

- reconstruction accuracy: $\sum_k S_k + r_{\mathrm{final}}$ matches the input signal;
- component-wise MSE, MAE, and max absolute error against clean-reference
  decompositions;
- `H` tuning over a small grid;
- sequential versus threaded timing for local robust fits.
