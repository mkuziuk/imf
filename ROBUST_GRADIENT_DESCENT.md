# Robust Gradient-Descent IMF

This note explains the robust gradient-descent method implemented in
`robust_gradient_descent_imf.ipynb`.

The goal is to compute intrinsic multiscale filtering (IMF) components when the
observed signal contains ordinary noise plus large contaminated observations. The
baseline local mean is sensitive to outliers, so the robust method replaces each local
mean fit with a smooth robust local location estimate.

## 1. IMF Recursion

The IMF decomposition is iterative. Let the observed signal be

$$
y = (y_1,\ldots,y_n).
$$

At the first stage, the residual is the original signal:

$$
r_1 = y.
$$

At stage $k$, a local smoother extracts an IMF component $S_k$ from the current
residual $r_k$. The residual is then updated by subtraction:

$$
r_{k+1} = r_k - S_k.
$$

The full reconstruction after $K$ stages is

$$
y = \sum_{k=1}^{K} S_k + r_{K+1}.
$$

In the notebook, this is implemented by `robust_gd_imf`:

```python
residual = np.asarray(y, dtype=float).copy()
imfs = []

for window_size in window_sizes:
    s_k = local_robust_gd_filter(residual, window_size, H, ...)
    imfs.append(s_k)
    residual = residual - s_k
```

The key point is that IMF stages are sequential: $S_{k+1}$ cannot be computed until
$r_{k+1}$ is known.

## 2. Local Robust Objective

For a fixed stage $k$ and time point $t$, the method looks at a local window around
$t$. The local fit is a scalar value $x$ that minimizes a weighted robust objective:

$$
\widehat S_k(t)
= \arg\min_x
  \sum_u w_{t,u}\,\rho_H\!\left(r_k(u)-x\right).
$$

Variables:

- $t$: target time index where the smoother is being evaluated.
- $u$: index of an observation inside the local window around $t$.
- $r_k(u)$: current residual value at index $u$.
- $x$: scalar local location estimate being optimized.
- $\widehat S_k(t)$: fitted robust smoother value at time $t$.
- $w_{t,u}$: normalized kernel weight for observation $u$ in the window around $t$.
- $\rho_H$: smooth robust loss.
- $H$: robustness scale parameter.

The implementation computes one such scalar problem per time point.

## 3. Epanechnikov Window Weights

The local weights come from the Epanechnikov kernel:

$$
K(v) = \frac{3}{4}(1-|v|)^2_+.
$$

For a window with radius $R$, each offset $j \in \{-R,\ldots,R\}$ is scaled as

$$
v_j = \frac{j}{R}.
$$

Then the discrete weights are

$$
\tilde w_j = \frac{3}{4}\max(0,1-|v_j|)^2,
$$

and normalized:

$$
w_j = \frac{\tilde w_j}{\sum_\ell \tilde w_\ell}.
$$

In code:

```python
offsets = np.arange(-radius, radius + 1)
u = offsets / radius
weights = 0.75 * np.maximum(0, 1 - np.abs(u)) ** 2
weights = weights / weights.sum()
```

The center of the window gets the largest weight. Values near the window boundary get
smaller weights.

## 4. Smooth Robust Loss

The PDF proposes a smoothed version of the absolute-value loss:

$$
\rho_H(r)
= r\,\operatorname{erf}\!\left(\frac{r}{\sqrt{2}H}\right)
  + \sqrt{\frac{2}{\pi}}\,H
    \exp\!\left(-\frac{r^2}{2H^2}\right).
$$

Variables and functions:

- $r$: local residual difference, $r = r_k(u)-x$.
- $H > 0$: smoothing and robustness scale.
- $\operatorname{erf}$: Gaussian error function.
- $\rho_H(r)$: smooth approximation to $|r|$.

Behavior:

- Near zero, $\rho_H$ behaves more like a quadratic loss.
- In the tails, it behaves more like absolute value.
- Smaller $H$ makes the method more median-like.
- Larger $H$ makes the method more mean-like.

The notebook evaluates a small grid:

$$
H \in \{0.25, 0.5, 1.0, 2.0\}\sigma.
$$

## 5. Score Function

Gradient descent uses the derivative of the loss:

$$
\psi_H(r)
= \rho_H'(r)
= \operatorname{erf}\!\left(\frac{r}{\sqrt{2}H}\right).
$$

This score is bounded:

$$
-1 \leq \psi_H(r) \leq 1.
$$

That boundedness is the robustness mechanism. A very large outlier cannot contribute
an arbitrarily large gradient. It can only contribute approximately $+1$ or $-1$ before
being weighted by $w_{t,u}$.

In code:

```python
def smooth_abs_score(residual, H):
    return erf_approx(residual / (SQRT_2 * H))
```

The notebook uses a vectorized Abramowitz-Stegun approximation for `erf` so the method
does not require SciPy.

## 6. Gradient of the Local Objective

For one local window, define the objective

$$
F_t(x)
= \sum_u w_{t,u}\,\rho_H\!\left(r_k(u)-x\right).
$$

Differentiate with respect to $x$:

$$
\frac{d}{dx}F_t(x)
= -\sum_u w_{t,u}\,
   \psi_H\!\left(r_k(u)-x\right).
$$

The minus sign appears because the loss is applied to $r_k(u)-x$.

Gradient descent updates by subtracting the gradient:

$$
x^{(m+1)}
= x^{(m)} - \eta F_t'(x^{(m)}).
$$

Substituting the derivative gives the update used in the notebook:

$$
x^{(m+1)}
= x^{(m)}
  + \eta \sum_u w_{t,u}\,
    \psi_H\!\left(r_k(u)-x^{(m)}\right).
$$

Variables:

- $m$: gradient iteration index.
- $x^{(m)}$: current local location estimate.
- $x^{(m+1)}$: next local location estimate.
- $\eta$: gradient step size.
- $F_t'(x)$: derivative of the local objective at time $t$.

In code:

```python
local_score = np.sum(
    row_weights * smooth_abs_score(windows - x[:, None], H),
    axis=1,
)
x_next = x + step * local_score
```

The implementation computes this update for many windows at once. Each row of
`windows` is one local problem, and `x` is a vector of all current local estimates.

## 7. Initialization

Each local problem starts from the ordinary median of its window:

$$
x^{(0)}_t = \operatorname{median}\{r_k(u): u \text{ in the window around } t\}.
$$

In code:

```python
x = np.median(windows, axis=1)
```

This is a stable robust starting point. It is already close to the solution when $H$ is
small.

## 8. Step Size

The implementation uses

$$
\eta = 0.95\,\frac{H}{\sqrt{2/\pi}}.
$$

This comes from the curvature bound for the smooth absolute-value loss:

$$
0 \leq \rho_H''(r) \leq \frac{\sqrt{2/\pi}}{H}.
$$

Because the weights are normalized, the local objective has the same global curvature
bound. A step slightly below the reciprocal of that bound is a conservative fixed step:

$$
\eta < \frac{H}{\sqrt{2/\pi}}.
$$

In code:

```python
step = 0.95 * H / SQRT_2_OVER_PI
```

## 9. Clipping

After every update, the estimate is clipped to the range of its local window:

$$
x^{(m+1)}
\leftarrow
\min\left\{
  \max\left(x^{(m+1)}, \min_u r_k(u)\right),
  \max_u r_k(u)
\right\}.
$$

In code:

```python
lower = windows.min(axis=1)
upper = windows.max(axis=1)
x_next = np.clip(x + step * local_score, lower, upper)
```

This prevents numerical overshoot from moving the local estimate outside the observed
local data range.

## 10. Stopping Rule

The iteration stops when the largest update across all local windows is small:

$$
\max_t |x_t^{(m+1)} - x_t^{(m)}|
\leq
\mathrm{tol}\left(1+\max_t |x_t^{(m+1)}|\right).
$$

In code:

```python
max_delta = float(np.max(np.abs(x_next - x)))
x = x_next
if max_delta <= tol * (1.0 + float(np.max(np.abs(x)))):
    break
```

The default maximum number of gradient iterations is `max_iter=60`.

## 11. Vectorization

The local problems are independent once the current residual $r_k$ is fixed. The
notebook exploits this by storing all windows in a matrix:

$$
\texttt{windows}[t, j] = \text{the } j\text{th value in the window around }t.
$$

Then the score update is computed for all time points in a chunk at once:

```python
smooth_abs_score(windows - x[:, None], H)
```

Here:

- `windows` has shape `(n_windows, window_size)`;
- `x[:, None]` broadcasts each local estimate across its window;
- the result has one score value per window entry;
- multiplying by `row_weights` applies the kernel weights;
- summing over `axis=1` gives one score per window.

This removes the inner Python loop over time points.

## 12. Parallelization

There are two levels of parallelism.

### Local Fits Inside One IMF Stage

For fixed $r_k$, all local fits are independent:

$$
\widehat S_k(t_1),\widehat S_k(t_2),\ldots,\widehat S_k(t_n)
$$

can be computed separately because each solves its own one-dimensional optimization
problem.

The notebook splits the window matrix into chunks:

```python
ranges = [
    (start, min(start + chunk_size, n))
    for start in range(0, n, chunk_size)
]
```

Then it processes those chunks with `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for start, values in executor.map(_robust_filter_chunk, args):
        out[start:start + len(values)] = values
```

Each worker computes gradient descent for its subset of local windows.

### Benchmark Cases

The notebook also parallelizes independent benchmark trials. Different contamination
probabilities, contamination scales, and random repeats do not depend on each other, so
they can be run concurrently.

## 13. Why IMF Stages Are Not Parallelized

The decomposition stages cannot be freely parallelized because stage $k+1$ depends on
the residual from stage $k$:

$$
r_{k+1} = r_k - S_k.
$$

So the sequence must be:

1. compute $S_1$ from $r_1$;
2. compute $r_2 = r_1 - S_1$;
3. compute $S_2$ from $r_2$;
4. continue.

Only the local fits inside each fixed stage are independent.

## 14. Relationship to Linear Operator Commutativity

`IMF.pdf` proves that for the linear mean-filter case, the operators $W^{(k)}$ and
$A^{(k)}$ commute under a regular wrap design.

That result does not apply directly to the robust gradient-descent smoother.

Reason: the robust smoother is nonlinear. It is defined by an optimization problem:

$$
\widehat S_k(t)
= \arg\min_x
  \sum_u w_{t,u}\,\rho_H(r_k(u)-x),
$$

not by multiplication with a fixed matrix $W^{(k)}$.

Therefore:

- linear mean IMF can be studied with fixed operators;
- robust gradient-descent IMF must be treated as an iterative nonlinear procedure;
- parallelism comes from independent local optimization problems, not from operator
  commutativity.

## 15. Output

After all stages, the notebook returns:

```python
return {
    "imfs": imfs,
    "residual": residual,
    "reconstruction": imfs.sum(axis=0) + residual,
    "H": float(H),
}
```

The reconstruction check is

$$
\left\| y - \left(\sum_{k=1}^{K}S_k + r_{K+1}\right) \right\|_\infty.
$$

In the notebook run, this is near floating-point precision, which confirms that the
decomposition subtracts and reconstructs consistently.
