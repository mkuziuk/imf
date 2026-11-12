"""Bias-variance Monte Carlo behind slides 11, 12 and 14 of irmf_talk.

Run from the imf repository root.  Uses the numerical cells of
experiments/real-vs-calculated/gd_imf_real_vs_calculated_limits_noise_only.ipynb
unchanged (signal, window schedule, weights, lookup-grid GD robust fit).

Setting: sigma=0.2, p=0.2, H=2*sigma, contamination C = +-s*Exp(1),
s/sigma in RATIOS, R independent repeats (seeds 910000+r).  For each repeat the
same Gaussian, mask, exponential and sign draws are reused across s (common
random numbers).  Two observation models:
  additive: Y = X + G + M*C
  masked:   Y = G + C where M=1, X + G elsewhere
Errors e_k = IMF_k(Y) - IMF_k(X) use each method's own clean reference.
Output: mc_bias_variance.npz with per-stage bias^2, variance, MSE (averaged
over t) and per-repeat stage RMSE.
"""
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB = "experiments/real-vs-calculated/gd_imf_real_vs_calculated_limits_noise_only.ipynb"
CELLS = [3, 5, 7, 9, 10, 14, 16]
RATIOS = np.array([0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20.0])
R, SIGMA, P = 100, 0.2, 0.2
H = 2 * SIGMA


def load(fn, idx):
    nb = json.load(open(fn))
    ns = {"__name__": "mc", "__file__": fn}
    for i in idx:
        exec(compile("".join(nb["cells"][i]["source"]), f"{fn}:{i}", "exec"), ns)
        ns["display"] = lambda *a, **k: None
        plt.close("all")
    return ns


def main():
    ns = load(NB, CELLS)
    x, ws, n = ns["x_clean"], ns["window_sizes"], ns["n"]
    K = len(ws)

    def lin(y):
        return ns["linear_imf_with_history"](y, ws)["imfs"]

    def rob(y):
        return ns["robust_gd_imf_with_history"](y, ws, H=H, grid=ns["lookup_grid"],
                                                max_iter=60, tol=1e-6, max_workers=12)["imfs"]

    fits = {"linear": lin, "robust": rob}
    refs = {m: f(x) for m, f in fits.items()}
    shape = (2, 2, len(RATIOS), K, n)
    S1, S2 = np.zeros(shape), np.zeros(shape)
    rm = np.zeros((2, 2, len(RATIOS), R, K))
    for r in range(R):
        g = np.random.default_rng(910000 + r)
        G = SIGMA * g.normal(size=n)
        M = g.random(n) < P
        E = g.exponential(size=n) * g.choice([-1.0, 1.0], size=n)
        for si, rt in enumerate(RATIOS):
            C = M * (rt * SIGMA) * E
            ys = [x + G + C, np.where(M, G + C, x + G)]
            for mi, y in enumerate(ys):
                for ki, m in enumerate(fits):
                    e = fits[m](y) - refs[m]
                    S1[mi, ki, si] += e
                    S2[mi, ki, si] += e ** 2
                    rm[mi, ki, si, r] = np.sqrt((e ** 2).mean(1))
    bias, mse = S1 / R, S2 / R
    var = mse - bias ** 2
    np.savez("mc_bias_variance.npz", ratios=RATIOS, bias2=(bias ** 2).mean(-1),
             var=var.mean(-1), mse=mse.mean(-1), rmse_rep=rm)
    # axes: [model (additive, masked), method (linear, robust), scale, stage]


if __name__ == "__main__":
    main()
