"""
Module 17: uncertainty, out-of-distribution detection and active learning.

The "simulation" is the classic pin-fin efficiency formula (aluminium pin, 3 mm diameter):
    efficiency = tanh(mL) / (mL),   m = sqrt(4 h / (k D))
Inputs: convection coefficient h [W/m^2.K] and fin length L [mm]. It's cheap, so we
know the truth everywhere and can check every uncertainty claim.

Training designs cover h = 10-300, L = 5-40 mm. We then ask about:
  in-distribution designs   inside that box
  out-of-distribution (OOD) h = 300-600 and L = 40-80 mm, outside the box

  1. Gaussian process: does its uncertainty grow outside the box? Is it honest?
  2. Deep ensemble of 5 networks: does their disagreement flag OOD designs?
  3. Active learning: choose each new simulation where the GP is least sure,
     vs choosing at random.

Run:  python uq_active.py      (about 1 minute)
"""

import warnings

import numpy as np
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
K_AL, D = 200.0, 0.003
BOX = np.array([[10.0, 300.0], [5.0, 40.0]])          # training range: h, L[mm]
OOD_BOX = np.array([[300.0, 600.0], [40.0, 80.0]])


def fin_efficiency(h, L_mm):
    mL = np.sqrt(4 * h / (K_AL * D)) * L_mm / 1000
    return np.tanh(mL) / mL


def to_unit(X):
    """Scale inputs so the training box is [0, 1]^2 (OOD points land outside it)."""
    return (X - BOX[:, 0]) / (BOX[:, 1] - BOX[:, 0])


def sample(box, n, seed):
    U = qmc.LatinHypercube(d=2, seed=seed).random(n)
    return box[:, 0] + U * (box[:, 1] - box[:, 0])


def fit_gp(X, y):
    kernel = ConstantKernel(1.0) * Matern(length_scale=[0.5, 0.5], nu=2.5) + WhiteKernel(1e-8, (1e-12, 1e-4))
    return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3,
                                    random_state=0).fit(to_unit(X), y)


def gp_stats(gp, X, y):
    mean, std = gp.predict(to_unit(X), return_std=True)
    err = np.abs(mean - y)
    coverage = np.mean(err <= 2 * std)                  # should be ~95% if the std is honest
    return err, std, coverage


def main():
    X_tr = sample(BOX, 15, seed=0)
    y_tr = fin_efficiency(*X_tr.T)
    X_in = sample(BOX, 500, seed=1)
    X_out = sample(OOD_BOX, 500, seed=2)
    y_in, y_out = fin_efficiency(*X_in.T), fin_efficiency(*X_out.T)
    print(f"Training: 15 designs in h = 10-300, L = 5-40 mm; efficiency from "
          f"{y_tr.min():.2f} to {y_tr.max():.2f}\n")

    # ---- 1. Gaussian process uncertainty --------------------------------------
    gp = fit_gp(X_tr, y_tr)
    print("1) Gaussian process, 15 training designs")
    print(f"{'':>22} {'mean |error|':>13} {'mean predicted std':>19} {'truth within +/-2 std':>22}")
    for name, X, y in [("in distribution", X_in, y_in), ("out of distribution", X_out, y_out)]:
        err, std, cov = gp_stats(gp, X, y)
        print(f"{name:>22} {err.mean():13.4f} {std.mean():19.4f} {100 * cov:21.0f}%")
    print("   Outside the box the GP's error grows, and so does its own uncertainty estimate.")
    print("   Check whether that uncertainty is honest: ideally ~95% of truths fall within 2 std.\n")

    # ---- 2. deep ensemble ---------------------------------------------------------
    X_tr40 = sample(BOX, 40, seed=3)
    y_tr40 = fin_efficiency(*X_tr40.T)
    mu, sd = y_tr40.mean(), y_tr40.std()                  # standardize the target (module 10)
    members = [MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=5000, tol=1e-9, random_state=s)
               .fit(to_unit(X_tr40), (y_tr40 - mu) / sd) for s in range(5)]
    print("2) Deep ensemble: 5 networks, same 40 designs, different random starting weights")
    print(f"{'':>22} {'mean |error|':>13} {'mean disagreement':>18} {'error / disagreement':>21}")
    dis = {}
    for name, X, y in [("in distribution", X_in, y_in), ("out of distribution", X_out, y_out)]:
        preds = np.array([m.predict(to_unit(X)) * sd + mu for m in members])
        err = np.abs(preds.mean(0) - y)
        dis[name] = preds.std(0)
        print(f"{name:>22} {err.mean():13.4f} {dis[name].mean():18.4f} {err.mean() / dis[name].mean():20.1f}x")
    threshold = np.percentile(dis["in distribution"], 95)
    flagged = np.mean(dis["out of distribution"] > threshold)
    print(f"   Flag a design when disagreement exceeds the 95th percentile seen in distribution:")
    print(f"   {100 * flagged:.0f}% of the OOD designs get flagged (5% of in-distribution ones, by construction).")
    print("   But look at the last column: outside the box the real error is several times")
    print("   bigger than the disagreement. Ensembles detect OOD well; they size the error badly.\n")

    # ---- 3. active learning -------------------------------------------------------
    print("3) Active learning: 5 starting designs, then add one simulation at a time")
    pool = sample(BOX, 2000, seed=4)
    start = sample(BOX, 5, seed=5)
    budgets = [5, 10, 15, 20, 30]
    curves = {}
    for strategy in ["random", "most uncertain"]:
        rng = np.random.default_rng(6)
        X = start.copy()
        errs = {}
        while len(X) <= budgets[-1]:
            y = fin_efficiency(*X.T)
            g = fit_gp(X, y)
            if len(X) in budgets:
                errs[len(X)] = np.abs(g.predict(to_unit(X_in)) - y_in).max()
            if strategy == "random":
                nxt = pool[rng.integers(len(pool))]
            else:
                _, std = g.predict(to_unit(pool), return_std=True)
                nxt = pool[np.argmax(std)]
            X = np.vstack([X, nxt])
        curves[strategy] = errs
    print(f"{'worst error on 500 test designs':>32} " + " ".join(f"{f'n = {b}':>9}" for b in budgets))
    for strategy, errs in curves.items():
        print(f"{strategy:>32} " + " ".join(f"{errs[b]:9.4f}" for b in budgets))
    print("   Asking the GP where it is least sure puts simulations where they reduce the")
    print("   worst-case error fastest, typically the edges and corners of the design space.")


if __name__ == "__main__":
    main()
